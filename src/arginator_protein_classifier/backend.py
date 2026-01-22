from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
import uvicorn
import os
import shutil
import uuid
import pandas as pd
from hydra import compose, initialize
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
from contextlib import asynccontextmanager
import wandb

# --- IMPORTS ---
from src.arginator_protein_classifier.convertfa import load_t5_model, run_conversion
from src.arginator_protein_classifier.inference import run_inference
from src.arginator_protein_classifier.umap_plot import UMAPEmbeddingVisualizer
from src.arginator_protein_classifier.train import get_secret

# Global variables
MODEL = None
VOCAB = None
CFG: DictConfig = None
JOBS = {}

# Fetch key and set it as Env Var so WandB finds it automatically
try:
    # Only fetch if not already set (allows local runs to still work)
    if "WANDB_API_KEY" not in os.environ:
        print("Fetching WandB key from Secret Manager...")
        api_key = get_secret("arginator", "WANDB_API_KEY")
        os.environ["WANDB_API_KEY"] = api_key.strip() # .strip() removes accidental newlines
        wandb.login(key=api_key.strip())
        
except Exception as e:
    print(f"Could not fetch secret: {e}")

# --- NEW LIFESPAN WAY ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup Logic (Runs before app starts)
    global MODEL, VOCAB, CFG
    
    print("Initializing Hydra...")
        # 1. Resolve Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # 2. Initialize Hydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Example: "paths.data=/gcs/bucket;paths.model_dir=/gcs/bucket/models"
    overrides_str = os.environ.get("HYDRA_OVERRIDES", "")
    overrides = [o.strip() for o in overrides_str.split(";")] if overrides_str else []

    with initialize(version_base=None, config_path="../../configs"):
        # Pass the overrides here
        CFG = compose(config_name="train_config", overrides=overrides)

    print(f"Loaded Config with overrides: {overrides}")

    print("Loading T5 Model...")
    MODEL, VOCAB = await run_in_threadpool(load_t5_model, model_dir=CFG.paths.t5_model_dir)
    print("Startup Complete.")
    
    yield  # Hand over control to the application
    
    # 2. Shutdown Logic (Runs when app stops)
    # If you had any DB connections or pools to close, you would do it here.
    print("Shutting down...")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

# Pass lifespan into the app creation
app = FastAPI(lifespan=lifespan)

def update_job_progress_scaled(job_id, current, total, start_percent, end_percent):
    if total > 0:
        raw_fraction = current / total
        # Calculate how much of the "global" bar this task occupies
        span = end_percent - start_percent
        scaled_progress = int(start_percent + (raw_fraction * span))
        JOBS[job_id]["progress"] = scaled_progress

def process_file_task(job_id: str, temp_fa: str, saved_h5: str, classification_type: str):
    try:
        # STEP 1: CONVERSION (0% -> 60%)
        # We tell the callback to map the conversion progress to the 0-60 range
        progress_callback = lambda c, t: update_job_progress_scaled(job_id, c, t, 0, 60)

        print(f"[{job_id}] Starting Conversion...")
        run_conversion(
            seq_path=temp_fa, 
            emb_path=saved_h5, 
            model=MODEL, 
            vocab=VOCAB,
            per_protein=True,
            callback=progress_callback
        )

        # STEP 2: INFERENCE (60% -> 80%)
        JOBS[job_id]["progress"] = 65 # Jump to 65% to show step change

        ENTITY = "eleni-iriondo2-danmarks-tekniske-universitet-dtu-org"
        PROJECT = "wandb-registry-arginator_models"

        if classification_type == 'Multiclass':
            # Format: entity/project/collection
            model_collection_name = f"{ENTITY}/{PROJECT}/multiclass_models"
        else:
            model_collection_name = f"{ENTITY}/{PROJECT}/binary_models"

        # 2. Download the latest artifact
        try:
            print(f"Downloading latest model from: {model_collection_name}...")
            
            api = wandb.Api()
            
            # Pass the full path here
            artifact = api.artifact(f"{model_collection_name}:latest", type="model")
            
            # Initialize API (make sure you are logged in or env var WANDB_API_KEY is set)
            api = wandb.Api()
            
            # Fetch the artifact. 'latest' is the alias for the most recent version.
            artifact = api.artifact(f"{model_collection_name}:latest", type="model")
            
            # Download to the specific directory your config expects
            download_dir = artifact.download(root=".models")
            
            # WandB often preserves the filename you saved it with (e.g., "best-checkpoint.ckpt")
            # You might need to find the .ckpt file if the name isn't fixed to "model.ckpt"
            downloaded_files = [f for f in os.listdir(download_dir) if f.endswith(".ckpt")]
            
            if not downloaded_files:
                raise FileNotFoundError("No .ckpt file found in the downloaded artifact.")
                
            # Use the downloaded file
            weights_path = os.path.join(download_dir, downloaded_files[0])
            print(f"Model downloaded to: {weights_path}")

        except wandb.errors.CommError as e:
            raise RuntimeError(f"Failed to access WandB registry. Check your API key and permissions. Error: {e}")

        run_inference(
            checkpoint_path=weights_path,
            data_path=saved_h5,
            output_dir=CFG.paths.data_inference_dir,
            job_id=job_id,
        )
        
        output_csv = os.path.join(CFG.paths.data_inference_dir, f"{job_id}_results.csv")
        
        if not os.path.exists(output_csv):
            raise FileNotFoundError(f"Inference failed to generate output file: {output_csv}")

        df = pd.read_csv(output_csv)
        
      
        JOBS[job_id]["progress"] = 80 # Update progress before starting UMAP
        
        file_map = {
            "card_A": CFG.paths.card_A,
            "card_B": CFG.paths.card_B,
            "card_C": CFG.paths.card_C,
            "card_D": CFG.paths.card_D,
            "query": saved_h5,
        }

        visualizer = UMAPEmbeddingVisualizer()
        
        # Determine output path
        umap_output = os.path.join(CFG.paths.data_inference_dir, f"{job_id}_umap_plot.png")
        
        visualizer.run(
            file_map, 
            binary_mode=(classification_type == "Binary"), 
            output_filename=umap_output
        )

        # STEP 4: COMPLETE (100%)
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100 # Finally hit 100%
        
        JOBS[job_id]["result"] = {
            "classification_type": classification_type,
            "csv_path": output_csv,
            "preview": df.head(5).to_dict() 
        }

    except Exception as e:
        print(f"[{job_id}] Error: {e}")
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
    
    finally:
        if os.path.exists(temp_fa): 
            os.remove(temp_fa)
@app.post("/submit_job")
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    classification_type: str = Form(...)
):
    job_id = str(uuid.uuid4())
    
    # Ensure inference dir exists
    os.makedirs(CFG.paths.data_inference_dir, exist_ok=True)

    temp_fa = os.path.join(CFG.paths.data_inference_dir, f"{job_id}_{file.filename}")
    saved_h5 = os.path.join(CFG.paths.data_inference_dir, f"{job_id}.h5")

    try:
        with open(temp_fa, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    JOBS[job_id] = {
        "status": "processing", 
        "progress": 0, 
        "result": None,
        "filename": file.filename
    }

    background_tasks.add_task(
        process_file_task, job_id, temp_fa, saved_h5, classification_type
    )

    return {"job_id": job_id, "status": "submitted"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """New endpoint to download the CSV results"""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not finished yet")

    result = job.get("result", {})
    csv_path = result.get("csv_path")

    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=500, detail="Result file not found on server")

    return FileResponse(
        path=csv_path, 
        filename=f"{job_id}_results.csv", 
        media_type='text/csv'
    )

@app.get("/download_plot/{job_id}")
async def download_plot(job_id: str):
    """Endpoint to retrieve the generated UMAP plot."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not finished yet")

    # Construct the expected path based on your process_file_task naming convention
    plot_path = os.path.join(CFG.paths.data_inference_dir, f"{job_id}_umap_plot.png")

    if not os.path.exists(plot_path):
        # It's possible the plot failed even if the job succeeded, handle gracefully
        raise HTTPException(status_code=404, detail="Plot file not found on server")

    return FileResponse(plot_path, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)