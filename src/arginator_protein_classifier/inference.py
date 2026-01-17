import logging
import os
import torch
import h5py
import pandas as pd
import hydra
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from arginator_protein_classifier.model import Lightning_Model
from omegaconf import DictConfig, ListConfig

torch.serialization.add_safe_globals([DictConfig, ListConfig])
log = logging.getLogger(__name__)

# Helper to find config path (needed for the CLI wrapper)
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
config_path = os.path.join(project_root, "configs")

# --- 1. THE DATASET CLASS ---
class InferenceDataset(Dataset):
    def __init__(self, h5_path: Path):
        self.h5_path = Path(h5_path)
        self.samples = []
        if not self.h5_path.exists():
            raise FileNotFoundError(f"File {self.h5_path} does not exist.")
            
        with h5py.File(self.h5_path, "r") as hf:
            for key in hf.keys():
                data_numpy = hf[key][:]
                embedding = torch.from_numpy(data_numpy).float()
                self.samples.append((embedding, key))     
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

load_dotenv() 

# --- 2. THE PURE PYTHON FUNCTION (Importable) ---
def run_inference(checkpoint_path: str, data_path: str, batch_size: int = 32, output_dir: str = None):
    """
    Pure Python function to run inference. 
    Can be imported and called from any other script.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check paths
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load Model
    log.info(f"Loading model from {checkpoint_path}...")
    model = Lightning_Model.load_from_checkpoint(checkpoint_path, weights_only = False)
    model.to(DEVICE)
    model.eval()

    # AUTO-DETECT TASK (No config needed!)
    # We use the model's saved hyperparameters to determine the task
    output_dim = model.hparams.output_dim
    task_name = "binary" if output_dim == 2 else "multiclass"
    log.info(f"Auto-detected task: {task_name} (output_dim={output_dim})")

    # Define Label Map based on detected task
    label_map = {}
    if task_name == "binary":
        label_map = {0: "Non-Betalactamase", 1: "Betalactamase"}
    else:
        # For multiclass, we still need the names from ENV or fallback
        classes_str = os.environ.get("CLASSES", "")
        if classes_str:
            class_names = [c.strip() for c in classes_str.split(",")]
            label_map = {i: name for i, name in enumerate(class_names)}
        else:
            log.warning("Multiclass detected but no CLASSES in .env. Using numeric labels.")
            label_map = {i: f"Class_{i}" for i in range(output_dim)}

    # Handle Input Data (File vs Folder)
    path_obj = Path(data_path)
    h5_files = [path_obj] if path_obj.is_file() else list(path_obj.glob("*.h5"))
    
    if not h5_files:
        log.error(f"No .h5 files found in {data_path}")
        return

    # Loop through files
    for h5_file in h5_files:
        log.info(f"Processing {h5_file.name}...")
        dataset = InferenceDataset(h5_file)
        # num_workers=0 is safer for simple scripts to avoid multiprocess overhead/bugs
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

        results = []
        with torch.no_grad():
            for batch in dataloader:
                embeddings, ids = batch
                embeddings = embeddings.to(DEVICE)
                
                logits = model(embeddings)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                probs_np = probs.cpu().numpy()
                preds_np = preds.cpu().numpy()
                
                for i in range(len(ids)):
                    pred_idx = preds_np[i]
                    row = {
                        "protein_id": ids[i],
                        "predicted_label": label_map.get(pred_idx, f"Class_{pred_idx}"),
                        "confidence_score": probs_np[i][pred_idx]
                    }
                    if task_name == "binary":
                        row["prob_betalactamase"] = probs_np[i][1]
                    results.append(row)

        # Save output
        save_dir = Path(output_dir) if output_dir else path_obj.parent
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            
        output_csv = save_dir / f"predictions_{h5_file.stem}_{task_name}.csv"
        pd.DataFrame(results).to_csv(output_csv, index=False)
        log.info(f"Saved predictions to {output_csv}")


# --- 3. THE HYDRA WRAPPER (CLI Only) ---
@hydra.main(version_base=None, config_path=config_path, config_name="train_config")
def main(cfg: DictConfig):
    """
    Wrapper for command line usage. 
    Usage: python inference.py
    """
    # Define defaults or read from config if you want
    # For now, let's hardcode the path logic or read from a specific cfg field if it existed
    # But since you want to pass arguments manually in other scripts, 
    # we usually just point to the constants or CLI args here.
    
    # You can update these strings manually for CLI runs, 
    # OR add them to your config.yaml
    ckpt = "/Users/emilianotorres/DTU_Masters/ARGinator/src/arginator_protein_classifier/outputs/2026-01-17/13-55-21/best-checkpoint.ckpt"
    data = "/Users/emilianotorres/Downloads/"
    
    run_inference(
        checkpoint_path=ckpt,
        data_path=data,
        batch_size=cfg.experiment.batch_size,
    )

if __name__ == "__main__":
    main()