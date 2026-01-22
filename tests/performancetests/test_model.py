import os
import time
import torch
import logging
from pathlib import Path
import sys
import pytest
from dotenv import load_dotenv
import wandb

# 3. Import DIRECTLY from the package (do NOT use src.arginator...)
from src.arginator_protein_classifier.inference import run_inference
from src.arginator_protein_classifier.model import Lightning_Model

load_dotenv()
log = logging.getLogger(__name__)


def download_model_from_wandb(artifact_path: str, download_dir: str = None) -> str:
    """
    Download a model checkpoint from Weights & Biases.
    
    Args:
        artifact_path: W&B artifact path in format "entity/project/artifact_name:version"
        download_dir: Directory to download to. Defaults to temp directory.
    
    Returns:
        Path to the downloaded checkpoint file.
    """
    if download_dir is None:
        download_dir = Path("/tmp/wandb_artifacts")
    else:
        download_dir = Path(download_dir)
    
    download_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
        log.info(f"Downloading artifact from wandb: {artifact_path}")
        artifact = api.artifact(artifact_path)
        artifact_dir = artifact.download(root=str(download_dir))
        
        # Find the .ckpt file in the downloaded directory
        ckpt_files = list(Path(artifact_dir).glob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt files found in {artifact_dir}")
        
        ckpt_path = str(ckpt_files[0])
        log.info(f"Model checkpoint downloaded to: {ckpt_path}")
        return ckpt_path
    
    except Exception as e:
        log.error(f"Failed to download model from wandb: {e}")
        raise


@pytest.fixture
def model_checkpoint():
    """Load a model checkpoint for testing, downloading from wandb if needed."""
    # First check if local path is available
    ckpt_path = os.getenv("TEST_CHECKPOINT")
    
    # If not, try to download from wandb
    if not ckpt_path or not os.path.exists(ckpt_path):
        wandb_artifact = os.getenv("WANDB_ARTIFACT")
        if wandb_artifact:
            try:
                ckpt_path = download_model_from_wandb(wandb_artifact)
            except Exception as e:
                pytest.skip(f"Could not download model from wandb: {e}")
        else:
            # Fall back to default local path
            log.warning("Failed to load the model")
            pytest.skip(f"Checkpoint not found")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Lightning_Model.load_from_checkpoint(ckpt_path, weights_only=False)
    model.to(DEVICE)
    model.eval()
    return model


@pytest.fixture
def sample_data():
    """Provide sample embeddings for testing."""
    # Return sample embeddings matching the model's expected input dimension
    return torch.randn(10, 1024)  # Adjust dimension based on your model


def test_model_inference_speed(model_checkpoint, sample_data):
    """Test that model inference is fast enough."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint.to(DEVICE)
    
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model_checkpoint(sample_data.to(DEVICE))
    end = time.time()
    
    inference_time = end - start
    log.info(f"100 inferences completed in {inference_time:.4f} seconds")
    
    # Assert inference is fast enough (adjust threshold as needed)
    assert inference_time < 10, f"Inference too slow: {inference_time}s"


def test_run_inference_function(model_checkpoint):
    """Test the run_inference function with sample data."""
    ckpt_path = os.getenv(
        "TEST_CHECKPOINT",
        "/Users/emilianotorres/DTU_Masters/ARGinator/src/arginator_protein_classifier/outputs/2026-01-17/13-55-21/best-checkpoint.ckpt"
    )
    
    # You'll need sample .h5 files for this test
    data_path = os.getenv("TEST_DATA_PATH")
    if not data_path or not os.path.exists(data_path):
        pytest.skip("TEST_DATA_PATH not set or data not available")
    
    output_dir = Path("/tmp/inference_test_output")
    output_dir.mkdir(exist_ok=True)
    
    start = time.time()
    run_inference(
        checkpoint_path=ckpt_path,
        data_path=data_path,
        batch_size=32,
        output_dir=str(output_dir),
        job_id="test_run"
    )
    end = time.time()
    
    log.info(f"run_inference completed in {end - start:.4f} seconds")
    
    # Check output file was created
    output_file = output_dir / "test_run_results.csv"
    assert output_file.exists(), f"Output file not created at {output_file}"


def test_model_output_shape(model_checkpoint, sample_data):
    """Test that model outputs have the expected shape."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint.to(DEVICE)
    
    with torch.no_grad():
        output = model_checkpoint(sample_data.to(DEVICE))
    
    expected_output_dim = model_checkpoint.hparams.output_dim
    assert output.shape == (sample_data.shape[0], expected_output_dim), \
        f"Output shape {output.shape} doesn't match expected ({sample_data.shape[0]}, {expected_output_dim})"
    
    log.info(f"Model output shape: {output.shape}")


if __name__ == "__main__":
    pytest.main()