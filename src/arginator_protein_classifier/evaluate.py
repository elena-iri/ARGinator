import torch
import hydra
import logging
import os
from omegaconf import DictConfig
from arginator_protein_classifier.data import get_dataloaders
from arginator_protein_classifier.model import Model

log = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Get path to configs
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
config_path = os.path.join(project_root, "configs")

# We reuse the same config file as training to ensure consistency
@hydra.main(version_base=None, config_path=config_path, config_name="train_config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model using Hydra configuration."""
    
    hparams = cfg.experiment
    paths = cfg.paths
    
    log.info("Evaluating model performance...")
    
    # 1. Load Model Architecture
    # CRITICAL: We use the config values, so the architecture matches exactly what was trained
    model = Model(
        input_dim=1024, 
        output_dim=2, 
        dropout_rate=hparams.dropout_rate 
    ).to(DEVICE)
    
    # 2. Load Weights
    try:
        model.load_state_dict(torch.load(paths.model_filename, map_location=DEVICE))
        log.info(f"Loaded weights from {paths.model_filename}")
    except FileNotFoundError:
        log.error(f"Error: Model file '{paths.model_filename}' not found.")
        log.error("If your model is in a Hydra output folder, provide the full path:")
        log.error("Usage: uv run src/.../evaluate.py paths.model_filename=/path/to/model.pth")
        return

    # 3. Load Test Data
    _, _, test_dataloader = get_dataloaders(paths.data, batch_size=hparams.batch_size)

    # 4. Evaluation Loop
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)
            
    log.info(f"Test Set Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    evaluate()