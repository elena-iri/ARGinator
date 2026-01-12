import torch
import typer

from arginator_protein_classifier.data import get_dataloaders
from arginator_protein_classifier.model import Model

log = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@app.command()
def evaluate(model_checkpoint: str = "models/model.pth", data_path: str = "data") -> None:
    """Evaluate a trained model."""
    print("Evaluating model performance...")

    # 1. Load Model Architecture
    model = Model(input_dim=1024, output_dim=2, dropout_rate=0.2).to(DEVICE)

    # 2. Load Weights (Created by train.py)
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

    print(f"Test Set Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    app()
