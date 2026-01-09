import torch
import typer
from arginator_protein_classifier.data import get_dataloaders
from arginator_protein_classifier.model import Model

app = typer.Typer()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@app.command()
def evaluate(model_checkpoint: str = "models/model.pth", data_path: str = ".data") -> None:
    """Evaluate a trained model."""
    print("Evaluating model performance...")
    
    # 1. Load Model Architecture
    model = Model(input_dim=1024, output_dim=2, dropout_rate=0.2).to(DEVICE)
    
    # 2. Load Weights (Created by train.py)
    try:
        model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file '{model_checkpoint}' not found. Run train.py first!")
        raise typer.Exit(code=1)

    # 3. Load Test Data
    _, _, test_dataloader = get_dataloaders(data_path, batch_size=32)

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