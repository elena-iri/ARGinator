import torch
import typer
import matplotlib.pyplot as plt
import os
from arginator_protein_classifier.model import Model
from arginator_protein_classifier.data import get_dataloaders

# Initialize Typer app
app = typer.Typer()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@app.command()
def train(
    lr: float = 1e-3,
    dropout_rate: float = 0.2,
    batch_size: int = 32, 
    epochs: int = 5,
    model_path: str = "models/model.pth",
    data_path: str = "data"
) -> None:
    """
    Train the model. 
    Usage: uv run src/arginator_protein_classifier/train.py --lr 0.01 --epochs 20
    """
    print("Training Day and Night")
    print(f"Configuration: {lr=}, {batch_size=}, {epochs=}, {dropout_rate=}")

    # 1. Setup Model (Definitions from model.py)
    model = Model(input_dim=1024, output_dim=2, dropout_rate=dropout_rate).to(DEVICE)

    # 2. Setup Data (Ingredients from data.py)
    train_dataloader, _, _ = get_dataloaders(data_path, batch_size=batch_size)

    # 3. Setup Training Tools
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    statistics = {"train_loss": [], "train_accuracy": []}

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch: {epoch}, iter {i}, loss: {loss.item():.4f}, Acc: {accuracy:.2f}")
    
    print("Training Complete")
    
    # 5. Save Artifacts
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    os.makedirs("reports/figures", exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")
    print("Plot saved to reports/figures/training_statistics.png")

if __name__ == "__main__":
    app()