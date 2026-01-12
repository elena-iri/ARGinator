import torch
import hydra
import logging
import os
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from arginator_protein_classifier.model import Model
from arginator_protein_classifier.data import get_dataloaders

# 1. Setup Logger
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# Get absolute path to configs
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
config_path = os.path.join(project_root, "configs")

@hydra.main(version_base=None, config_path=config_path, config_name="train_config")
def train(cfg: DictConfig) -> None:
    """Train the model using Hydra configuration."""
    
    # Get the hydra output directory
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Saving outputs to: {output_dir}")

    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    hparams = cfg.experiment
    torch.manual_seed(hparams.seed)
    
    log.info("Training Day and Night")
    
    # SETUP MODEL
    model = Model(
        input_dim=1024, 
        output_dim=2, 
        dropout_rate=hparams.dropout_rate
    ).to(DEVICE)

    # SETUP DATA
    train_dataloader, _, _ = get_dataloaders(
        cfg.paths.data, 
        batch_size=hparams.batch_size,
        seed=cfg.processing.seed
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    statistics = {"train_loss": [], "train_accuracy": []}

    # TRAINING LOOP
    for epoch in range(hparams.epochs):
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
                log.info(f"Epoch: {epoch}, iter {i}, loss: {loss.item():.4f}, Acc: {accuracy:.2f}")
    
    log.info("Training Complete")
    
    # SAVE ARTIFACTS TO HYDRA FOLDER
    # Force the filename to be inside the output_dir
    # cfg.paths.model_filename is just "model.pth"
    model_save_path = os.path.join(output_dir, cfg.paths.model_filename)
    
    torch.save(model.state_dict(), model_save_path)
    log.info(f"Model saved to {model_save_path}")
    
    # Save Plot to the same folder
    plot_save_path = os.path.join(output_dir, "training_statistics.png")
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    
    fig.savefig(plot_save_path)
    log.info(f"Plot saved to {plot_save_path}")

if __name__ == "__main__":
    train()