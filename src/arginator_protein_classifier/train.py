import torch
import wandb
import hydra
import logging
import os
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

    run = wandb.init(
        project="arginator_protein_classifier",
        config={"lr": hparams.lr,
                "dropout_rate": hparams.dropout_rate,
                "batch_size": hparams.batch_size,
                "epochs": hparams.epochs,
                "seed": hparams.seed},
    )

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

            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            if i % 100 == 0:
                log.info(f"Epoch: {epoch}, iter {i}, loss: {loss.item():.4f}, Acc: {accuracy:.2f}")
                
                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads)})
    
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

    #Saving artifacts to wandb

    final_accuracy = accuracy_score(target, y_pred.argmax(dim=1))
    final_precision = precision_score(target, y_pred.argmax(dim=1), average="weighted")
    final_recall = recall_score(target, y_pred.argmax(dim=1), average="weighted")
    final_f1 = f1_score(target, y_pred.argmax(dim=1), average="weighted")

    # first we save the model to a file then log it as an artifact
    #torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="arginator_binary_classifier_model",
        type="model",
        description="A model trained to classify ProtT5 protein embedddings into beta-lactamase or non-beta-lactamases",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file(model_save_path)
    run.log_artifact(artifact)

if __name__ == "__main__":
    train()