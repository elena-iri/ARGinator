import logging
import os
from re import split
from dotenv import load_dotenv
import hydra
import matplotlib.pyplot as plt
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

import wandb
from arginator_protein_classifier.data import TL_Dataset
from arginator_protein_classifier.model import Lightning_Model

from google.cloud import secretmanager

def get_secret(project_id, secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

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

# from arginator_protein_classifier.model import Model
# from arginator_protein_classifier.data import get_dataloaders
# 1. Setup Logger
log = logging.getLogger(__name__)
load_dotenv()
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

    wandb_logger = WandbLogger(
        entity = os.environ["WANDB_ENTITY"],
        project=os.environ["WANDB_PROJECT"],
        config=OmegaConf.to_container(cfg.experiment, resolve=True),
        log_model=True,  # Automatically logs model checkpoints
    )

    # run = wandb.init(
    #     project="arginator_protein_classifier",
    #     #config={"lr": hparams.lr,
    #      #       "dropout_rate": hparams.dropout_rate,
    #      #       "batch_size": hparams.batch_size,
    #       #      "epochs": hparams.epochs,
    #        #     "seed": hparams.seed},
    #     config=OmegaConf.to_container(cfg.experiment, resolve=True),
    # )
    log.info("Training Day and Night")

    root_data_folder = os.path.join(project_root, ".data")

    data = TL_Dataset(
        data_path=cfg.paths.data,
        task=cfg.task.name,
        batch_size=cfg.experiment.batch_size,
        split_ratios=cfg.experiment.splits,
        seed=cfg.processing.seed,
        output_folder=root_data_folder,
    )

    model = Lightning_Model(
        input_dim=1024,
        output_dim=cfg.task.output_dim,
        dropout_rate=cfg.experiment.dropout_rate,
        optimizer_config=cfg.optimizer,
        loss_fn=cfg.experiment.loss_function,
    )

    # 3. callbacks (Optional but recommended)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=output_dir,
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
    )

    # Trainer. We can also use hydra to instantiate the Trainer by modifying config file (too lazy to do it)
    trainer = Trainer(
        max_epochs=cfg.experiment.epochs,
        accelerator="auto",  # Automatically detects GPU/MPS/CPU
        devices=1,
        default_root_dir=output_dir,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )

    trainer.fit(model, data)

    log.info("Training Complete")

    # 7. Post-Training Evaluation
    # Retrieve dataloaders from the DataModule for final manual eval
    # (Assuming TL_Dataset has these methods/properties, typical for DataModules)
    is_binary = cfg.task.output_dim == 2
    avg_method = "binary" if is_binary else "macro"

    if hasattr(data, "test_dataloader"):
        test_loader = data.test_dataloader()
    else:
        # Fallback if TL_Dataset constructs them differently
        data.setup()
        test_loader = data.test_dataloader()

    # Calculate metrics using the helper function
    test_metrics = evaluate(model, test_loader, DEVICE, average_method=avg_method)

    final_test_accuracy = test_metrics["accuracy"]
    final_test_precision = test_metrics["precision"]
    final_test_recall = test_metrics["recall"]
    final_test_f1 = test_metrics["f1"]

    log.info(
        f"Final Test Metrics: Accuracy: {final_test_accuracy:.4f}, Precision: {final_test_precision:.4f}, Recall: {final_test_recall:.4f}, F1: {final_test_f1:.4f}"
    )
    best_model_path = checkpoint_callback.best_model_path

    if best_model_path:
        # 2. Create an artifact for the registry
        artifact = wandb.Artifact(
            name="arginator_production_model", # The name in the registry
            type="model",
            description="Best model from training run",
            metadata=test_metrics # Optional: attach metrics to the artifact
        )
        
        # 3. Add the file and log it
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)
        
        # 4. Link it to the Registered Model collection
        # Replace 'my-registry' with your desired registry name
        wandb.run.link_artifact(artifact, f"wandb-registry-arginator_models/{cfg.task.name}_models")
        
        log.info(f"Best model linked to registry: arginator_registry")
    #WandB Summary
    wandb.summary["test_accuracy"] = final_test_accuracy
    wandb.summary["test_f1"] = final_test_f1

    # 8. Plot ROC Curve
    if is_binary:
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(
            y_true=test_metrics["targets"].cpu(),
            y_score=test_metrics["logits"][:, 1].cpu(),  # assuming binary classification positive class index 1
            name="Final ROC curve (test set)",
            ax=ax,
            plot_chance_level=True,
        )
        wandb.log({"roc_curve": wandb.Image(fig)})
        plt.close(fig)
    else:
        log.info("Skipping Binary ROC Curve plot for multiclass setting.")

    # Note: WandbLogger with log_model=True automatically handles artifact saving for the best model.


def evaluate(model, dataloader, device, average_method="binary"):
    model.eval()
    model.to(device)

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            all_logits.append(logits)
            all_targets.append(y)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    preds = logits.argmax(dim=1)

    metrics = {
        "accuracy": accuracy_score(targets.cpu(), preds.cpu()),
        "precision": precision_score(targets.cpu(), preds.cpu(), average=average_method),
        "recall": recall_score(targets.cpu(), preds.cpu(), average=average_method),
        "f1": f1_score(targets.cpu(), preds.cpu(), average=average_method),
        "logits": logits,
        "targets": targets,
    }
    return metrics


if __name__ == "__main__":
    train()