import logging
import os
from re import split

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
        project="arginator_protein_classifier",
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

    data = TL_Dataset(
        data_path=cfg.paths.data,
        task=cfg.task.name,
        batch_size=cfg.experiment.batch_size,
        split_ratios=cfg.experiment.splits,
        seed=cfg.processing.seed,
        output_folder=cfg.paths.data,
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

    # Update WandB Summary
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
    # # SETUP MODEL
    # model = Model(
    #     input_dim=1024,
    #     output_dim=cfg.task.output_dim,
    #     dropout_rate=hparams.dropout_rate
    # ).to(DEVICE)

    # # SETUP DATA
    # train_dataloader, _, _ = get_dataloaders(
    #     cfg.paths.data,
    #     task = cfg.task.name,
    #     batch_size=hparams.batch_size,
    #     seed=cfg.processing.seed
    # )

    # #loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = instantiate(cfg.experiment.loss_function)
    # #optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    # optimizer = instantiate(cfg.optimizer, params=model.parameters())
    # statistics = {"train_loss": [], "train_accuracy": []}

    # # TRAINING LOOP
    # for epoch in range(hparams.epochs):
    #     model.train()
    #     for i, (img, target) in enumerate(train_dataloader):
    #         img, target = img.to(DEVICE), target.to(DEVICE)

    #         optimizer.zero_grad()
    #         y_pred = model(img)
    #         loss = loss_fn(y_pred, target)
    #         loss.backward()
    #         optimizer.step()

    #         statistics["train_loss"].append(loss.item())
    #         accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
    #         statistics["train_accuracy"].append(accuracy)

    #         if i % 100 == 0:
    #             log.info(f"Epoch: {epoch}, iter {i}, loss: {loss.item():.4f}, Acc: {accuracy:.2f}")
#     # SETUP DATA
#     train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
#         cfg.paths.data,
#         batch_size=hparams.batch_size,
#         seed=cfg.processing.seed
#     )

#     loss_fn = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
#     statistics = {"train_loss": [], "train_accuracy": [], "val_accuracy": [], "val_loss": []}

#     # TRAINING LOOP
#     for epoch in range(hparams.epochs):
#         for i, (img, target) in enumerate(train_dataloader):
#             model.train()
#             img, target = img.to(DEVICE), target.to(DEVICE)

#             optimizer.zero_grad()
#             y_pred = model(img)
#             loss = loss_fn(y_pred, target)
#             loss.backward()
#             optimizer.step()

#             statistics["train_loss"].append(loss.item())
#             accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
#             statistics["train_accuracy"].append(accuracy)

#             wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

#             if i % 100 == 0:
#                 log.info(f"Epoch: {epoch}, iter {i}, loss: {loss.item():.4f}, Acc: {accuracy:.2f}")

#                 # add a plot of histogram of the gradients
#                 grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
#                 wandb.log({"gradients": wandb.Histogram(grads)})

#         #evalutaion after each epoch
#         model.eval()
#         #correct, total = 0, 0
#         with torch.no_grad():
#             for val_img, val_target in val_dataloader:
#                 val_img, val_target = val_img.to(DEVICE), val_target.to(DEVICE)
#                 y_pred_val = model(val_img)
#                 val_loss = loss_fn(y_pred_val, val_target)
#                     #correct += (y_pred_test.argmax(dim=1) == test_target).float().sum().item()
#                     #total += test_target.size(0)
#                 statistics["val_loss"].append(val_loss.item())
#                 val_accuracy = (y_pred_val.argmax(dim=1) == val_target).float().mean().item()
#                 statistics["val_accuracy"].append(val_accuracy)

#                 wandb.log({"val_accuracy": val_accuracy, "val_loss": val_loss.item()})

# log.info("Training Complete")

# # SAVE ARTIFACTS TO HYDRA FOLDER
# # Force the filename to be inside the output_dir
# # cfg.paths.model_filename is just "model.pth"
# training_filename = f"model_{cfg.task.name}.pth"
# model_save_path = os.path.join(output_dir, training_filename)

# torch.save(model.state_dict(), model_save_path)
# log.info(f"Model saved to {model_save_path}")

# # Save Plot to the same folder
# training_plot_filename = f"training_plot_{cfg.task.name}.png"
# plot_save_path = os.path.join(output_dir, training_plot_filename)

# fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# axs[0].plot(statistics["train_loss"])
# axs[0].set_title("Train loss")
# axs[1].plot(statistics["train_accuracy"])
# axs[1].set_title("Train accuracy")

# fig.savefig(plot_save_path)
# log.info(f"Plot saved to {plot_save_path}")

# Calculate final metrics on training and tesst data
#     train_metrics = evaluate(model, train_dataloader, DEVICE)

#     final_train_accuracy = train_metrics["accuracy"]
#     final_train_precision = train_metrics["precision"]
#     final_train_recall = train_metrics["recall"]
#     final_train_f1 = train_metrics["f1"]

#     test_metrics = evaluate(model, test_dataloader, DEVICE)

#     final_test_accuracy = test_metrics["accuracy"]
#     final_test_precision = test_metrics["precision"]
#     final_test_recall = test_metrics["recall"]
#     final_test_f1 = test_metrics["f1"]

#     val_metrics = evaluate(model, val_dataloader, DEVICE)

#     final_val_accuracy = val_metrics["accuracy"]
#     final_val_precision = val_metrics["precision"]
#     final_val_recall = val_metrics["recall"]
#     final_val_f1 = val_metrics["f1"]

#     log.info(f"Final Training Metrics: Accuracy: {final_train_accuracy:.4f}, Precision: {final_train_precision:.4f}, Recall: {final_train_recall:.4f}, F1: {final_train_f1:.4f}")
#     log.info(f"Final Test Metrics: Accuracy: {final_test_accuracy:.4f}, Precision: {final_test_precision:.4f}, Recall: {final_test_recall:.4f}, F1: {final_test_f1:.4f}")
#     log.info(f"Final Validation Metrics: Accuracy: {final_val_accuracy:.4f}, Precision: {final_val_precision:.4f}, Recall: {final_val_recall:.4f}, F1: {final_val_f1:.4f}")

#     fig, ax = plt.subplots()
#     RocCurveDisplay.from_predictions(
#         y_true=test_metrics["targets"].cpu(),
#         y_score=test_metrics["logits"][:, 1].cpu(),  # positive class
#         name="Final ROC curve (test set)",
#         ax=ax,
#         plot_chance_level=True,
#     )


#         # alternatively use wandb.log({"roc": wandb.Image(plt)}
#     wandb.log({"roc_curve": wandb.Image(fig)})
#     plt.close(fig)  # close the plot to avoid memory leaks and overlapping figures


#     #Saving artifacts to wandb

#     # first we save the model to a file then log it as an artifact
#     #torch.save(model.state_dict(), "model.pth")
#     artifact = wandb.Artifact(
#         name="arginator_binary_classifier_model",
#         type="model",
#         description="A model trained to classify ProtT5 protein embedddings into beta-lactamase or non-beta-lactamases",
#         metadata={"train_accuracy": final_train_accuracy, "train_precision": final_train_precision, "train_recall": final_train_recall, "train_f1": final_train_f1, "test_accuracy": final_test_accuracy, "test_precision": final_test_precision, "test_recall": final_test_recall, "test_f1": final_test_f1, "val_accuracy": final_val_accuracy, "val_precision": final_val_precision, "val_recall": final_val_recall, "val_f1": final_val_f1},
#     )
#     artifact.add_file(model_save_path)
#     run.log_artifact(artifact)


# def evaluate(model, dataloader, device):
#     model.eval()

#     all_logits = []
#     all_targets = []

#     with torch.no_grad():
#         for x, y in dataloader:
#             x = x.to(device)
#             y = y.to(device)

#             logits = model(x)

#             all_logits.append(logits)
#             all_targets.append(y)

#     logits = torch.cat(all_logits, dim=0)
#     targets = torch.cat(all_targets, dim=0)

#     # predictions
#     preds = logits.argmax(dim=1)

#     metrics = {
#         "accuracy": accuracy_score(targets.cpu(), preds.cpu()),
#         "precision": precision_score(targets.cpu(), preds.cpu(), average="binary"),
#         "recall": recall_score(targets.cpu(), preds.cpu(), average="binary"),
#         "f1": f1_score(targets.cpu(), preds.cpu(), average="binary"),
#         "logits": logits,      # keep for ROC
#         "targets": targets,
#     }

#     return metrics
