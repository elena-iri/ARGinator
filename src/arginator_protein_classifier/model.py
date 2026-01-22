import logging
import os

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from matplotlib.pylab import logistic
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy, F1Score, Precision, Recall

# class Model(nn.Module):
#     def __init__(self, input_dim, output_dim, dropout_rate)->None:
#         super().__init__()
#         self.input_dim = input_dim,
#         self.output_dim = output_dim,
#         self.fc1 = nn.Linear(1024, 256)
#         self.dropout=nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, output_dim)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x


class Lightning_Model(LightningModule):
    def __init__(self, input_dim, output_dim, dropout_rate, loss_fn, optimizer_config) -> None:
        super().__init__()
        self.save_hyperparameters()  # save args for checkpointing
        # architecture
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=1)
        # Loss & Optimizer are to change according to hydra
        self.loss_fn = instantiate(loss_fn)
        self.optimizer_config = optimizer_config

        # --- FIX: Dynamic Task Definition ---
        if output_dim == 2:
            task_type = "binary"
            # num_classes is ignored for binary in newer torchmetrics versions,
            # but good to keep clean kwargs
            metrics_kwargs = {"task": "binary"}
        else:
            task_type = "multiclass"
            metrics_kwargs = {"task": "multiclass", "num_classes": output_dim}

        # Metrics
        self.train_acc = Accuracy(**metrics_kwargs)
        self.train_prec = Precision(**metrics_kwargs)
        self.train_rec = Recall(**metrics_kwargs)
        self.train_f1 = F1Score(**metrics_kwargs)

        self.val_acc = Accuracy(**metrics_kwargs)
        self.val_prec = Precision(**metrics_kwargs)
        self.val_rec = Recall(**metrics_kwargs)
        self.val_f1 = F1Score(**metrics_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        logits = self(data)
        loss = self.loss_fn(logits, target)

        # Logging
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, target)
        self.train_prec(preds, target)
        self.train_rec(preds, target)
        self.train_f1(preds, target)

        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": self.train_acc,
                "train_precision": self.train_prec,
                "train_recall": self.train_rec,
                "train_f1": self.train_f1,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_acc(preds, y)
        self.val_prec(preds, y)
        self.val_rec(preds, y)
        self.val_f1(preds, y)

        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": self.val_acc,
                "val_precision": self.val_prec,
                "val_recall": self.val_rec,
                "val_f1": self.val_f1,
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_config, params=self.parameters())
        return optimizer

    # Get absolute path to configs


current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
config_path = os.path.join(project_root, "configs")


@hydra.main(version_base=None, config_path=config_path, config_name="train_config")

def main(cfg: DictConfig) -> None:
    x = torch.rand(1, 1024)
    print(x.shape[1])
    loss_fn = cfg.experiment.loss_function
    optimizer = cfg.optimizer
    mock = Lightning_Model(x.shape[1], 2, 0.3, loss_fn, optimizer)
    output = mock.forward(x)
    print(output.shape)


if __name__ == "__main__":
    main()
