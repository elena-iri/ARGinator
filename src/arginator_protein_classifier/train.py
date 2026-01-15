from re import split
import torch
import hydra
import logging
import os
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from arginator_protein_classifier.model import Lightning_Model
from arginator_protein_classifier.data import TL_Dataset

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

    data = TL_Dataset(data_path=cfg.paths.data,
                      task = cfg.task.name,
                      batch_size = cfg.experiment.batch_size,
                      split_ratios = (0.7, 0.15, 0.15),
                      seed = cfg.processing.seed)

    model = Lightning_Model(input_dim=1024, 
                            output_dim = cfg.task.output_dim,
                            dropout_rate = cfg.experiment.dropout_rate,
                            optimizer = cfg.optimizer,
                            loss_fn = cfg.experiment.loss_function)
    
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
        accelerator="auto", # Automatically detects GPU/MPS/CPU
        devices=1,
        default_root_dir=output_dir,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, data)
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

if __name__ == "__main__":
    train()