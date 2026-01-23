import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

# Import the train function
from arginator_protein_classifier.train import train


@patch("arginator_protein_classifier.train.ModelCheckpoint")     # 9. NEW: Mock Checkpoint Callback
@patch("arginator_protein_classifier.train.wandb")               # 8. Mock global wandb module
@patch("arginator_protein_classifier.train.plt")                 # 7. Mock Plotting
@patch("arginator_protein_classifier.train.evaluate")            # 6. Mock Evaluate helper
@patch("arginator_protein_classifier.train.Lightning_Model")     # 5. Mock Model Class
@patch("arginator_protein_classifier.train.Trainer")             # 4. Mock Lightning Trainer
@patch("arginator_protein_classifier.train.WandbLogger")         # 3. Mock WandB Logger
@patch("arginator_protein_classifier.train.TL_Dataset")          # 2. Mock DataModule
@patch("arginator_protein_classifier.train.HydraConfig")         # 1. Mock Hydra
def test_train_flow(
    mock_hydra, 
    mock_dataset_cls, 
    mock_wandb_logger_cls, 
    mock_trainer_cls, 
    mock_model_cls, 
    mock_evaluate, 
    mock_plt, 
    mock_wandb_module,
    mock_checkpoint_cls,  # <--- Capture the new mock here
    tmp_path
):
    """
    Tests the training function logic by mocking all external dependencies.
    """

    # 1. Setup Dummy Config
    cfg = OmegaConf.create({
        "experiment": {
            "seed": 42,
            "dropout_rate": 0.1,
            "batch_size": 2,
            "epochs": 1,
            "splits": [0.8, 0.1, 0.1],
            "loss_function": "torch.nn.CrossEntropyLoss"
        },
        "task": {
            "name": "binary",
            "output_dim": 2
        },
        "optimizer": {
             "_target_": "torch.optim.Adam",
             "lr": 0.001
        },
        "paths": {
            "data": str(tmp_path / "dummy_data"),
            "model_filename": "model.pth"
        },
        "processing": {
            "seed": 42
        }
    })

    # 2. Setup Mocks
    mock_hydra.get.return_value.runtime.output_dir = str(tmp_path)
    
    # Configure plt.subplots
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_ax.plot.return_value = [MagicMock()] # Return list for unpacking
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    
    # Mock DataModule
    mock_dm_instance = mock_dataset_cls.return_value
    mock_dm_instance.test_dataloader.return_value = [("input", "target")]
    
    # Mock Trainer & Model
    mock_trainer_instance = mock_trainer_cls.return_value
    mock_model_instance = mock_model_cls.return_value

    # --- FIX: Ensure ModelCheckpoint provides a path ---
    # This ensures 'if best_model_path:' evaluates to True
    mock_checkpoint_cls.return_value.best_model_path = "dummy/path/best.ckpt"

    # Mock Evaluate return values
    mock_evaluate.return_value = {
        "accuracy": 0.95,
        "precision": 0.90,
        "recall": 0.85,
        "f1": 0.88,
        "logits": torch.tensor([[0.1, 0.9], [0.9, 0.1]]), 
        "targets": torch.tensor([1, 0])           
    }
    
    # Mock Global WandB Object
    mock_wandb_module.summary = {} 
    mock_wandb_module.run = MagicMock()

    # 3. Execution
    with patch.dict(os.environ, {
        "WANDB_API_KEY": "dummy_key", 
        "WANDB_ENTITY": "dummy_entity", 
        "WANDB_PROJECT": "dummy_project"
    }):
        train.__wrapped__(cfg)

    # 4. Assertions
    mock_dataset_cls.assert_called_once()
    mock_trainer_cls.assert_called_once()
    mock_trainer_instance.fit.assert_called_once_with(mock_model_instance, mock_dm_instance)
    
    # Check that plotting was triggered
    mock_plt.subplots.assert_called_once()
    
    # Check that artifact logging was attempted
    assert mock_wandb_module.log_artifact.called or mock_wandb_module.run.link_artifact.called