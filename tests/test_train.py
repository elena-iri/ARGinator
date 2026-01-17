import os
from unittest.mock import MagicMock, patch

import pytest
import torch

# Import the train function
from arginator_protein_classifier.train import train
from omegaconf import OmegaConf


@patch("arginator_protein_classifier.train.HydraConfig")  # 1. Mock Hydra runtime
@patch("arginator_protein_classifier.train.TL_Dataset")  # 2. Mock Data Module
@patch("arginator_protein_classifier.train.Trainer")  # 3. Mock Lightning Trainer
@patch("arginator_protein_classifier.train.wandb")  # 4. Mock WandB to prevent API calls
@patch("arginator_protein_classifier.train.WandbLogger")  # 5. Mock the Logger Class
def test_train(mock_wandb_logger, mock_wandb, mock_trainer, mock_tl_dataset, mock_hydra_config, tmp_path):
    """
    Tests the training loop by bypassing the @hydra.main decorator,
    mocking the Lightning Trainer, and verifying connections.
    """

    # --- 1: Create a Dummy Config ---
    # Must match the structure in train.py (experiment, task, optimizer, etc.)
    cfg = OmegaConf.create(
        {
            "experiment": {
                "seed": 42,
                "dropout_rate": 0.1,
                "batch_size": 2,
                "epochs": 1,
                "lr": 0.001,
                "splits": [0.7, 0.15, 0.15],
                # Mock the loss function config for instantiation
                "loss_function": {"_target_": "torch.nn.CrossEntropyLoss"},
            },
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "task": {"name": "Binary", "output_dim": 2},
            "paths": {
                "data": str(tmp_path / "data"),
                # "model_filename" is no longer used directly in train.py, handled by Checkpoint
            },
            "processing": {
                "seed": 42,
                "force": False,  # Added this as it's often used in processing
            },
        }
    )

    # --- 2: Mock Hydra Runtime ---
    # Return the temp path as the output directory
    mock_hydra_config.get.return_value.runtime.output_dir = str(tmp_path)

    # --- 3: Mock Data Loading ---
    # Create fake tensors for evaluation loop
    fake_img = torch.randn(2, 1024)
    fake_target = torch.tensor([0, 1], dtype=torch.long)

    # Create a mock dataloader that acts like a list of batches
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = [(fake_img, fake_target)]
    mock_loader.__len__.return_value = 1

    # Configure the TL_Dataset instance to return this loader
    # The train.py code calls data.test_dataloader()
    mock_dataset_instance = mock_tl_dataset.return_value
    mock_dataset_instance.test_dataloader.return_value = mock_loader

    # --- 4: Mock Trainer ---
    # We want to ensure trainer.fit() is called, but we don't want it to actually run.
    mock_trainer_instance = mock_trainer.return_value

    # --- 5: Mock WandB ---
    # prevent wandb.summary["test_accuracy"] from crashing
    mock_wandb.summary = {}

    # --- 6: Run the Function ---
    # Use .__wrapped__ to bypass the @hydra.main decorator
    train.__wrapped__(cfg)

    # --- 7: Assertions ---

    # A. Verify Dataset Initialization
    mock_tl_dataset.assert_called_once()
    # Check if the path passed to dataset matches config
    _, kwargs = mock_tl_dataset.call_args
    assert kwargs["data_path"] == cfg.paths.data
    assert kwargs["task"] == "Binary"

    # B. Verify Trainer Initialization & Fit
    mock_trainer.assert_called_once()
    mock_trainer_instance.fit.assert_called_once()

    # Check that fit was called with (model, data)
    # args[0] is model, args[1] is data
    args, _ = mock_trainer_instance.fit.call_args
    model_arg = args[0]
    data_arg = args[1]

    # Verify the model passed to trainer has correct dims
    assert model_arg.input_dim == 1024
    assert model_arg.output_dim == 2

    # Verify the data passed is our mock dataset
    assert data_arg == mock_dataset_instance

    # C. Verify Evaluation Logic (Post-Training)
    # The code calls evaluate() which iterates over test_loader.
    # Since we mocked the test_loader to yield 1 batch, the loop should complete.
    # We can check if wandb.summary was updated
    assert "test_accuracy" in mock_wandb.summary, "WandB summary was not updated with test accuracy"
    assert "test_f1" in mock_wandb.summary, "WandB summary was not updated with F1 score"

    # D. Verify WandB Logging
    # Check that wandb.log was called (likely for the ROC curve or metrics)
    # Note: Depending on binary vs multiclass, roc curve logic might differ.
    if cfg.task.output_dim == 2:
        # Check if wandb.Image was created (part of ROC logging)
        assert mock_wandb.Image.called or mock_wandb.log.called
