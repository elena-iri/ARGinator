import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
import torch
import os

# Import the train function
from arginator_protein_classifier.train import train

@patch("arginator_protein_classifier.train.HydraConfig")  # 1. Mock Hydra's runtime config
@patch("arginator_protein_classifier.train.get_dataloaders")  # 2. Mock data loading
@patch("arginator_protein_classifier.train.plt")  # 3. Mock plotting (so no windows open)
def test_train(mock_plt, mock_get_dataloaders, mock_hydra_config, tmp_path):
    """
    Tests the training loop by bypassing the @hydra.main decorator 
    and mocking the data/environment.
    """
    
    # 1: Create a Dummy Config
    # We mimic the structure of your config.yaml
    cfg = OmegaConf.create({
        "experiment": {
            "seed": 42,
            "dropout_rate": 0.1,
            "batch_size": 2,
            "epochs": 1,  # Run just 1 epoch for speed
            "lr": 0.001
        },
        "paths": {
            "data": "dummy/path",
            "model_filename": "model.pth"
        },
        "processing": {
            "seed": 42
        }
    })

    # 2: Mock the Hydra Runtime Output Directory
    # When code calls HydraConfig.get().runtime.output_dir, return our temp folder
    mock_hydra_config.get.return_value.runtime.output_dir = str(tmp_path)

    # 3: Mock the Data
    # Create fake tensors that match Model input (1024 dim)
    # Batch size 2, input dim 1024
    fake_img = torch.randn(2, 1024) 
    fake_target = torch.randint(0, 2, (2,)) # Binary targets (0 or 1)
    
    # Create a mock dataloader that yields one batch and then stops
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = [(fake_img, fake_target)]
    mock_loader.__len__.return_value = 1
    
    # get_dataloaders returns (train, val, test). We only need train for this script.
    mock_get_dataloaders.return_value = (mock_loader, mock_loader, mock_loader)

    mock_fig = MagicMock()
    mock_axs = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axs)

    # 4: Run the Function
    # IMPORTANT: Use .__wrapped__ to bypass the @hydra.main decorator!
    train.__wrapped__(cfg)

    # 5: Assertions
    
    # A. Check if the model file was actually created
    expected_model_path = tmp_path / "model.pth"
    assert expected_model_path.exists(), "Model file was not saved!"

    # B. Check if the plot logic was triggered
    # Since 'fig' is a mock, it won't write a real file. 
    # Instead, we verify that .savefig() was called exactly once.
    expected_plot_path = tmp_path / "training_statistics.png"
    
    # Check that savefig was called
    assert mock_fig.savefig.called, "Plot savefig was never called!"
    
    # Verify it was called with the correct path
    args, _ = mock_fig.savefig.call_args
    assert str(expected_plot_path) in str(args[0]), f"Plot saved to wrong path: {args[0]}"