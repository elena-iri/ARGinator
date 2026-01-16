import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
import torch
from arginator_protein_classifier.evaluate import evaluate

@patch("arginator_protein_classifier.evaluate.get_dataloaders")           # Becomes Arg 3
@patch("arginator_protein_classifier.evaluate.torch.load")                # Becomes Arg 2
@patch("arginator_protein_classifier.evaluate.Model")                     # Becomes Arg 1
@patch("arginator_protein_classifier.evaluate.DEVICE", torch.device("cpu")) # No Arg
def test_evaluate(mock_model_class, mock_torch_load, mock_get_dataloaders):
    """
    Test the evaluation script.
    """
    # 1: Config
    cfg = OmegaConf.create({
        "experiment": {
            "dropout_rate": 0.1,
            "batch_size": 2,
        },
        "paths": {
            "data": "dummy/path",
            "model_filename": "dummy_model.pth"
        }
    })

    # 2: Mock the Model
    # Now 'mock_model_class' actually holds the Model class mock
    mock_model_instance = MagicMock()
    mock_model_class.return_value = mock_model_instance
    
    # Ensure model.to(DEVICE) returns the model itself
    mock_model_instance.to.return_value = mock_model_instance

    # Setup the forward pass return value
    fake_prediction = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    mock_model_instance.return_value = fake_prediction

    # 3: Mock Data
    fake_img = torch.randn(2, 1024)
    fake_target = torch.tensor([1, 0])
    
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = [(fake_img, fake_target)]
    
    # Now 'mock_get_dataloaders' actually holds the Dataloader function mock
    mock_get_dataloaders.return_value = (None, None, mock_loader)

    # 4: Run
    evaluate.__wrapped__(cfg)

    # 5: Assertions
    assert mock_model_instance.to.called
    assert mock_model_instance.eval.called
    assert mock_model_instance.called