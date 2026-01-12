import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
import torch
from arginator_protein_classifier.evaluate import evaluate

@patch("arginator_protein_classifier.evaluate.get_dataloaders")
@patch("arginator_protein_classifier.evaluate.torch.load")
@patch("arginator_protein_classifier.evaluate.Model")
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
    mock_model_instance = MagicMock()
    mock_model_class.return_value = mock_model_instance
    
    # Ensure model.to(DEVICE) returns the model itself (Chainable method)
    mock_model_instance.to.return_value = mock_model_instance

    # Setup the forward pass return value
    # Prediction: [Class 1 (0.9), Class 0 (0.8)]
    fake_prediction = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    mock_model_instance.return_value = fake_prediction

    # 3: Mock Data
    fake_img = torch.randn(2, 1024)
    # Targets match the prediction indices [1, 0]
    fake_target = torch.tensor([1, 0])
    
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = [(fake_img, fake_target)]
    mock_get_dataloaders.return_value = (None, None, mock_loader)

    # 4: Run
    evaluate.__wrapped__(cfg)

    # 5: Assertions
    # Verify the model was moved to device
    assert mock_model_instance.to.called
    # Verify evaluation mode was set
    assert mock_model_instance.eval.called
    # Verify forward pass happened
    assert mock_model_instance.called