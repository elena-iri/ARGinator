from unittest.mock import MagicMock

import pytest
import torch

from arginator_protein_classifier.train import evaluate


def test_evaluate_metrics():
    """
    Test the evaluate() helper function from train.py.
    We mock the model and dataloader to return known values and check if 
    accuracy/precision/recall/f1 are calculated correctly.
    """
    
    # 1. Mock the Model
    mock_model = MagicMock()
    
    # Define what the model 'predicts' (Logits)
    # Batch size = 2, Classes = 2
    # Sample 0: [0.1, 0.9] -> Argmax = 1 (Predicted Class 1)
    # Sample 1: [0.8, 0.2] -> Argmax = 0 (Predicted Class 0)
    mock_logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    
    # Configure the mock to return these logits when called: model(x)
    mock_model.return_value = mock_logits

    # 2. Create Dummy Data
    # Inputs (random, doesn't matter as model is mocked)
    fake_inputs = torch.randn(2, 1024)
    
    # Targets (Ground Truth)
    # Sample 0: 1 (Match)
    # Sample 1: 0 (Match)
    # This scenario represents 100% Accuracy
    fake_targets = torch.tensor([1, 0])
    
    # Mock Dataloader (Iterable list of batches)
    mock_dataloader = [(fake_inputs, fake_targets)]

    # 3. Execution
    device = torch.device("cpu")
    
    # We test the "binary" average method
    metrics = evaluate(mock_model, mock_dataloader, device, average_method="binary")

    # 4. Assertions
    
    # Ensure model was put in eval mode
    mock_model.eval.assert_called_once()
    
    # Check Metrics (Should be perfect)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    
    # Check that logits and targets were passed through (for ROC plotting)
    assert torch.equal(metrics["logits"], mock_logits)
    assert torch.equal(metrics["targets"], fake_targets)

def test_evaluate_multiclass_mismatch():
    """
    Test evaluate with a mismatch to ensure metrics calculate correctly
    even when predictions are wrong.
    """
    mock_model = MagicMock()
    
    # Prediction: [Class 1, Class 0]
    # We simulate a model that is completely wrong (swapped classes)
    mock_logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]]) 
    mock_model.return_value = mock_logits

    # Ground Truth: [Class 0, Class 1]
    # - Sample 0: True 0, Pred 1 (Wrong)
    # - Sample 1: True 1, Pred 0 (Wrong)
    # This ensures both classes exist in targets (avoids ZeroDivision warning)
    fake_targets = torch.tensor([0, 1])
    
    mock_dataloader = [(torch.randn(2, 10), fake_targets)]
    
    metrics = evaluate(mock_model, mock_dataloader, torch.device("cpu"), average_method="macro")
    
    # Since both predictions are wrong, accuracy should be 0.0
    assert metrics["accuracy"] == 0.0