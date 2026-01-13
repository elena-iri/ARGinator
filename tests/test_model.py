import pytest
import torch
from arginator_protein_classifier.model import Model

# Define constants matching your project structure
INPUT_DIM = 1024
OUTPUT_DIM = 2

@pytest.fixture
def model():
    """
    Pytest fixture to create a fresh model instance for each test.
    """
    return Model(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, dropout_rate=0.1)

@pytest.mark.parametrize("batch_size", [1, 32, 64])
def test_model_forward_shape(model, batch_size):
    """
    Test that the model accepts the correct input shape 
    and returns the correct output shape for various batch sizes.
    """
    # Create random input tensor [Batch_Size, Input_Dim]
    x = torch.randn(batch_size, INPUT_DIM)
    
    # Forward pass
    y = model(x)
    
    # Assertions
    assert y.shape == (batch_size, OUTPUT_DIM), \
        f"Expected output shape {(batch_size, OUTPUT_DIM)}, got {y.shape}"
    
    assert not torch.isnan(y).any(), "Model output contains NaNs!"

def test_model_backward(model):
    """
    Smoke test: Can we calculate gradients?
    This ensures the computational graph is connected correctly.
    """
    batch_size = 4
    x = torch.randn(batch_size, INPUT_DIM)
    target = torch.randint(0, OUTPUT_DIM, (batch_size,))
    
    # Forward
    y = model(x)
    loss = torch.nn.functional.cross_entropy(y, target)
    
    # Backward
    loss.backward()
    
    # Check if gradients exist for a key layer (e.g., first layer weights)
    # We iterate parameters to find one with grad
    has_grads = False
    for param in model.parameters():
        if param.grad is not None:
            has_grads = True
            assert param.grad.shape == param.shape
            break
            
    assert has_grads, "Gradients were not computed. Is requires_grad=True?"

@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_model_dropout_initialization(dropout):
    """
    Test that we can initialize the model with different hyperparameters.
    """
    model = Model(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, dropout_rate=dropout)
    assert model is not None