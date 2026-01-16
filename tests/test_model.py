import pytest
import torch
from omegaconf import DictConfig
from arginator_protein_classifier.model import Lightning_Model

# Define constants matching your project structure
INPUT_DIM = 1024
OUTPUT_DIM = 2

@pytest.fixture
def hydra_mocks():
    """
    Creates dummy Hydra DictConfigs for loss and optimizer.
    This allows instantiation to work without loading real config files.
    """
    # Mock config for Loss Function (e.g., CrossEntropyLoss)
    loss_cfg = DictConfig({'_target_': 'torch.nn.CrossEntropyLoss'})
    
    # Mock config for Optimizer (e.g., Adam)
    # Note: 'params' is injected by the model, so we don't need it here
    opt_cfg = DictConfig({'_target_': 'torch.optim.Adam', 'lr': 0.001})
    
    return loss_cfg, opt_cfg

@pytest.fixture
def model(hydra_mocks):
    """
    Pytest fixture to create a fresh Lightning_Model instance for each test.
    """
    loss_cfg, opt_cfg = hydra_mocks
    return Lightning_Model(
        input_dim=INPUT_DIM, 
        output_dim=OUTPUT_DIM, 
        dropout_rate=0.1, 
        loss_fn=loss_cfg, 
        optimizer_config=opt_cfg
    )

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
    
    # Check for NaNs (smoke test for numerical stability)
    assert not torch.isnan(y).any(), "Model output contains NaNs!"

def test_model_backward(model):
    """
    Smoke test: Can we calculate gradients?
    This ensures the computational graph is connected correctly.
    """
    batch_size = 4
    x = torch.randn(batch_size, INPUT_DIM)
    
    # Create dummy targets (integers for CrossEntropy)
    target = torch.randint(0, OUTPUT_DIM, (batch_size,))
    
    # Forward
    logits = model(x)
    
    # Calculate loss manually using the model's instantiated loss function
    loss = model.loss_fn(logits, target)
    
    # Backward
    loss.backward()
    
    # Check if gradients exist for a key layer (e.g., first layer weights)
    has_grads = False
    for param in model.parameters():
        if param.grad is not None:
            has_grads = True
            assert param.grad.shape == param.shape
            break
            
    assert has_grads, "Gradients were not computed. Is requires_grad=True?"

@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_model_initialization(dropout, hydra_mocks):
    """
    Test that we can initialize the model with different hyperparameters.
    """
    loss_cfg, opt_cfg = hydra_mocks
    model = Lightning_Model(
        input_dim=INPUT_DIM, 
        output_dim=OUTPUT_DIM, 
        dropout_rate=dropout,
        loss_fn=loss_cfg,
        optimizer_config=opt_cfg
    )
    assert model is not None
    assert model.dropout.p == dropout

def test_optimizer_configuration(model):
    """
    Test that configure_optimizers returns a valid optimizer instance.
    """
    optimizer = model.configure_optimizers()
    
    # Check if it is a torch optimizer
    assert isinstance(optimizer, torch.optim.Optimizer)
    
    # Check if the optimizer has the parameters of the model
    # (Checking the number of parameter groups usually suffices)
    assert len(optimizer.param_groups) > 0