import pytest
import torch
import h5py
import numpy as np
import os
from pathlib import Path
from unittest.mock import patch

# IMPORTANT: We need to mock environment variables BEFORE importing data.py
# because MyDataset loads them in __init__ or relies on dotenv.
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set environment variables required by MyDataset."""
    monkeypatch.setenv("FILE_PATTERN", "card_class_{}_embeddings.h5")
    # For binary task, class names might not be strictly needed but good to have
    monkeypatch.setenv("CLASSES", "a, b, c, d")

# Now import the module under test
from arginator_protein_classifier.data import get_dataloaders

# HELPER FUNCTION
def create_dummy_h5(folder, filename, num_samples=10):
    """
    Creates an H5 file where each sample is a separate key.
    """
    filepath = folder / filename
    
    with h5py.File(filepath, "w") as f:
        for i in range(num_samples):
            # Shape is (1024,) to match model input
            data = np.random.randn(1024).astype('float32')
            f.create_dataset(f"protein_{i}", data=data)
            
    return filepath

# DATALOADERS TEST FUNCTION
def test_get_dataloaders(tmp_path):
    """
    Test that data loaders are created correctly from H5 files.
    """
    # 1. Setup: Create dummy files
    # In data.py: label = 0 if "non" in file_path.name else 1
    
    # Create 'Non-Protein' data (Label 0)
    create_dummy_h5(tmp_path, "non_protein_data.h5", num_samples=20)
    
    # Create 'Protein' data (Label 1)
    create_dummy_h5(tmp_path, "positive_protein_data.h5", num_samples=20)

    # 2. Execution: Call get_dataloaders pointing to the temp folder
    # Total samples = 40. 
    # With split (0.7, 0.15, 0.15) -> Train: 28, Val: 6, Test: 6
    batch_size = 4
    
    # get_dataloaders(data_path, task="Binary", batch_size=32, split_ratios=..., seed=42)
    
    splits = [0.7, 0.15, 0.15]
    
    train_dl, val_dl, test_dl = get_dataloaders(
        data_path=str(tmp_path), 
        batch_size=batch_size, 
        split_ratios=splits,
        seed=42
    )

    # 3. Assertions: Basic Checks
    assert train_dl is not None
    assert len(train_dl.dataset) == 28, f"Expected 28 train samples, got {len(train_dl.dataset)}"
    assert len(val_dl.dataset) == 6
    assert len(test_dl.dataset) == 6

    # 4. Assertions: Check Data Shapes
    # Fetch one batch to verify the collate_fn works
    x, y = next(iter(train_dl))

    # Check Input (Embeddings)
    # Should be [batch_size, 1024]
    assert x.shape == (batch_size, 1024), f"Expected input shape ({batch_size}, 1024), got {x.shape}"
    assert x.dtype == torch.float32

    # Check Targets (Labels)
    # Should be [batch_size]
    assert y.shape == (batch_size,), f"Expected target shape ({batch_size},), got {y.shape}"
    assert y.dtype == torch.int64
    
    # Check that we actually got both labels (0 and 1) in the dataset
    all_labels = [label for _, label in train_dl.dataset]
    assert 0 in all_labels and 1 in all_labels, "Dataset should contain both classes (0 and 1)"