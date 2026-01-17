import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from arginator_protein_classifier.data import get_dataloaders


# 1. Helper Function (Unchanged, but ensures filename compatibility)
def create_dummy_h5(folder, filename, num_samples=10):
    """
    Creates an H5 file where each sample is a separate key.
    """
    filepath = folder / filename

    with h5py.File(filepath, "w") as f:
        for i in range(num_samples):
            # Create a random 1D vector (1024,)
            data = np.random.randn(1024).astype("float32")
            f.create_dataset(f"protein_{i}", data=data)

    return filepath


# 2. Main Test Function
def test_get_dataloaders(tmp_path, monkeypatch):
    """
    Test that data loaders are created correctly from H5 files using the new data.py logic.
    """
    # --- A. Mock Environment Variables ---
    # MyDataset.__init__ crashes if these keys are missing from os.environ.
    # We use monkeypatch to temporarily set them just for this test.
    monkeypatch.setenv("FILE_PATTERN", "card_class_{}_embeddings.h5")
    monkeypatch.setenv("CLASSES", "class_a,class_b")

    # --- B. Setup Dummy Data ---
    # In your binary logic: label = 0 if "non" in file_path.name else 1

    # Label 0 (Negative)
    create_dummy_h5(tmp_path, "non_protein_data.h5", num_samples=20)

    # Label 1 (Positive)
    create_dummy_h5(tmp_path, "positive_protein_data.h5", num_samples=20)

    # --- C. Execution ---
    # Total samples = 40.
    # Splits: Train=0.7 (28), Val=0.15 (6), Test=0.15 (6)
    batch_size = 4

    # Note: We must pass 'split_ratios' now as it is required by your logic
    train_dl, val_dl, test_dl = get_dataloaders(
        data_path=str(tmp_path), task="Binary", batch_size=batch_size, split_ratios=[0.7, 0.15, 0.15], seed=42
    )

    # --- D. Basic Assertions ---
    assert train_dl is not None
    assert len(train_dl.dataset) == 28, f"Expected 28 train samples, got {len(train_dl.dataset)}"
    assert len(val_dl.dataset) == 6
    assert len(test_dl.dataset) == 6

    # --- E. Check Data Shapes & Types ---
    # Fetch one batch to verify the collate_fn and MyDataset.__getitem__ work
    x, y = next(iter(train_dl))

    # Check Input (Embeddings) -> [batch_size, 1024]
    assert x.shape == (batch_size, 1024), f"Expected input shape ({batch_size}, 1024), got {x.shape}"
    assert x.dtype == torch.float32

    # Check Targets (Labels) -> [batch_size]
    assert y.shape == (batch_size,), f"Expected target shape ({batch_size},), got {y.shape}"

    # Your new code returns labels as 'int', PyTorch generally expects Long/Int64 for CrossEntropy
    # If your dataset returns simple ints, DataLoader converts them.
    # Just ensure it's not a string or object.

    # --- F. Verify Class Balance/Content ---
    # We check the full training dataset to ensure we have both classes
    # train_dl.dataset is a 'Subset', so we access the underlying dataset via indices

    # Extract labels from the Subset
    # subset.dataset is the full MyDataset
    # subset.indices are the indices for this split
    full_dataset = train_dl.dataset.dataset
    train_indices = train_dl.dataset.indices

    train_labels = [full_dataset.data[i][1] for i in train_indices]

    assert 0 in train_labels and 1 in train_labels, "Training set should contain both classes (0 and 1)"

    # --- G. Verify Preprocessing File Creation ---
    # MyDataset should have created a .pt file in the tmp_path
    processed_file = tmp_path / "processed_data_binary.pt"
    assert processed_file.exists(), "MyDataset did not save the processed .pt file"
