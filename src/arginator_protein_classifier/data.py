from dotenv import load_dotenv
import glob
import os
import numpy as np
from pathlib import Path
import torch
import typer
import h5py
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

# Initialize Typer
app = typer.Typer()

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path, output_folder: Path = None, force_process: bool = False) -> None:
        self.data_path = Path(data_path)
        self.output_folder = Path(output_folder) if output_folder else self.data_path
        self.processed_file = self.output_folder / "processed_data.pt"
        self.data = []

        # Logic Update: Process if missing OR if force_process is True
        if force_process or not self.processed_file.exists():
            self.preprocess(self.output_folder)
        
        print(f"Loading data from {self.processed_file}")
        self.data = torch.load(self.processed_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        embedding, label = self.data[index]
        return embedding, label

    def preprocess(self, output_folder: Path) -> None:
        """Reads H5 files, assigns labels, and saves to .pt file."""
        output_folder.mkdir(parents=True, exist_ok=True)
        all_samples = []
        
        # Find all .h5 files
        h5_files = list(self.data_path.glob("*.h5"))
        
        if not h5_files:
            print(f"No .h5 files found in {self.data_path}")
            return

        print(f"Found {len(h5_files)} H5 files. Processing...")

        for file_path in h5_files:
            # Labeling Logic: 0 if 'non', else 1
            label = 0 if "non" in file_path.name else 1
            
            try:
                with h5py.File(file_path, 'r') as hf:
                    for key in hf.keys():
                        data_numpy = hf[key][:] 
                        embedding = torch.from_numpy(data_numpy).float()
                        all_samples.append((embedding, label))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        # Save to disk
        torch.save(all_samples, output_folder / "processed_data.pt")
        print(f"Saved {len(all_samples)} samples to {output_folder / 'processed_data.pt'}")

def get_dataloaders(data_path, batch_size=32, split_ratios=(0.7, 0.15, 0.15)):
    """Creates train, val, and test dataloaders."""
    # Instantiate dataset (will load cached data automatically)
    full_dataset = MyDataset(data_path)
    total_size = len(full_dataset)
    
    train_ratio, val_ratio, test_ratio = split_ratios
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Total samples: {total_size}")
    print(f"Splitting into: Train ({train_size}), Val ({val_size}), Test ({test_size})")

    generator = torch.Generator().manual_seed(42) 
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Typer Command
@app.command()
def main(
    data_path: str = ".data", 
    output_folder: str = ".data", 
    force: bool = typer.Option(False, "--force", "-f", help="Force regenerate the processed data.")
):
    """
    Process raw .h5 data into a single .pt file.
    """
    print(f"Processing data from {data_path}...")
    # We trigger the init with the force flag
    MyDataset(Path(data_path), Path(output_folder), force_process=force)
    print("Data processing complete.")

if __name__ == "__main__":
    app()