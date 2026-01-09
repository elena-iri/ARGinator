from dotenv import load_dotenv
import glob
import os
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import typer
import h5py
from torch.utils.data import Dataset, Dataloader, ConcatDataset, random_split

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path, output_folder: Path = None) -> None:
        self.data_path = Path(data_path)
        self.output_folder = Path(output_folder) if output_folder else self.data_path
        self.processed_file = self.output_folder/"processed_data.pt"
        self.data = []

        if not self.processed_file.exists():
            self.preprocess(self.output_folder)
        
        print(f"Loading data from {self.processed_file}")
        self.data = torch.load(self.processed_file)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        embedding, label = self.data[index]
        return embedding, label

    # def preprocess(self, output_folder: Path) -> None:
    #     """Preprocess the raw data and save it to the output folder."""

    def preprocess(self, output_folder: Path) -> None:
            """
            Preprocess the raw h5 data:
            1. Reads H5 files.
            2. Assigns label 1 if filename contains 'card_class', else 0.
            3. Saves a list of (embedding, label) tuples to a .pt file.
            """
            output_folder.mkdir(parents=True, exist_ok=True)
            all_samples = []
            
            # Find all .h5 files
            h5_files = list(self.data_path.glob("*.h5"))
            
            if not h5_files:
                print("No .h5 files found in the data path.")
                return

            print(f"Found {len(h5_files)} H5 files. Processing...")

            for file_path in h5_files:
                # --- Labeling Logic ---
                # If it is betalactamese, it's Positive (1).
                # Otherwise (non-betalactamese), it's Negative (0).
                label = 0 if "non" in file_path.name else 1
                
                try:
                    with h5py.File(file_path, 'r') as hf:
                        # Iterate over the keys (identifiers) in the h5 file
                        for key in hf.keys():
                            # Read the embedding
                            #print(key)
                            data_numpy = hf[key][:] 
                            #print(data_numpy[:3])
                            # Convert to FloatTensor
                            embedding = torch.from_numpy(data_numpy).float()
                            # Add to list as tuple: (Tensor(1024), int)
                            all_samples.append((embedding, label))
                            
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

            # Save the unified dataset to disk
            torch.save(all_samples, output_folder / "processed_data.pt")
            print(f"Saved {len(all_samples)} samples to {output_folder / 'processed_data.pt'}")
                    
            print(f"Preprocessing complete. Data saved to {output_folder}")

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)

if __name__ == "__main__":
    #typer.run(preprocess)
    load_dotenv()
    folder_path = os.environ["DATA_PATH"]
    data = MyDataset(folder_path)

