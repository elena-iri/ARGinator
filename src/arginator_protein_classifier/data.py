import logging
import h5py
import torch
import hydra
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from omegaconf import DictConfig

# Initialize Logger
log = logging.getLogger(__name__)

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path, output_folder: Path = None, force_process: bool = False) -> None:
        self.data_path = Path(data_path)
        self.output_folder = Path(output_folder) if output_folder else self.data_path
        self.processed_file = self.output_folder / "processed_data.pt"
        self.data = []

        if force_process or not self.processed_file.exists():
            self.preprocess(self.output_folder)
        
        log.info(f"Loading data from {self.processed_file}")
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
        
        # sort the list to ensure deterministic order 
        h5_files = sorted(list(self.data_path.glob("*.h5")))
        
        if not h5_files:
            log.warning(f"No .h5 files found in {self.data_path}")
            return

        log.info(f"Found {len(h5_files)} H5 files. Processing...")

        for file_path in h5_files:
            label = 0 if "non" in file_path.name else 1
            try:
                with h5py.File(file_path, 'r') as hf:
                    for key in hf.keys():
                        data_numpy = hf[key][:] 
                        embedding = torch.from_numpy(data_numpy).float()
                        all_samples.append((embedding, label))
            except Exception as e:
                log.error(f"Error reading {file_path}: {e}")

        torch.save(all_samples, output_folder / "processed_data.pt")
        log.info(f"Saved {len(all_samples)} samples to {output_folder / 'processed_data.pt'}")

# CRITICAL CHANGE: Accept seed argument
def get_dataloaders(data_path, batch_size=32, split_ratios=(0.7, 0.15, 0.15), seed=42):
    """Creates train, val, and test dataloaders."""
    full_dataset = MyDataset(Path(data_path))
    total_size = len(full_dataset)
    
    train_ratio, val_ratio, test_ratio = split_ratios
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    log.info(f"Splitting {total_size} samples: Train({train_size}), Val({val_size}), Test({test_size}) with seed {seed}")

    # Use the passed seed for the random split
    generator = torch.Generator().manual_seed(seed) 
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Get path to configs (Dynamic Absolute Path)
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
config_path = os.path.join(project_root, "configs")

# Hydra Main
@hydra.main(version_base=None, config_path=config_path, config_name="train_config")
def main(cfg: DictConfig):
    """
    Process raw .h5 data into a single .pt file.
    """
    data_path = Path(cfg.paths.data)
    force = cfg.processing.force
    seed = cfg.processing.seed
    
    # Set seed for any global operations
    torch.manual_seed(seed)
    
    log.info(f"Processing data from {data_path}...")
    
    # Ensure folder exists
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset
    MyDataset(data_path, output_folder=data_path, force_process=force)
    
    log.info("Data processing complete.")

if __name__ == "__main__":
    main()