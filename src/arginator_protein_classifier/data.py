from pathlib import Path

import h5py
import torch
import typer
from torch.utils.data import DataLoader, Dataset, random_split

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
            log.warning(f"No .h5 files found in {self.data_path}")
            return

        log.info(f"Found {len(h5_files)} H5 files. Processing...")

        for file_path in h5_files:
            label = 0 if "non" in file_path.name else 1
            try:
                with h5py.File(file_path, "r") as hf:
                    for key in hf.keys():
                        data_numpy = hf[key][:]
                        embedding = torch.from_numpy(data_numpy).float()
                        all_samples.append((embedding, label))
            except Exception as e:
                log.error(f"Error reading {file_path}: {e}")

        torch.save(all_samples, output_folder / "processed_data.pt")
        log.info(f"Saved {len(all_samples)} samples to {output_folder / 'processed_data.pt'}")


def get_dataloaders(data_path, batch_size=32, split_ratios=(0.7, 0.15, 0.15)):
    """Creates train, val, and test dataloaders."""
    full_dataset = MyDataset(Path(data_path))
    total_size = len(full_dataset)

    train_ratio, val_ratio, test_ratio = split_ratios
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    print(f"Total samples: {total_size}")
    print(f"Splitting into: Train ({train_size}), Val ({val_size}), Test ({test_size})")

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Typer Command
@app.command()
def main(
    data_path: str = "data",
    output_folder: str = "data",
    force: bool = typer.Option(False, "--force", "-f", help="Force regenerate the processed data."),
):
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
    app()
