from pkgutil import get_data
from dotenv import load_dotenv
import logging
import h5py
from sympy import Array
import torch
import hydra
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset
from omegaconf import DictConfig
# from operator import itemgetter
import numpy as np
from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
# import tasks

# Initialize Logger
log = logging.getLogger(__name__)
# Initialize environment variable of interest
class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path, output_folder: Path = None, force_process: bool = False, task: str = "Binary") -> None:
        self.data_path = Path(data_path)
        self.output_folder = Path(output_folder) if output_folder else self.data_path
        self.processed_file = self.output_folder / f"processed_data_{task.lower()}.pt"
        self.data = []
        self.task = task.lower()
        load_dotenv()
        self.file_pattern = os.environ["FILE_PATTERN"]
        classes = os.environ.get("CLASSES", "")
        self.class_names = [c.strip() for c in classes.split(',')] if classes else []
        # labels = os.environ[""]
        if not self.class_names:
            log.warning("No CLASS_NAMES found in .env. Labeling might fail.")

        if force_process or not self.processed_file.exists():
            self.preprocess(self.output_folder)
        
        log.info(f"Loading data from {self.processed_file}")
        self.data = torch.load(self.processed_file)

        labels = [sample[1] for sample in self.data]
        if task == "binary":
            log.info(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
        elif task =="multiclass":
            class_counts = np.bincount(labels)
            for idx, count in enumerate(class_counts):
                log.info(f"Class {idx}: {count}")   

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        embedding, label = self.data[index]
        return embedding, label
    
    # def _labeling(self, file_name: str) -> int:
    #     """
    #     Determines the label for a file based on the task and class configuration.
    #     """
    #     # --- Task: Binary ---
    #     # Convention: "non" class is 0 (Negative), everything else is 1 (Positive)
    #     if self.task == 'binary':
    #         if "non" in file_name.lower():
    #             return 0
    #         return 1
    #     elif self.task == 'multiclass':
    #         # --- Task: Multi-class ---
    #         # We match the file name against our explicit list from .env
    #         # e.g., if 'class_b' is in filename, and 'b' is at index 1 in class_names, label is 1.
    #         for idx, class_name in enumerate(self.class_names):
    #             if class_name in file_name:
    #                 return idx
    #         log.warning(f"Could not determine label for file {file_name}")
    #         return -1
    
    def _labeling(self, file_name: str) -> int:
        """
        Determines the label for a file based on the task and class configuration.
        """
        # --- Task: Binary ---
        if self.task == 'binary':
            if "non" in file_name.lower():
                return 0
            return 1
            
        # --- Task: Multi-class ---
        elif self.task == 'multiclass':
            for idx, class_name in enumerate(self.class_names):
                # Use the pattern to construct exactly what the file should look like
                # e.g., "card_class_{}_embeddings.h5" -> "card_class_a_embeddings.h5"
                expected_filename = self.file_pattern.format(class_name)
                
                # Check for exact match (or if it ends with it, to be safe against paths)
                if file_name == expected_filename or file_name.endswith(expected_filename):
                    return idx
                    
            log.warning(f"Could not determine label for file {file_name}")
            return -1         
    

    def preprocess(self, output_folder: Path) -> None:
        """Reads H5 files, assigns labels, and saves to .pt file."""
        output_folder.mkdir(parents=True, exist_ok=True)
        all_samples = []
        
        # sort the list to ensure deterministic order 
        h5_files = sorted(list(self.data_path.glob("*.h5")))
        
        if not h5_files:
            log.warning(f"No .h5 files found in {self.data_path}")
            return

        log.info(f"Found {len(h5_files)} H5 files. Processing for {self.task} setting")

        for file_path in h5_files:
            label = self._labeling(file_path.name)

            if label == -1:
                continue
            log.info(f"processing {file_path.name} with label {label}")
            """Printing (idx, file_path) returns 
            0 .data/card_class_a_embeddings.h5
            1 .data/card_class_b_embeddings.h5
            2 .data/card_class_c_embeddings.h5
            3 .data/card_class_d_embeddings.h5
            4 .data/non_betalactamase_embeddings_60k.h5"""
            # if self.task == "Binary":
            #     label = 0 if "non" in file_path.name else 1
            
            try:
                with h5py.File(file_path, 'r') as hf:
                    for key in hf.keys():
                        data_numpy = hf[key][:] 
                        embedding = torch.from_numpy(data_numpy).float()
                        all_samples.append((embedding, label))
            except Exception as e:
                log.error(f"Error reading {file_path}: {e}")

        #torch.save(all_samples, output_folder / "processed_data.pt")
        torch.save(all_samples, self.processed_file)
        log.info(f"Saved {len(all_samples)} samples to {output_folder / 'processed_data.pt'}")

    def get_labels(self) -> torch.tensor:
        """
        Helper to get all labels as a simple list/tensor. 
        Crucial for calculating class weights.
        """
        # self.data is a list of (embedding, label) tuples.
        # We extract just the labels.
        return [sample[1] for sample in self.data]

def get_dataloaders(data_path, task: str = "Binary", batch_size=32, 
                    split_ratios: list = None, seed: int = 42):
    """Creates train, val, and test dataloaders."""
    full_dataset = MyDataset(Path(data_path), task=task)
    labels = full_dataset.get_labels()
    indices = np.arange(len(full_dataset))
    labels = [int(label) for label in labels]
    val_test_ratio = split_ratios[1] + split_ratios[2]
    # First split: Separate Train from (Val + Test)
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices, labels, test_size = val_test_ratio, stratify = labels, random_state = seed
    )

    # Second split: Separate Val from Test
    # We need to adjust the ratio because we are splitting the 'temp' subset now
    # If val is 0.15 and test is 0.15, they are equal, so we split 50/50
    val_ratio_adjusted = split_ratios[1] / (split_ratios[1] + split_ratios[2])

    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=(1 - val_ratio_adjusted), 
        stratify=temp_labels, # Stratify based on the labels in the temp set
        random_state=seed
    )

    log.info(f"Stratified Split: Train({len(train_idx)}), Val({len(val_idx)}), Test({len(test_idx)})")

    # 4. Create PyTorch Subsets
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)

# ---------------------------------------------------------
    # Handle Imbalance by Calculating Weights for TRAIN subset only
    # ---------------------------------------------------------
    # We have a stratified train set (still imbalanced, but consistent).
    # We apply the Sampler here to fix the imbalance during training.
    
    log.info("Calculating weights for WeightedRandomSampler (Train set only)...")
    
    # Note: We already have 'train_labels' from the sklearn split above!
    # No need to iterate through indices again.
    
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    
    # Assign weights to samples
    sample_weights = [class_weights[t] for t in train_labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=2
    )
    
    # Val/Test loaders should NOT be balanced (they must represent real world)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

# Get path to configs (Dynamic Absolute Path)
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
config_path = os.path.join(project_root, "configs")

class TL_Dataset(pl.LightningDataModule):
    def __init__(self, data_path, task, batch_size, split_ratios, seed, output_folder):
        super().__init__()
        self.data_path = Path(data_path)
        self.task = task
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.seed = seed
        self.output_path = output_folder

    def setup(self, stage=None):
        """does the split and replaces get_dataloaders"""
        full_dataset = MyDataset(self.data_path, task=self.task)
        labels = full_dataset.get_labels()
        
        # Split indices (using your logic)
        indices = np.arange(len(full_dataset))
        labels = [int(label) for label in labels]
        val_test_ratio = self.split_ratios[1] + self.split_ratios[2]
        
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            indices, labels, test_size=val_test_ratio, stratify=labels, random_state=self.seed
        )

        val_ratio_adjusted = self.split_ratios[1] / (self.split_ratios[1] + self.split_ratios[2])
        val_idx, test_idx = train_test_split(
            temp_idx, 
            test_size=(1 - val_ratio_adjusted), 
            stratify=temp_labels, 
            random_state=self.seed
        )

        # Create Subsets
        self.train_subset = Subset(full_dataset, train_idx)
        self.val_subset = Subset(full_dataset, val_idx)
        self.test_subset = Subset(full_dataset, test_idx)

        # Calculate Sampler Weights (Train only)
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[t] for t in train_labels]
        
        self.train_sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_subset, 
            batch_size=self.batch_size, 
            sampler=self.train_sampler,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(self.val_subset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_subset, batch_size=self.batch_size, num_workers=2)

# Hydra Main
@hydra.main(version_base=None, config_path=config_path, config_name="train_config")
def main(cfg: DictConfig):
    """
    Process raw .h5 data into a single .pt file.
    """
    data_path = Path(cfg.paths.data)
    force = cfg.processing.force
    seed = cfg.processing.seed
    hydra_config = HydraConfig.get()
    task = hydra_config.runtime.choices.task
    batch_size = cfg.experiment.batch_size
    split_ratios = cfg.experiment.splits
    # Set seed for any global operations
    torch.manual_seed(seed)
    # Show the task we are going for when building datasets
    log.info(f"Selected task configuration: {task}")
    # Ensure folder exists
    log.info(f"Processing data from {data_path} (task={task})...")
    data_path.mkdir(parents=True, exist_ok=True) 
    # Initialize dataset
    dataset = TL_Dataset(data_path, task, batch_size, split_ratios, seed, data_path)
    log.info(f"Processing data from {data_path} (task={task})...")
    dataset.setup()
    log.info(f"Created General Dataset. Dataset is of type{type(dataset)}")
    log.info(f"Creating Training Dataset")
    train_loader = dataset.train_subset
    log.info(f"Training Dataset has shape {train_loader.__len__}")
    val_loader = dataset.val_subset
    log.info(f"Validation Dataset has shape {val_loader.__len__}")
    test_loader = dataset.test_subset
    log.info(f"Test Dataset has shape {test_loader.__len__}")
    log.info("Testing Train Dataloader creation method")
    tr_dl = dataset.train_dataloader()
    log.info(f"Number of batches in train loader: {len(tr_dl)}")
    #first batch
    first_batch = next(iter(tr_dl))
    log.info(f"First batch shape (features): {first_batch[0].shape}")
    # train_loader, val_loader, test_loader = get_dataloaders(data_path, task=task, seed=seed)
    # first_batch = next(iter(train_loader))
    # log.info(f"Loader worked. Batch Shape: {first_batch[0].shape}")
if __name__ == "__main__":
    main()
