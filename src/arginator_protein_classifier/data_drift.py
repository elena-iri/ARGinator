# Data Drift Detection Script
import logging
import os
from pathlib import Path
from typing import Tuple

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.report import Report
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from scipy.stats import entropy
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)


def compute_embedding_statistics(embeddings_array: np.ndarray) -> dict:
    """
    Compute L2 norm, L1 norm, and entropy for each embedding.
    
    Args:
        embeddings_array: Array of shape (n_samples, embedding_dim)
        
    Returns:
        Dictionary with 'l2_norm', 'l1_norm', 'entropy' keys, each containing arrays of shape (n_samples,)
    """
    # L2 norm: Euclidean norm
    l2_norms = np.linalg.norm(embeddings_array, ord=2, axis=1)
    
    # L1 norm: Manhattan distance
    l1_norms = np.linalg.norm(embeddings_array, ord=1, axis=1)
    
    # Entropy: Treat absolute values normalized by L1 as probability distribution
    entropies = []
    for emb in embeddings_array:
        # Normalize absolute values to form a probability distribution
        abs_emb = np.abs(emb)
        prob_dist = abs_emb / (np.sum(abs_emb) + 1e-10)
        ent = entropy(prob_dist)
        entropies.append(ent)
    
    return {
        "l2_norm": l2_norms,
        "l1_norm": l1_norms,
        "entropy": np.array(entropies),
    }


def load_reference_data(processed_data_path: Path) -> Tuple[pd.DataFrame, PCA]:
    """
    Load reference data from processed tensor file and convert to DataFrame.
    Applies PCA for dimensionality reduction.
    
    Args:
        processed_data_path: Path to processed_data_*.pt file
        
    Returns:
        Tuple of (reference_df, fitted_pca)
    """
    log.info(f"Loading reference data from {processed_data_path}")
    
    # Load the processed tensor data
    data = torch.load(processed_data_path)
    
    embeddings = []
    labels = []
    
    for embedding, label in data:
        embeddings.append(embedding.numpy())
        labels.append(label)
    
    embeddings_array = np.array(embeddings)
    log.info(f"Loaded {len(embeddings)} samples with original shape {embeddings_array[0].shape}")
    
    # Fit PCA on reference data to 20 components
    pca = PCA(n_components=20)
    embeddings_pca = pca.fit_transform(embeddings_array)
    
    # Log variance explained
    variance_explained = pca.explained_variance_ratio_
    cumsum_variance = np.cumsum(variance_explained)
    
    log.info(f"Variance explained by each component (first 5):")
    for i, var in enumerate(variance_explained[:5]):
        log.info(f"  PC{i+1}: {var:.4f} ({cumsum_variance[i]:.4f} cumulative)")
    
    log.info(f"Total variance explained by 20 PCs: {cumsum_variance[-1]:.4f}")
    
    # Compute embedding statistics
    stats = compute_embedding_statistics(embeddings_array)
    
    # Convert to DataFrame with PC features + statistics
    df_dict = {f"PC_{i}": embeddings_pca[:, i] for i in range(20)}
    df_dict["l2_norm"] = stats["l2_norm"]
    df_dict["l1_norm"] = stats["l1_norm"]
    df_dict["entropy"] = stats["entropy"]
    df_dict["label"] = labels
    
    reference_df = pd.DataFrame(df_dict)
    log.info(f"Reference DataFrame shape after PCA and statistics: {reference_df.shape}")
    
    return reference_df, pca


def generate_mock_current_data(num_samples: int, embedding_dim: int = 1024, pca: PCA = None) -> pd.DataFrame:
    """
    Generate mock current data using random embeddings with "non" label.
    Applies the same PCA transformation as reference data.
    
    Args:
        num_samples: Number of samples to generate
        embedding_dim: Original embedding dimension (default 1024)
        pca: Fitted PCA object from reference data
        
    Returns:
        DataFrame with mock current data
    """
    if pca is None:
        raise ValueError("PCA object must be provided")
    
    # Generate random embeddings in original space
    random_embeddings = torch.rand(num_samples, embedding_dim)
    random_embeddings_array = random_embeddings.numpy()
    
    # Transform using fitted PCA
    random_embeddings_pca = pca.transform(random_embeddings_array)
    
    # Compute embedding statistics on original random embeddings
    stats = compute_embedding_statistics(random_embeddings_array)
    
    # All samples labeled as "non" (label = 0)
    labels = np.zeros(num_samples, dtype=int)
    
    # Convert to DataFrame with PC features + statistics
    df_dict = {f"PC_{i}": random_embeddings_pca[:, i] for i in range(20)}
    df_dict["l2_norm"] = stats["l2_norm"]
    df_dict["l1_norm"] = stats["l1_norm"]
    df_dict["entropy"] = stats["entropy"]
    df_dict["label"] = labels
    
    current_df = pd.DataFrame(df_dict)
    log.info(f"Generated mock current data with shape: {current_df.shape}")
    
    return current_df


#def save_data_to_csv(df: pd.DataFrame, output_path: Path, name: str = "data") -> None:
 #   """Save DataFrame to CSV file."""
  #  output_path.parent.mkdir(parents=True, exist_ok=True)
  #  df.to_csv(output_path, index=False)
  #  log.info(f"Saved {name} to {output_path}")


def generate_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate Evidently data drift report.
    
    Args:
        reference_df: Reference data DataFrame
        current_df: Current data DataFrame
        output_path: Path to save HTML report
    """
    log.info("Generating Evidently Data Drift Report...")
    
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    log.info(f"Report saved to {output_path}")


def load_current_data(h5_path: Path, results_path: Path, pca: PCA, task: str) -> pd.DataFrame:
    """
    Load current data from H5 embeddings file and corresponding model predictions CSV.
    Applies PCA transformation and computes statistics.
    
    Args:
        h5_path: Path to H5 file containing embeddings and protein IDs
        results_path: Path to CSV file with columns [protein_id, predicted_label]
        pca: Fitted PCA object from reference data
        task: Task name ("binary" or "multiclass") for label mapping
        
    Returns:
        DataFrame with PCA features, statistics, and predicted labels
    """
    load_dotenv()
    
    # Build label map based on task
    label_map = {}
    if task.lower() == "binary":
        label_map = {0: "Non-Betalactamase", 1: "Betalactamase"}
    else:
        # For multiclass
        classes_str = os.environ.get("CLASSES", "")
        if classes_str:
            class_names = [c.strip() for c in classes_str.split(",")]
            label_map = {i: name for i, name in enumerate(class_names)}
        else:
            log.warning("Multiclass detected but no CLASSES in .env. Using numeric labels.")
            # Will handle numeric labels later
    
    # Load embeddings from H5
    log.info(f"Loading embeddings from {h5_path}")
    embeddings_list = []
    protein_ids = []
    
    with h5py.File(h5_path, "r") as hf:
        for key in hf.keys():
            data_numpy = hf[key][:]
            embeddings_list.append(data_numpy)
            protein_ids.append(key)
    
    embeddings_array = np.array(embeddings_list)
    log.info(f"Loaded {len(embeddings_array)} embeddings from H5")
    
    # Load predictions from CSV
    log.info(f"Loading predictions from {results_path}")
    predictions_df = pd.read_csv(results_path)
    log.info(f"Loaded predictions for {len(predictions_df)} samples")
    
    # Map protein IDs to predictions
    # Create a dict for quick lookup
    pred_dict = {}
    for _, row in predictions_df.iterrows():
        protein_id = row.iloc[0]  # First column is protein ID
        pred_label_str = row.iloc[1]  # Second column is predicted label
        pred_dict[protein_id] = pred_label_str
    
    # Match embeddings with predictions
    matched_embeddings = []
    matched_labels = []
    mismatched_count = 0
    
    for i, prot_id in enumerate(protein_ids):
        if prot_id in pred_dict:
            matched_embeddings.append(embeddings_array[i])
            pred_label_str = pred_dict[prot_id]
            # Convert string label back to numeric if needed
            if task.lower() == "binary":
                label_numeric = 1 if pred_label_str == "Betalactamase" else 0
            else:
                # For multiclass, find the label in the map
                label_numeric = None
                for idx, name in label_map.items():
                    if name == pred_label_str:
                        label_numeric = idx
                        break
                if label_numeric is None:
                    log.warning(f"Could not find label {pred_label_str} in label map")
                    label_numeric = -1
            matched_labels.append(label_numeric)
        else:
            mismatched_count += 1
    
    if mismatched_count > 0:
        log.warning(f"Found {mismatched_count} embeddings with no matching predictions")
    
    matched_embeddings = np.array(matched_embeddings)
    log.info(f"Matched {len(matched_embeddings)} embeddings with predictions")
    
    # Transform with PCA
    embeddings_pca = pca.transform(matched_embeddings)
    
    # Compute statistics
    stats = compute_embedding_statistics(matched_embeddings)
    
    # Build DataFrame
    df_dict = {f"PC_{i}": embeddings_pca[:, i] for i in range(20)}
    df_dict["l2_norm"] = stats["l2_norm"]
    df_dict["l1_norm"] = stats["l1_norm"]
    df_dict["entropy"] = stats["entropy"]
    df_dict["label"] = matched_labels
    
    current_df = pd.DataFrame(df_dict)
    log.info(f"Current data DataFrame shape: {current_df.shape}")
    
    return current_df


# Get path to configs (Dynamic Absolute Path)
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
config_path = os.path.join(project_root, "configs")



@hydra.main(version_base=None, config_path=config_path, config_name="train_config")
def main(cfg: DictConfig):
    """
    Main pipeline: Load reference data, generate mock current data or load actual, and create report.
    
    Args:
        cfg: Hydra configuration
            - Use +current_data_jobid="<job_id>" to load real data, or omit for mock data
    """
    logging.basicConfig(level=logging.INFO)
    hydra_config = HydraConfig.get()
    task = hydra_config.runtime.choices.task

    log.info(f"Starting Data Drift Detection Pipeline for the {task} model...")
    output_dir = Path("./reports")

    processed_data_path = Path(f".data/processed_data_{task}.pt")
    
    # Load reference data and fit PCA
    reference_df, pca = load_reference_data(processed_data_path)
    
    # Generate mock or load current data
    num_mock_samples = len(reference_df)
    current_data_jobid = cfg.get("current_data_jobid", None)
    
    if current_data_jobid is None:
        current_df = generate_mock_current_data(
            num_samples=num_mock_samples,
            embedding_dim=1024,  # Original embedding dimension
            pca=pca
        )
        log.info("Using mock current data")
    else:
        # Load current data from specified job ID
        current_data_h5_path = Path(f".data/inference/{current_data_jobid}.h5")
        current_data_results_path = Path(f".data/inference/{current_data_jobid}_results.csv")
        log.info(f"Loading current data from job ID: {current_data_jobid}")
        current_df = load_current_data(current_data_h5_path, current_data_results_path, pca, task)
    
    # Save to CSVs
    #save_data_to_csv(reference_df, output_dir / "reference_data.csv", name="reference data")
    #save_data_to_csv(current_df, output_dir / "current_data.csv", name="current data")
    
    # Generate drift report
    generate_drift_report(reference_df, current_df, output_dir / "data_drift_report.html")
    
    log.info("Data drift detection pipeline completed!")

if __name__ == "__main__":
    # Example usage
    #uv run src/arginator_protein_classifier/data_drift.py +current_data_jobid="171c2ea7-ec04-4e83-bda5-4e0a7da77ed5"
    #By default it will generate mock data
    # Update these paths as needed
    main()