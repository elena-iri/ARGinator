import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib
from collections import defaultdict
import logging
import os

# Configuration
DATA_DIR = Path("data")
EXPECTED_DIM = 1024
MIN_CLASS_RATIO = 0.10
REPORT_FILE = "data_report.md"
PLOT_FILE = "distribution.png"

# --- CONFIGURATION FROM ENV OR DEFAULTS ---
# We grab these directly here so we don't rely on data.py's heavy init
TASK = os.getenv("TASK", "binary").lower()
FILE_PATTERN = os.getenv("FILE_PATTERN", "card_class_{}_embeddings.h5")
CLASSES = os.getenv("CLASSES", "").split(",") if os.getenv("CLASSES") else []
CLASSES = [c.strip() for c in CLASSES]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DataValidator")

def get_label(filename):
    """
    Standalone labeling logic replicated from data.py to avoid 
    instantiating the heavy dataset class during CI.
    """
    # Binary Logic
    if TASK == "binary":
        if "non" in filename.lower():
            return 0, "Non-Protein (0)"
        return 1, "Protein (1)"
    
    # Multiclass Logic
    elif TASK == "multiclass":
        for idx, class_name in enumerate(CLASSES):
            expected_filename = FILE_PATTERN.format(class_name)
            if filename == expected_filename or filename.endswith(expected_filename):
                return idx, f"{class_name} ({idx})"
        
        return -1, "Unknown"
    
    return -1, "Unknown"

def check_data():
    log.info(f"🔍 Starting Data Quality Checks in {DATA_DIR}...")
    log.info(f"ℹ️  Task: {TASK}, Classes: {CLASSES}")

    h5_files = list(DATA_DIR.glob("*.h5"))
    if not h5_files:
        log.error("❌ No .h5 files found!")
        sys.exit(1)

    total_samples = 0
    class_counts = defaultdict(int)
    sample_hashes = set()
    
    errors = {"sanity": [], "leakage": [], "distribution": []}

    # --- 1. Validation Loop ---
    for file_path in h5_files:
        label, label_name = get_label(file_path.name)
        
        # Skip unknown files
        if label == -1: 
            log.warning(f"Skipping unknown file: {file_path.name}")
            continue

        try:
            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    data = f[key][:]
                    total_samples += 1
                    class_counts[label_name] += 1

                    # Sanity Check
                    if data.shape != (EXPECTED_DIM,):
                        errors["sanity"].append(f"{file_path.name}/{key}: Invalid shape {data.shape}")
                    if np.isnan(data).any():
                        errors["sanity"].append(f"{file_path.name}/{key}: Contains NaNs")

                    # Leakage Check (Hashing)
                    data_hash = hashlib.md5(data.tobytes()).hexdigest()
                    if data_hash in sample_hashes:
                        errors["leakage"].append(f"{file_path.name}/{key}: Duplicate sample")
                    else:
                        sample_hashes.add(data_hash)

        except Exception as e:
            errors["sanity"].append(f"{file_path.name}: Read Error ({str(e)})")

    # --- 2. Distribution Check ---
    ratios = {}
    for label, count in class_counts.items():
        ratio = count / total_samples if total_samples > 0 else 0
        ratios[label] = ratio
        if ratio < MIN_CLASS_RATIO:
            errors["distribution"].append(f"Class '{label}' is {ratio:.1%} (< {MIN_CLASS_RATIO:.0%})")

    # --- 3. Generate Plot ---
    if total_samples > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(list(class_counts.keys()), list(class_counts.values()), color=['#ff9999', '#66b3ff'])
        ax.set_title(f"Class Distribution (N={total_samples})")
        plt.savefig(PLOT_FILE)
    else:
        log.error("No samples found processing!")
        sys.exit(1)

    # --- 4. Generate Report ---
    with open(REPORT_FILE, "w") as f:
        f.write("# Data Quality Report\n\n")
        f.write("## 1. Distribution\n")
        f.write(f"![Distribution]({PLOT_FILE})\n\n")
        
        for label, count in class_counts.items():
            status = "✅" if ratios[label] >= MIN_CLASS_RATIO else "❌ Imbalanced"
            f.write(f"- **{label}**: {count} ({ratios[label]:.1%}) {status}\n")

        if any(errors.values()):
            f.write("\n## 2. Critical Issues\n")
            if errors["sanity"]: f.write(f"- **Sanity**: {len(errors['sanity'])} errors\n")
            if errors["leakage"]: f.write(f"- **Leakage**: {len(errors['leakage'])} duplicates\n")
            if errors["distribution"]: f.write(f"- **Balance**: {len(errors['distribution'])} issues\n")
        else:
            f.write("\n## 2. Status\n✅ All checks passed.")

    if any(errors.values()):
        log.error("❌ Data checks failed.")
        sys.exit(1)

if __name__ == "__main__":
    check_data()