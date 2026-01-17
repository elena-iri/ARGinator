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
DATA_DIR = Path(os.getenv("DATA_PATH", ".data"))
EXPECTED_DIM = 1024
MIN_CLASS_RATIO = 0.10
REPORT_FILE = "data_report.md"

# Output images
PLOT_BINARY = "distribution_binary.png"
PLOT_MULTICLASS = "distribution_multiclass.png"

# --- CONFIGURATION ---
FILE_PATTERN = os.getenv("FILE_PATTERN", "card_class_{}_embeddings.h5")
CLASSES = os.getenv("CLASSES", "").split(",") if os.getenv("CLASSES") else []
CLASSES = [c.strip() for c in CLASSES]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("DataValidator")

def get_binary_label(filename):
    """Always returns a label based on 'non' vs others."""
    if "non" in filename.lower():
        return 0, "Non-Protein (0)"
    return 1, "Protein (1)"

def get_multiclass_label(filename):
    """Returns label index if filename matches a class in CLASSES, else -1."""
    if not CLASSES:
        return -1, "No Classes Defined"
        
    for idx, class_name in enumerate(CLASSES):
        # Check if class name is in filename (e.g. "card_class_a_embeddings")
        expected_part = class_name
        
        # More robust check: use the pattern if possible, or simple substring
        if expected_part in filename:
             return idx, f"{class_name} ({idx})"
             
    return -1, "Unknown/Background"

def plot_distribution(counts, title, filename):
    """Helper to generate a bar chart."""
    if not counts:
        return False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(counts.keys())
    values = list(counts.values())
    
    # Dynamic colors
    colors = plt.cm.Paired(np.linspace(0, 1, len(labels)))
    
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Samples")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return True

def check_data():
    log.info(f"🔍 Starting Data Quality Checks in {DATA_DIR}...")
    
    if not DATA_DIR.exists():
        log.error(f"❌ Directory {DATA_DIR} does not exist!")
        sys.exit(1)

    h5_files = list(DATA_DIR.glob("*.h5"))
    if not h5_files:
        log.error(f"❌ No .h5 files found in {DATA_DIR}!")
        sys.exit(1)

    total_samples = 0
    
    # Dual Tracking
    binary_counts = defaultdict(int)
    multiclass_counts = defaultdict(int)
    
    sample_hashes = set()
    errors = {"sanity": [], "leakage": [], "distribution": []}

    # --- 1. Validation Loop ---
    for file_path in h5_files:
        # Get BOTH labels
        bin_lbl, bin_name = get_binary_label(file_path.name)
        multi_lbl, multi_name = get_multiclass_label(file_path.name)

        try:
            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    data = f[key][:]
                    total_samples += 1
                    
                    # Update BOTH counters
                    binary_counts[bin_name] += 1
                    
                    # Only count multiclass if it's a known class (skip background/unknown)
                    if multi_lbl != -1:
                        multiclass_counts[multi_name] += 1

                    # Sanity & Leakage (Same as before)
                    if data.shape != (EXPECTED_DIM,):
                        errors["sanity"].append(f"{file_path.name}/{key}: Invalid shape {data.shape}")
                    if np.isnan(data).any():
                        errors["sanity"].append(f"{file_path.name}/{key}: Contains NaNs")

                    data_hash = hashlib.md5(data.tobytes()).hexdigest()
                    if data_hash in sample_hashes:
                        errors["leakage"].append(f"{file_path.name}/{key}: Duplicate sample")
                    else:
                        sample_hashes.add(data_hash)

        except Exception as e:
            errors["sanity"].append(f"{file_path.name}: Read Error ({str(e)})")

    # --- 2. Generate Plots ---
    has_bin_plot = plot_distribution(binary_counts, f"Binary Distribution (N={total_samples})", PLOT_BINARY)
    has_multi_plot = plot_distribution(multiclass_counts, f"Multiclass Distribution (Subset N={sum(multiclass_counts.values())})", PLOT_MULTICLASS)

    # --- 3. Check Ratios (Focus on Binary for Failure) ---
    # We mainly fail CI if the Binary split is broken. Multiclass is often sparse.
    for label, count in binary_counts.items():
        ratio = count / total_samples
        if ratio < MIN_CLASS_RATIO:
            errors["distribution"].append(f"Binary Class '{label}' is {ratio:.1%} (< {MIN_CLASS_RATIO:.0%})")

    # --- 4. Generate Report ---
    with open(REPORT_FILE, "w") as f:
        f.write("# Data Quality Report\n\n")
        f.write(f"**Total Samples:** {total_samples}\n\n")

        # Section A: Binary
        f.write("## 1. Binary Overview (Protein vs Non-Protein)\n")
        if has_bin_plot:
            f.write(f"![Binary Distribution]({PLOT_BINARY})\n\n")
        for label, count in binary_counts.items():
            ratio = count / total_samples
            status = "✅" if ratio >= MIN_CLASS_RATIO else "❌ Imbalanced"
            f.write(f"- **{label}**: {count} ({ratio:.1%}) {status}\n")

        # Section B: Multiclass
        f.write("\n## 2. Multiclass Breakdown\n")
        if multiclass_counts:
            if has_multi_plot:
                f.write(f"![Multiclass Distribution]({PLOT_MULTICLASS})\n\n")
            for label, count in sorted(multiclass_counts.items()):
                f.write(f"- **{label}**: {count}\n")
        else:
            f.write("_No specific classes detected (Check CLASSES env var if this is unexpected)._\n")

        # Section C: Errors
        if any(errors.values()):
            f.write("\n## 3. Critical Issues\n")
            if errors["sanity"]: f.write(f"- **Sanity**: {len(errors['sanity'])} errors\n")
            if errors["leakage"]: f.write(f"- **Leakage**: {len(errors['leakage'])} duplicates\n")
            if errors["distribution"]: f.write(f"- **Balance**: {len(errors['distribution'])} issues\n")
        else:
            f.write("\n## 3. Status\n✅ All data checks passed.")

    if any(errors.values()):
        log.error("❌ Data checks failed.")
        sys.exit(1)

if __name__ == "__main__":
    check_data()