import h5py
import matplotlib
import numpy as np
import pandas as pd
import umap

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class UMAPEmbeddingVisualizer:
    """Visualize protein embeddings using UMAP dimensionality reduction."""
    
    def __init__(self, n_neighbors=200, min_dist=0.01, metric="cosine", random_state=42):
        """
        Initialize the UMAP visualizer.
        
        Args:
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            metric: Distance metric for UMAP
            om_state: Random seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.umap_model = None
        self.data = None
        self.labels = None
        self.X_umap = None
    
    def load_embeddings(self, h5_path, label):
        """Load embeddings from H5 file."""
        embeddings = []
        ids = []
        with h5py.File(h5_path, "r") as f:
            for seq_id in f.keys():
                vec = np.array(f[seq_id])
                embeddings.append(vec)
                ids.append(seq_id)
        return pd.DataFrame({
            "id": ids,
            "embedding": embeddings,
            "label": label
        })
    
    def load_all_datasets(self, file_map):
        """Load all datasets from file mapping."""
        dfs = []
        for label, path in file_map.items():
            dfs.append(self.load_embeddings(path, label))
        self.data = pd.concat(dfs, ignore_index=True)
        return self.data
    
    def prepare_data(self, binary_mode=False):
        """
        Stack embeddings and prepare labels.
        
        Args:
            binary_mode: If True, convert all non-query labels to 'card'
        """
        X = np.vstack(self.data["embedding"].values)
        labels = self.data["label"].values
        
        if binary_mode:
            labels = np.array([lbl if lbl == "query" else "card" for lbl in labels])
        
        self.labels = labels
        return X
    
    def fit_transform(self, X):
        """Fit UMAP model and transform embeddings."""
        self.umap_model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state
        )
        self.X_umap = self.umap_model.fit_transform(X)
        return self.X_umap
    
    def plot_umap(self, title, filename=None):
        """
        Plot UMAP visualization.
        
        Args:
            title: Title for the plot
            filename: Path to save the plot (if None, displays the plot)
        """
        if self.X_umap is None:
            raise ValueError("UMAP not fitted yet. Call fit_transform first.")
        
        plt.figure(figsize=(8, 6))
        for lbl in np.unique(self.labels):
            mask = self.labels == lbl
            x = self.X_umap[mask, 0]
            y = self.X_umap[mask, 1]
            plt.scatter(x, y, label=lbl, alpha=0.5, s=2)
        
        plt.legend()
        plt.title(title)
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300)
            print(f"Plot saved to {filename}")
        else:
            plt.show()
        
        plt.close()
    
    def run(self, file_map, binary_mode=False, output_filename=None):
        """
        Execute full pipeline: load data, fit UMAP, and plot.
        
        Args:
            file_map: Dictionary mapping labels to H5 file paths
            binary_mode: If True, convert to binary classification
            output_filename: Path to save the plot
        """
        print("Loading embeddings...")
        self.load_all_datasets(file_map)
        
        print("Preparing data...")
        X = self.prepare_data(binary_mode=binary_mode)
        
        print("Fitting UMAP...")
        self.fit_transform(X)
        
        print("Plotting UMAP...")
        title = "UMAP on PLM Embeddings of reference card proteins and query (cosine distance metric)"
        self.plot_umap(title, filename=output_filename)


# Usage example
if __name__ == "__main__":
    file_map = {
        "card_A": ".data/card_class_a_embeddings.h5",
        "card_B": ".data/card_class_b_embeddings.h5",
        "card_C": ".data/card_class_c_embeddings.h5",
        "card_D": ".data/card_class_d_embeddings.h5",
        "query": ".data/query_embeddings.h5",
    }
    
    visualizer = UMAPEmbeddingVisualizer()
    visualizer.run(file_map, binary_mode=True, output_filename=".data/umap_card_query.png")