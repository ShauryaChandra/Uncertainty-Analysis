"""Scenario reduction and dimensionality reduction."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def reduce_scenarios(scenarios: np.ndarray, n_clusters: int = 8,
                     random_state: int = 42) -> np.ndarray:
    """Return K-means cluster centroids as representative scenarios."""
    data = np.asarray(scenarios, dtype=float)
    if data.ndim != 2 or data.shape[0] < 1:
        raise ValueError("scenarios must be a 2D array with at least one row.")
    if not 1 <= n_clusters <= data.shape[0]:
        raise ValueError("n_clusters must be between 1 and number of scenarios.")
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(data)
    return model.cluster_centers_

def pca_projection(scenarios: np.ndarray, n_components: int = 2):
    """Standardize scenario features and project scenarios onto PCA components."""
    data = np.asarray(scenarios, dtype=float)
    if data.ndim != 2 or min(data.shape) < n_components:
        raise ValueError("Not enough rows/features for requested PCA components.")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(data)
    model = PCA(n_components=n_components, random_state=42)
    projected = model.fit_transform(standardized)
    return projected, model.explained_variance_ratio_, model
