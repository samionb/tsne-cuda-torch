"""Shared helpers for TorchTSNE tests."""

import numpy as np
from scipy.sparse import csr_array
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances


def make_blobs_data(n_samples=50, n_features=4, centers=4, random_state=0):
    """Create deterministic float32 blob data for tests."""
    x, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=1.2,
        random_state=random_state,
    )
    return x.astype(np.float32)


def make_sparse_distance_graph(x: np.ndarray, n_neighbors: int) -> csr_array:
    """Create a fixed-width sparse squared-distance graph for tests."""
    distances = pairwise_distances(x, squared=True).astype(np.float32)
    order = np.argsort(distances, axis=1)[:, 1 : n_neighbors + 1]
    values = np.take_along_axis(distances, order, axis=1)
    indptr = np.arange(0, x.shape[0] * n_neighbors + 1, n_neighbors, dtype=np.int64)
    return csr_array((values.ravel(), order.ravel(), indptr), shape=(x.shape[0], x.shape[0]))
