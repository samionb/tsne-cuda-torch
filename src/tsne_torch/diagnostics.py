"""Metrics and diagnostics helpers for TorchTSNE."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sklearn
import torch
from sklearn.neighbors import NearestNeighbors


def knn_overlap(reference: np.ndarray, candidate: np.ndarray, n_neighbors: int = 10) -> float:
    """
    Compute the mean k-nearest-neighbor overlap between two spaces.

    See also: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html

    :param reference: Reference coordinates or features.
    :param candidate: Candidate embedding to compare against the reference neighborhoods.
    :param n_neighbors: Number of neighbors to compare per sample.

    :return: Mean neighborhood-overlap ratio in ``[0, 1]``.
    """
    ref_nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(reference)
    cand_nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(candidate)
    ref_idx = ref_nn.kneighbors(return_distance=False)[:, 1:]
    cand_idx = cand_nn.kneighbors(return_distance=False)[:, 1:]

    overlaps = [
        len(set(ref_row.tolist()) & set(cand_row.tolist())) / float(n_neighbors)
        for ref_row, cand_row in zip(ref_idx, cand_idx)
    ]
    return float(np.mean(overlaps))


def library_versions() -> dict[str, str]:
    """
    Collect the core numeric library versions used by the benchmark runner.

    :return: Mapping from library name to version string.
    """
    return {
        'numpy': np.__version__,
        'scikit-learn': sklearn.__version__,
        'torch': torch.__version__,
    }


@dataclass
class FitDiagnostics:
    """Minimal fit diagnostics exposed by the estimator."""

    backend: str
    timings: dict
    device: str
    memory: dict = field(default_factory=dict)
