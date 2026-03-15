"""High-performance Torch/CUDA t-SNE implementation."""

from . import memory
from .affinity import (
    binary_search_perplexity_torch,
    joint_probabilities_from_squared_distances,
    build_sparse_affinity_from_sparse_precomputed,
)
from .estimator import TorchTSNE
from .exact_backend import exact_kl_divergence_objective

__all__ = [
    'TorchTSNE',
    'binary_search_perplexity_torch',
    'build_sparse_affinity_from_sparse_precomputed',
    'joint_probabilities_from_squared_distances',
    'exact_kl_divergence_objective',
    'memory',
]
