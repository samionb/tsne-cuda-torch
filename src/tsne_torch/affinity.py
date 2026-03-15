"""Affinity and probability helpers for TorchTSNE."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from scipy.sparse import csr_array
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

MACHINE_EPSILON = np.finfo(np.double).eps
PERPLEXITY_TOLERANCE = 1e-5
MAX_PERPLEXITY_STEPS = 100
CUDA_KNN_BATCH_SIZE = 768


@dataclass
class SparseAffinity:
    """Sparse affinity container shared by the FFT backend."""

    matrix: csr_array
    rows: np.ndarray


def _synchronize(device: torch.device | None) -> None:
    """
    Synchronize the active CUDA stream when timing a CUDA-backed operation.

    :param device: Torch device associated with the timed operation.

    :return: None.
    """
    if device is not None and device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _timed_start(device: torch.device | None) -> float:
    """
    Start a timing window with CUDA synchronization when needed.

    :param device: Torch device associated with the timed operation.

    :return: Start timestamp from ``perf_counter``.
    """
    _synchronize(device)
    return perf_counter()


def _timed_stop(start: float, device: torch.device | None) -> float:
    """
    Stop a timing window with CUDA synchronization when needed.

    :param start: Start timestamp returned by ``_timed_start``.
    :param device: Torch device associated with the timed operation.

    :return: Elapsed wall-clock duration in seconds.
    """
    _synchronize(device)
    return perf_counter() - start


def _add_timing(timings: dict | None, key: str, duration: float) -> None:
    """
    Accumulate a named duration into an optional timing dictionary.

    :param timings: Mutable timing dictionary, or ``None`` to disable accumulation.
    :param key: Timing bucket name.
    :param duration: Duration in seconds to add.

    :return: None.
    """
    if timings is None:
        return
    timings[key] = timings.get(key, 0.0) + float(duration)


def _to_torch_with_optional_transfer_timing(
    values: np.ndarray,
    *,
    device: torch.device,
    timings: dict | None,
) -> torch.Tensor:
    """
    Materialize a NumPy array as a float32 Torch tensor while isolating CUDA upload time.

    :param values: NumPy array to materialize.
    :param device: Target Torch device.
    :param timings: Optional timing dictionary updated with transfer duration on CUDA.

    :return: Torch tensor view or copy on the requested device.
    """
    if device.type == 'cuda':
        start = _timed_start(device)
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        _add_timing(timings, 'host_device_transfer', _timed_stop(start, device))
        return tensor
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _build_dense_knn_squared_distances_cuda(
    x,
    *,
    n_neighbors: int,
    device: torch.device,
    timings: dict | None,
    batch_size: int = CUDA_KNN_BATCH_SIZE,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Build an exact Euclidean top-k neighbor graph with batched Torch CUDA kernels.

    :param x: Dense feature matrix.
    :param n_neighbors: Number of nearest neighbors to keep per sample.
    :param device: CUDA device used for the batched distance sweep.
    :param timings: Optional timing dictionary updated with compute and transfer durations.
    :param batch_size: Query batch size for the pairwise-distance sweep.

    :return: Tuple ``(neighbor_indices, squared_distances)``.
    """
    x_array = np.asarray(x, dtype=np.float32)
    n_samples = x_array.shape[0]
    if n_samples <= n_neighbors:
        raise ValueError('n_neighbors must be smaller than the number of samples')

    x_tensor = _to_torch_with_optional_transfer_timing(x_array, device=device, timings=timings)
    squared_norms = torch.sum(x_tensor * x_tensor, dim=1)
    distances_tensor = torch.empty((n_samples, n_neighbors), dtype=torch.float32, device=device)
    neighbor_indices = np.empty((n_samples, n_neighbors), dtype=np.int64)

    for start_row in range(0, n_samples, batch_size):
        stop_row = min(start_row + batch_size, n_samples)
        query = x_tensor[start_row:stop_row]

        start = _timed_start(device)
        dist = squared_norms[start_row:stop_row, None] + squared_norms[None, :] - 2.0 * (query @ x_tensor.T)
        dist.clamp_min_(0.0)
        local_rows = torch.arange(stop_row - start_row, device=device)
        dist[local_rows, start_row + local_rows] = torch.inf
        values, neighbors = torch.topk(dist, k=n_neighbors, dim=1, largest=False, sorted=True)
        distances_tensor[start_row:stop_row] = values
        _add_timing(timings, 'knn_build', _timed_stop(start, device))

        start = _timed_start(device)
        neighbor_indices[start_row:stop_row] = neighbors.cpu().numpy().astype(np.int64, copy=False)
        _add_timing(timings, 'host_device_transfer', _timed_stop(start, device))

    return neighbor_indices, distances_tensor


def compute_learning_rate(
    n_samples: int,
    early_exaggeration: float,
    learning_rate: float | str,
) -> float:
    """
    Compute the effective t-SNE learning rate with sklearn-compatible semantics.

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    :param n_samples: Number of input samples.
    :param early_exaggeration: Early exaggeration factor used by t-SNE.
    :param learning_rate: Requested learning rate, or ``'auto'`` to mirror sklearn behavior.

    :return: Effective learning rate as a float.
    """
    if learning_rate == 'auto':
        return float(max(n_samples / early_exaggeration / 4.0, 50.0))
    return float(learning_rate)


def binary_search_perplexity_torch(
    sqdistances: torch.Tensor,
    desired_perplexity: float,
    verbose: int = 0,
    max_steps: int = MAX_PERPLEXITY_STEPS,
    tolerance: float = PERPLEXITY_TOLERANCE,
) -> torch.Tensor:
    """
    Solve the conditional Gaussian bandwidth search for a target perplexity on a Torch device.

    Reference: https://jmlr.org/papers/v9/vandermaaten08a.html

    :param sqdistances: Squared-distance matrix or neighbor-distance matrix on a Torch device.
    :param desired_perplexity: Target perplexity for each sample.
    :param verbose: Verbosity level. When non-zero, prints a mean sigma summary.
    :param max_steps: Maximum number of binary-search iterations.
    :param tolerance: Entropy tolerance for the perplexity search.

    :return: Conditional probability matrix with one row per sample.
    """
    if sqdistances.ndim != 2:
        raise ValueError('sqdistances must be a 2D tensor')

    device = sqdistances.device
    n_samples, n_neighbors = sqdistances.shape
    using_neighbors = n_neighbors < n_samples
    working = sqdistances.to(dtype=torch.float64)

    beta = torch.ones((n_samples, 1), dtype=torch.float64, device=device)
    beta_min = torch.full_like(beta, -torch.inf)
    beta_max = torch.full_like(beta, torch.inf)
    desired_entropy = torch.log(torch.tensor(desired_perplexity, dtype=torch.float64, device=device))

    mask = None
    if not using_neighbors and n_samples == n_neighbors:
        mask = torch.eye(n_samples, dtype=torch.bool, device=device)

    probabilities = torch.zeros_like(working)

    for _ in range(max_steps):
        probabilities = torch.exp(-working * beta)
        if mask is not None:
            probabilities = probabilities.masked_fill(mask, 0.0)

        sum_pi = probabilities.sum(dim=1, keepdim=True).clamp_min(MACHINE_EPSILON)
        probabilities = probabilities / sum_pi
        entropy = torch.log(sum_pi.squeeze(1)) + beta.squeeze(1) * (working * probabilities).sum(dim=1)
        entropy_diff = entropy - desired_entropy

        done = torch.abs(entropy_diff) <= tolerance
        if torch.all(done):
            break

        greater = (entropy_diff > 0.0).unsqueeze(1)
        beta_min = torch.where(greater, beta, beta_min)
        beta_max = torch.where(~greater, beta, beta_max)

        beta = torch.where(
            greater & torch.isinf(beta_max),
            beta * 2.0,
            torch.where(
                greater,
                (beta + beta_max) / 2.0,
                torch.where(
                    torch.isinf(beta_min),
                    beta / 2.0,
                    (beta + beta_min) / 2.0,
                ),
            ),
        )

    if verbose:
        mean_sigma = torch.sqrt(torch.tensor(n_samples, dtype=torch.float64, device=device) / beta.mean())
        print(f'[TorchTSNE] Mean sigma: {mean_sigma.item():.6f}')

    return probabilities.to(dtype=torch.float32)


def squared_euclidean_distances_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Compute a dense squared Euclidean distance matrix using Torch tensor algebra.

    :param x: Input feature matrix with shape ``(n_samples, n_features)``.

    :return: Dense squared-distance matrix with shape ``(n_samples, n_samples)``.
    """
    squared_norm = (x * x).sum(dim=1, keepdim=True)
    distances = squared_norm + squared_norm.T - 2.0 * (x @ x.T)
    distances.clamp_(min=0.0)
    return distances


def joint_probabilities_from_squared_distances(
    sqdistances: torch.Tensor,
    perplexity: float,
    verbose: int = 0,
    timings: dict | None = None,
) -> torch.Tensor:
    """
    Build the dense symmetric joint probability matrix used by exact t-SNE.

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    :param sqdistances: Squared-distance matrix on the target Torch device.
    :param perplexity: Requested perplexity.
    :param verbose: Verbosity level passed to the perplexity search.
    :param timings: Optional timing dictionary updated with sub-stage durations.

    :return: Symmetric probability matrix ``P``.
    """
    start = _timed_start(sqdistances.device)
    conditional = binary_search_perplexity_torch(sqdistances, perplexity, verbose=verbose)
    _add_timing(timings, 'perplexity_search', _timed_stop(start, sqdistances.device))

    start = _timed_start(sqdistances.device)
    p = conditional + conditional.T
    p = p / torch.clamp(p.sum(), min=MACHINE_EPSILON)
    p.fill_diagonal_(0.0)
    p = torch.clamp(p, min=MACHINE_EPSILON)
    _add_timing(timings, 'symmetrize', _timed_stop(start, sqdistances.device))
    return p


def compute_dense_squared_distances(
    x,
    *,
    metric: str | callable,
    metric_params: dict | None,
    n_jobs: int | None,
) -> np.ndarray:
    """
    Compute a dense squared-distance matrix with sklearn-compatible metric handling.

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

    :param x: Input samples.
    :param metric: Distance metric name or callable.
    :param metric_params: Optional metric keyword arguments.
    :param n_jobs: Parallel job count forwarded to sklearn when supported.

    :return: Dense squared-distance matrix as a NumPy array.
    """
    if metric == 'euclidean':
        return pairwise_distances(x, metric=metric, squared=True)

    metric_params_ = metric_params or {}
    distances = pairwise_distances(x, metric=metric, n_jobs=n_jobs, **metric_params_)
    if np.any(distances < 0):
        raise ValueError('All distances should be positive, the metric given is not correct')
    distances = np.asarray(distances, dtype=np.float64)
    distances **= 2
    return distances


def build_sparse_affinity_from_knn(
    x,
    *,
    perplexity: float,
    metric: str | callable,
    metric_params: dict | None,
    n_jobs: int | None,
    device: torch.device,
    verbose: int = 0,
    timings: dict | None = None,
) -> SparseAffinity:
    """
    Build sparse joint affinities from a k-nearest-neighbor graph for the FFT backend.

    See also: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html

    :param x: Input samples.
    :param perplexity: Requested perplexity.
    :param metric: Distance metric name or callable.
    :param metric_params: Optional metric keyword arguments.
    :param n_jobs: Parallel job count forwarded to sklearn when supported.
    :param device: Torch device used for the perplexity search.
    :param verbose: Verbosity level passed to the perplexity search.
    :param timings: Optional timing dictionary updated with sub-stage durations.

    :return: Sparse affinity container shared by the FFT backend.
    """
    n_samples = x.shape[0]
    n_neighbors = min(n_samples - 1, int(3.0 * perplexity + 1))

    if device.type == 'cuda' and metric == 'euclidean':
        neighbor_indices, distances_tensor = _build_dense_knn_squared_distances_cuda(
            x,
            n_neighbors=n_neighbors,
            device=device,
            timings=timings,
        )
        start = _timed_start(device)
        probs = binary_search_perplexity_torch(
            distances_tensor,
            perplexity,
            verbose=verbose,
        )
        _add_timing(timings, 'perplexity_search', _timed_stop(start, device))

        start = _timed_start(device)
        probs_cpu = probs.cpu().numpy()
        _add_timing(timings, 'host_device_transfer', _timed_stop(start, device))

        indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors, dtype=np.int64)
        start = _timed_start(None)
        p = csr_array((probs_cpu.ravel(), neighbor_indices.ravel(), indptr), shape=(n_samples, n_samples))
        p = p + p.T
        p /= max(p.sum(), MACHINE_EPSILON)
        rows = np.repeat(np.arange(n_samples, dtype=np.int64), np.diff(p.indptr))
        _add_timing(timings, 'symmetrize', _timed_stop(start, None))
        return SparseAffinity(matrix=p, rows=rows)

    knn = NearestNeighbors(
        algorithm='auto',
        n_neighbors=n_neighbors,
        metric=metric,
        metric_params=metric_params,
        n_jobs=n_jobs,
    )
    knn.fit(x)
    start = _timed_start(None)
    distances_nn = knn.kneighbors_graph(mode='distance')
    _add_timing(timings, 'knn_build', _timed_stop(start, None))
    distances_nn.sort_indices()

    distances_nn.data = np.square(np.asarray(distances_nn.data, dtype=np.float64))

    distances_data = distances_nn.data.reshape(n_samples, -1)
    distances_tensor = _to_torch_with_optional_transfer_timing(distances_data, device=device, timings=timings)
    start = _timed_start(device)
    probs = binary_search_perplexity_torch(
        distances_tensor,
        perplexity,
        verbose=verbose,
    )
    _add_timing(timings, 'perplexity_search', _timed_stop(start, device))

    start = _timed_start(device)
    probs_cpu = probs.cpu().numpy()
    _add_timing(timings, 'host_device_transfer', _timed_stop(start, device))

    start = _timed_start(None)
    p = csr_array(
        (probs_cpu.ravel(), distances_nn.indices, distances_nn.indptr),
        shape=(n_samples, n_samples),
    )
    p = p + p.T
    p /= max(p.sum(), MACHINE_EPSILON)
    rows = np.repeat(np.arange(n_samples, dtype=np.int64), np.diff(p.indptr))
    _add_timing(timings, 'symmetrize', _timed_stop(start, None))
    return SparseAffinity(matrix=p, rows=rows)


def build_sparse_affinity_from_precomputed(
    distances: np.ndarray,
    *,
    perplexity: float,
    device: torch.device,
    verbose: int = 0,
    timings: dict | None = None,
) -> SparseAffinity:
    """
    Build sparse affinities from a dense precomputed distance matrix.

    :param distances: Dense pairwise distance matrix.
    :param perplexity: Requested perplexity.
    :param device: Torch device used for the perplexity search.
    :param verbose: Verbosity level passed to the perplexity search.
    :param timings: Optional timing dictionary updated with sub-stage durations.

    :return: Sparse affinity container shared by the FFT backend.
    """
    n_samples = distances.shape[0]
    n_neighbors = min(n_samples - 1, int(3.0 * perplexity + 1))
    start = _timed_start(None)
    order = np.argsort(distances, axis=1)[:, 1: n_neighbors + 1]
    _add_timing(timings, 'knn_build', _timed_stop(start, None))
    sorted_distances = np.take_along_axis(distances, order, axis=1)
    sorted_distances_tensor = _to_torch_with_optional_transfer_timing(sorted_distances, device=device, timings=timings)
    start = _timed_start(device)
    probs = binary_search_perplexity_torch(
        sorted_distances_tensor,
        perplexity,
        verbose=verbose,
    )
    _add_timing(timings, 'perplexity_search', _timed_stop(start, device))

    start = _timed_start(device)
    probs_cpu = probs.cpu().numpy()
    _add_timing(timings, 'host_device_transfer', _timed_stop(start, device))

    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors, dtype=np.int64)
    start = _timed_start(None)
    p = csr_array((probs_cpu.ravel(), order.ravel(), indptr), shape=(n_samples, n_samples))
    p = p + p.T
    p /= max(p.sum(), MACHINE_EPSILON)
    rows = np.repeat(np.arange(n_samples, dtype=np.int64), np.diff(p.indptr))
    _add_timing(timings, 'symmetrize', _timed_stop(start, None))
    return SparseAffinity(matrix=p, rows=rows)


def build_sparse_affinity_from_sparse_precomputed(
    distances: csr_array,
    *,
    perplexity: float,
    device: torch.device,
    verbose: int = 0,
    timings: dict | None = None,
) -> SparseAffinity:
    """
    Build sparse affinities from a sparse precomputed distance graph.

    :param distances: Sparse precomputed distance graph in CSR form.
    :param perplexity: Requested perplexity.
    :param device: Torch device used for the perplexity search.
    :param verbose: Verbosity level passed to the perplexity search.
    :param timings: Optional timing dictionary updated with sub-stage durations.

    :return: Sparse affinity container shared by the FFT backend.
    """
    distances = distances.tocsr(copy=True)
    distances.sort_indices()

    row_lengths = np.diff(distances.indptr)
    if len(row_lengths) == 0 or np.any(row_lengths != row_lengths[0]):
        raise ValueError('Sparse precomputed distance graph must have the same number of neighbors in every row')

    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, row_lengths[0]).astype(np.float32, copy=False)
    distances_tensor = _to_torch_with_optional_transfer_timing(distances_data, device=device, timings=timings)
    start = _timed_start(device)
    probs = binary_search_perplexity_torch(
        distances_tensor,
        perplexity,
        verbose=verbose,
    )
    _add_timing(timings, 'perplexity_search', _timed_stop(start, device))

    start = _timed_start(device)
    probs_cpu = probs.cpu().numpy()
    _add_timing(timings, 'host_device_transfer', _timed_stop(start, device))

    start = _timed_start(None)
    p = csr_array((probs_cpu.ravel(), distances.indices, distances.indptr), shape=distances.shape)
    p = p + p.T
    p /= max(p.sum(), MACHINE_EPSILON)
    rows = np.repeat(np.arange(n_samples, dtype=np.int64), np.diff(p.indptr))
    _add_timing(timings, 'symmetrize', _timed_stop(start, None))
    return SparseAffinity(matrix=p, rows=rows)
