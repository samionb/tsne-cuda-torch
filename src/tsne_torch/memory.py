"""Memory estimation and safety checks for TorchTSNE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


FLOAT32_BYTES = 4
INT64_BYTES = 8
CPU_HEADROOM_RATIO = 0.90
CUDA_HEADROOM_RATIO = 0.85
RUNTIME_OVERHEAD_RATIO = 1.50


@dataclass
class MemoryEstimate:
    """Estimated runtime memory requirement for one backend invocation."""

    backend: str
    device: str
    required_bytes: int
    available_bytes: int | None
    safe_budget_bytes: int | None
    fits: bool | None
    details: dict[str, int]
    metadata: dict[str, int]

    def as_dict(self) -> dict:
        """Return a JSON-friendly representation for diagnostics."""
        return {
            'backend': self.backend,
            'device': self.device,
            'required_bytes': int(self.required_bytes),
            'available_bytes': None if self.available_bytes is None else int(self.available_bytes),
            'safe_budget_bytes': None if self.safe_budget_bytes is None else int(self.safe_budget_bytes),
            'fits': self.fits,
            'details': {key: int(value) for key, value in self.details.items()},
            'metadata': {key: int(value) for key, value in self.metadata.items()},
        }


def format_num_bytes(num_bytes: int | None) -> str:
    """
    Format a byte count into a human-readable string.

    :param num_bytes: Byte count, or ``None`` when unknown.

    :return: Human-readable memory string.
    """
    if num_bytes is None:
        return 'unknown'

    value = float(num_bytes)
    units = ('B', 'KiB', 'MiB', 'GiB', 'TiB')
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f'{value:.1f} {unit}'
        value /= 1024.0
    return f'{value:.1f} TiB'


def available_memory_bytes(device: torch.device) -> int | None:
    """
    Query the currently available memory on the requested device.

    :param device: Target Torch device.

    :return: Available memory in bytes, or ``None`` when it cannot be queried.
    """
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            return None
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return int(free_bytes)

    if psutil is None:
        return None
    return int(psutil.virtual_memory().available)


def _dense_matrix_bytes(n_rows: int, n_cols: int, dtype_bytes: int = FLOAT32_BYTES) -> int:
    """
    Compute the memory footprint of a dense matrix.

    :param n_rows: Number of rows.
    :param n_cols: Number of columns.
    :param dtype_bytes: Bytes per element.

    :return: Matrix footprint in bytes.
    """
    return int(n_rows) * int(n_cols) * int(dtype_bytes)


def _dense_input_tensor_bytes(x) -> int:
    """
    Estimate the footprint of a dense float32 tensor materialized from the estimator input.

    :param x: Dense estimator input.

    :return: Dense input tensor footprint in bytes.
    """
    return _dense_matrix_bytes(*x.shape)


def _dense_exact_details(x, *, n_components: int) -> tuple[dict[str, int], dict[str, int]]:
    """
    Estimate the dominant allocation categories for the dense exact backend.

    :param x: Input data passed to the estimator.
    :param n_components: Embedding dimensionality.

    :return: Tuple ``(detail_bytes, metadata)``.
    """
    n_samples = int(x.shape[0])
    details = {
        'distance_tensor_bytes': _dense_matrix_bytes(n_samples, n_samples),
        'probability_tensor_bytes': _dense_matrix_bytes(n_samples, n_samples),
        # dist, inv, q, and weights all live at this scale inside the exact objective.
        'objective_workspace_bytes': 4 * _dense_matrix_bytes(n_samples, n_samples),
        # params, update, gains, and gradient.
        'optimizer_state_bytes': 4 * _dense_matrix_bytes(n_samples, n_components),
        'input_tensor_bytes': _dense_input_tensor_bytes(x),
    }
    return details, {}


def _fft_dense_details(x, *, n_components: int, grid_size: int) -> tuple[dict[str, int], dict[str, int]]:
    """
    Estimate the dominant allocation categories for the dense FFT backend.

    :param x: Input data passed to the estimator.
    :param n_components: Embedding dimensionality.
    :param grid_size: Side length of the FFT grid.

    :return: Tuple ``(detail_bytes, metadata)``.
    """
    n_samples = int(x.shape[0])
    conv_size = 2 * grid_size - 1
    details = {
        'affinity_tensor_bytes': _dense_matrix_bytes(n_samples, n_samples),
        'distance_tensor_bytes': _dense_matrix_bytes(n_samples, n_samples),
        # diff, dist, q, and attractive weights are all materialized at this scale.
        'objective_workspace_bytes': 5 * _dense_matrix_bytes(n_samples, n_samples),
        'optimizer_state_bytes': 4 * _dense_matrix_bytes(n_samples, n_components),
        'grid_workspace_bytes': 8 * _dense_matrix_bytes(conv_size, conv_size),
        'input_tensor_bytes': _dense_input_tensor_bytes(x),
    }
    return details, {}


def _fft_sparse_details(
    x,
    *,
    metric: str,
    n_components: int,
    grid_size: int,
    perplexity: float,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Estimate the dominant allocation categories for the sparse FFT backend.

    :param x: Input data passed to the estimator.
    :param metric: Distance metric mode.
    :param n_components: Embedding dimensionality.
    :param grid_size: Side length of the FFT grid.
    :param perplexity: Requested perplexity used to infer neighbor count when needed.

    :return: Tuple ``(detail_bytes, metadata)``.
    """
    n_samples = int(x.shape[0])
    conv_size = 2 * grid_size - 1
    if metric == 'precomputed' and sp.issparse(x):
        affinity_nnz = int(min(n_samples * max(n_samples - 1, 0), 2 * x.nnz))
    else:
        n_neighbors = min(n_samples - 1, int(3.0 * perplexity + 1))
        affinity_nnz = int(min(n_samples * max(n_samples - 1, 0), 2 * n_samples * n_neighbors))

    details = {
        'affinity_index_bytes': affinity_nnz * (2 * INT64_BYTES + FLOAT32_BYTES),
        'edge_workspace_bytes': affinity_nnz * (n_components + 2) * FLOAT32_BYTES,
        'optimizer_state_bytes': 4 * _dense_matrix_bytes(n_samples, n_components),
        'grid_workspace_bytes': 8 * _dense_matrix_bytes(conv_size, conv_size),
    }

    if metric != 'precomputed':
        details['input_tensor_bytes'] = _dense_input_tensor_bytes(x)
    elif not sp.issparse(x):
        details['input_tensor_bytes'] = _dense_input_tensor_bytes(x)
    else:
        details['input_tensor_bytes'] = 0
    return details, {'affinity_nnz_estimate': affinity_nnz}


def estimate_tsne_memory(
    x,
    *,
    method: str,
    metric: str,
    n_components: int,
    perplexity: float,
    device: torch.device,
    grid_size: int = 256,
) -> MemoryEstimate:
    """
    Estimate runtime memory requirements for a specific TorchTSNE backend.

    :param x: Input data passed to the estimator.
    :param method: Requested t-SNE method.
    :param metric: Distance metric mode.
    :param n_components: Embedding dimensionality.
    :param perplexity: Requested perplexity.
    :param device: Target Torch device.
    :param grid_size: Side length of the FFT grid for FFT-based backends.

    :return: Memory estimate object with required bytes, available bytes, and fit status.
    """
    if method == 'exact':
        backend = 'torch_exact'
        details, metadata = _dense_exact_details(x, n_components=n_components)
    elif method == 'fft' and int(x.shape[0]) <= 256:
        backend = 'torch_fft_dense'
        details, metadata = _fft_dense_details(x, n_components=n_components, grid_size=grid_size)
    elif method == 'fft':
        backend = 'torch_fft'
        details, metadata = _fft_sparse_details(
            x,
            metric=metric,
            n_components=n_components,
            grid_size=grid_size,
            perplexity=perplexity,
        )
    else:
        backend = f'sklearn_{method}'
        details = {}
        metadata = {}

    required_bytes = int(sum(details.values()) * RUNTIME_OVERHEAD_RATIO)
    metadata = {**metadata, 'runtime_overhead_ratio_x100': int(RUNTIME_OVERHEAD_RATIO * 100)}
    available_bytes = available_memory_bytes(device)
    safe_budget_bytes = None
    fits = None
    if available_bytes is not None:
        headroom_ratio = CUDA_HEADROOM_RATIO if device.type == 'cuda' else CPU_HEADROOM_RATIO
        safe_budget_bytes = int(available_bytes * headroom_ratio)
        fits = required_bytes <= safe_budget_bytes

    return MemoryEstimate(
        backend=backend,
        device=str(device),
        required_bytes=required_bytes,
        available_bytes=available_bytes,
        safe_budget_bytes=safe_budget_bytes,
        fits=fits,
        details=details,
        metadata=metadata,
    )


def build_memory_error_message(
    estimate: MemoryEstimate,
    *,
    n_samples: int,
    method: str,
) -> str:
    """
    Build a readable error message for runs that exceed the estimated memory budget.

    :param estimate: Memory estimate that failed the fit check.
    :param n_samples: Number of input samples.
    :param method: Requested t-SNE method.

    :return: Human-readable ``MemoryError`` message.
    """
    suggestions = ['reduce n_samples']
    if method != 'fft' and not estimate.backend.startswith('torch_fft'):
        suggestions.append("switch to method='fft'")
    if estimate.device == 'cuda':
        suggestions.append("retry on device='cpu'")
    if method == 'exact' or estimate.backend == 'torch_fft_dense':
        suggestions.append('use a sparse precomputed graph for large datasets')
    return (
        f"Estimated {estimate.backend} memory requirement for n_samples={n_samples} is "
        f'{format_num_bytes(estimate.required_bytes)}, but only {format_num_bytes(estimate.available_bytes)} '
        f'appears available on {estimate.device} (safe budget {format_num_bytes(estimate.safe_budget_bytes)}). '
        f"Try to {', '.join(suggestions)}."
    )
