"""FFT-inspired approximate backend for TorchTSNE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as func

from .affinity import SparseAffinity

MACHINE_EPSILON = np.finfo(np.double).eps


@dataclass
class SparseAffinityTensors:
    """Torch views of sparse affinities used by the FFT backend."""

    rows: torch.Tensor
    cols: torch.Tensor
    values: torch.Tensor


def sparse_affinity_to_torch(
    affinity: SparseAffinity,
    *,
    device: torch.device,
) -> SparseAffinityTensors:
    """
    Convert a sparse affinity container into Torch tensors on the requested device.

    :param affinity: Sparse affinity container backed by a SciPy CSR matrix.
    :param device: Destination Torch device.

    :return: Torch tensor views of rows, columns, and values.
    """
    matrix = affinity.matrix
    return SparseAffinityTensors(
        rows=torch.as_tensor(affinity.rows, dtype=torch.long, device=device),
        cols=torch.as_tensor(matrix.indices, dtype=torch.long, device=device),
        values=torch.as_tensor(matrix.data, dtype=torch.float32, device=device),
    )


def _linear_convolution_fft(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Compute a 2D linear convolution using Torch FFT primitives.

    See also: https://docs.pytorch.org/docs/stable/fft.html

    :param image: Input image or grid.
    :param kernel: Convolution kernel.

    :return: Convolved tensor cropped back to the input image size.
    """
    conv_shape = (image.shape[-2] + kernel.shape[-2] - 1, image.shape[-1] + kernel.shape[-1] - 1)
    image_fft = torch.fft.rfftn(image, s=conv_shape)
    kernel_fft = torch.fft.rfftn(kernel, s=conv_shape)
    convolved = torch.fft.irfftn(image_fft * kernel_fft, s=conv_shape)
    start_y = kernel.shape[-2] // 2
    start_x = kernel.shape[-1] // 2
    end_y = start_y + image.shape[-2]
    end_x = start_x + image.shape[-1]
    return convolved[start_y:end_y, start_x:end_x]


def _splat_points_to_grid(coords: torch.Tensor, grid_size: int) -> torch.Tensor:
    """
    Bilinearly splat 2D point coordinates into a square occupancy grid.

    :param coords: Point coordinates expressed in grid coordinates.
    :param grid_size: Side length of the square grid.

    :return: Occupancy grid with bilinearly accumulated point weights.
    """
    x = coords[:, 0].clamp(0.0, grid_size - 1.001)
    y = coords[:, 1].clamp(0.0, grid_size - 1.001)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = (x0 + 1).clamp(max=grid_size - 1)
    y1 = (y0 + 1).clamp(max=grid_size - 1)

    wx = x - x0.float()
    wy = y - y0.float()

    weights = [
        ((1.0 - wx) * (1.0 - wy), x0, y0),
        (wx * (1.0 - wy), x1, y0),
        ((1.0 - wx) * wy, x0, y1),
        (wx * wy, x1, y1),
    ]

    grid = torch.zeros(grid_size * grid_size, dtype=coords.dtype, device=coords.device)
    for weight, xi, yi in weights:
        flat_index = yi * grid_size + xi
        grid.index_add_(0, flat_index, weight)
    return grid.view(grid_size, grid_size)


def _sample_grid(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Sample a regular grid at arbitrary coordinates with bilinear interpolation.

    :param field: Grid to sample from.
    :param coords: Coordinates expressed in grid space.

    :return: Sampled values for each requested coordinate.
    """
    grid_size = field.shape[-1]
    x = coords[:, 0] / max(grid_size - 1, 1) * 2.0 - 1.0
    y = coords[:, 1] / max(grid_size - 1, 1) * 2.0 - 1.0
    sample_grid = torch.stack((x, y), dim=-1).view(1, -1, 1, 2)
    sampled = func.grid_sample(
        field.view(1, 1, grid_size, grid_size),
        sample_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
    return sampled.view(-1)


def approximate_negative_forces_fft(
    y: torch.Tensor,
    *,
    grid_size: int,
    degrees_of_freedom: float,
):
    """
    Approximate the t-SNE negative force field with FFT-based grid convolutions.

    :param y: Current 2D embedding.
    :param grid_size: Side length of the convolution grid.
    :param degrees_of_freedom: Student-t degrees of freedom for the embedding distribution.

    :return: Tuple ``(sum_q, negative_force)``.
    """
    if y.shape[1] != 2:
        raise ValueError('FFT backend only supports n_components=2')

    device = y.device
    dtype = y.dtype
    min_xy = y.min(dim=0).values
    max_xy = y.max(dim=0).values
    span = torch.max(max_xy - min_xy).clamp_min(torch.tensor(1e-6, dtype=dtype, device=device))
    padding = 0.05 * span + 1e-6
    step = (span + 2.0 * padding) / max(grid_size - 1, 1)
    origin = min_xy - padding
    coords = (y - origin) / step

    occupancy = _splat_points_to_grid(coords, grid_size)

    deltas = torch.arange(-(grid_size - 1), grid_size, dtype=dtype, device=device) * step
    yy, xx = torch.meshgrid(deltas, deltas, indexing='ij')
    r2 = xx.pow(2) + yy.pow(2)
    exponent = (degrees_of_freedom + 1.0) / 2.0
    q_kernel = degrees_of_freedom / (degrees_of_freedom + r2)
    if degrees_of_freedom != 1.0:
        q_kernel = q_kernel.pow(exponent)
    force_kernel_x = xx * q_kernel.pow(2)
    force_kernel_y = yy * q_kernel.pow(2)

    potential_grid = _linear_convolution_fft(occupancy, q_kernel)
    force_x_grid = _linear_convolution_fft(occupancy, force_kernel_x)
    force_y_grid = _linear_convolution_fft(occupancy, force_kernel_y)

    sampled_potential = _sample_grid(potential_grid, coords) - 1.0
    sampled_force_x = _sample_grid(force_x_grid, coords)
    sampled_force_y = _sample_grid(force_y_grid, coords)

    neg_force = torch.stack((sampled_force_x, sampled_force_y), dim=1)
    sum_q = sampled_potential.sum().clamp_min(MACHINE_EPSILON)
    return sum_q, neg_force


def fft_kl_divergence_objective(
    params: torch.Tensor,
    affinity: SparseAffinityTensors,
    degrees_of_freedom: float,
    *,
    grid_size: int = 128,
    skip_num_points: int = 0,
    compute_error: bool = True,
):
    """
    Evaluate the sparse FFT t-SNE objective with sparse attraction and FFT repulsion.

    :param params: Current embedding with shape ``(n_samples, 2)``.
    :param affinity: Sparse affinity tensors.
    :param degrees_of_freedom: Student-t degrees of freedom for the embedding distribution.
    :param grid_size: Side length of the FFT grid.
    :param skip_num_points: Number of leading points whose gradients should be forced to zero.
    :param compute_error: Whether to compute the KL divergence value in addition to the gradient.

    :return: Tuple ``(kl_divergence, gradient)``.
    """
    if params.shape[1] != 2:
        raise ValueError('FFT backend only supports n_components=2')

    rows = affinity.rows
    cols = affinity.cols
    values = affinity.values
    diff = params[rows] - params[cols]
    dist = torch.sum(diff * diff, dim=1)

    exponent = (degrees_of_freedom + 1.0) / 2.0
    q = degrees_of_freedom / (degrees_of_freedom + dist)
    if degrees_of_freedom != 1.0:
        q = q.pow(exponent)

    attractive_force = torch.zeros_like(params)
    attractive_force.index_add_(0, rows, values.unsqueeze(1) * q.unsqueeze(1) * diff)

    sum_q, negative_force = approximate_negative_forces_fft(
        params,
        grid_size=grid_size,
        degrees_of_freedom=degrees_of_freedom,
    )

    if compute_error:
        kl_divergence = torch.sum(
            values * torch.log(torch.clamp(values, min=MACHINE_EPSILON) / torch.clamp(q / sum_q, min=MACHINE_EPSILON))
        ).item()
    else:
        kl_divergence = float('nan')

    grad = attractive_force - negative_force / sum_q
    grad *= 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom

    if skip_num_points:
        grad = grad.clone()
        grad[:skip_num_points] = 0

    return kl_divergence, grad


def fft_kl_divergence_dense_objective(
    params: torch.Tensor,
    affinity: torch.Tensor,
    degrees_of_freedom: float,
    *,
    grid_size: int = 128,
    skip_num_points: int = 0,
    compute_error: bool = True,
):
    """
    Evaluate the dense FFT t-SNE objective with dense attraction and FFT repulsion.

    :param params: Current embedding with shape ``(n_samples, 2)``.
    :param affinity: Dense symmetric affinity matrix ``P``.
    :param degrees_of_freedom: Student-t degrees of freedom for the embedding distribution.
    :param grid_size: Side length of the FFT grid.
    :param skip_num_points: Number of leading points whose gradients should be forced to zero.
    :param compute_error: Whether to compute the KL divergence value in addition to the gradient.

    :return: Tuple ``(kl_divergence, gradient)``.
    """
    if params.shape[1] != 2:
        raise ValueError('FFT backend only supports n_components=2')

    diff = params[:, None, :] - params[None, :, :]
    dist = torch.sum(diff * diff, dim=2)

    exponent = (degrees_of_freedom + 1.0) / 2.0
    q = degrees_of_freedom / (degrees_of_freedom + dist)
    if degrees_of_freedom != 1.0:
        q = q.pow(exponent)
    q.fill_diagonal_(0.0)

    attractive_weights = affinity * q
    attractive_force = params * attractive_weights.sum(dim=1, keepdim=True) - attractive_weights @ params

    sum_q, negative_force = approximate_negative_forces_fft(
        params,
        grid_size=grid_size,
        degrees_of_freedom=degrees_of_freedom,
    )

    if compute_error:
        q_norm = torch.clamp(q / sum_q, min=MACHINE_EPSILON)
        kl_divergence = torch.sum(
            affinity * torch.log(torch.clamp(affinity, min=MACHINE_EPSILON) / q_norm)
        ).item()
    else:
        kl_divergence = float('nan')

    grad = attractive_force - negative_force / sum_q
    grad *= 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom

    if skip_num_points:
        grad = grad.clone()
        grad[:skip_num_points] = 0

    return kl_divergence, grad
