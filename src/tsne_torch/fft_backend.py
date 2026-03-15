"""FFT-inspired approximate backend for TorchTSNE."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

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
    return _linear_convolution_fft_from_image_fft(image_fft, kernel, image_shape=image.shape)


def _linear_convolution_fft_from_image_fft(
    image_fft: torch.Tensor,
    kernel: torch.Tensor,
    *,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """
    Compute a 2D linear convolution when the image FFT has already been materialized.

    :param image_fft: Frequency-domain representation of the input image.
    :param kernel: Convolution kernel.
    :param image_shape: Spatial shape of the original image.

    :return: Convolved tensor cropped back to the input image size.
    """
    conv_shape = (image_shape[-2] + kernel.shape[-2] - 1, image_shape[-1] + kernel.shape[-1] - 1)
    kernel_fft = torch.fft.rfftn(kernel, s=conv_shape)
    convolved = torch.fft.irfftn(image_fft * kernel_fft, s=conv_shape)
    start_y = kernel.shape[-2] // 2
    start_x = kernel.shape[-1] // 2
    end_y = start_y + image_shape[-2]
    end_x = start_x + image_shape[-1]
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

    grid = torch.zeros(grid_size * grid_size, dtype=coords.dtype, device=coords.device)
    flat_indices = torch.cat(
        (
            y0 * grid_size + x0,
            y0 * grid_size + x1,
            y1 * grid_size + x0,
            y1 * grid_size + x1,
        ),
        dim=0,
    )
    flat_weights = torch.cat(
        (
            (1.0 - wx) * (1.0 - wy),
            wx * (1.0 - wy),
            (1.0 - wx) * wy,
            wx * wy,
        ),
        dim=0,
    )
    grid.index_add_(0, flat_indices, flat_weights)
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


def _sample_grid_channels(fields: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Sample multiple regular grids at arbitrary coordinates with one bilinear interpolation call.

    :param fields: Grids to sample from with shape ``(n_channels, height, width)``.
    :param coords: Coordinates expressed in grid space.

    :return: Sampled values with shape ``(n_points, n_channels)``.
    """
    grid_size = fields.shape[-1]
    x = coords[:, 0] / max(grid_size - 1, 1) * 2.0 - 1.0
    y = coords[:, 1] / max(grid_size - 1, 1) * 2.0 - 1.0
    sample_grid = torch.stack((x, y), dim=-1).view(1, -1, 1, 2)
    sampled = func.grid_sample(
        fields.unsqueeze(0),
        sample_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()


def _timed_start(device: torch.device, timings: dict | None):
    """
    Start an optional profiling window for an FFT backend objective stage.

    :param device: Torch device executing the stage.
    :param timings: Optional timing dictionary. Profiling is disabled when ``None``.

    :return: CPU timestamp, CUDA event pair, or ``None`` when profiling is disabled.
    """
    if timings is None:
        return None
    if device.type == 'cuda' and torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return start_event, end_event
    return perf_counter()


def _timed_stop(start_token, device: torch.device, timings: dict | None, key: str) -> None:
    """
    Stop an optional profiling window and accumulate the elapsed duration.

    :param start_token: Token returned by ``_timed_start``.
    :param device: Torch device executing the stage.
    :param timings: Mutable timing dictionary receiving the elapsed duration.
    :param key: Timing bucket name.

    :return: None.
    """
    if timings is None or start_token is None:
        return
    if device.type == 'cuda' and torch.cuda.is_available():
        start_event, end_event = start_token
        end_event.record()
        end_event.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000.0
    else:
        elapsed = perf_counter() - start_token
    timings[key] = timings.get(key, 0.0) + float(elapsed)


def approximate_negative_forces_fft(
    y: torch.Tensor,
    *,
    grid_size: int,
    degrees_of_freedom: float,
    timings: dict | None = None,
):
    """
    Approximate the t-SNE negative force field with FFT-based grid convolutions.

    :param y: Current 2D embedding.
    :param grid_size: Side length of the convolution grid.
    :param degrees_of_freedom: Student-t degrees of freedom for the embedding distribution.
    :param timings: Optional timing dictionary receiving sampled negative-force stage timings.

    :return: Tuple ``(sum_q, negative_force)``.
    """
    if y.shape[1] != 2:
        raise ValueError('FFT backend only supports n_components=2')

    device = y.device
    dtype = y.dtype
    total_start = _timed_start(device, timings)

    stage_start = _timed_start(device, timings)
    min_xy = y.min(dim=0).values
    max_xy = y.max(dim=0).values
    span = torch.max(max_xy - min_xy).clamp_min(torch.tensor(1e-6, dtype=dtype, device=device))
    padding = 0.05 * span + 1e-6
    step = (span + 2.0 * padding) / max(grid_size - 1, 1)
    origin = min_xy - padding
    coords = (y - origin) / step
    _timed_stop(stage_start, device, timings, 'negative_force_coords')

    stage_start = _timed_start(device, timings)
    occupancy = _splat_points_to_grid(coords, grid_size)
    _timed_stop(stage_start, device, timings, 'negative_force_splat')

    stage_start = _timed_start(device, timings)
    deltas = torch.arange(-(grid_size - 1), grid_size, dtype=dtype, device=device) * step
    yy, xx = torch.meshgrid(deltas, deltas, indexing='ij')
    r2 = xx.pow(2) + yy.pow(2)
    if degrees_of_freedom == 1.0:
        q_kernel = r2.add(1.0).reciprocal_()
    else:
        exponent = (degrees_of_freedom + 1.0) / 2.0
        q_kernel = degrees_of_freedom / (degrees_of_freedom + r2)
        q_kernel = q_kernel.pow(exponent)
    q_kernel_sq = q_kernel * q_kernel
    force_kernel_x = xx * q_kernel_sq
    force_kernel_y = yy * q_kernel_sq
    _timed_stop(stage_start, device, timings, 'negative_force_kernel_build')

    stage_start = _timed_start(device, timings)
    conv_shape = (occupancy.shape[-2] + q_kernel.shape[-2] - 1, occupancy.shape[-1] + q_kernel.shape[-1] - 1)
    occupancy_fft = torch.fft.rfftn(occupancy, s=conv_shape)
    potential_grid = _linear_convolution_fft_from_image_fft(occupancy_fft, q_kernel, image_shape=occupancy.shape)
    force_x_grid = _linear_convolution_fft_from_image_fft(occupancy_fft, force_kernel_x, image_shape=occupancy.shape)
    force_y_grid = _linear_convolution_fft_from_image_fft(occupancy_fft, force_kernel_y, image_shape=occupancy.shape)
    _timed_stop(stage_start, device, timings, 'negative_force_fft_convolution')

    stage_start = _timed_start(device, timings)
    sampled_fields = _sample_grid_channels(
        torch.stack((potential_grid, force_x_grid, force_y_grid), dim=0),
        coords,
    )
    _timed_stop(stage_start, device, timings, 'negative_force_grid_sample')

    stage_start = _timed_start(device, timings)
    sum_q = (sampled_fields[:, 0].sum() - float(y.shape[0])).clamp_min(MACHINE_EPSILON)
    neg_force = sampled_fields[:, 1:3].contiguous()
    _timed_stop(stage_start, device, timings, 'negative_force_combine')
    _timed_stop(total_start, device, timings, 'negative_force_total')
    return sum_q, neg_force


def fft_kl_divergence_objective(
    params: torch.Tensor,
    affinity: SparseAffinityTensors,
    degrees_of_freedom: float,
    *,
    grid_size: int = 128,
    skip_num_points: int = 0,
    compute_error: bool = True,
    timings: dict | None = None,
):
    """
    Evaluate the sparse FFT t-SNE objective with sparse attraction and FFT repulsion.

    :param params: Current embedding with shape ``(n_samples, 2)``.
    :param affinity: Sparse affinity tensors.
    :param degrees_of_freedom: Student-t degrees of freedom for the embedding distribution.
    :param grid_size: Side length of the FFT grid.
    :param skip_num_points: Number of leading points whose gradients should be forced to zero.
    :param compute_error: Whether to compute the KL divergence value in addition to the gradient.
    :param timings: Optional timing dictionary receiving sampled objective stage timings.

    :return: Tuple ``(kl_divergence, gradient)``.
    """
    if params.shape[1] != 2:
        raise ValueError('FFT backend only supports n_components=2')

    device = params.device
    rows = affinity.rows
    cols = affinity.cols
    values = affinity.values

    stage_start = _timed_start(device, timings)
    diff = params[rows] - params[cols]
    dist = torch.sum(diff * diff, dim=1)
    if degrees_of_freedom == 1.0:
        q = dist.add(1.0).reciprocal_()
        grad_scale = 4.0
    else:
        exponent = (degrees_of_freedom + 1.0) / 2.0
        q = degrees_of_freedom / (degrees_of_freedom + dist)
        q = q.pow(exponent)
        grad_scale = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom

    attractive_force = torch.zeros_like(params)
    attractive_force.index_add_(0, rows, values.unsqueeze(1) * q.unsqueeze(1) * diff)
    _timed_stop(stage_start, device, timings, 'attractive_force')

    sum_q, negative_force = approximate_negative_forces_fft(
        params,
        grid_size=grid_size,
        degrees_of_freedom=degrees_of_freedom,
        timings=timings,
    )

    if compute_error:
        stage_start = _timed_start(device, timings)
        kl_divergence = torch.sum(
            values * torch.log(torch.clamp(values, min=MACHINE_EPSILON) / torch.clamp(q / sum_q, min=MACHINE_EPSILON))
        ).item()
        _timed_stop(stage_start, device, timings, 'error_eval')
    else:
        kl_divergence = float('nan')

    stage_start = _timed_start(device, timings)
    grad = attractive_force - negative_force / sum_q
    grad *= grad_scale

    if skip_num_points:
        grad = grad.clone()
        grad[:skip_num_points] = 0
    _timed_stop(stage_start, device, timings, 'gradient_finalize')

    return kl_divergence, grad


def fft_kl_divergence_dense_objective(
    params: torch.Tensor,
    affinity: torch.Tensor,
    degrees_of_freedom: float,
    *,
    grid_size: int = 128,
    skip_num_points: int = 0,
    compute_error: bool = True,
    timings: dict | None = None,
):
    """
    Evaluate the dense FFT t-SNE objective with dense attraction and FFT repulsion.

    :param params: Current embedding with shape ``(n_samples, 2)``.
    :param affinity: Dense symmetric affinity matrix ``P``.
    :param degrees_of_freedom: Student-t degrees of freedom for the embedding distribution.
    :param grid_size: Side length of the FFT grid.
    :param skip_num_points: Number of leading points whose gradients should be forced to zero.
    :param compute_error: Whether to compute the KL divergence value in addition to the gradient.
    :param timings: Optional timing dictionary receiving sampled objective stage timings.

    :return: Tuple ``(kl_divergence, gradient)``.
    """
    if params.shape[1] != 2:
        raise ValueError('FFT backend only supports n_components=2')

    device = params.device
    stage_start = _timed_start(device, timings)
    diff = params[:, None, :] - params[None, :, :]
    dist = torch.sum(diff * diff, dim=2)

    if degrees_of_freedom == 1.0:
        q = dist.add(1.0).reciprocal_()
        grad_scale = 4.0
    else:
        exponent = (degrees_of_freedom + 1.0) / 2.0
        q = degrees_of_freedom / (degrees_of_freedom + dist)
        q = q.pow(exponent)
        grad_scale = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    q.fill_diagonal_(0.0)

    attractive_weights = affinity * q
    attractive_force = params * attractive_weights.sum(dim=1, keepdim=True) - attractive_weights @ params
    _timed_stop(stage_start, device, timings, 'attractive_force')

    sum_q, negative_force = approximate_negative_forces_fft(
        params,
        grid_size=grid_size,
        degrees_of_freedom=degrees_of_freedom,
        timings=timings,
    )

    if compute_error:
        stage_start = _timed_start(device, timings)
        q_norm = torch.clamp(q / sum_q, min=MACHINE_EPSILON)
        kl_divergence = torch.sum(
            affinity * torch.log(torch.clamp(affinity, min=MACHINE_EPSILON) / q_norm)
        ).item()
        _timed_stop(stage_start, device, timings, 'error_eval')
    else:
        kl_divergence = float('nan')

    stage_start = _timed_start(device, timings)
    grad = attractive_force - negative_force / sum_q
    grad *= grad_scale

    if skip_num_points:
        grad = grad.clone()
        grad[:skip_num_points] = 0
    _timed_stop(stage_start, device, timings, 'gradient_finalize')

    return kl_divergence, grad
