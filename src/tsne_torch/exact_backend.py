"""Exact Torch backend for TorchTSNE."""

from __future__ import annotations

import numpy as np
import torch

MACHINE_EPSILON = np.finfo(np.double).eps


def exact_kl_divergence_objective(
    params: torch.Tensor,
    p: torch.Tensor,
    degrees_of_freedom: float,
    *,
    skip_num_points: int = 0,
    compute_error: bool = True,
    timings: dict | None = None,
):
    """
    Evaluate the exact t-SNE KL objective and gradient on Torch tensors.

    Reference: https://jmlr.org/papers/v9/vandermaaten08a.html

    :param params: Current embedding with shape ``(n_samples, n_components)``.
    :param p: Dense symmetric affinity matrix ``P``.
    :param degrees_of_freedom: Student-t degrees of freedom for the embedding distribution.
    :param skip_num_points: Number of leading points whose gradients should be forced to zero.
    :param compute_error: Whether to compute the KL divergence value in addition to the gradient.
    :param timings: Optional objective timing dictionary. Unused for the exact backend.

    :return: Tuple ``(kl_divergence, gradient)``.
    """
    _ = timings
    y = params
    sum_y = (y * y).sum(dim=1, keepdim=True)
    dist = sum_y + sum_y.T - 2.0 * (y @ y.T)
    dist.clamp_(min=0.0)

    exponent = (degrees_of_freedom + 1.0) / 2.0
    inv = degrees_of_freedom / (degrees_of_freedom + dist)
    if degrees_of_freedom != 1.0:
        inv = inv.pow(exponent)
    inv.fill_diagonal_(0.0)

    sum_q = inv.sum().clamp_min(MACHINE_EPSILON)
    q = inv / sum_q
    q = torch.clamp(q, min=MACHINE_EPSILON)

    if compute_error:
        kl_divergence = torch.sum(p * torch.log(torch.clamp(p, min=MACHINE_EPSILON) / q)).item()
    else:
        kl_divergence = float('nan')

    weights = (p - q) * inv
    grad = y * weights.sum(dim=1, keepdim=True) - weights @ y
    grad *= 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom

    if skip_num_points:
        grad = grad.clone()
        grad[:skip_num_points] = 0

    return kl_divergence, grad
