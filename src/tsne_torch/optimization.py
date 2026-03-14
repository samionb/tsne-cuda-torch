"""Optimizer helpers for TorchTSNE."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch


def gradient_descent(
    objective,
    p0: torch.Tensor,
    *,
    it: int,
    max_iter: int,
    n_iter_check: int = 1,
    n_iter_without_progress: int = 300,
    momentum: float = 0.8,
    learning_rate: float = 200.0,
    min_gain: float = 0.01,
    min_grad_norm: float = 1e-7,
    verbose: int = 0,
    args=None,
    kwargs=None,
):
    """
    Run the sklearn-style batch gradient descent loop on Torch tensors.

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    :param objective: Callable returning ``(error, gradient)`` for the current parameters.
    :param p0: Initial parameter tensor.
    :param it: Starting iteration index.
    :param max_iter: Maximum optimization iteration.
    :param n_iter_check: Frequency for convergence and timing checks.
    :param n_iter_without_progress: Allowed check intervals without error improvement.
    :param momentum: Momentum coefficient.
    :param learning_rate: Optimizer learning rate.
    :param min_gain: Lower bound for adaptive gains.
    :param min_grad_norm: Gradient-norm threshold used for early stopping.
    :param verbose: Verbosity level.
    :param args: Positional arguments forwarded to the objective.
    :param kwargs: Keyword arguments forwarded to the objective.

    :return: Tuple ``(params, error, last_iteration, diagnostics)``.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.clone()
    update = torch.zeros_like(p)
    gains = torch.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it
    timings = []
    stopped_reason = 'max_iter'
    tic = perf_counter()

    for i in range(it, max_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        objective_kwargs = dict(kwargs)
        objective_kwargs['compute_error'] = check_convergence or i == max_iter - 1

        error, grad = objective(p, *args, **objective_kwargs)

        inc = update * grad < 0.0
        gains = torch.where(inc, gains + 0.2, gains * 0.8)
        gains.clamp_(min=min_gain)
        grad = grad * gains
        update = momentum * update - learning_rate * grad
        p = p + update

        if check_convergence:
            toc = perf_counter()
            duration = toc - tic
            tic = toc
            timings.append(duration / n_iter_check)
            grad_norm = torch.linalg.vector_norm(grad).item()

            if verbose >= 2:
                print(
                    '[TorchTSNE] Iteration %d: error = %.7f, gradient norm = %.7f (%s iterations in %0.3fs)'
                    % (i + 1, error, grad_norm, n_iter_check, duration)
                )

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                stopped_reason = 'no_progress'
                if verbose >= 2:
                    print(
                        '[TorchTSNE] Iteration %d: did not make any progress during the last %d episodes. Finished.'
                        % (i + 1, n_iter_without_progress)
                    )
                break

            if grad_norm <= min_grad_norm:
                stopped_reason = 'grad_norm'
                if verbose >= 2:
                    print(
                        '[TorchTSNE] Iteration %d: gradient norm %f. Finished.'
                        % (i + 1, grad_norm)
                    )
                break

    diagnostics = {
        'iteration_times': timings,
        'median_iteration_time': float(np.median(timings)) if timings else 0.0,
        'stopped_reason': stopped_reason,
    }
    return p, float(error), i, diagnostics
