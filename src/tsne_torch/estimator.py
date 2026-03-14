"""sklearn-style high-performance estimator for Torch/CUDA t-SNE."""

from __future__ import annotations

import logging
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_non_negative

from .affinity import (
    build_sparse_affinity_from_knn,
    build_sparse_affinity_from_precomputed,
    build_sparse_affinity_from_sparse_precomputed,
    compute_dense_squared_distances,
    compute_learning_rate,
    joint_probabilities_from_squared_distances,
    squared_euclidean_distances_torch,
)
from .diagnostics import FitDiagnostics
from .exact_backend import exact_kl_divergence_objective
from .fft_backend import (
    fft_kl_divergence_dense_objective,
    fft_kl_divergence_objective,
    sparse_affinity_to_torch,
)
from .memory import build_memory_error_message, estimate_tsne_memory, format_num_bytes
from .optimization import gradient_descent

LOGGER = logging.getLogger('tsne_torch.estimator')


class TorchTSNE(TransformerMixin, BaseEstimator):
    """High-performance Torch/CUDA variant of sklearn's TSNE."""

    _EXPLORATION_MAX_ITER = 250
    _N_ITER_CHECK = 50
    _FFT_GRID_SIZE = 256

    def __init__(
        self,
        n_components=2,
        *,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate='auto',
        max_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric='euclidean',
        metric_params=None,
        init='pca',
        verbose=0,
        random_state=None,
        method='exact',
        angle=0.5,
        n_jobs=None,
        device='auto',
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.device = device

    def _resolve_device(self) -> torch.device:
        """
        Resolve the effective Torch device for the current estimator configuration.

        :return: Resolved Torch device.
        """
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        return torch.device(self.device)

    def _check_params_vs_input(self, x):
        """
        Validate estimator parameters against the provided input.

        :param x: Prepared estimator input.

        :return: None.
        """
        if self.perplexity >= x.shape[0]:
            raise ValueError(f'perplexity ({self.perplexity}) must be less than n_samples ({x.shape[0]})')
        if self.max_iter < self._EXPLORATION_MAX_ITER:
            raise ValueError('max_iter must be at least 250')
        if self.method not in {'exact', 'barnes_hut', 'fft'}:
            raise ValueError("method must be one of {'exact', 'barnes_hut', 'fft'}")
        if self.device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError("device must be one of {'auto', 'cpu', 'cuda'}")
        if self.n_components < 1:
            raise ValueError('n_components must be at least 1')
        if self.method == 'barnes_hut' and self.n_components > 3:
            raise ValueError("'n_components' should be inferior to 4 for the barnes_hut algorithm")

    def _fallback_to_sklearn(self, x, reason: str):
        """
        Delegate fitting to scikit-learn when a TorchTSNE backend is unsupported.

        :param x: Estimator input.
        :param reason: Human-readable fallback reason stored in diagnostics.

        :return: Fitted embedding as a NumPy array.
        """
        if self.verbose:
            LOGGER.info('[TorchTSNE] Falling back to sklearn TSNE: %s', reason)

        sklearn_method = 'barnes_hut' if self.method == 'barnes_hut' else 'exact'
        fallback = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            early_exaggeration=self.early_exaggeration,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            n_iter_without_progress=self.n_iter_without_progress,
            min_grad_norm=self.min_grad_norm,
            metric=self.metric,
            metric_params=self.metric_params,
            init=self.init,
            verbose=self.verbose,
            random_state=self.random_state,
            method=sklearn_method,
            angle=self.angle,
            n_jobs=self.n_jobs,
        )
        start = perf_counter()
        embedding = fallback.fit_transform(x)
        duration = perf_counter() - start
        self.embedding_ = embedding.astype(np.float32, copy=False)
        self.kl_divergence_ = float(fallback.kl_divergence_)
        self.learning_rate_ = float(fallback.learning_rate_)
        self.n_iter_ = int(fallback.n_iter_)
        self.diagnostics_ = FitDiagnostics(
            backend=f'sklearn_{sklearn_method}',
            timings={'total': duration, 'fallback_reason': reason},
            device='cpu',
        )
        return self.embedding_

    def _estimate_memory(self, x, device: torch.device):
        """
        Estimate backend memory requirements for the current input and device.

        :param x: Estimator input.
        :param device: Candidate Torch device.

        :return: Memory estimate object.
        """
        return estimate_tsne_memory(
            x,
            method=self.method,
            metric=self.metric,
            n_components=self.n_components,
            perplexity=self.perplexity,
            device=device,
            grid_size=self._FFT_GRID_SIZE,
        )

    def _resolve_device_with_memory_checks(self, x):
        """
        Resolve the execution device and enforce memory preflight checks.

        :param x: Estimator input.

        :return: Tuple ``(device, memory_estimate)``.
        """
        device = self._resolve_device()
        estimate = self._estimate_memory(x, device)

        if self.device == 'auto' and device.type == 'cuda' and estimate.fits is False:
            cpu_device = torch.device('cpu')
            cpu_estimate = self._estimate_memory(x, cpu_device)
            if cpu_estimate.fits is not False:
                cpu_estimate.metadata = {**cpu_estimate.metadata, 'auto_device_fallback': 1}
                LOGGER.warning(
                    '[TorchTSNE] Falling back from CUDA to CPU because the CUDA memory estimate '
                    '(%s required, %s safe budget) does not fit.',
                    format_num_bytes(estimate.required_bytes),
                    format_num_bytes(estimate.safe_budget_bytes),
                )
                return cpu_device, cpu_estimate

        if estimate.fits is False:
            raise MemoryError(
                build_memory_error_message(
                    estimate,
                    n_samples=int(x.shape[0]),
                    method=self.method,
                )
            )

        return device, estimate

    def _prepare_input(self, x):
        """
        Normalize and validate raw estimator input before backend dispatch.

        :param x: Raw estimator input.

        :return: Prepared input array or sparse matrix.
        """
        if self.metric == 'precomputed':
            if isinstance(self.init, str) and self.init == 'pca':
                raise ValueError('The parameter init="pca" cannot be used with metric="precomputed".')
            if x.shape[0] != x.shape[1]:
                raise ValueError('X should be a square distance matrix')
            check_non_negative(
                x,
                "TorchTSNE.fit(). With metric='precomputed', X should contain positive distances.",
            )
            if sp.issparse(x):
                return x
            return np.asarray(x)

        return x

    def _initialize_embedding(self, x, n_samples: int, random_state) -> np.ndarray:
        """
        Build the initial embedding using sklearn-compatible rules.

        :param x: Estimator input.
        :param n_samples: Number of samples to embed.
        :param random_state: sklearn-style random-state helper.

        :return: Initial embedding as a NumPy array.
        """
        if isinstance(self.init, np.ndarray):
            init = np.asarray(self.init, dtype=np.float32)
            if init.shape != (n_samples, self.n_components):
                raise ValueError(
                    f'init ndarray must have shape ({n_samples}, {self.n_components}), got {init.shape}'
                )
            return init

        if self.init == 'random':
            return (1e-4 * random_state.standard_normal((n_samples, self.n_components))).astype(np.float32)

        if self.init != 'pca':
            raise ValueError('init must be "pca", "random", or an ndarray')

        data = x.toarray() if sp.issparse(x) else np.asarray(x)
        pca = PCA(n_components=self.n_components, random_state=random_state)
        embedded = pca.fit_transform(data).astype(np.float32, copy=False)
        std = np.std(embedded[:, 0])
        if std == 0.0:
            return embedded
        return embedded / std * 1e-4

    def _run_optimization(self, objective, params, p_data, degrees_of_freedom, backend_kwargs):
        """
        Run the two-stage sklearn-style optimization schedule.

        :param objective: Objective callable returning ``(error, gradient)``.
        :param params: Initial embedding parameters on the target device.
        :param p_data: Affinity data passed to the objective.
        :param degrees_of_freedom: Student-t degrees of freedom.
        :param backend_kwargs: Extra keyword arguments for the backend objective.

        :return: Tuple ``(params, kl_divergence, iteration, stage1_diag, stage2_diag)``.
        """
        opt_args = {
            'it': 0,
            'n_iter_check': self._N_ITER_CHECK,
            'min_grad_norm': self.min_grad_norm,
            'learning_rate': self.learning_rate_,
            'verbose': self.verbose,
            'kwargs': dict(backend_kwargs),
            'args': [self._scale_affinities(p_data, self.early_exaggeration), degrees_of_freedom],
            'n_iter_without_progress': self._EXPLORATION_MAX_ITER,
            'max_iter': self._EXPLORATION_MAX_ITER,
            'momentum': 0.5,
        }

        params, kl_divergence, it, stage1_diag = gradient_descent(objective, params, **opt_args)

        p_data = self._scale_affinities(opt_args['args'][0], 1.0 / self.early_exaggeration)
        stage2_diag = {'iteration_times': [], 'median_iteration_time': 0.0, 'stopped_reason': 'skipped'}
        remaining = self.max_iter - self._EXPLORATION_MAX_ITER
        if remaining > 0 and it + 1 < self.max_iter:
            opt_args['max_iter'] = self.max_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = self.n_iter_without_progress
            opt_args['args'] = [p_data, degrees_of_freedom]
            params, kl_divergence, it, stage2_diag = gradient_descent(objective, params, **opt_args)

        return params, kl_divergence, it, stage1_diag, stage2_diag

    @staticmethod
    def _scale_affinities(p_data, scale: float):
        """
        Scale dense or sparse affinity data without changing its structure.

        :param p_data: Dense affinity tensor or sparse affinity tensor container.
        :param scale: Multiplicative scale factor.

        :return: Scaled affinity object.
        """
        if hasattr(p_data, 'rows') and hasattr(p_data, 'cols') and hasattr(p_data, 'values'):
            return type(p_data)(rows=p_data.rows, cols=p_data.cols, values=p_data.values * scale)
        return p_data * scale

    def _run_exact_backend(self, x, *, random_state, device: torch.device):
        """
        Execute the dense exact Torch backend.

        :param x: Estimator input.
        :param random_state: sklearn-style random-state helper.
        :param device: Target Torch device.

        :return: Tuple ``(params, kl_divergence, iteration, backend_name, timings, stage1, stage2)``.
        """
        timings = {}
        if self.metric == 'precomputed':
            if sp.issparse(x):
                raise TypeError('TorchTSNE with method="exact" does not accept sparse precomputed distance matrix.')
            distances = np.asarray(x, dtype=np.float64)
            start = perf_counter()
            distances_tensor = torch.as_tensor(distances, dtype=torch.float32, device=device)
            timings['distance_build'] = perf_counter() - start
        elif self.metric == 'euclidean' and not sp.issparse(x):
            start = perf_counter()
            x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
            distances_tensor = squared_euclidean_distances_torch(x_tensor)
            timings['distance_build'] = perf_counter() - start
        else:
            start = perf_counter()
            distances = compute_dense_squared_distances(
                x,
                metric=self.metric,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs,
            )
            timings['distance_build'] = perf_counter() - start
            distances_tensor = torch.as_tensor(distances, dtype=torch.float32, device=device)

        start = perf_counter()
        p_matrix = joint_probabilities_from_squared_distances(distances_tensor, self.perplexity, self.verbose)
        timings['affinity_build'] = perf_counter() - start

        init = self._initialize_embedding(x, distances_tensor.shape[0], random_state)
        params = torch.as_tensor(init, dtype=torch.float32, device=device)
        degrees_of_freedom = float(max(self.n_components - 1, 1))

        start = perf_counter()
        params, kl_divergence, it, stage1, stage2 = self._run_optimization(
            exact_kl_divergence_objective,
            params,
            p_matrix,
            degrees_of_freedom,
            {'skip_num_points': 0},
        )
        timings['optimization'] = perf_counter() - start
        return params, kl_divergence, it, 'torch_exact', timings, stage1, stage2

    def _run_fft_backend(self, x, *, random_state, device: torch.device):
        """
        Execute the FFT-based approximate backend.

        :param x: Estimator input.
        :param random_state: sklearn-style random-state helper.
        :param device: Target Torch device.

        :return: Backend result tuple, or a NumPy embedding when falling back to sklearn.
        """
        if self.n_components != 2:
            return self._fallback_to_sklearn(x, 'fft backend only supports n_components=2')

        timings = {}
        if x.shape[0] <= 256:
            if self.metric == 'precomputed':
                if sp.issparse(x):
                    return self._fallback_to_sklearn(x, 'fft backend requires dense precomputed distances')
                distances_tensor = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float32, device=device)
            elif self.metric == 'euclidean' and not sp.issparse(x):
                x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
                distances_tensor = squared_euclidean_distances_torch(x_tensor)
            else:
                distances = compute_dense_squared_distances(
                    x,
                    metric=self.metric,
                    metric_params=self.metric_params,
                    n_jobs=self.n_jobs,
                )
                distances_tensor = torch.as_tensor(distances, dtype=torch.float32, device=device)

            start = perf_counter()
            p_matrix = joint_probabilities_from_squared_distances(distances_tensor, self.perplexity, self.verbose)
            timings['affinity_build'] = perf_counter() - start

            init = self._initialize_embedding(x, distances_tensor.shape[0], random_state)
            params = torch.as_tensor(init, dtype=torch.float32, device=device)
            degrees_of_freedom = float(max(self.n_components - 1, 1))

            start = perf_counter()
            params, kl_divergence, it, stage1, stage2 = self._run_optimization(
                fft_kl_divergence_dense_objective,
                params,
                p_matrix,
                degrees_of_freedom,
                {'skip_num_points': 0, 'grid_size': self._FFT_GRID_SIZE},
            )
            timings['optimization'] = perf_counter() - start
            return params, kl_divergence, it, 'torch_fft_dense', timings, stage1, stage2

        if self.metric == 'precomputed':
            if sp.issparse(x):
                start = perf_counter()
                affinity = build_sparse_affinity_from_sparse_precomputed(
                    x,
                    perplexity=self.perplexity,
                    device=device,
                    verbose=self.verbose,
                )
                timings['affinity_build'] = perf_counter() - start
            else:
                distances = np.asarray(x, dtype=np.float64)
                start = perf_counter()
                affinity = build_sparse_affinity_from_precomputed(
                    distances,
                    perplexity=self.perplexity,
                    device=device,
                    verbose=self.verbose,
                )
                timings['affinity_build'] = perf_counter() - start
        else:
            start = perf_counter()
            affinity = build_sparse_affinity_from_knn(
                x,
                perplexity=self.perplexity,
                metric=self.metric,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs,
                device=device,
                verbose=self.verbose,
            )
            timings['affinity_build'] = perf_counter() - start

        affinity_torch = sparse_affinity_to_torch(affinity, device=device)
        init = self._initialize_embedding(x, affinity.matrix.shape[0], random_state)
        params = torch.as_tensor(init, dtype=torch.float32, device=device)
        degrees_of_freedom = float(max(self.n_components - 1, 1))

        start = perf_counter()
        params, kl_divergence, it, stage1, stage2 = self._run_optimization(
            fft_kl_divergence_objective,
            params,
            affinity_torch,
            degrees_of_freedom,
            {'skip_num_points': 0, 'grid_size': self._FFT_GRID_SIZE},
        )
        timings['optimization'] = perf_counter() - start
        return params, kl_divergence, it, 'torch_fft', timings, stage1, stage2

    def fit_transform(self, x, y=None):
        """
        Fit the estimator and return the embedding.

        Reference: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

        :param x: Input samples, precomputed distances, or sparse precomputed graph.
        :param y: Ignored. Present for sklearn transformer compatibility.

        :return: Embedded coordinates as a NumPy array.
        """
        x = self._prepare_input(x)
        self._check_params_vs_input(x)
        random_state = check_random_state(self.random_state)
        self.learning_rate_ = compute_learning_rate(x.shape[0], self.early_exaggeration, self.learning_rate)

        if self.method == 'barnes_hut':
            return self._fallback_to_sklearn(x, 'barnes_hut is delegated to sklearn in tsne-torch')

        device, memory_estimate = self._resolve_device_with_memory_checks(x)

        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        total_start = perf_counter()
        if self.method == 'fft':
            result = self._run_fft_backend(x, random_state=random_state, device=device)
            if isinstance(result, np.ndarray):
                return result
            params, kl_divergence, it, backend_name, timings, stage1, stage2 = result
        else:
            params, kl_divergence, it, backend_name, timings, stage1, stage2 = self._run_exact_backend(
                x,
                random_state=random_state,
                device=device,
            )

        self.embedding_ = params.detach().cpu().numpy().astype(np.float32, copy=False)
        self.kl_divergence_ = float(kl_divergence)
        self.n_iter_ = int(it)
        timings['total'] = perf_counter() - total_start
        timings['stage1'] = stage1
        timings['stage2'] = stage2
        timings['peak_cuda_memory'] = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' and torch.cuda.is_available() else 0
        self.diagnostics_ = FitDiagnostics(
            backend=backend_name,
            timings=timings,
            device=str(device),
            memory=memory_estimate.as_dict(),
        )
        return self.embedding_

    def fit(self, x, y=None):
        """
        Fit the estimator and store the learned embedding on the instance.

        :param x: Input samples, precomputed distances, or sparse precomputed graph.
        :param y: Ignored. Present for sklearn estimator compatibility.

        :return: Fitted estimator instance.
        """
        self.fit_transform(x, y=y)
        return self
