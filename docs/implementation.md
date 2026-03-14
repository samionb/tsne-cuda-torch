# Implementation Notes

This document holds the implementation detail that was moved out of the main [README](../README.md). It focuses on how the package is structured, how the backends work, and where the performance gains come from.

## Table Of Contents

- [Project Structure](#project-structure)
- [API Compatibility Philosophy](#api-compatibility-philosophy)
- [Exact Backend](#exact-backend)
- [FFT Backend](#fft-backend)
- [Memory Validation And Device Selection](#memory-validation-and-device-selection)
- [Where The Optimization Comes From](#where-the-optimization-comes-from)
- [Fallback Behavior](#fallback-behavior)
- [Correctness Strategy](#correctness-strategy)

## Project Structure

Main files:

- [`src/tsne_torch/__init__.py`](../src/tsne_torch/__init__.py)
  Package exports for `TorchTSNE` and the main helper functions.
- [`src/tsne_torch/estimator.py`](../src/tsne_torch/estimator.py)
  sklearn-style estimator shell, validation, backend dispatch, memory preflight, and diagnostics.
- [`src/tsne_torch/affinity.py`](../src/tsne_torch/affinity.py)
  Dense distance helpers, vectorized perplexity search, dense/sparse probability construction, and sparse affinity builders.
- [`src/tsne_torch/exact_backend.py`](../src/tsne_torch/exact_backend.py)
  Exact KL divergence objective and gradient in Torch.
- [`src/tsne_torch/fft_backend.py`](../src/tsne_torch/fft_backend.py)
  FFT-based repulsive-force approximation, sparse attraction handling, grid splatting, interpolation, and convolution.
- [`src/tsne_torch/optimization.py`](../src/tsne_torch/optimization.py)
  Torch port of sklearn-style gradient descent with momentum, gains, convergence checks, and timing diagnostics.
- [`src/tsne_torch/memory.py`](../src/tsne_torch/memory.py)
  Runtime memory estimation, budget checks, and readable fail-fast errors.
- [`src/tsne_torch/diagnostics.py`](../src/tsne_torch/diagnostics.py)
  Fit diagnostics and helper metrics such as k-NN overlap.
- [`src/tsne_torch/benchmarking.py`](../src/tsne_torch/benchmarking.py)
  Reusable benchmark logic, dataset builders, JSON export, and PNG chart generation.

## API Compatibility Philosophy

`TorchTSNE` is intentionally shaped to feel like sklearn first and a custom research package second.

That means:

- constructor parameters follow sklearn naming and behavior where practical,
- fitted attributes mirror familiar sklearn outputs,
- validation rules stay close to sklearn expectations,
- deliberately unsupported paths do not invent new semantics,
- unsupported paths fall back explicitly to sklearn instead.

The package currently exposes:

- `method="exact"`
- `method="fft"`
- `method="barnes_hut"` via sklearn delegation
- `device="auto" | "cpu" | "cuda"`

## Exact Backend

The exact backend is dense by design.

Pipeline:

1. Build squared distances.
2. Run a batched binary search to match the requested perplexity.
3. Symmetrize and normalize the joint probability matrix `P`.
4. Initialize the embedding with sklearn-like `random` or [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) behavior.
5. Optimize the exact KL objective on Torch tensors.

Important details:

- Euclidean distances use a GEMM-friendly formulation based on `x @ x.T`.
- The KL objective and gradient are written manually rather than using Torch autograd.
- The optimizer follows sklearn's two-stage schedule: early exaggeration first, then the normal phase with updated momentum.

## FFT Backend

The FFT backend is the scalable approximation path.

For small `n`, it can still operate with dense affinities. For larger `n`, it is intended to run with sparse affinities and keep the repulsive-force approximation on a regular grid.

Pipeline:

1. Build a sparse affinity graph from k-nearest neighbors or sparse precomputed distances.
2. Convert the sparse structure into Torch tensors.
3. Splat the current 2D embedding onto a grid.
4. Use [`torch.fft`](https://docs.pytorch.org/docs/stable/fft.html) to convolve the grid with kernels that approximate the repulsive field.
5. Interpolate the field back to point locations.
6. Combine sparse attractive forces with FFT-approximated repulsive forces.
7. Optimize with the same sklearn-like schedule used by the exact backend.

Important constraints:

- The FFT backend currently supports only `n_components=2`.
- Sparse precomputed graphs are the intended path for very large datasets.

## Memory Validation And Device Selection

Large t-SNE workloads are memory-heavy, so `TorchTSNE` performs a memory preflight before entering the expensive kernels.

The preflight:

- estimates backend-specific tensor usage,
- queries available CPU RAM or free CUDA memory,
- applies headroom instead of using the full reported budget,
- raises a readable `MemoryError` if the run is unlikely to fit,
- can fall back from CUDA to CPU when `device="auto"` and only the GPU budget fails.

This matters especially for:

- dense exact t-SNE,
- dense FFT on smaller problems,
- multi-million-edge sparse FFT runs.

## Where The Optimization Comes From

The speedups come from moving the numerically dominant pieces into fast Torch kernels and keeping data on the selected device for as much of the pipeline as possible.

Main sources of acceleration:

- vectorized pairwise distance construction,
- batched device-side perplexity search,
- manual objective evaluation without autograd overhead,
- CUDA-resident optimization state,
- FFT-based repulsive-force approximation via `torch.fft`,
- sparse attraction handling for the scalable path.

In practice:

- `method="exact"` benefits most on medium dense problems,
- `method="fft"` benefits most on large sparse problems.

## Fallback Behavior

Fallbacks are explicit and intentional.

Current examples:

- `method="barnes_hut"` is delegated to sklearn.
- Unsupported combinations for the Torch FFT path fall back to sklearn exact or Barnes-Hut, depending on the requested mode.
- Memory-driven fallback can move `device="auto"` from CUDA to CPU when only the GPU budget fails.

The goal is to preserve expected behavior rather than silently changing semantics or failing late.

## Correctness Strategy

The test suite checks both low-level numerical behavior and end-to-end quality.

Coverage includes:

- parameter validation,
- learning-rate auto behavior,
- exact `P` probability agreement with sklearn,
- exact gradient agreement with sklearn internals,
- deterministic exact runs with fixed seeds,
- exact-vs-sklearn quality checks,
- FFT-vs-exact quality checks,
- early stopping behavior,
- CUDA execution coverage,
- memory fail-fast behavior,
- chart-generation regression tests.

Run the suite with:

```bash
pytest -q
```
