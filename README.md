# TorchTSNE

`TorchTSNE` is a high-performance, scikit-learn-style [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) implementation built on vanilla [PyTorch](https://pytorch.org/). It keeps the public feel of `sklearn.manifold.TSNE` while moving the heavy numeric work into Torch tensor operations and [CUDA](https://docs.pytorch.org/docs/stable/notes/cuda.html) when available.

The project stays in pure Python: no custom C++ or CUDA extensions, no `nvcc`, and no separately compiled helper libraries beyond the PyTorch wheel you install.

## What It Supports

- `method="exact"` for dense exact t-SNE on Torch tensors
- `method="fft"` for the scalable sparse + `torch.fft` path
- `method="barnes_hut"` for sklearn-compatible API parity via explicit sklearn fallback
- `device="auto" | "cpu" | "cuda"`

Practical priorities:

- sklearn-like constructor semantics and fitted attributes
- correctness checks against scikit-learn behavior
- CUDA acceleration for the numerically dominant kernels
- memory preflight checks before expensive runs

## Dependencies

Required runtime dependencies:

- `torch`
- `numpy`
- `scipy`
- `scikit-learn`

Optional dependencies:

- `matplotlib` for benchmark figures
- `psutil` for memory-budget estimation
- `torchvision` for the optional `MNIST 60k` benchmark

Compatibility:

- Python `3.8+`
- CPU-only and CUDA-enabled PyTorch builds are both supported

## Installation

1. Install the correct [PyTorch build](https://pytorch.org/get-started/locally/) for your CPU or CUDA environment.
2. Install the package:

```bash
pip install -e .[dev]
```

For benchmark-only usage:

```bash
pip install -e .[bench]
```

If you want to run the optional `MNIST 60k` benchmark, install a matching `torchvision` build for your selected `torch` wheel.

## Usage

```python
from tsne_torch import TorchTSNE

model = TorchTSNE(
    method='exact',
    device='cuda',
    init='random',
    learning_rate='auto',
    perplexity=30.0,
    max_iter=300,
    random_state=0,
)

embedding = model.fit_transform(x)
print(model.kl_divergence_)
print(model.embedding_.shape)
```

For large sparse runs, use `method='fft'` with `metric='precomputed'` and pass a sparse distance graph.

## How It Works

- The front-end estimator mirrors sklearn closely and keeps familiar fitted attributes such as `embedding_`, `kl_divergence_`, `n_iter_`, and `learning_rate_`.
- The exact backend builds dense affinities in Torch and evaluates the KL objective and gradient manually, without autograd overhead.
- The FFT backend uses sparse affinities plus `torch.fft` grid convolution to approximate the repulsive field efficiently at large scale, with reused occupancy FFTs and batched grid sampling to keep per-iteration overhead down.
- The optimizer follows sklearn's two-stage schedule with early exaggeration, gains, momentum, and convergence checks.
- A memory preflight estimates CPU or CUDA budget before the run starts and can fall back from GPU to CPU when `device='auto'`.
- Unsupported or intentionally unported cases fall back explicitly to sklearn instead of silently changing semantics.

## Benchmark Highlights

The full benchmark report, methodology, hardware details, full tables, and artifact index live in [docs/benchmarks/README.md](./docs/benchmarks/README.md). The figures below are the three main anchor results.

### Scaling Sweep

The scaling sweep shows the key trend in this project: the CUDA FFT path grows with a much smaller log-log runtime slope than sklearn Barnes-Hut, so speedup keeps increasing as sample count rises.

![TorchTSNE scaling sweep chart](./docs/benchmarks/benchmark_scaling_sweep_1k_100k_scaling_sweep_sparse_graph_f512.png)

### MNIST 60k Embedding Comparison

On a shared sparse graph built from the `MNIST` training split, the CUDA FFT path preserves visually consistent digit clusters while delivering a large runtime advantage over sklearn Barnes-Hut.

![TorchTSNE MNIST embedding comparison](./docs/benchmarks/benchmark_mnist_60k_embeddings.png)

### Medium Dense Comparison

For medium dense inputs, the exact CUDA backend is the clearest apples-to-apples comparison against sklearn exact: large speedup, similar trustworthiness, and fully GPU-resident optimization.

![TorchTSNE medium dense benchmark chart](./docs/benchmarks/benchmark_memory_smoke.png)

### At A Glance

| Scenario | Main takeaway |
| --- | --- |
| Scaling sweep (`1k -> 250k`) | Post-crossover fits show sklearn runtime growing roughly like `n^1.266`, while `tsne-torch` FFT CUDA grows closer to `n^0.399`; measured CUDA speedup reaches `232.75x` at `250000` samples. |
| `MNIST 60k` | `tsne_torch_fft_cuda` is `86.37x` faster than sklearn Barnes-Hut on the shared sparse graph, with comparable neighborhood-preservation quality. |
| `2048 x 512` medium dense | `tsne_torch_exact_cuda` is about `94.4x` faster than sklearn exact while staying close on trustworthiness and k-NN overlap. |

## Docs

- [Benchmark report](./docs/benchmarks/README.md)
  Full benchmark methodology, hardware details, metric definitions, tables, and artifact index.
- [Implementation note](./docs/implementation.md)
  Internal architecture, backend behavior, fallback policy, memory validation, and correctness approach.
- [Benchmark CLI](./benchmarks/run_benchmarks.py)
  Thin entrypoint for local benchmark runs.

## Current Limitations

- The FFT backend currently supports only 2D embeddings.
- Barnes-Hut is delegated to sklearn rather than reimplemented here.
- `kl_divergence` can overflow on very large sparse runs, so trustworthiness and k-NN overlap are more reliable at that scale.
- The million-sample benchmark uses sparse graph inputs rather than a fully materialized dense feature matrix.

## License

This project is licensed under the [BSD 3-Clause License](./LICENSE). Provenance and sklearn attribution notes are recorded in [NOTICE](./NOTICE).
