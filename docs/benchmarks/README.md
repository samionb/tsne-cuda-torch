# Benchmark Report

This document holds the full benchmark story for `TorchTSNE`: validated environment, methodology, metric definitions, full tables, and links to every curated benchmark artifact in this folder.

The main [README](../../README.md) keeps only the high-signal landing-page highlights.

## Table Of Contents

- [Overview](#overview)
- [Test Environment](#test-environment)
- [Methodology](#methodology)
- [Metrics](#metrics)
- [Scaling Sweep](#scaling-sweep)
- [MNIST 60k](#mnist-60k)
- [Medium Dense](#medium-dense)
- [Large Sparse](#large-sparse)
- [Largest Shared Run](#largest-shared-run)
- [Artifact Index](#artifact-index)

## Overview

The benchmark suite is designed to answer three practical questions:

- Where does CUDA acceleration materially beat sklearn on comparable inputs?
- How does the FFT path scale relative to sklearn Barnes-Hut?
- What accuracy and memory tradeoffs appear as problem size grows?

The curated artifacts in this directory are single-run snapshots from the validated workstation below. They are useful engineering references, not universal constants.

## Test Environment

Validated workstation:

- CPU: `AMD Ryzen 7 9700X 8-Core Processor`
- CPU topology: `8 physical cores / 16 logical processors`
- GPU: `NVIDIA GeForce RTX 5070 Ti`
- System memory: `31.16 GiB RAM`
- Operating system: `Windows 11 Pro 25H2 (OS Build 26200.8037)`
- Python environment: `conda env1`
- Python version: `3.9.23`
- Numeric stack: `numpy 2.0.2`, `scikit-learn 1.6.1`, `torch 2.8.0+cu128`

These values matter because the large sparse FFT runs depend heavily on:

- available host RAM,
- free CUDA memory,
- CPU throughput during sparse graph preparation,
- GPU throughput during the FFT optimization loop.

## Methodology

Benchmark runner:

- reusable logic: [`src/tsne_torch/benchmarking.py`](../../src/tsne_torch/benchmarking.py)
- CLI wrapper: [`benchmarks/run_benchmarks.py`](../../benchmarks/run_benchmarks.py)
- console entry point: `tsne-torch-benchmark`

General methodology:

- CUDA is warmed up before timed runs.
- CUDA is synchronized around timed regions.
- Each run records end-to-end duration, backend timings, quality metrics, and memory diagnostics.
- Benchmarks emit both JSON results and PNG charts.
- For very large datasets, only the baselines that are still realistic to run are compared.

Important caveat:

- The scaling sweep artifact names still contain `1k_100k` for historical reasons, but the stored results now include the `250000` sample point as well.

## Metrics

### Runtime Metrics

- `median_duration`
  End-to-end runtime for a baseline.
- `timings["affinity_build"]`
  Time spent building dense or sparse affinities.
- `timings["optimization"]`
  Time spent inside the iterative optimization loop.
- `timings["stage1"]` and `timings["stage2"]`
  Early-exaggeration and post-exaggeration timing and stop-reason data.
- `median_iteration_time`
  Median steady-state iteration time inside a phase.

### Accuracy Metrics

- [`trustworthiness`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html)
  Local-neighborhood preservation score. Higher is better.
- `k-NN overlap`
  Direct nearest-neighbor-set retention score. Higher is better.
- `kl_divergence`
  The t-SNE objective value. Lower is better when numerically stable.

Large-scale caveat:

- On the largest sparse runs, `kl_divergence` can overflow. In those cases the most reliable signals are trustworthiness and k-NN overlap.

### Memory Metrics

- `peak_cuda_memory`
  Peak allocated CUDA memory observed during the run.
- `memory.required_bytes`
  Runtime memory estimate from the estimator preflight.
- `memory.fits`
  Whether the preflight estimate fit within the safe budget.

## Scaling Sweep

Artifacts:

- Figure: [`benchmark_scaling_sweep_1k_100k_scaling_sweep_sparse_graph_f512.png`](./benchmark_scaling_sweep_1k_100k_scaling_sweep_sparse_graph_f512.png)
- JSON: [`benchmark_scaling_sweep_1k_100k.json`](./benchmark_scaling_sweep_1k_100k.json)

Methodology:

- datasets: `blobs_1000_f512_sweep_graph`, `blobs_5000_f512_sweep_graph`, `blobs_10000_f512_sweep_graph`, `blobs_50000_f512_sweep_graph`, `blobs_75000_f512_sweep_graph`, `blobs_100000_f512_sweep_graph`, `blobs_250000_f512_sweep_graph`
- input shape: `512` features with a fixed-width `48`-neighbor sparse distance graph
- compared baselines: `sklearn_barnes_hut`, `tsne_torch_fft_cpu`, `tsne_torch_fft_cuda`
- benchmark mode: `--device cuda --repeats 1`

![TorchTSNE scaling sweep chart](./benchmark_scaling_sweep_1k_100k_scaling_sweep_sparse_graph_f512.png)

Results:

| Samples | Graph NNZ | sklearn Barnes-Hut (s) | tsne-torch FFT CPU (s) | tsne-torch FFT CUDA (s) | CPU Speedup vs sklearn | CUDA Speedup vs sklearn |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1000` | `48,000` | `0.427` | `2.473` | `0.946` | `0.17x` | `0.45x` |
| `5000` | `240,000` | `4.111` | `3.667` | `0.535` | `1.12x` | `7.69x` |
| `10000` | `480,000` | `10.131` | `5.187` | `0.587` | `1.95x` | `17.25x` |
| `50000` | `2,400,000` | `80.253` | `21.980` | `1.127` | `3.65x` | `71.21x` |
| `75000` | `3,600,000` | `124.121` | `24.392` | `1.316` | `5.09x` | `94.34x` |
| `100000` | `4,800,000` | `170.758` | `31.569` | `1.449` | `5.41x` | `117.86x` |
| `250000` | `12,000,000` | `614.384` | `76.720` | `2.640` | `8.01x` | `232.75x` |

Power-law fit analysis for the post-crossover regime (`n >= 5000`):

- `t_sklearn(n) ~= 8.568e-05 * n^1.266`
- `t_fft_cpu(n) ~= 4.500e-03 * n^0.776`
- `t_fft_cuda(n) ~= 1.589e-02 * n^0.399`
- `S_fft_cpu(n) ~= 1.904e-02 * n^0.490`
- `S_fft_cuda(n) ~= 5.391e-03 * n^0.867`

Headline observations:

- sklearn Barnes-Hut is still fastest at `1000` samples; the CUDA FFT path overtakes it by `5000` samples.
- The CUDA runtime slope is materially smaller than sklearn's, so the speedup continues to widen with sample count.
- `tsne_torch_fft_cuda` reaches `232.75x` measured speedup at `250000` samples.

## MNIST 60k

Artifacts:

- Runtime chart: [`benchmark_mnist_60k.png`](./benchmark_mnist_60k.png)
- Embedding chart: [`benchmark_mnist_60k_embeddings.png`](./benchmark_mnist_60k_embeddings.png)
- JSON: [`benchmark_mnist_60k.json`](./benchmark_mnist_60k.json)

Methodology:

- dataset: `torchvision.datasets.MNIST(train=True)` with `60,000` training samples
- preprocessing: flattened and normalized to `[0, 1]`, then reduced with `PCA(50)` only for one-time graph construction
- shared benchmark input: exact sparse `48`-NN squared-distance graph built once from the PCA-reduced features
- compared baselines: `sklearn_barnes_hut`, `tsne_torch_fft_cpu`, `tsne_torch_fft_cuda`
- benchmark mode: `--device cuda --repeats 1`

![TorchTSNE MNIST runtime chart](./benchmark_mnist_60k.png)

![TorchTSNE MNIST embedding comparison](./benchmark_mnist_60k_embeddings.png)

Results:

| Baseline | Median Runtime (s) | Trustworthiness | k-NN Overlap | KL Divergence | Peak GPU Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sklearn_barnes_hut` | `91.224` | `0.928501` | `0.244850` | `overflow` | `0 MB` |
| `tsne_torch_fft_cpu` | `14.939` | `0.931572` | `0.244050` | `overflow` | `0 MB` |
| `tsne_torch_fft_cuda` | `1.056` | `0.931414` | `0.244550` | `overflow` | `211.7 MB` |

Headline observations:

- `tsne_torch_fft_cuda` is `86.37x` faster than sklearn Barnes-Hut on the shared graph.
- `tsne_torch_fft_cpu` is `6.11x` faster than sklearn Barnes-Hut.
- Quality stays tightly clustered across the three runs.
- The scatterplots show visually consistent digit clusters across methods after independent embedding normalization for plotting.

## Medium Dense

Artifacts:

- Figure: [`benchmark_memory_smoke.png`](./benchmark_memory_smoke.png)
- JSON: [`benchmark_memory_smoke.json`](./benchmark_memory_smoke.json)

Dataset:

- `blobs_2048_f512`

![TorchTSNE medium dense benchmark chart](./benchmark_memory_smoke.png)

Results:

| Baseline | Median Runtime (s) | Trustworthiness | k-NN Overlap | KL Divergence | Peak GPU Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sklearn_exact` | `18.099` | `0.991370` | `0.237050` | `0.986602` | `0 MB` |
| `sklearn_barnes_hut` | `1.519` | `0.991340` | `0.235650` | `0.985958` | `0 MB` |
| `tsne_torch_exact_cpu` | `3.537` | `0.991196` | `0.232150` | `0.986734` | `0 MB` |
| `tsne_torch_exact_cuda` | `0.192` | `0.991256` | `0.233350` | `0.987457` | `160.2 MB` |
| `tsne_torch_fft_cuda` | `0.637` | `0.991190` | `0.233200` | `0.980551` | `36.0 MB` |

Headline observations:

- `tsne_torch_exact_cuda` is about `94.4x` faster than sklearn exact.
- `tsne_torch_fft_cuda` is about `28.4x` faster than sklearn exact.
- Quality remains close across methods on trustworthiness and neighborhood retention.

## Large Sparse

Artifacts:

- Figure: [`benchmark_100k_cuda.png`](./benchmark_100k_cuda.png)
- JSON: [`benchmark_100k_cuda.json`](./benchmark_100k_cuda.json)

Dataset:

- `blobs_100000_f512_graph`

![TorchTSNE 100k sparse benchmark chart](./benchmark_100k_cuda.png)

Results:

| Baseline | Median Runtime (s) | Trustworthiness | k-NN Overlap | KL Divergence | Peak GPU Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sklearn_barnes_hut` | `193.786` | `0.996057` | `0.313150` | `overflow` | `0 MB` |
| `tsne_torch_fft_cpu` | `31.487` | `0.996087` | `0.315750` | `overflow` | `0 MB` |
| `tsne_torch_fft_cuda` | `2.376` | `0.996072` | `0.316300` | `overflow` | `480.2 MB` |

Headline observations:

- `tsne_torch_fft_cuda` is about `81.6x` faster than sklearn Barnes-Hut.
- `tsne_torch_fft_cpu` is about `6.2x` faster than sklearn Barnes-Hut.
- At this scale, trustworthiness and k-NN overlap are more reliable than `kl_divergence`.

## Largest Shared Run

Artifacts:

- Figure: [`benchmark_1m_shared.png`](./benchmark_1m_shared.png)
- JSON: [`benchmark_1m_shared.json`](./benchmark_1m_shared.json)

Dataset:

- `blobs_1000000_f512_graph_shared`

![TorchTSNE 1M shared benchmark chart](./benchmark_1m_shared.png)

Results:

| Baseline | Median Runtime (s) | Trustworthiness | k-NN Overlap | KL Divergence | Peak GPU Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sklearn_barnes_hut` | `3891.127` | `0.999542` | `0.684350` | `overflow` | `0 MB` |
| `tsne_torch_fft_cpu` | `297.077` | `0.999546` | `0.681350` | `overflow` | `0 MB` |
| `tsne_torch_fft_cuda` | `10.204` | `0.999546` | `0.682800` | `overflow` | `4787.3 MB` |

Headline observations:

- `1000000` samples is the largest size validated here for both sklearn and TorchTSNE.
- `tsne_torch_fft_cuda` is about `381x` faster than sklearn Barnes-Hut.
- The shared million-sample run is feasible on this workstation only because the benchmark uses a sparse precomputed graph rather than a dense feature matrix.

## Artifact Index

| Artifact | Purpose |
| --- | --- |
| [`benchmark_scaling_sweep_1k_100k_scaling_sweep_sparse_graph_f512.png`](./benchmark_scaling_sweep_1k_100k_scaling_sweep_sparse_graph_f512.png) | Scaling sweep figure through `250000` samples |
| [`benchmark_scaling_sweep_1k_100k.json`](./benchmark_scaling_sweep_1k_100k.json) | Scaling sweep raw results and fit analysis |
| [`benchmark_mnist_60k.png`](./benchmark_mnist_60k.png) | `MNIST 60k` runtime / quality / memory comparison |
| [`benchmark_mnist_60k_embeddings.png`](./benchmark_mnist_60k_embeddings.png) | `MNIST 60k` side-by-side 2D embedding scatterplots |
| [`benchmark_mnist_60k.json`](./benchmark_mnist_60k.json) | `MNIST 60k` raw results |
| [`benchmark_memory_smoke.png`](./benchmark_memory_smoke.png) | Medium dense comparison figure |
| [`benchmark_memory_smoke.json`](./benchmark_memory_smoke.json) | Medium dense raw results |
| [`benchmark_100k_cuda.png`](./benchmark_100k_cuda.png) | `100000 x 512` sparse benchmark figure |
| [`benchmark_100k_cuda.json`](./benchmark_100k_cuda.json) | `100000 x 512` sparse raw results |
| [`benchmark_1m_shared.png`](./benchmark_1m_shared.png) | Largest shared sklearn/tsne-torch comparison figure |
| [`benchmark_1m_shared.json`](./benchmark_1m_shared.json) | Largest shared raw results |
