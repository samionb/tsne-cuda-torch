"""Benchmark utilities and CLI for TorchTSNE."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from statistics import median
from time import perf_counter

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_array
import torch
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits, make_blobs
from sklearn.manifold import TSNE, trustworthiness

from .diagnostics import knn_overlap, library_versions
from .estimator import TorchTSNE
from .memory import format_num_bytes

LOGGER = logging.getLogger('tsne_torch.benchmarking')

_KL_OVERFLOW_THRESHOLD = 1e100

BASELINE_LABELS = {
    'sklearn_exact': 'sklearn exact',
    'sklearn_barnes_hut': 'sklearn Barnes-Hut',
    'tsne_torch_exact_cpu': 'tsne-torch exact CPU',
    'tsne_torch_exact_cuda': 'tsne-torch exact CUDA',
    'tsne_torch_fft_cpu': 'tsne-torch FFT CPU',
    'tsne_torch_fft_cuda': 'tsne-torch FFT CUDA',
}

SCALING_SWEEP_SAMPLES = (1000, 5000, 10000, 50000, 75000, 100000, 250000)
SCALING_SWEEP_GROUP = 'scaling_sweep_sparse_graph_f512'
SCALING_SWEEP_BASELINES = (
    'sklearn_barnes_hut',
    'tsne_torch_fft_cpu',
    'tsne_torch_fft_cuda',
)
SCALING_SWEEP_FIT_MIN_SAMPLE_COUNT = 5000
MNIST_PCA_COMPONENTS = 50
MNIST_GRAPH_NEIGHBORS = 48
MNIST_BENCHMARK_DATASET = 'mnist_train_60000_graph'


def configure_logging():
    """
    Configure process-wide logging for benchmark CLI usage.

    :return: None.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    else:
        logging.getLogger().setLevel(logging.INFO)


def _is_valid_kl(value: float) -> bool:
    """
    Check whether a KL divergence value is numerically meaningful for reporting.

    :param value: KL divergence value to validate.

    :return: ``True`` when the value is finite and below the overflow threshold.
    """
    return np.isfinite(value) and value < _KL_OVERFLOW_THRESHOLD


def _format_kl(value: float) -> str:
    """
    Format a KL divergence value for concise benchmark logs and charts.

    :param value: KL divergence value to format.

    :return: Formatted KL string, or ``'overflow'`` / ``'nan'`` for unstable values.
    """
    if np.isnan(value):
        return 'nan'
    if not np.isfinite(value) or value >= _KL_OVERFLOW_THRESHOLD:
        return 'overflow'
    return f'{value:.6f}'


def _format_baseline_label(label: str) -> str:
    """
    Convert an internal baseline identifier into a chart-friendly label.

    :param label: Internal baseline name.

    :return: Human-readable label for plots and summaries.
    """
    return BASELINE_LABELS.get(label, label.replace('_', ' '))


def _metric_limits(values, *, floor: float = 0.0, ceil: float = 1.0, min_pad: float = 0.01):
    """
    Compute a readable y-axis range for clustered metric values.

    :param values: Metric values to visualize.
    :param floor: Lower hard bound for the axis.
    :param ceil: Upper hard bound for the axis.
    :param min_pad: Minimum padding to apply around the data range.

    :return: Tuple ``(lower_limit, upper_limit)``.
    """
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return floor, ceil

    lower = float(finite.min())
    upper = float(finite.max())
    span = upper - lower
    pad = max(span * 0.2, min_pad)
    lower = max(floor, lower - pad)
    upper = min(ceil, upper + pad)
    if upper <= lower:
        upper = min(ceil, lower + min_pad)
    return lower, upper


def _annotate_bars(ax, bars, labels, *, rotation: int = 0, fontsize: int = 9):
    """
    Add compact value labels above a Matplotlib bar collection.

    :param ax: Matplotlib axis receiving the annotations.
    :param bars: Bar artists returned by ``ax.bar``.
    :param labels: Text labels to place above the bars.
    :param rotation: Text rotation in degrees.
    :param fontsize: Annotation font size.

    :return: None.
    """
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=fontsize,
            rotation=rotation,
        )


def _chart_output_path(base_output: Path, dataset_name: str, *, single_dataset: bool) -> Path:
    """
    Choose a stable PNG output path for a dataset chart.

    :param base_output: Benchmark JSON output path.
    :param dataset_name: Dataset identifier.
    :param single_dataset: Whether the run contains exactly one dataset.

    :return: PNG output path for the chart.
    """
    if single_dataset:
        return base_output.with_suffix('.png')
    return base_output.with_name(f'{base_output.stem}_{dataset_name}.png')


def _embedding_chart_output_path(base_output: Path, dataset_name: str, *, single_dataset: bool) -> Path:
    """
    Choose a stable PNG output path for an embedding-comparison chart.

    :param base_output: Benchmark JSON output path.
    :param dataset_name: Dataset identifier.
    :param single_dataset: Whether the run contains exactly one dataset.

    :return: PNG output path for the embedding chart.
    """
    if single_dataset:
        return base_output.with_name(f'{base_output.stem}_embeddings.png')
    return base_output.with_name(f'{base_output.stem}_{dataset_name}_embeddings.png')


def _normalize_embedding_for_plot(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D embedding for more comparable side-by-side visualization.

    :param embedding: Embedding array with shape ``(n_samples, 2)``.

    :return: Centered and scale-normalized embedding.
    """
    embedding = np.asarray(embedding, dtype=np.float32)
    centered = embedding - np.mean(embedding, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if scale <= 0.0 or not np.isfinite(scale):
        return centered
    return centered / scale


def fit_power_law_curve(
    x_values,
    y_values,
    *,
    min_sample_count: int | None = None,
):
    """
    Fit a power-law curve of the form ``y = coefficient * x ** exponent`` in log-log space.

    :param x_values: Independent variable values.
    :param y_values: Positive dependent variable values.
    :param min_sample_count: Optional lower bound for the x-range included in the fit.

    :return: Fit metadata dictionary, or ``None`` when there are too few valid points.
    """
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x_array) & np.isfinite(y_array) & (x_array > 0.0) & (y_array > 0.0)
    if min_sample_count is not None:
        valid &= x_array >= float(min_sample_count)
    if int(valid.sum()) < 2:
        return None

    log_x = np.log10(x_array[valid])
    log_y = np.log10(y_array[valid])
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    predicted = exponent * log_x + intercept
    ss_res = float(np.sum((log_y - predicted) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        'coefficient': float(10.0**intercept),
        'exponent': float(exponent),
        'r2_log10': float(r2),
        'n_points': int(valid.sum()),
        'min_sample_count': int(min_sample_count) if min_sample_count is not None else None,
    }


def evaluate_power_law_curve(x_values, fit: dict):
    """
    Evaluate a previously fitted power-law model.

    :param x_values: Independent variable values.
    :param fit: Fit dictionary returned by ``fit_power_law_curve``.

    :return: NumPy array containing the fitted values.
    """
    x_array = np.asarray(x_values, dtype=np.float64)
    return fit['coefficient'] * np.power(x_array, fit['exponent'])


def format_power_law_equation(fit: dict, *, symbol: str) -> str:
    """
    Format a fitted power-law model for logs, charts, and README summaries.

    :param fit: Fit dictionary returned by ``fit_power_law_curve``.
    :param symbol: Left-hand-side symbol used in the formatted equation.

    :return: Compact equation string.
    """
    return f'{symbol} ~= {fit["coefficient"]:.3e} * n^{fit["exponent"]:.3f} (R^2={fit["r2_log10"]:.3f})'


def analyze_scaling_sweep(summary_rows: list[dict]):
    """
    Compute post-crossover power-law fits for runtime and speedup across the scaling sweep.

    :param summary_rows: Sweep summary rows keyed by sample count.

    :return: Analysis dictionary containing runtime and speedup fit metadata.
    """
    summary_rows = sorted(summary_rows, key=lambda row: row['sample_count'])
    sample_counts = np.asarray([row['sample_count'] for row in summary_rows], dtype=np.float64)
    analysis = {
        'fit_min_sample_count': SCALING_SWEEP_FIT_MIN_SAMPLE_COUNT,
        'runtime_power_law': {},
        'speedup_power_law': {},
    }

    sklearn_fit = None
    for baseline in SCALING_SWEEP_BASELINES:
        durations = np.asarray(
            [row.get(f'{baseline}_duration', np.nan) for row in summary_rows],
            dtype=np.float64,
        )
        fit = fit_power_law_curve(
            sample_counts,
            durations,
            min_sample_count=SCALING_SWEEP_FIT_MIN_SAMPLE_COUNT,
        )
        if fit is None:
            continue
        analysis['runtime_power_law'][baseline] = fit
        if baseline == 'sklearn_barnes_hut':
            sklearn_fit = fit

    for baseline in SCALING_SWEEP_BASELINES[1:]:
        speedups = np.asarray(
            [row.get(f'{baseline}_speedup_vs_sklearn', np.nan) for row in summary_rows],
            dtype=np.float64,
        )
        fit = fit_power_law_curve(
            sample_counts,
            speedups,
            min_sample_count=SCALING_SWEEP_FIT_MIN_SAMPLE_COUNT,
        )
        if fit is None:
            continue
        if sklearn_fit is not None and baseline in analysis['runtime_power_law']:
            fit['derived_exponent_from_runtime_slopes'] = (
                sklearn_fit['exponent'] - analysis['runtime_power_law'][baseline]['exponent']
            )
        analysis['speedup_power_law'][baseline] = fit

    return analysis


def save_dataset_chart(name: str, profile: dict, rows: list[dict], output_path: Path):
    """
    Render and save a PNG comparison chart for one benchmark dataset.

    :param name: Dataset identifier.
    :param profile: Dataset profile metadata used in the chart subtitle.
    :param rows: Benchmark result rows for the dataset.
    :param output_path: PNG output path.

    :return: Saved output path, or ``None`` when Matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning('Matplotlib is not available; skipping chart generation for %s', name)
        return None

    plt.style.use('default')
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not cycle_colors:
        cycle_colors = [f'C{i}' for i in range(10)]

    display_name = profile.get('display_name', name)
    labels = [_format_baseline_label(row['baseline']) for row in rows]
    runtimes = np.asarray([row['median_duration'] for row in rows], dtype=np.float64)
    trust = np.asarray([row['trustworthiness'] for row in rows], dtype=np.float64)
    overlap = np.asarray([row['knn_overlap'] for row in rows], dtype=np.float64)
    estimated_mem = np.asarray(
        [row.get('memory', {}).get('required_bytes', np.nan) / (1024 * 1024) for row in rows],
        dtype=np.float64,
    )
    peak_mem = np.asarray(
        [row['diagnostics'].get('peak_cuda_memory', 0) / (1024 * 1024) for row in rows],
        dtype=np.float64,
    )

    reference_row = None
    for preferred in ('sklearn_exact', 'sklearn_barnes_hut', rows[0]['baseline']):
        reference_row = next((row for row in rows if row['baseline'] == preferred), None)
        if reference_row is not None:
            break
    reference_duration = reference_row['median_duration']
    speedups = reference_duration / runtimes

    colors = [cycle_colors[i % len(cycle_colors)] for i in range(len(rows))]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle('TorchTSNE Benchmark Comparison', fontsize=18, fontweight='bold')
    subtitle = (
        f'Dataset: {display_name} | input={profile["input_shape"]} | '
        f'fit_input={profile.get("fit_input_shape", profile["input_shape"])} | metric={profile["metric"]}'
    )
    if profile.get('benchmark_note'):
        subtitle = f'{subtitle} | {profile["benchmark_note"]}'
    fig.text(
        0.5,
        0.965,
        subtitle,
        ha='center',
        fontsize=11,
    )

    runtime_ax = axes[0, 0]
    runtime_bars = runtime_ax.bar(x, runtimes, color=colors, edgecolor='#2f2f2f', linewidth=0.8)
    if np.nanmax(runtimes) / max(np.nanmin(runtimes), 1e-9) >= 8.0:
        runtime_ax.set_yscale('log')
        runtime_ax.set_ylabel('Median Runtime (seconds, log scale)')
    else:
        runtime_ax.set_ylabel('Median Runtime (seconds)')
    runtime_ax.set_title('Runtime Comparison')
    runtime_ax.set_xticks(x, labels, rotation=25, ha='right')
    runtime_ax.grid(axis='y', linestyle='--', alpha=0.35)
    runtime_labels = [f'{runtime:.2f}s\n{speedup:.1f}x' for runtime, speedup in zip(runtimes, speedups)]
    _annotate_bars(runtime_ax, runtime_bars, runtime_labels, fontsize=8)
    runtime_ax.text(
        0.02,
        0.98,
        f'Speedup annotations are relative to {reference_row["baseline"]}',
        transform=runtime_ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'edgecolor': '#b0b0b0'},
    )

    trust_ax = axes[0, 1]
    trust_bars = trust_ax.bar(x, trust, color=colors, edgecolor='#2f2f2f', linewidth=0.8)
    trust_ax.set_title('Trustworthiness')
    trust_ax.set_ylabel('Score')
    trust_ax.set_xticks(x, labels, rotation=25, ha='right')
    trust_ax.set_ylim(*_metric_limits(trust, floor=0.0, ceil=1.0, min_pad=0.002))
    trust_ax.grid(axis='y', linestyle='--', alpha=0.35)
    _annotate_bars(trust_ax, trust_bars, [f'{value:.4f}' for value in trust], fontsize=8)

    overlap_ax = axes[1, 0]
    overlap_bars = overlap_ax.bar(x, overlap, color=colors, edgecolor='#2f2f2f', linewidth=0.8)
    overlap_ax.set_title('Neighborhood Retention (k-NN Overlap)')
    overlap_ax.set_ylabel('Score')
    overlap_ax.set_xticks(x, labels, rotation=25, ha='right')
    overlap_ax.set_ylim(*_metric_limits(overlap, floor=0.0, ceil=1.0, min_pad=0.02))
    overlap_ax.grid(axis='y', linestyle='--', alpha=0.35)
    _annotate_bars(overlap_ax, overlap_bars, [f'{value:.4f}' for value in overlap], fontsize=8)

    memory_ax = axes[1, 1]
    width = 0.38
    est_bars = memory_ax.bar(
        x - width / 2.0,
        estimated_mem,
        width=width,
        color=cycle_colors[0],
        edgecolor='#2f2f2f',
        linewidth=0.8,
        label='Estimated Runtime Memory',
    )
    peak_bars = memory_ax.bar(
        x + width / 2.0,
        peak_mem,
        width=width,
        color=cycle_colors[1 % len(cycle_colors)],
        edgecolor='#2f2f2f',
        linewidth=0.8,
        label='Measured Peak GPU Memory',
    )
    memory_ax.set_title('Memory Comparison')
    memory_ax.set_ylabel('Memory (MiB)')
    memory_ax.set_xticks(x, labels, rotation=25, ha='right')
    memory_ax.grid(axis='y', linestyle='--', alpha=0.35)
    memory_ax.legend(loc='upper right', frameon=True)
    finite_memory = np.asarray(
        [value for value in np.concatenate((estimated_mem, peak_mem)) if np.isfinite(value)],
        dtype=np.float64,
    )
    memory_ax.set_ylim(0.0, max(float(finite_memory.max()) * 1.15, 1.0) if finite_memory.size else 1.0)
    _annotate_bars(
        memory_ax,
        est_bars,
        ['n/a' if not np.isfinite(value) else f'{value:.1f}' for value in estimated_mem],
        fontsize=8,
    )
    _annotate_bars(memory_ax, peak_bars, [f'{value:.1f}' for value in peak_mem], fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    LOGGER.info('Saved benchmark chart to %s', output_path)
    return output_path


def save_embedding_comparison_chart(
    name: str,
    profile: dict,
    rows: list[dict],
    embeddings_by_baseline: dict[str, np.ndarray],
    output_path: Path,
):
    """
    Render and save a side-by-side embedding scatterplot comparison.

    :param name: Dataset identifier.
    :param profile: Dataset profile metadata.
    :param rows: Benchmark result rows for the dataset.
    :param embeddings_by_baseline: Mapping from baseline label to 2D embedding.
    :param output_path: PNG output path.

    :return: Saved output path, or ``None`` when the dataset lacks labels or Matplotlib is unavailable.
    """
    labels = profile.get('plot_labels')
    if labels is None:
        return None

    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        LOGGER.warning('Matplotlib is not available; skipping embedding chart generation for %s', name)
        return None

    available_rows = [row for row in rows if row['baseline'] in embeddings_by_baseline]
    if not available_rows:
        return None

    labels = np.asarray(labels)
    if labels.shape[0] != next(iter(embeddings_by_baseline.values())).shape[0]:
        LOGGER.warning('Label length mismatch for %s; skipping embedding chart generation', name)
        return None

    plt.style.use('default')
    display_name = profile.get('display_name', name)
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab10', len(unique_labels))
    color_lookup = {label: cmap(index) for index, label in enumerate(unique_labels)}

    normalized_embeddings = {
        row['baseline']: _normalize_embedding_for_plot(embeddings_by_baseline[row['baseline']])
        for row in available_rows
    }
    all_points = np.vstack([normalized_embeddings[row['baseline']] for row in available_rows])
    limit = float(np.max(np.abs(all_points))) * 1.08 if all_points.size else 1.0
    limit = max(limit, 1.0)

    fig, axes = plt.subplots(1, len(available_rows), figsize=(6.1 * len(available_rows), 6.2), constrained_layout=False)
    if len(available_rows) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.04, right=0.88, bottom=0.08, top=0.82, wspace=0.08)
    fig.suptitle(f'{display_name} 2D Embedding Comparison', fontsize=17, fontweight='bold', y=0.97)
    fig.text(
        0.5,
        0.925,
        'Embeddings are centered and scale-normalized independently for visual comparison',
        ha='center',
        fontsize=10.5,
    )

    for ax, row in zip(axes, available_rows):
        baseline = row['baseline']
        embedding = normalized_embeddings[baseline]
        colors = [color_lookup[label] for label in labels]
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=colors,
            s=2.0,
            alpha=0.65,
            linewidths=0.0,
            rasterized=True,
        )
        ax.set_title(
            f'{_format_baseline_label(baseline)}\n'
            f'{row["median_duration"]:.3f}s | trust={row["trustworthiness"]:.4f}',
            fontsize=11,
        )
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        for spine in ax.spines.values():
            spine.set_alpha(0.35)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=str(label), markerfacecolor=color_lookup[label], markersize=7)
        for label in unique_labels
    ]
    fig.legend(
        handles=legend_handles,
        title='Digit Label',
        loc='center right',
        bbox_to_anchor=(0.985, 0.52),
        frameon=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    LOGGER.info('Saved embedding comparison chart to %s', output_path)
    return output_path


def save_scaling_sweep_chart(name: str, summary_rows: list[dict], output_path: Path, analysis: dict | None = None):
    """
    Render and save a log-scale scaling chart across multiple dataset sizes.

    :param name: Sweep identifier.
    :param summary_rows: Sweep summary rows keyed by sample size.
    :param output_path: PNG output path.
    :param analysis: Optional precomputed sweep analysis dictionary.

    :return: Saved output path, or ``None`` when Matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning('Matplotlib is not available; skipping scaling sweep chart generation for %s', name)
        return None

    plt.style.use('default')
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not cycle_colors:
        cycle_colors = [f'C{i}' for i in range(10)]

    summary_rows = sorted(summary_rows, key=lambda row: row['sample_count'])
    analysis = analyze_scaling_sweep(summary_rows) if analysis is None else analysis
    sample_counts = np.asarray([row['sample_count'] for row in summary_rows], dtype=np.float64)
    graph_nnz = np.asarray([row['graph_nnz'] for row in summary_rows], dtype=np.float64)

    runtime_series = {}
    for index, baseline in enumerate(SCALING_SWEEP_BASELINES):
        durations = np.asarray(
            [row.get(f'{baseline}_duration', np.nan) for row in summary_rows],
            dtype=np.float64,
        )
        if np.any(np.isfinite(durations)):
            runtime_series[baseline] = {
                'durations': durations,
                'color': cycle_colors[index % len(cycle_colors)],
                'label': _format_baseline_label(baseline),
            }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.80, wspace=0.12)
    fig.suptitle('TorchTSNE vs scikit-learn Scaling Sweep', fontsize=17, fontweight='bold', y=0.985)
    fig.text(
        0.5,
        0.952,
        'Sparse 512-feature benchmark, 48-neighbor graph, log-scale runtime profiling',
        ha='center',
        fontsize=10.5,
    )

    runtime_ax = axes[0]
    for baseline, series in runtime_series.items():
        runtime_ax.plot(
            sample_counts,
            series['durations'],
            marker='o',
            linewidth=2.2,
            markersize=6,
            color=series['color'],
            label=series['label'],
        )
        fit = analysis['runtime_power_law'].get(baseline)
        if fit is not None:
            fit_x = sample_counts[sample_counts >= float(fit['min_sample_count'])]
            runtime_ax.plot(
                fit_x,
                evaluate_power_law_curve(fit_x, fit),
                linestyle='--',
                linewidth=1.6,
                alpha=0.85,
                color=series['color'],
            )
    runtime_ax.set_xscale('log')
    runtime_ax.set_yscale('log')
    runtime_ax.set_title('Runtime vs Sample Count')
    runtime_ax.set_xlabel('Samples')
    runtime_ax.set_ylabel('Median Runtime (seconds)')
    runtime_ax.grid(True, which='both', linestyle='--', alpha=0.35)
    runtime_ax.legend(loc='upper left', frameon=True)
    runtime_fit_lines = [f'Power-law fits for n >= {analysis["fit_min_sample_count"]:,}:']
    for baseline in SCALING_SWEEP_BASELINES:
        fit = analysis['runtime_power_law'].get(baseline)
        if fit is None:
            continue
        runtime_fit_lines.append(
            f'{_format_baseline_label(baseline)}: a={fit["exponent"]:.3f}, R^2={fit["r2_log10"]:.3f}'
        )
    runtime_ax.text(
        0.98,
        0.03,
        '\n'.join(runtime_fit_lines),
        transform=runtime_ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=8.4,
        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'edgecolor': '#b0b0b0'},
    )

    speedup_ax = axes[1]
    for index, baseline in enumerate(SCALING_SWEEP_BASELINES[1:], start=1):
        speedups = np.asarray(
            [row.get(f'{baseline}_speedup_vs_sklearn', np.nan) for row in summary_rows],
            dtype=np.float64,
        )
        if not np.any(np.isfinite(speedups)):
            continue
        speedup_ax.plot(
            sample_counts,
            speedups,
            marker='o',
            linewidth=2.2,
            markersize=6,
            color=cycle_colors[index % len(cycle_colors)],
            label=f'{_format_baseline_label(baseline)} vs sklearn Barnes-Hut',
        )
        fit = analysis['speedup_power_law'].get(baseline)
        if fit is not None:
            fit_x = sample_counts[sample_counts >= float(fit['min_sample_count'])]
            speedup_ax.plot(
                fit_x,
                evaluate_power_law_curve(fit_x, fit),
                linestyle='--',
                linewidth=1.6,
                alpha=0.85,
                color=cycle_colors[index % len(cycle_colors)],
            )
    speedup_ax.set_xscale('log')
    speedup_ax.set_title('Speedup vs scikit-learn Barnes-Hut')
    speedup_ax.set_xlabel('Samples')
    speedup_ax.set_ylabel('Speedup (x)')
    speedup_ax.grid(True, which='both', linestyle='--', alpha=0.35)
    speedup_ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.16),
        frameon=True,
        ncol=1,
    )

    nnz_note = (
        f'Graph edges sweep from {int(graph_nnz.min()):,} to {int(graph_nnz.max()):,} '
        f'(48 neighbors per sample)'
    )
    speedup_ax.text(
        0.02,
        0.98,
        nnz_note,
        transform=speedup_ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'edgecolor': '#b0b0b0'},
    )
    speedup_fit_lines = [f'Power-law speedup fits for n >= {analysis["fit_min_sample_count"]:,}:']
    for baseline in SCALING_SWEEP_BASELINES[1:]:
        fit = analysis['speedup_power_law'].get(baseline)
        if fit is None:
            continue
        speedup_fit_lines.append(
            f'{_format_baseline_label(baseline)}: beta={fit["exponent"]:.3f}, R^2={fit["r2_log10"]:.3f}'
        )
    speedup_ax.text(
        0.02,
        0.05,
        '\n'.join(speedup_fit_lines),
        transform=speedup_ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=8.4,
        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'edgecolor': '#b0b0b0'},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)
    LOGGER.info('Saved scaling sweep chart to %s', output_path)
    return output_path


def summarize_dataset_results(name: str, profile: dict, rows: list[dict]):
    """
    Log a readable speed, quality, and memory summary for a benchmark dataset.

    :param name: Dataset identifier.
    :param profile: Dataset profile metadata.
    :param rows: Benchmark result rows for the dataset.

    :return: None.
    """
    LOGGER.info(
        'Dataset %s | metric=%s | input_shape=%s | fit_input_shape=%s',
        name,
        profile['metric'],
        profile['input_shape'],
        profile.get('fit_input_shape', profile['input_shape']),
    )
    if profile.get('graph_nnz') is not None:
        LOGGER.info('Dataset %s uses sparse graph input with nnz=%s', name, profile['graph_nnz'])

    reference_row = None
    for preferred in ('sklearn_exact', 'sklearn_barnes_hut', rows[0]['baseline']):
        reference_row = next((row for row in rows if row['baseline'] == preferred), None)
        if reference_row is not None:
            break
    fastest_row = min(rows, key=lambda row: row['median_duration'])
    best_trust_row = max(rows, key=lambda row: row['trustworthiness'])
    valid_kl_rows = [row for row in rows if _is_valid_kl(row['kl_divergence'])]
    best_kl_row = min(valid_kl_rows, key=lambda row: row['kl_divergence']) if valid_kl_rows else None

    LOGGER.info(
        'Highlights for %s | fastest=%s (%.3fs) | best_trust=%s (%.6f) | best_kl=%s (%s)',
        name,
        fastest_row['baseline'],
        fastest_row['median_duration'],
        best_trust_row['baseline'],
        best_trust_row['trustworthiness'],
        best_kl_row['baseline'] if best_kl_row is not None else 'unreliable',
        _format_kl(best_kl_row['kl_divergence']) if best_kl_row is not None else 'overflow',
    )

    LOGGER.info('Accuracy summary for %s:', name)
    for row in rows:
        LOGGER.info(
            '  %-22s trust=%.6f knn_overlap=%.6f kl=%s',
            row['baseline'],
            row['trustworthiness'],
            row['knn_overlap'],
            _format_kl(row['kl_divergence']),
        )

    LOGGER.info('Speed summary for %s:', name)
    for row in rows:
        speedup = reference_row['median_duration'] / row['median_duration']
        peak_mem = row['diagnostics'].get('peak_cuda_memory', 0)
        estimate = row.get('memory', {})
        LOGGER.info(
            '  %-22s median=%.3fs speedup_vs_%s=%.2fx peak_gpu_mem=%.1f MB estimated=%s fits=%s',
            row['baseline'],
            row['median_duration'],
            reference_row['baseline'],
            speedup,
            peak_mem / (1024 * 1024),
            format_num_bytes(estimate.get('required_bytes')),
            estimate.get('fits'),
        )


def summarize_scaling_sweep(name: str, summary_rows: list[dict]):
    """
    Log a compact scaling summary across multiple sample sizes.

    :param name: Sweep identifier.
    :param summary_rows: Sweep summary rows.

    :return: Sweep analysis dictionary.
    """
    LOGGER.info('Scaling sweep summary for %s:', name)
    for row in summary_rows:
        LOGGER.info(
            '  samples=%-8s sklearn_bh=%.3fs fft_cpu=%.3fs fft_cuda=%.3fs cuda_speedup=%.2fx',
            row['sample_count'],
            row.get('sklearn_barnes_hut_duration', np.nan),
            row.get('tsne_torch_fft_cpu_duration', np.nan),
            row.get('tsne_torch_fft_cuda_duration', np.nan),
            row.get('tsne_torch_fft_cuda_speedup_vs_sklearn', np.nan),
        )
    analysis = analyze_scaling_sweep(summary_rows)
    LOGGER.info('Post-crossover power-law fits for %s (n >= %s):', name, analysis['fit_min_sample_count'])
    for baseline in SCALING_SWEEP_BASELINES:
        fit = analysis['runtime_power_law'].get(baseline)
        if fit is None:
            continue
        LOGGER.info('  runtime %-22s %s', baseline, format_power_law_equation(fit, symbol='t(n)'))
    for baseline in SCALING_SWEEP_BASELINES[1:]:
        fit = analysis['speedup_power_law'].get(baseline)
        if fit is None:
            continue
        LOGGER.info('  speedup %-22s %s', baseline, format_power_law_equation(fit, symbol='S(n)'))
    return analysis


def build_scaling_sweep_summary(name: str, rows: list[dict]) -> list[dict]:
    """
    Build compact per-sample summary rows for a scaling sweep.

    :param name: Sweep identifier.
    :param rows: Dataset result rows belonging to the sweep.

    :return: Sorted summary rows.
    """
    del name
    by_sample = {}
    for row in rows:
        sample_count = row.get('sample_count')
        if sample_count is None:
            continue
        by_sample.setdefault(sample_count, []).append(row)

    summary_rows = []
    for sample_count in sorted(by_sample):
        sample_rows = by_sample[sample_count]
        baseline_map = {row['baseline']: row for row in sample_rows}
        reference = baseline_map.get('sklearn_barnes_hut')
        summary = {
            'sample_count': sample_count,
            'graph_nnz': sample_rows[0].get('graph_nnz'),
        }
        for baseline in SCALING_SWEEP_BASELINES:
            result = baseline_map.get(baseline)
            if result is None:
                continue
            summary[f'{baseline}_duration'] = result['median_duration']
            summary[f'{baseline}_trustworthiness'] = result['trustworthiness']
            summary[f'{baseline}_knn_overlap'] = result['knn_overlap']
            summary[f'{baseline}_peak_gpu_memory'] = result['diagnostics'].get('peak_cuda_memory', 0)
            if reference is not None:
                summary[f'{baseline}_speedup_vs_sklearn'] = (
                    reference['median_duration'] / result['median_duration']
                )
        summary_rows.append(summary)
    return summary_rows


def collect_scaling_sweeps(dataset_rows: list[dict]) -> dict[str, list[dict]]:
    """
    Group benchmark result rows by scaling sweep identifier.

    :param dataset_rows: Flat benchmark result rows.

    :return: Mapping from sweep name to grouped rows.
    """
    sweeps = {}
    for row in dataset_rows:
        sweep_group = row.get('sweep_group')
        if sweep_group is None:
            continue
        sweeps.setdefault(sweep_group, []).append(row)
    return sweeps


def sort_csr_graph_by_row_values(graph: csr_array) -> csr_array:
    """
    Sort each CSR row by ascending distance value.

    :param graph: CSR sparse graph with per-row edge weights.

    :return: Row-sorted CSR sparse graph.
    """
    graph = csr_array(graph)
    indices = graph.indices.copy()
    data = graph.data.copy()
    indptr = graph.indptr.copy()

    for row_index in range(graph.shape[0]):
        start, stop = indptr[row_index], indptr[row_index + 1]
        if stop - start <= 1:
            continue
        order = np.argsort(data[start:stop], kind='mergesort')
        data[start:stop] = data[start:stop][order]
        indices[start:stop] = indices[start:stop][order]

    return csr_array((data, indices, indptr), shape=graph.shape)


def build_cluster_sampled_distance_graph(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    n_neighbors: int,
    random_state: int,
    batch_size: int = 256,
) -> csr_array:
    """
    Build a fixed-width sparse distance graph by sampling neighbors inside each cluster.

    :param x: Dense feature matrix.
    :param labels: Cluster labels aligned with ``x``.
    :param n_neighbors: Number of sampled neighbors per sample.
    :param random_state: Random seed controlling neighbor sampling.
    :param batch_size: Number of samples processed per cluster batch.

    :return: CSR sparse distance graph.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(labels)
    n_samples = x.shape[0]

    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors, dtype=np.int64)
    indices = np.empty(n_samples * n_neighbors, dtype=np.int32)
    data = np.empty(n_samples * n_neighbors, dtype=np.float32)
    column_offsets = np.arange(n_neighbors, dtype=np.int64)

    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if len(members) <= n_neighbors:
            raise ValueError(f'Cluster {label} has only {len(members)} members, need more than {n_neighbors}')

        for start in range(0, len(members), batch_size):
            batch = members[start : start + batch_size]
            self_pos = np.searchsorted(members, batch)
            sampled_pos = rng.integers(0, len(members) - 1, size=(len(batch), n_neighbors), dtype=np.int64)
            sampled_pos += sampled_pos >= self_pos[:, None]
            neighbors = members[sampled_pos]

            diff = x[batch][:, None, :] - x[neighbors]
            distances = np.einsum('bij,bij->bi', diff, diff, optimize=True).astype(np.float32, copy=False)

            flat_rows = (batch[:, None] * n_neighbors + column_offsets).ravel()
            indices[flat_rows] = neighbors.ravel()
            data[flat_rows] = distances.ravel()

    graph = csr_array((data, indices, indptr), shape=(n_samples, n_samples))
    return sort_csr_graph_by_row_values(graph)


def build_synthetic_cluster_graph(
    *,
    n_samples: int,
    n_features: int,
    n_clusters: int,
    cluster_std: float,
    n_neighbors: int,
    random_state: int,
    batch_size: int = 4096,
) -> csr_array:
    """
    Build a synthetic sparse cluster graph without materializing a full dense feature matrix.

    :param n_samples: Number of samples to model.
    :param n_features: Virtual feature dimensionality used for the sampled distance distribution.
    :param n_clusters: Number of synthetic clusters.
    :param cluster_std: Synthetic cluster standard deviation.
    :param n_neighbors: Number of sampled neighbors per sample.
    :param random_state: Random seed controlling graph generation.
    :param batch_size: Number of samples processed per cluster batch.

    :return: CSR sparse distance graph.
    """
    rng = np.random.default_rng(random_state)
    if n_samples <= n_neighbors:
        raise ValueError('n_samples must be greater than n_neighbors')

    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors, dtype=np.int64)
    indices = np.empty(n_samples * n_neighbors, dtype=np.int32)
    data = np.empty(n_samples * n_neighbors, dtype=np.float32)
    column_offsets = np.arange(n_neighbors, dtype=np.int64)
    distance_scale = np.float32(2.0 * cluster_std * cluster_std)

    for label in range(n_clusters):
        members = np.arange(label, n_samples, n_clusters, dtype=np.int64)
        if len(members) <= n_neighbors:
            raise ValueError(f'Cluster {label} has only {len(members)} members, need more than {n_neighbors}')

        for start in range(0, len(members), batch_size):
            batch = members[start : start + batch_size]
            self_pos = np.arange(start, start + len(batch), dtype=np.int64)
            sampled_pos = rng.integers(0, len(members) - 1, size=(len(batch), n_neighbors), dtype=np.int64)
            sampled_pos += sampled_pos >= self_pos[:, None]

            neighbors = members[sampled_pos]
            distances = distance_scale * rng.chisquare(df=n_features, size=(len(batch), n_neighbors)).astype(
                np.float32,
                copy=False,
            )

            flat_rows = (batch[:, None] * n_neighbors + column_offsets).ravel()
            indices[flat_rows] = neighbors.ravel()
            data[flat_rows] = distances.ravel()

    return csr_array((data, indices, indptr), shape=(n_samples, n_samples))


def build_quality_reference_subset(
    *,
    n_samples: int,
    n_features: int,
    n_clusters: int,
    cluster_std: float,
    subset_size: int,
    random_state: int,
) -> np.ndarray:
    """
    Generate a representative dense subset used for quality metrics on very large benchmarks.

    :param n_samples: Number of samples represented by the synthetic benchmark.
    :param n_features: Feature dimensionality.
    :param n_clusters: Number of synthetic clusters.
    :param cluster_std: Synthetic cluster standard deviation.
    :param subset_size: Number of dense samples to generate for metric evaluation.
    :param random_state: Random seed controlling subset generation.

    :return: Dense feature subset used for trustworthiness and k-NN overlap.
    """
    rng = np.random.default_rng(random_state)
    centers = rng.normal(scale=6.0, size=(n_clusters, n_features)).astype(np.float32)
    subset_size = min(subset_size, n_samples)
    subset_indices = np.arange(subset_size, dtype=np.int64)
    labels = subset_indices % n_clusters
    noise = rng.normal(scale=cluster_std, size=(subset_size, n_features)).astype(np.float32)
    return centers[labels] + noise


def load_mnist_training_data(data_root: Path, *, download: bool):
    """
    Load the MNIST training split as a flattened float32 matrix in ``[0, 1]``.

    :param data_root: Local dataset cache directory used by ``torchvision``.
    :param download: Whether missing dataset files may be downloaded.

    :return: Tuple ``(images, labels)`` with shapes ``(60000, 784)`` and ``(60000,)``.
    """
    try:
        from torchvision.datasets import MNIST
    except ImportError as exc:
        raise ImportError('MNIST benchmark requires torchvision to be installed.') from exc

    dataset = MNIST(root=str(data_root), train=True, download=download)
    images = dataset.data.numpy().reshape(len(dataset), -1).astype(np.float32) / 255.0
    labels = dataset.targets.numpy().astype(np.int64)
    return images, labels


def build_exact_topk_distance_graph(
    x: np.ndarray,
    *,
    n_neighbors: int,
    device: str,
    batch_size: int = 768,
):
    """
    Build an exact sparse top-k squared-distance graph with batched Torch matmul kernels.

    :param x: Dense feature matrix.
    :param n_neighbors: Number of nearest neighbors to keep per sample.
    :param device: Torch device name used for distance batches.
    :param batch_size: Query batch size for the pairwise-distance sweep.

    :return: CSR sparse squared-distance graph.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] <= n_neighbors:
        raise ValueError('n_neighbors must be smaller than the number of samples')

    torch_device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=torch_device)
    squared_norms = torch.sum(x_tensor * x_tensor, dim=1)

    n_samples = x.shape[0]
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors, dtype=np.int64)
    indices = np.empty(n_samples * n_neighbors, dtype=np.int32)
    distances = np.empty(n_samples * n_neighbors, dtype=np.float32)

    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        query = x_tensor[start:stop]
        dist = squared_norms[start:stop, None] + squared_norms[None, :] - 2.0 * (query @ x_tensor.T)
        dist.clamp_min_(0.0)
        local_rows = torch.arange(stop - start, device=torch_device)
        dist[local_rows, start + local_rows] = torch.inf
        values, neighbors = torch.topk(dist, k=n_neighbors, dim=1, largest=False, sorted=True)

        row_slice = slice(start * n_neighbors, stop * n_neighbors)
        indices[row_slice] = neighbors.cpu().numpy().astype(np.int32, copy=False).ravel()
        distances[row_slice] = values.cpu().numpy().astype(np.float32, copy=False).ravel()

        if torch_device.type == 'cuda':
            torch.cuda.synchronize(torch_device)
        if start == 0 or stop == n_samples or ((start // batch_size) + 1) % 10 == 0:
            LOGGER.info(
                'Built exact k-NN graph batch %s/%s on %s',
                (start // batch_size) + 1,
                int(np.ceil(n_samples / batch_size)),
                torch_device,
            )

    return csr_array((distances, indices, indptr), shape=(n_samples, n_samples))


def load_or_build_mnist_sparse_graph(*, random_state: int, dataset_build_device: str):
    """
    Load or build a cached PCA-reduced exact sparse graph for the MNIST training split.

    :param random_state: Random seed used for PCA.
    :param dataset_build_device: Requested device used while constructing the graph.

    :return: Tuple ``(raw_images, labels, sparse_graph)``.
    """
    data_root = Path('data') / 'torchvision'
    cache_dir = Path('data') / 'benchmarks'
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_path = cache_dir / (
        f'mnist_train_60000_knn{MNIST_GRAPH_NEIGHBORS}_pca{MNIST_PCA_COMPONENTS}_seed{random_state}.npz'
    )

    download = not graph_path.exists()
    try:
        raw_images, labels = load_mnist_training_data(data_root, download=download)
    except RuntimeError:
        raw_images, labels = load_mnist_training_data(data_root, download=True)

    if graph_path.exists():
        LOGGER.info('Loading cached MNIST sparse graph from %s', graph_path)
        return raw_images, labels, csr_array(sp.load_npz(graph_path))

    LOGGER.info('Building MNIST sparse graph from raw training images')
    pca = PCA(n_components=MNIST_PCA_COMPONENTS, svd_solver='randomized', random_state=random_state)
    reduced = pca.fit_transform(raw_images).astype(np.float32, copy=False)
    graph = build_exact_topk_distance_graph(
        reduced,
        n_neighbors=MNIST_GRAPH_NEIGHBORS,
        device=dataset_build_device,
    )
    sp.save_npz(graph_path, sp.csr_matrix(graph))
    LOGGER.info('Saved MNIST sparse graph cache to %s', graph_path)
    return raw_images, labels, graph


def build_datasets(
    random_state: int,
    selected: list[str] | None = None,
    *,
    dataset_build_device: str = 'cpu',
) -> dict[str, dict]:
    """
    Build deterministic benchmark dataset profiles.

    :param random_state: Random seed used for deterministic dataset generation.
    :param selected: Optional list of dataset names to build.
    :param dataset_build_device: Device hint used for expensive dataset preprocessing.

    :return: Mapping from dataset name to benchmark profile.
    """
    rng = np.random.RandomState(random_state)
    requested = None if selected is None else set(selected)
    datasets = {}

    def include(name: str) -> bool:
        return requested is None or name in requested

    def include_only_when_selected(name: str) -> bool:
        return requested is not None and name in requested

    for n_samples in (512, 2000, 5000, 10000):
        dataset_name = f'blobs_{n_samples}'
        if not include(dataset_name):
            continue
        x, _ = make_blobs(
            n_samples=n_samples,
            n_features=32,
            centers=20,
            cluster_std=1.3,
            random_state=rng,
        )
        x = x.astype(np.float32)
        datasets[dataset_name] = {
            'fit_input': x,
            'quality_reference': x,
            'metric': 'euclidean',
            'input_shape': list(x.shape),
            'max_iter': 300,
            'perplexity': 30.0,
            'scalable_only': False,
        }

    if include('blobs_2048_f512'):
        x_large, _ = make_blobs(
            n_samples=2048,
            n_features=512,
            centers=32,
            cluster_std=1.1,
            random_state=rng,
        )
        x_large = x_large.astype(np.float32)
        datasets['blobs_2048_f512'] = {
            'fit_input': x_large,
            'quality_reference': x_large,
            'metric': 'euclidean',
            'input_shape': list(x_large.shape),
            'max_iter': 300,
            'perplexity': 30.0,
            'scalable_only': False,
        }

    if include('blobs_100000_f512_graph'):
        x_huge, labels_huge = make_blobs(
            n_samples=100000,
            n_features=512,
            centers=64,
            cluster_std=1.1,
            random_state=rng,
        )
        x_huge = x_huge.astype(np.float32)
        graph_huge = build_cluster_sampled_distance_graph(
            x_huge,
            labels_huge,
            n_neighbors=48,
            random_state=random_state,
        )
        datasets['blobs_100000_f512_graph'] = {
            'fit_input': graph_huge,
            'quality_reference': x_huge,
            'metric': 'precomputed',
            'input_shape': list(x_huge.shape),
            'fit_input_shape': list(graph_huge.shape),
            'graph_nnz': int(graph_huge.nnz),
            'max_iter': 250,
            'perplexity': 15.0,
            'scalable_only': True,
        }

    for n_samples in SCALING_SWEEP_SAMPLES:
        dataset_name = f'blobs_{n_samples}_f512_sweep_graph'
        if not include(dataset_name):
            continue
        x_sweep, labels_sweep = make_blobs(
            n_samples=n_samples,
            n_features=512,
            centers=16,
            cluster_std=1.1,
            random_state=rng,
        )
        x_sweep = x_sweep.astype(np.float32)
        graph_sweep = build_cluster_sampled_distance_graph(
            x_sweep,
            labels_sweep,
            n_neighbors=48,
            random_state=random_state + n_samples,
        )
        datasets[dataset_name] = {
            'fit_input': graph_sweep,
            'quality_reference': x_sweep[: min(len(x_sweep), 2000)].copy(),
            'metric': 'precomputed',
            'input_shape': list(x_sweep.shape),
            'fit_input_shape': list(graph_sweep.shape),
            'graph_nnz': int(graph_sweep.nnz),
            'max_iter': 250,
            'perplexity': 15.0,
            'scalable_only': True,
            'sweep_group': SCALING_SWEEP_GROUP,
            'sample_count': n_samples,
        }

    if include('blobs_1000000_f512_graph'):
        graph_million = build_synthetic_cluster_graph(
            n_samples=1_000_000,
            n_features=512,
            n_clusters=128,
            cluster_std=1.1,
            n_neighbors=48,
            random_state=random_state,
        )
        quality_subset_million = build_quality_reference_subset(
            n_samples=1_000_000,
            n_features=512,
            n_clusters=128,
            cluster_std=1.1,
            subset_size=2000,
            random_state=random_state + 1,
        )
        datasets['blobs_1000000_f512_graph'] = {
            'fit_input': graph_million,
            'quality_reference': quality_subset_million,
            'metric': 'precomputed',
            'input_shape': [1_000_000, 512],
            'fit_input_shape': list(graph_million.shape),
            'graph_nnz': int(graph_million.nnz),
            'max_iter': 250,
            'perplexity': 15.0,
            'scalable_only': True,
            'disable_sklearn_baselines': True,
        }

    if include('digits'):
        digits = load_digits()
        digits_x = digits.data.astype(np.float32)
        datasets['digits'] = {
            'fit_input': digits_x,
            'quality_reference': digits_x,
            'plot_labels': digits.target.astype(np.int64),
            'metric': 'euclidean',
            'input_shape': list(digits_x.shape),
            'max_iter': 300,
            'perplexity': 30.0,
            'scalable_only': False,
        }

    if include_only_when_selected(MNIST_BENCHMARK_DATASET):
        mnist_x, mnist_labels, mnist_graph = load_or_build_mnist_sparse_graph(
            random_state=random_state,
            dataset_build_device=dataset_build_device,
        )
        datasets[MNIST_BENCHMARK_DATASET] = {
            'fit_input': mnist_graph,
            'quality_reference': mnist_x,
            'plot_labels': mnist_labels,
            'metric': 'precomputed',
            'input_shape': list(mnist_x.shape),
            'fit_input_shape': list(mnist_graph.shape),
            'graph_nnz': int(mnist_graph.nnz),
            'max_iter': 250,
            'perplexity': 15.0,
            'scalable_only': True,
            'display_name': 'MNIST Train 60k',
            'benchmark_note': f'PCA({MNIST_PCA_COMPONENTS}) exact {MNIST_GRAPH_NEIGHBORS}-NN sparse graph',
        }
    return datasets


def run_model(model, fit_input, quality_reference):
    """
    Fit one benchmark baseline and collect timing, quality, and memory diagnostics.

    :param model: sklearn TSNE or TorchTSNE instance.
    :param fit_input: Input passed to ``fit_transform``.
    :param quality_reference: Reference features used for trustworthiness and k-NN overlap.

    :return: Tuple containing embedding, duration, diagnostics, KL value, trustworthiness, overlap, and memory info.
    """
    start = perf_counter()
    embedding = model.fit_transform(fit_input)
    duration = perf_counter() - start

    if isinstance(model, TSNE):
        diagnostics = {'backend': 'sklearn', 'timings': {'total': duration}, 'peak_cuda_memory': 0}
        kl_divergence = float(model.kl_divergence_)
    else:
        diagnostics = {
            'backend': model.diagnostics_.backend,
            'timings': model.diagnostics_.timings,
            'peak_cuda_memory': model.diagnostics_.timings.get('peak_cuda_memory', 0),
        }
        kl_divergence = float(model.kl_divergence_)

    subset = min(len(quality_reference), 2000)
    trust = trustworthiness(quality_reference[:subset], embedding[:subset], n_neighbors=10)
    overlap = knn_overlap(quality_reference[:subset], embedding[:subset], n_neighbors=10)
    memory = {} if isinstance(model, TSNE) else model.diagnostics_.memory
    return embedding, duration, diagnostics, kl_divergence, trust, overlap, memory


def warm_up_cuda_runtime():
    """
    Warm up CUDA kernels commonly exercised by the tsne-torch backends.

    :return: None.
    """
    if not torch.cuda.is_available():
        return

    device = torch.device('cuda')
    x = torch.randn(256, 128, device=device)
    _ = x @ x.T
    _ = torch.cdist(x[:64], x[:64]).pow(2)
    grid = torch.randn(128, 128, device=device)
    kernel = torch.randn(128, 128, device=device)
    _ = torch.fft.irfftn(torch.fft.rfftn(grid) * torch.fft.rfftn(kernel), s=grid.shape)
    torch.cuda.synchronize(device)


def build_baselines(device: str, profile: dict):
    """
    Create benchmark model factories for one dataset profile.

    :param device: Requested benchmark device mode.
    :param profile: Dataset profile controlling method availability and iteration counts.

    :return: Sequence of ``(label, model_factory)`` pairs.
    """
    common = {
        'init': 'random',
        'learning_rate': 'auto',
        'random_state': 0,
        'max_iter': profile['max_iter'],
        'metric': profile['metric'],
        'perplexity': profile['perplexity'],
        'n_jobs': 1,
    }
    if profile['scalable_only']:
        baselines = [('tsne_torch_fft_cpu', lambda: TorchTSNE(method='fft', device='cpu', **common))]
        if not profile.get('disable_sklearn_baselines'):
            baselines.insert(0, ('sklearn_barnes_hut', lambda: TSNE(method='barnes_hut', **common)))
        if device == 'cuda' and torch.cuda.is_available():
            baselines.append(('tsne_torch_fft_cuda', lambda: TorchTSNE(method='fft', device='cuda', **common)))
        return baselines

    baselines = [
        ('sklearn_exact', lambda: TSNE(method='exact', **common)),
        ('sklearn_barnes_hut', lambda: TSNE(method='barnes_hut', **common)),
        ('tsne_torch_exact_cpu', lambda: TorchTSNE(method='exact', device='cpu', **common)),
    ]
    if device == 'cuda' and torch.cuda.is_available():
        baselines.extend(
            [
                ('tsne_torch_exact_cuda', lambda: TorchTSNE(method='exact', device='cuda', **common)),
                ('tsne_torch_fft_cuda', lambda: TorchTSNE(method='fft', device='cuda', **common)),
            ]
        )
    return baselines


def benchmark_dataset(name, profile: dict, repeats: int, device: str):
    """
    Benchmark all supported baselines for a single dataset profile.

    :param name: Dataset identifier.
    :param profile: Dataset profile metadata and inputs.
    :param repeats: Number of repeated runs per baseline.
    :param device: Requested benchmark device mode.

    :return: Tuple ``(rows, embeddings_by_baseline)``.
    """
    fit_input = profile['fit_input']
    quality_reference = profile['quality_reference']
    baselines = build_baselines(device, profile)
    dataset_results = []
    embeddings_by_baseline = {}
    for label, model_factory in baselines:
        durations = []
        metrics = None
        if device == 'cuda' and torch.cuda.is_available() and label.endswith('_cuda'):
            if profile['metric'] == 'precomputed' and sp.issparse(fit_input):
                LOGGER.info('Skipping per-model warm-up for %s on sparse precomputed input', label)
            else:
                warm_model = model_factory()
                if profile['metric'] == 'precomputed':
                    warm_input = fit_input[:256, :256]
                else:
                    warm_input = fit_input[: min(len(fit_input), 256)]
                warm_model.fit_transform(warm_input)
                torch.cuda.synchronize()
        for _ in range(repeats):
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            model = model_factory()
            embedding, duration, diagnostics, kl_divergence, trust, overlap, memory = run_model(
                model,
                fit_input,
                quality_reference,
            )
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            durations.append(duration)
            metrics = {
                'embedding_shape': list(embedding.shape),
                'diagnostics': diagnostics,
                'memory': memory,
                'kl_divergence': kl_divergence,
                'trustworthiness': trust,
                'knn_overlap': overlap,
            }
            embeddings_by_baseline[label] = embedding.astype(np.float32, copy=False)
        dataset_results.append(
            {
                'dataset': name,
                'input_shape': profile['input_shape'],
                'fit_input_shape': profile.get('fit_input_shape', profile['input_shape']),
                'metric': profile['metric'],
                'graph_nnz': profile.get('graph_nnz'),
                'sample_count': profile.get('sample_count'),
                'sweep_group': profile.get('sweep_group'),
                'baseline': label,
                'durations': durations,
                'median_duration': median(durations),
                **metrics,
            }
        )
    return dataset_results, embeddings_by_baseline


def main(argv: list[str] | None = None):
    """
    Run the TorchTSNE benchmark CLI entrypoint.

    :param argv: Optional argument list used instead of ``sys.argv``.

    :return: None.
    """
    configure_logging()
    parser = argparse.ArgumentParser(description='Benchmark TorchTSNE against sklearn TSNE.')
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/benchmark_results.json'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=None,
        help='Optional dataset names to benchmark. Defaults to the full benchmark matrix.',
    )
    args = parser.parse_args(argv)

    datasets = build_datasets(args.seed, selected=args.datasets, dataset_build_device=args.device)
    results = {
        'library_versions': library_versions(),
        'device': args.device,
        'datasets': [],
        'charts': [],
        'sweeps': [],
    }
    LOGGER.info('Running TorchTSNE benchmark with libraries=%s', results['library_versions'])
    LOGGER.info('Requested datasets: %s', list(datasets.keys()))

    if args.device == 'cuda' and torch.cuda.is_available():
        warm_up_cuda_runtime()
        LOGGER.info('CUDA warm-up completed on device=%s', torch.cuda.get_device_name(0))

    for name, profile in datasets.items():
        dataset_rows, dataset_embeddings = benchmark_dataset(name, profile, args.repeats, args.device)
        results['datasets'].extend(dataset_rows)
        summarize_dataset_results(name, profile, dataset_rows)
        chart_path = _chart_output_path(args.output, name, single_dataset=len(datasets) == 1)
        saved_chart = save_dataset_chart(name, profile, dataset_rows, chart_path)
        if saved_chart is not None:
            results['charts'].append(str(saved_chart))
        embedding_chart_path = _embedding_chart_output_path(args.output, name, single_dataset=len(datasets) == 1)
        saved_embedding_chart = save_embedding_comparison_chart(
            name,
            profile,
            dataset_rows,
            dataset_embeddings,
            embedding_chart_path,
        )
        if saved_embedding_chart is not None:
            results['charts'].append(str(saved_embedding_chart))

    scaling_sweeps = collect_scaling_sweeps(results['datasets'])
    for sweep_name, sweep_rows in scaling_sweeps.items():
        summary_rows = build_scaling_sweep_summary(sweep_name, sweep_rows)
        analysis = summarize_scaling_sweep(sweep_name, summary_rows)
        chart_path = args.output.with_name(f'{args.output.stem}_{sweep_name}.png')
        saved_chart = save_scaling_sweep_chart(sweep_name, summary_rows, chart_path, analysis=analysis)
        results['sweeps'].append(
            {
                'name': sweep_name,
                'chart': str(saved_chart) if saved_chart is not None else None,
                'rows': summary_rows,
                'analysis': analysis,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)
    LOGGER.info('Saved benchmark results to %s', args.output)


if __name__ == '__main__':
    main()
