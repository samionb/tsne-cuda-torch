from pathlib import Path
from time import sleep
import uuid

import pytest

import numpy as np
from scipy.sparse import csr_array

from tsne_torch.benchmarking import (
    analyze_scaling_sweep,
    build_datasets,
    save_dataset_chart,
    save_embedding_comparison_chart,
    save_scaling_sweep_chart,
)


def test_benchmark_chart_is_written():
    pytest.importorskip('matplotlib')

    rows = [
        {
            'baseline': 'sklearn_exact',
            'median_duration': 10.0,
            'trustworthiness': 0.98,
            'knn_overlap': 0.31,
            'diagnostics': {'peak_cuda_memory': 0},
            'memory': {},
        },
        {
            'baseline': 'tsne_torch_exact_cuda',
            'median_duration': 0.5,
            'trustworthiness': 0.99,
            'knn_overlap': 0.34,
            'diagnostics': {'peak_cuda_memory': 128 * 1024 * 1024},
            'memory': {'required_bytes': 96 * 1024 * 1024},
        },
        {
            'baseline': 'tsne_torch_fft_cuda',
            'median_duration': 0.9,
            'trustworthiness': 0.985,
            'knn_overlap': 0.33,
            'diagnostics': {'peak_cuda_memory': 48 * 1024 * 1024},
            'memory': {'required_bytes': 32 * 1024 * 1024},
        },
    ]
    profile = {
        'input_shape': [2048, 512],
        'fit_input_shape': [2048, 512],
        'metric': 'euclidean',
    }

    output_path = Path(__file__).resolve().parent / f'_tmp_chart_{uuid.uuid4().hex}.png'
    try:
        saved_path = save_dataset_chart('blobs_2048_f512', profile, rows, output_path)

        assert saved_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        for _ in range(5):
            try:
                output_path.unlink(missing_ok=True)
                break
            except PermissionError:
                sleep(0.1)


def test_scaling_sweep_chart_is_written():
    pytest.importorskip('matplotlib')

    summary_rows = [
        {
            'sample_count': 1000,
            'graph_nnz': 48000,
            'sklearn_barnes_hut_duration': 0.8,
            'tsne_torch_fft_cpu_duration': 0.2,
            'tsne_torch_fft_cuda_duration': 0.05,
            'tsne_torch_fft_cpu_speedup_vs_sklearn': 4.0,
            'tsne_torch_fft_cuda_speedup_vs_sklearn': 16.0,
        },
        {
            'sample_count': 5000,
            'graph_nnz': 240000,
            'sklearn_barnes_hut_duration': 6.0,
            'tsne_torch_fft_cpu_duration': 1.1,
            'tsne_torch_fft_cuda_duration': 0.2,
            'tsne_torch_fft_cpu_speedup_vs_sklearn': 5.45,
            'tsne_torch_fft_cuda_speedup_vs_sklearn': 30.0,
        },
    ]

    output_path = Path(__file__).resolve().parent / f'_tmp_sweep_chart_{uuid.uuid4().hex}.png'
    try:
        saved_path = save_scaling_sweep_chart('scaling_sweep_sparse_graph_f512', summary_rows, output_path)

        assert saved_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        for _ in range(5):
            try:
                output_path.unlink(missing_ok=True)
                break
            except PermissionError:
                sleep(0.1)


def test_scaling_sweep_analysis_reports_power_law_fits():
    summary_rows = [
        {
            'sample_count': 5000,
            'graph_nnz': 240000,
            'sklearn_barnes_hut_duration': 4.0,
            'tsne_torch_fft_cpu_duration': 2.0,
            'tsne_torch_fft_cuda_duration': 0.8,
            'tsne_torch_fft_cpu_speedup_vs_sklearn': 2.0,
            'tsne_torch_fft_cuda_speedup_vs_sklearn': 5.0,
        },
        {
            'sample_count': 10000,
            'graph_nnz': 480000,
            'sklearn_barnes_hut_duration': 10.0,
            'tsne_torch_fft_cpu_duration': 3.0,
            'tsne_torch_fft_cuda_duration': 1.0,
            'tsne_torch_fft_cpu_speedup_vs_sklearn': 3.3333333333,
            'tsne_torch_fft_cuda_speedup_vs_sklearn': 10.0,
        },
        {
            'sample_count': 50000,
            'graph_nnz': 2400000,
            'sklearn_barnes_hut_duration': 80.0,
            'tsne_torch_fft_cpu_duration': 15.0,
            'tsne_torch_fft_cuda_duration': 2.0,
            'tsne_torch_fft_cpu_speedup_vs_sklearn': 5.3333333333,
            'tsne_torch_fft_cuda_speedup_vs_sklearn': 40.0,
        },
    ]

    analysis = analyze_scaling_sweep(summary_rows)

    assert analysis['fit_min_sample_count'] == 5000
    assert analysis['runtime_power_law']['sklearn_barnes_hut']['exponent'] > 0.0
    assert analysis['runtime_power_law']['tsne_torch_fft_cuda']['exponent'] > 0.0
    assert analysis['speedup_power_law']['tsne_torch_fft_cuda']['exponent'] > 0.0


def test_embedding_comparison_chart_is_written():
    pytest.importorskip('matplotlib')

    rows = [
        {
            'baseline': 'sklearn_barnes_hut',
            'median_duration': 8.0,
            'trustworthiness': 0.93,
        },
        {
            'baseline': 'tsne_torch_fft_cuda',
            'median_duration': 0.8,
            'trustworthiness': 0.94,
        },
    ]
    profile = {
        'display_name': 'MNIST Train 60k',
        'plot_labels': np.asarray([0, 1, 0, 1, 2, 2]),
    }
    embeddings = {
        'sklearn_barnes_hut': np.asarray(
            [[0.0, 0.1], [1.0, 0.9], [0.2, -0.1], [1.1, 1.0], [-0.7, 0.8], [-0.6, 0.7]],
            dtype=np.float32,
        ),
        'tsne_torch_fft_cuda': np.asarray(
            [[0.1, 0.2], [0.9, 1.0], [0.0, -0.2], [1.2, 0.8], [-0.8, 0.7], [-0.5, 0.9]],
            dtype=np.float32,
        ),
    }

    output_path = Path(__file__).resolve().parent / f'_tmp_embedding_chart_{uuid.uuid4().hex}.png'
    try:
        saved_path = save_embedding_comparison_chart(
            'mnist_train_60000_graph',
            profile,
            rows,
            embeddings,
            output_path,
        )

        assert saved_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        for _ in range(5):
            try:
                output_path.unlink(missing_ok=True)
                break
            except PermissionError:
                sleep(0.1)


def test_embedding_comparison_chart_supports_many_classes_without_full_legend():
    pytest.importorskip('matplotlib')

    rows = [
        {
            'baseline': 'sklearn_barnes_hut',
            'median_duration': 8.0,
            'trustworthiness': 0.62,
        },
        {
            'baseline': 'tsne_torch_fft_cuda',
            'median_duration': 0.9,
            'trustworthiness': 0.64,
        },
    ]
    labels = np.arange(30, dtype=np.int64)
    profile = {
        'display_name': 'CIFAR-100 Train 50k',
        'plot_label_title': 'Fine Label',
        'plot_labels': labels,
    }
    embeddings = {
        'sklearn_barnes_hut': np.stack([labels.astype(np.float32), np.sin(labels)], axis=1),
        'tsne_torch_fft_cuda': np.stack([labels.astype(np.float32), np.cos(labels)], axis=1),
    }

    output_path = Path(__file__).resolve().parent / f'_tmp_embedding_chart_many_{uuid.uuid4().hex}.png'
    try:
        saved_path = save_embedding_comparison_chart(
            'cifar100_train_50000_graph',
            profile,
            rows,
            embeddings,
            output_path,
        )

        assert saved_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        for _ in range(5):
            try:
                output_path.unlink(missing_ok=True)
                break
            except PermissionError:
                sleep(0.1)


def test_build_datasets_includes_selected_cifar_profiles(monkeypatch):
    fake_graph = csr_array(
        (
            np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            np.asarray([1, 0, 1, 0], dtype=np.int32),
            np.asarray([0, 2, 4], dtype=np.int64),
        ),
        shape=(2, 2),
    )
    fake_images = np.zeros((2, 3072), dtype=np.float32)
    fake_labels = np.asarray([0, 1], dtype=np.int64)

    monkeypatch.setattr(
        'tsne_torch.benchmarking.load_or_build_cifar10_sparse_graph',
        lambda **kwargs: (fake_images, fake_labels, fake_graph),
    )
    monkeypatch.setattr(
        'tsne_torch.benchmarking.load_or_build_cifar100_sparse_graph',
        lambda **kwargs: (fake_images, fake_labels, fake_graph),
    )

    datasets = build_datasets(
        0,
        selected=['cifar10_train_50000_graph', 'cifar100_train_50000_graph'],
        dataset_build_device='cpu',
    )

    assert 'cifar10_train_50000_graph' in datasets
    assert 'cifar100_train_50000_graph' in datasets
    assert datasets['cifar10_train_50000_graph']['display_name'] == 'CIFAR-10 Train 50k'
    assert datasets['cifar100_train_50000_graph']['plot_label_title'] == 'Fine Label'
