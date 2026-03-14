import pytest

from _helpers import make_blobs_data, make_sparse_distance_graph
from tsne_torch import TorchTSNE
from tsne_torch import memory as memory_utils


def test_memory_diagnostics_are_reported_for_exact_backend():
    x = make_blobs_data(n_samples=35, n_features=4, centers=4, random_state=40)
    model = TorchTSNE(
        method='exact',
        init='random',
        learning_rate='auto',
        perplexity=8,
        max_iter=300,
        random_state=0,
        device='cpu',
    )

    model.fit_transform(x)

    memory = model.diagnostics_.memory
    assert memory['backend'] == 'torch_exact'
    assert memory['required_bytes'] > 0
    assert 'distance_tensor_bytes' in memory['details']
    assert memory['fits'] in {True, None}


def test_exact_backend_raises_memory_error_when_budget_is_too_small(monkeypatch):
    x = make_blobs_data(n_samples=64, n_features=8, centers=4, random_state=41)
    monkeypatch.setattr(memory_utils, 'available_memory_bytes', lambda device: 1024)

    model = TorchTSNE(
        method='exact',
        init='random',
        perplexity=8,
        learning_rate='auto',
        max_iter=300,
        random_state=0,
        device='cpu',
    )

    with pytest.raises(MemoryError, match='Estimated torch_exact memory requirement'):
        model.fit_transform(x)


def test_fft_sparse_backend_raises_memory_error_when_budget_is_too_small(monkeypatch):
    x = make_blobs_data(n_samples=300, n_features=4, centers=6, random_state=42)
    graph = make_sparse_distance_graph(x, n_neighbors=12)
    monkeypatch.setattr(memory_utils, 'available_memory_bytes', lambda device: 1024)

    model = TorchTSNE(
        method='fft',
        metric='precomputed',
        init='random',
        perplexity=3,
        learning_rate='auto',
        max_iter=300,
        random_state=0,
        device='cpu',
    )

    with pytest.raises(MemoryError, match='Estimated torch_fft memory requirement'):
        model.fit_transform(graph)
