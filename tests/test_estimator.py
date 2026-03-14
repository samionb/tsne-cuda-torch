import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances

from _helpers import make_blobs_data
from tsne_torch import TorchTSNE
from tsne_torch.diagnostics import knn_overlap


def test_validation_errors_and_auto_learning_rate():
    x = make_blobs_data(n_samples=10, n_features=3, centers=3, random_state=3)

    with pytest.raises(ValueError, match='perplexity'):
        TorchTSNE(perplexity=10).fit_transform(x)

    distances = pairwise_distances(x, squared=True)
    with pytest.raises(ValueError, match='init="pca"'):
        TorchTSNE(metric='precomputed', init='pca', perplexity=3).fit_transform(distances)

    with pytest.raises(ValueError, match='inferior to 4'):
        TorchTSNE(method='barnes_hut', n_components=4, perplexity=3).fit_transform(x)

    with pytest.raises(ValueError, match='square distance matrix'):
        TorchTSNE(metric='precomputed', init='random', perplexity=3).fit_transform(distances[:, :4])

    with pytest.raises(ValueError, match='positive distances'):
        bad_distances = distances.copy()
        bad_distances[0, 1] = -1.0
        TorchTSNE(metric='precomputed', init='random', perplexity=3).fit_transform(bad_distances)

    model = TorchTSNE(learning_rate='auto', perplexity=3, init='random', max_iter=300, random_state=0, device='cpu')
    model.fit_transform(x)
    assert model.learning_rate_ == pytest.approx(max(x.shape[0] / model.early_exaggeration / 4.0, 50.0))


def test_exact_backend_is_deterministic_and_tracks_iterations():
    x = make_blobs_data(n_samples=35, n_features=4, centers=4, random_state=4)
    model_a = TorchTSNE(method='exact', init='random', learning_rate='auto', perplexity=8, max_iter=300, random_state=0, device='cpu')
    model_b = TorchTSNE(method='exact', init='random', learning_rate='auto', perplexity=8, max_iter=300, random_state=0, device='cpu')

    embedding_a = model_a.fit_transform(x)
    embedding_b = model_b.fit_transform(x)

    assert_allclose(embedding_a, embedding_b, atol=1e-6, rtol=1e-6)
    assert model_a.n_iter_ == 299
    assert not np.isnan(model_a.kl_divergence_)


def test_exact_backend_quality_matches_sklearn():
    x = make_blobs_data(n_samples=45, n_features=5, centers=5, random_state=5)
    kwargs = {
        'init': 'random',
        'perplexity': 10,
        'learning_rate': 'auto',
        'max_iter': 350,
        'random_state': 0,
        'method': 'exact',
    }
    reference_model = TSNE(**kwargs)
    reference = reference_model.fit_transform(x)
    candidate_model = TorchTSNE(**kwargs, device='cpu')
    candidate = candidate_model.fit_transform(x)

    reference_trust = trustworthiness(x, reference, n_neighbors=5)
    candidate_trust = trustworthiness(x, candidate, n_neighbors=5)
    overlap = knn_overlap(reference, candidate, n_neighbors=10)
    kl_relative_error = abs(candidate_model.kl_divergence_ - reference_model.kl_divergence_) / max(
        abs(reference_model.kl_divergence_),
        1e-8,
    )

    assert abs(candidate_trust - reference_trust) <= 0.02
    assert overlap >= 0.85
    assert kl_relative_error <= 0.05


def test_fft_backend_quality_matches_exact():
    x = make_blobs_data(n_samples=60, n_features=5, centers=6, random_state=6)
    exact_model = TorchTSNE(
        method='exact',
        init='random',
        perplexity=10,
        learning_rate='auto',
        max_iter=300,
        random_state=0,
        n_jobs=1,
        device='cpu',
    )
    fft_model = TorchTSNE(
        method='fft',
        init='random',
        perplexity=10,
        learning_rate='auto',
        max_iter=300,
        random_state=0,
        n_jobs=1,
        device='cpu',
    )

    exact_embedding = exact_model.fit_transform(x)
    fft_embedding = fft_model.fit_transform(x)

    exact_trust = trustworthiness(x, exact_embedding, n_neighbors=5)
    fft_trust = trustworthiness(x, fft_embedding, n_neighbors=5)
    overlap = knn_overlap(exact_embedding, fft_embedding, n_neighbors=10)
    kl_relative_error = abs(fft_model.kl_divergence_ - exact_model.kl_divergence_) / max(
        abs(exact_model.kl_divergence_),
        1e-8,
    )

    assert abs(fft_trust - exact_trust) <= 0.03
    assert overlap >= 0.80
    assert kl_relative_error <= 0.30


def test_large_min_grad_norm_stops_early():
    x = make_blobs_data(n_samples=30, n_features=4, centers=4, random_state=7)
    model = TorchTSNE(
        method='exact',
        init='random',
        perplexity=8,
        learning_rate='auto',
        max_iter=300,
        random_state=0,
        min_grad_norm=10.0,
        device='cpu',
    )
    model.fit_transform(x)
    assert model.n_iter_ < 299
    assert model.diagnostics_.timings['stage1']['stopped_reason'] == 'grad_norm'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
def test_cuda_execution_path():
    x = make_blobs_data(n_samples=30, n_features=4, centers=4, random_state=8)
    model = TorchTSNE(
        method='exact',
        init='random',
        perplexity=8,
        learning_rate='auto',
        max_iter=300,
        random_state=0,
        device='cuda',
    )
    model.fit_transform(x)
    assert model.diagnostics_.device.startswith('cuda')
    assert model.diagnostics_.timings['peak_cuda_memory'] > 0
