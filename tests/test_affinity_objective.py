import torch
from numpy.testing import assert_allclose
from scipy.spatial.distance import squareform
from sklearn.manifold._t_sne import _joint_probabilities, _kl_divergence
from sklearn.metrics import pairwise_distances

from _helpers import make_blobs_data
from tsne_torch import exact_kl_divergence_objective, joint_probabilities_from_squared_distances
from tsne_torch.affinity import build_sparse_affinity_from_knn, build_sparse_affinity_from_precomputed
from tsne_torch.fft_backend import _splat_points_to_grid


def test_joint_probabilities_match_sklearn():
    x = make_blobs_data(n_samples=20, n_features=3, centers=3, random_state=1)
    distances = pairwise_distances(x, squared=True).astype('float32')
    torch_p = joint_probabilities_from_squared_distances(
        torch.tensor(distances, dtype=torch.float32),
        perplexity=5.0,
        verbose=0,
    ).cpu().numpy()
    sklearn_p = squareform(_joint_probabilities(distances, desired_perplexity=5.0, verbose=0))
    assert_allclose(torch_p, sklearn_p, atol=1e-5, rtol=1e-5)


def test_exact_gradient_matches_sklearn():
    x = make_blobs_data(n_samples=15, n_features=3, centers=3, random_state=2)
    distances = pairwise_distances(x, squared=True).astype('float32')
    torch_p = joint_probabilities_from_squared_distances(
        torch.tensor(distances, dtype=torch.float32),
        perplexity=4.0,
        verbose=0,
    )
    params = torch.randn((x.shape[0], 2), dtype=torch.float32)
    torch_kl, torch_grad = exact_kl_divergence_objective(params, torch_p, 1.0)

    sklearn_p = _joint_probabilities(distances, desired_perplexity=4.0, verbose=0)
    sklearn_kl, sklearn_grad = _kl_divergence(params.numpy().ravel(), sklearn_p, 1.0, x.shape[0], 2)

    assert_allclose(torch_kl, sklearn_kl, rtol=1e-5, atol=1e-5)
    assert_allclose(torch_grad.cpu().numpy().ravel(), sklearn_grad, rtol=1e-5, atol=1e-5)


def test_sparse_affinity_from_knn_squares_non_euclidean_distances():
    x = make_blobs_data(n_samples=18, n_features=3, centers=3, random_state=9)
    device = torch.device('cpu')
    knn_affinity = build_sparse_affinity_from_knn(
        x,
        perplexity=4.0,
        metric='manhattan',
        metric_params=None,
        n_jobs=1,
        device=device,
    )
    dense_distances = pairwise_distances(x, metric='manhattan')
    reference_affinity = build_sparse_affinity_from_precomputed(
        dense_distances**2,
        perplexity=4.0,
        device=device,
    )
    assert_allclose(knn_affinity.matrix.toarray(), reference_affinity.matrix.toarray(), atol=1e-6, rtol=1e-6)


def test_splat_points_to_grid_matches_four_corner_reference():
    coords = torch.tensor(
        [
            [0.2, 0.3],
            [1.75, 2.25],
            [3.95, 3.95],
            [-0.5, 1.25],
        ],
        dtype=torch.float32,
    )
    grid_size = 5
    actual = _splat_points_to_grid(coords, grid_size)

    x = coords[:, 0].clamp(0.0, grid_size - 1.001)
    y = coords[:, 1].clamp(0.0, grid_size - 1.001)
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = (x0 + 1).clamp(max=grid_size - 1)
    y1 = (y0 + 1).clamp(max=grid_size - 1)
    wx = x - x0.float()
    wy = y - y0.float()

    reference = torch.zeros((grid_size, grid_size), dtype=torch.float32)
    weights = (
        ((1.0 - wx) * (1.0 - wy), x0, y0),
        (wx * (1.0 - wy), x1, y0),
        ((1.0 - wx) * wy, x0, y1),
        (wx * wy, x1, y1),
    )
    for weight, xi, yi in weights:
        for sample_idx in range(coords.shape[0]):
            reference[yi[sample_idx], xi[sample_idx]] += weight[sample_idx]

    assert_allclose(actual.cpu().numpy(), reference.cpu().numpy(), atol=1e-7, rtol=1e-7)
