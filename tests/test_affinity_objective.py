import torch
from numpy.testing import assert_allclose
from scipy.spatial.distance import squareform
from sklearn.manifold._t_sne import _joint_probabilities, _kl_divergence
from sklearn.metrics import pairwise_distances

from _helpers import make_blobs_data
from tsne_torch import exact_kl_divergence_objective, joint_probabilities_from_squared_distances


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
