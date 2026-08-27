"""Exact spectral/Brenier graph-transport tests."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from conga.spectral import GMRFSparseCholeskyBackend, GMRFSpectralBackend


def _adjacency(n: int, edges) -> sparse.csc_matrix:
    edges = np.asarray(list(edges), dtype=np.int64)
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    return sparse.coo_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(n, n)
    ).tocsc()


def _path_precision(n: int, kappa: float) -> sparse.csc_matrix:
    adjacency = _adjacency(n, ((i, i + 1) for i in range(n - 1)))
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return sparse.diags(degree + kappa, format="csc") - adjacency


def _lattice_precision(side: int, kappa: float) -> sparse.csc_matrix:
    edges = []
    for i in range(side):
        for j in range(side):
            node = i * side + j
            if i + 1 < side:
                edges.append((node, node + side))
            if j + 1 < side:
                edges.append((node, node + 1))
    adjacency = _adjacency(side * side, edges)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return sparse.diags(degree + kappa, format="csc") - adjacency


def _check_fast_backend(base_precision, fast, eta):
    rng = np.random.default_rng(13)
    n = base_precision.shape[0]
    dense = GMRFSpectralBackend.dense(base_precision, fast.y)
    sparse_backend = GMRFSparseCholeskyBackend(base_precision, fast.y)

    probe = rng.normal(size=(3, n))
    diagonalization_error = np.max(np.abs(
        fast.analysis(probe @ base_precision)
        - fast.analysis(probe) * fast.eigenvalues
    ))
    assert diagonalization_error < 1e-11

    fast_value, fast_gradient = fast.collapsed_vg(eta)
    dense_value, dense_gradient = dense.collapsed_vg(eta)
    sparse_value, sparse_gradient = sparse_backend.collapsed_vg(eta)
    np.testing.assert_allclose(fast_value, dense_value, atol=1e-10, rtol=1e-11)
    np.testing.assert_allclose(fast_gradient, dense_gradient, atol=1e-10, rtol=1e-11)
    np.testing.assert_allclose(fast_value, sparse_value, atol=1e-10, rtol=1e-11)
    np.testing.assert_allclose(fast_gradient, sparse_gradient, atol=1e-9, rtol=1e-10)

    step = 1e-6
    finite_difference = np.empty(2)
    for coordinate in range(2):
        direction = np.zeros(2)
        direction[coordinate] = step
        right = fast.collapsed_vg(eta + direction)[0]
        left = fast.collapsed_vg(eta - direction)[0]
        finite_difference[coordinate] = (right - left) / (2.0 * step)
    np.testing.assert_allclose(fast_gradient, finite_difference, atol=2e-7, rtol=2e-7)

    fast_evaluation = fast.evaluate(eta)
    dense_evaluation = dense.evaluate(eta)
    sparse_evaluation = sparse_backend.evaluate(eta)
    x = rng.normal(size=(4, n))
    fast_z = fast.forward(x, fast_evaluation)
    dense_z = dense.forward(x, dense_evaluation)
    np.testing.assert_allclose(fast_z, dense_z, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(
        fast.inverse(fast_z, fast_evaluation), x, atol=2e-11, rtol=2e-11
    )

    centered = x - fast_evaluation.mean
    precision = (
        np.exp(-2.0 * eta[0]) * base_precision
        + np.exp(-2.0 * eta[1]) * sparse.eye(n, format="csc")
    )
    quadratic = np.sum(centered * (centered @ precision), axis=1)
    np.testing.assert_allclose(np.square(fast_z).sum(axis=1), quadratic, atol=2e-9)

    brenier_cost = fast.centered_quadratic_cost(eta)
    triangular_cost = sparse_backend.centered_quadratic_cost(sparse_evaluation)
    assert brenier_cost <= triangular_cost + 1e-12


def test_path_dct_is_exact_brenier_backend():
    n, kappa = 18, 0.2
    precision = _path_precision(n, kappa)
    y = np.random.default_rng(2).normal(size=n)
    fast = GMRFSpectralBackend.path(n, kappa, y)
    _check_fast_backend(precision, fast, np.log([0.35, 0.2]))


def test_lattice_dct_is_exact_brenier_backend():
    side, kappa = 5, 0.2
    precision = _lattice_precision(side, kappa)
    y = np.random.default_rng(4).normal(size=side * side)
    fast = GMRFSpectralBackend.lattice(side, kappa, y)
    _check_fast_backend(precision, fast, np.log([0.35, 0.2]))


def test_fast_backend_batches_core_dependent_reconstruction():
    n, kappa = 18, 0.2
    rng = np.random.default_rng(19)
    fast = GMRFSpectralBackend.path(n, kappa, rng.normal(size=n))
    etas = np.log(np.array([[0.35, 0.20], [0.50, 0.15], [0.25, 0.40]]))
    noise = rng.normal(size=(etas.shape[0], n))

    batched_value, batched_gradient = fast.collapsed_vg(etas)
    scalar_vg = [fast.collapsed_vg(eta) for eta in etas]
    np.testing.assert_allclose(
        batched_value, [value for value, _ in scalar_vg], atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        batched_gradient, [gradient for _, gradient in scalar_vg],
        atol=0.0, rtol=0.0,
    )

    batched = fast.evaluate(etas)
    reconstructed = fast.inverse(noise, batched)
    recovered = fast.forward(reconstructed, batched)
    np.testing.assert_allclose(recovered, noise, atol=2e-11, rtol=2e-11)

    for index, eta in enumerate(etas):
        scalar = fast.evaluate(eta)
        np.testing.assert_allclose(
            reconstructed[index], fast.inverse(noise[index], scalar),
            atol=2e-11, rtol=2e-11,
        )
        np.testing.assert_allclose(
            batched.precision_eigenvalues[index], scalar.precision_eigenvalues,
            atol=0.0, rtol=0.0,
        )
