"""General-graph sparse precision and Gaussian-block execution tests."""
from __future__ import annotations

import numpy as np
from scipy import sparse
import torch

from conga.core import DTYPE, TargetPosterior
from conga.inference import FisherHMC, Marginalized
from conga.inference.marginal import _CollapsedTarget
from conga.sparse import SparsePrecisionBackend
from conga.structure import GaussianBlock, SparseGaussianBlock


class _CycleGMRF(TargetPosterior):
    def __init__(self, n: int = 12):
        self.name = "cycle_gmrf"
        self.dim = n + 2
        self.n = n
        self.kappa = 0.3
        self.y = torch.linspace(-0.8, 1.1, n, dtype=DTYPE)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        u, v, x = z[..., 0], z[..., 1], z[..., 2:]
        edge = torch.roll(x, -1, dims=-1) - x
        prior_quad = edge.square().sum(-1) + self.kappa * x.square().sum(-1)
        obs_quad = (self.y - x).square().sum(-1)
        return (
            -self.n * u - 0.5 * torch.exp(-2 * u) * prior_quad
            - self.n * v - 0.5 * torch.exp(-2 * v) * obs_quad
            - 0.5 * u.square() - 0.5 * v.square()
        )

    def constrain(self, z: torch.Tensor):
        return {"state_sd": z[:, 0].exp(), "obs_sd": z[:, 1].exp(), "x": z[:, 2:]}


def _cycle_precision(n: int, u: float, v: float, kappa: float):
    rows = np.arange(n)
    adjacency = sparse.coo_matrix(
        (np.ones(2 * n), (np.r_[rows, rows], np.r_[np.roll(rows, -1), np.roll(rows, 1)])),
        shape=(n, n),
    ).tocsc()
    laplacian = sparse.diags(np.asarray(adjacency.sum(1)).reshape(-1)) - adjacency
    return np.exp(-2 * u) * (laplacian + kappa * sparse.eye(n)) + np.exp(-2 * v) * sparse.eye(n)


def test_sparse_precision_factor_is_exact_transport():
    precision = _cycle_precision(20, -0.4, -1.1, 0.2)
    backend = SparsePrecisionBackend(precision != 0)
    factor = backend.factor(precision, verify=True)
    dense = precision.toarray()
    rng = np.random.default_rng(4)
    rhs = rng.normal(size=(3, 20))
    assert np.max(np.abs(factor.solve(rhs) - np.linalg.solve(dense, rhs.T).T)) < 1e-11
    assert abs(factor.logdet - np.linalg.slogdet(dense)[1]) < 1e-11

    mean = rng.normal(size=20)
    x = rng.normal(size=(3, 20))
    z = factor.forward(x, mean)
    assert np.max(np.abs(factor.inverse(z, mean) - x)) < 1e-11
    dev = x - mean
    quadratic = np.einsum("bi,ij,bj->b", dev, dense, dev)
    assert np.max(np.abs(np.square(z).sum(1) - quadratic)) < 1e-10
    ii, jj = np.nonzero(precision.toarray())
    selected = factor.selected_inverse(ii, jj)
    dense_inverse = np.linalg.inv(dense)
    assert np.max(np.abs(selected - dense_inverse[ii, jj])) < 1e-11


def test_sparse_gaussian_block_matches_dense_black_box_collapse():
    target = _CycleGMRF()
    dense = GaussianBlock(target, torch.arange(2, target.dim), banded=False)
    sparse_block = SparseGaussianBlock(dense)
    expected = _cycle_precision(target.n, -0.3, -1.0, target.kappa) != 0
    assert np.array_equal(dense.pattern.numpy(), expected.toarray())

    etas = torch.tensor([[-0.3, -1.0], [0.2, -0.7]], dtype=DTYPE)
    evaluation = sparse_block.evaluate(etas, verify=True)
    lp_sparse = sparse_block.collapsed_log_post(etas, evaluation=evaluation)
    lp_dense = dense.collapsed_log_post(etas)
    assert float((lp_sparse - lp_dense).abs().max()) < 1e-10
    assert sparse_block.joint_identity_residual(etas) < 1e-9

    gen = torch.Generator().manual_seed(8)
    eps = torch.randn(2, target.n, dtype=DTYPE, generator=gen)
    x = sparse_block.inverse(etas, eps, evaluation=evaluation)
    recovered = sparse_block.forward(etas, x, evaluation=evaluation)
    assert float((recovered - eps).abs().max()) < 1e-11

    lp_vg, grad_sparse = sparse_block.collapsed_vg(etas)
    eta_grad = etas.clone().requires_grad_(True)
    lp_autograd = dense.collapsed_log_post(eta_grad)
    grad_dense = torch.autograd.grad(lp_autograd.sum(), eta_grad)[0]
    assert float((lp_vg - lp_autograd.detach()).abs().max()) < 1e-10
    assert float((grad_sparse - grad_dense).abs().max()) < 1e-9


def test_sparse_collapsed_target_supplies_exact_autograd_score():
    target = _CycleGMRF(n=8)
    dense = GaussianBlock(target, torch.arange(2, target.dim), banded=False)
    sparse_target = _CollapsedTarget(SparseGaussianBlock(dense))
    eta = torch.tensor([[-0.3, -1.0], [0.2, -0.7]], dtype=DTYPE,
                       requires_grad=True)
    lp = sparse_target.log_prob(eta)
    gradient = torch.autograd.grad(lp.sum(), eta)[0]

    dense_eta = eta.detach().requires_grad_(True)
    dense_lp = dense.collapsed_log_post(dense_eta)
    dense_gradient = torch.autograd.grad(dense_lp.sum(), dense_eta)[0]
    assert float((lp.detach() - dense_lp.detach()).abs().max()) < 1e-10
    assert float((gradient - dense_gradient).abs().max()) < 1e-9


def test_sparse_marginalized_nuts_smoke():
    __import__("pytest").importorskip("arviz")
    target = _CycleGMRF(n=8)
    sampler = FisherHMC(
        n_chains=2, n_warmup=12, n_draws=12, max_treedepth=3
    )
    fit = Marginalized(
        g_idx=torch.arange(2, target.dim), sampler=sampler, sparse=True
    ).fit(target, seed=3)
    draws = fit.sample(5)

    assert fit.algorithm == "Sparse-RB-FisherHMC"
    assert fit.diagnostics["sparse"] is True
    assert fit.diagnostics["d_h"] == 2
    assert fit.diagnostics["n_colors"] == 5
    assert len(fit.diagnostics["coordinate_bulk_ess"]) == 2
    assert len(fit.diagnostics["coordinate_rank_rhat"]) == 2
    assert np.isfinite(fit.diagnostics["max_rank_rhat"])
    assert draws["state_sd"].shape == (5,)
    assert draws["obs_sd"].shape == (5,)
    x = np.column_stack([draws[f"x[{i + 1}]"] for i in range(target.n)])
    assert x.shape == (5, target.n)
    assert np.isfinite(x).all()
