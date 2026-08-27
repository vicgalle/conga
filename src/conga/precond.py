"""The structure metric fitted from a value-and-gradient callable alone.

The metric of :mod:`conga.structure` leaves the sampler untouched -- it is
a fixed linear reparameterization -- so it does not need this repo's harness
to be useful: anything that exposes ``log_prob`` and gradients on an
unconstrained space can be preconditioned, and any sampler with a
reparameterization hook can consume the result. This module is the
PPL-agnostic half of that packaging.

The observation that keeps this module small: the structure machinery only
ever consumes *first-order* gradients of the joint log density -- the
precision by gradient differences, discovery by secant probes, and the
collapsed-posterior exploration by the gradient-free
:func:`conga.structure.demc`. So a value-and-gradient callable is a
sufficient bridge, and :class:`CallableTarget` adapts one into the
``TargetPosterior`` interface with a one-page ``autograd.Function``
(first-order only, which is all the metric path needs; the marginalized
emission's double-backward MAP stage is exactly what this bridge gives up).

:func:`fit_structure_metric` composes the full automatic pipeline -- block
discovery, the colored-probe collapse, a gradient-free ensemble pass on the
collapsed posterior, the Rao-Blackwellized covariance, and its Cholesky
whitening map -- against the external log density, so the metric lives in
whatever unconstrained parameterization the PPL chose and no model is
re-implemented anywhere.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .core import DTYPE, TargetPosterior
from .structure import (GaussianBlock, cost_aware_shrink, demc,
                        discover_gaussian_block, rb_joint_covariance,
                        whitening_cholesky)


# --------------------------------------------------------------------------- #
# bridge: batched (logp, grad) callable -> TargetPosterior
# --------------------------------------------------------------------------- #
class _LogpGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, fn):
        lp, g = fn(np.ascontiguousarray(z.detach().cpu().numpy(), dtype=np.float64))
        ctx.save_for_backward(torch.as_tensor(np.array(g), dtype=z.dtype))
        return torch.as_tensor(np.array(lp), dtype=z.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        (g,) = ctx.saved_tensors
        return grad_out.unsqueeze(-1) * g, None


class CallableTarget(TargetPosterior):
    """``TargetPosterior`` over a batched numpy ``X [K, D] -> (logp [K], grad [K, D])``.

    First-order differentiable only (``once_differentiable`` backward): the
    external function is a black box, so no graph exists to backpropagate
    through twice. Discovery, the colored collapse, and the RB metric never
    need more.
    """

    def __init__(self, logp_and_grad: Callable, dim: int, name: str = "external"):
        self._fn = logp_and_grad
        self.name = name
        self.dim = dim

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return _LogpGradFn.apply(z, self._fn)

    def constrain(self, z: torch.Tensor):
        return {"z": z}


# --------------------------------------------------------------------------- #
# the metric, fitted from a value-and-gradient callable alone
# --------------------------------------------------------------------------- #
@dataclass
class StructureMetric:
    """Whitening map ``x = mean + L^{-T} z`` plus construction diagnostics."""
    mean: np.ndarray                 # [D]
    L: np.ndarray                    # [D, D] lower; M = L L^T ~ precision
    g_idx: np.ndarray                # discovered block, after cost-aware shrink
    h_idx: np.ndarray                # collapsed coordinates
    n_colors: int                    # gradient passes per collapsed eval - 1
    identity_residual: float         # exact collapse identity at the RB draws
    rhat: float                      # split R-hat of the collapsed ensemble
    accept: float                    # DE-MC acceptance rate
    build_seconds: float

    def covariance(self) -> np.ndarray:
        """The RB joint covariance ``sigma = L^{-T} L^{-1}`` the map whitens by.

        Reconstructed from ``L`` (rather than kept alongside it) so it is
        consistent with the eigenvalue flooring of
        :func:`conga.structure.whitening_cholesky` -- this is the inverse
        mass matrix a sampler with a metric argument should be handed.
        """
        Linv = np.linalg.solve(self.L, np.eye(self.L.shape[0]))
        return np.ascontiguousarray(Linv.T @ Linv)


def check_whitened_gradient(logp_and_grad: Callable, dim: int,
                            metric: "StructureMetric", seed: int = 0,
                            eps: float = 1e-5) -> float:
    """Finite-difference check of the whitened gradient: for a few random z,
    compare ``d/dz [logp(mean + z @ Linv)]`` against ``g_x @ Linv.T``.

    This is the wiring check every plugin runs before handing the metric to
    its sampler: it catches a transform assembled against the wrong
    coordinate ordering (or a mean/L mix-up) while staying insensitive to a
    metric built from rough collapsed moments, which is merely suboptimal.
    Returns the worst relative error over the probes.
    """
    L = np.asarray(metric.L, dtype=np.float64)
    Linv = np.linalg.solve(L, np.eye(L.shape[0]))
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(3):
        z = 0.5 * rng.standard_normal(dim)
        k = int(rng.integers(dim))
        zp, zm = z.copy(), z.copy()
        zp[k] += eps
        zm[k] -= eps
        X = metric.mean + np.stack([z, zp, zm]) @ Linv
        lp, gx = logp_and_grad(X)
        gz = gx[0] @ Linv.T
        num = (lp[1] - lp[2]) / (2 * eps)
        worst = max(worst, abs(num - gz[k]) / max(1.0, abs(gz[k])))
    return worst


def fit_structure_metric(logp_and_grad: Callable, dim: int, *,
                         probe_points: Optional[np.ndarray] = None,
                         seed: int = 0, discovery_tol: float = 1e-8,
                         n_walkers: Optional[int] = None, n_burn: int = 2000,
                         n_sample: int = 4000, thin: int = 30, n_rb: int = 512,
                         min_block: int = 2, verbose: bool = True) -> StructureMetric:
    """The full automatic pipeline against an external log density.

    ``probe_points`` (e.g. the PPL's jittered initial points) anchor both
    discovery and pattern detection in the plausible region; without them the
    default random probes around the origin are used, which is adequate for
    the near-standardized unconstrained spaces PPLs produce but not
    guaranteed for badly scaled models.
    """
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    target = CallableTarget(logp_and_grad, dim)
    pts = None if probe_points is None else torch.as_tensor(
        np.asarray(probe_points, dtype=np.float64), dtype=DTYPE)

    disc = discover_gaussian_block(target, probe_points=pts,
                                   tol=discovery_tol, seed=seed)
    if len(disc.g_idx) < min_block:
        raise ValueError(
            f"no conditionally Gaussian block found (size {len(disc.g_idx)} "
            f"< {min_block}); structure preconditioning does not apply")
    if len(disc.h_idx) == 0:
        raise NotImplementedError(
            "the whole target is Gaussian; sample it directly")
    say(f"[structure] block: {len(disc.g_idx)} of {dim} coordinates "
        f"(kept-axis residual {disc.margins['max_kept_axis_residual']:.1e}, "
        f"rejected >= {disc.margins['min_conflict_residual']:.1e})")

    block = GaussianBlock(target, disc.g_idx, probe_etas=disc.probe_etas,
                          seed=seed)
    probe_base = disc.probe_etas
    # cost-aware refinement: hub coordinates whose dense pattern rows
    # degrade the coloring move to the core (a no-op on hub-free blocks)
    g_idx, shrink = cost_aware_shrink(block)
    if shrink["n_moves"]:
        say(f"[structure] cost-aware shrink: {shrink['n_moves']} hub "
            f"coordinate(s) to the core "
            f"({shrink['colors_before']} -> {shrink['colors_after']} colors)")
        mask = torch.ones(dim, dtype=torch.bool)
        mask[g_idx] = False
        probe_base = disc.base_points[:, mask]
        block = GaussianBlock(target, g_idx, probe_etas=probe_base,
                              pattern=shrink["pattern"], seed=seed)
    say(f"[structure] collapse: {block.n_colors} color(s), self-checks "
        f"sym {block.sym_err:.1e} / pattern {block.pattern_err:.1e}")

    # gradient-free exploration of the collapsed posterior, walkers seeded
    # from the certified probe points
    gen = torch.Generator().manual_seed(seed + 1)
    d_h = block.d_h
    if n_walkers is None:
        n_walkers = max(16, 4 * d_h)
    base = probe_base
    init = (base[torch.randint(base.shape[0], (n_walkers,), generator=gen)]
            + 0.5 * torch.randn(n_walkers, d_h, dtype=DTYPE, generator=gen))
    draws, rhat, acc = demc(block.collapsed_log_post, init, seed,
                            n_burn=n_burn, n_sample=n_sample, thin=thin)
    say(f"[structure] collapsed DE-MC: {draws.shape[0]} draws over "
        f"{d_h} dim(s), split R-hat {rhat:.2f}, accept {acc:.2f}")

    idx = torch.randperm(draws.shape[0], generator=gen)[:n_rb]
    etas = draws[idx]
    m, _, chol = block.conditional_moments(etas)
    covs = torch.cholesky_inverse(chol)
    resid = block.joint_identity_residual(etas[:64], gen=gen)
    say(f"[structure] collapse identity at the RB draws: {resid:.1e}")

    sigma = rb_joint_covariance(etas, m, covs.mean(0), block.h_idx, block.g_idx)
    L = whitening_cholesky(sigma)
    mean = torch.zeros(dim, dtype=DTYPE)
    mean[block.h_idx] = etas.mean(0)
    mean[block.g_idx] = m.mean(0)

    dt = time.time() - t0
    say(f"[structure] metric built in {dt:.1f}s")
    return StructureMetric(
        mean=mean.numpy(), L=L.numpy(),
        g_idx=block.g_idx.numpy(), h_idx=block.h_idx.numpy(),
        n_colors=block.n_colors, identity_residual=resid,
        rhat=rhat, accept=acc, build_seconds=dt)
