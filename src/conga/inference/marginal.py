"""Automatically marginalized sampling: the compiler's strongest emission.

``Marginalized`` wraps any :class:`InferenceAlgorithm` and a conditionally
Gaussian block -- indices supplied, or discovered from ``log_prob`` alone by
:func:`conga.structure.discover_gaussian_block`; the collapse itself is
derived automatically by :class:`conga.structure.GaussianBlock`, through
its banded (Kalman-scan) backend when the block's sparsity pattern permits, or
through :class:`conga.structure.SparseGaussianBlock` on a general graph.
The inner sampler runs on the *collapsed* posterior p(eta | y) --
low-dimensional, no scale--block funnel -- and the block is then drawn
*exactly* from p(x | eta, y) = N(m(eta), Lambda(eta)^{-1}), one conditional
draw per eta draw. The result is a sampler for the full joint posterior
whose only Monte Carlo error lives in the low-dimensional eta chain; the
block never mixes, it is integrated. With no constructor arguments at all,
``Marginalized().fit(target)`` is model-in, marginalized-sampler-out.

This is the block-Gaussian analogue of automatically marginalized MCMC
(Lai, Domke & Sheldon 2023), assembled from autodiff probes rather than
conjugacy pattern-matching, and it serves the benchmark twice:

  * as an *oracle row*: the accuracy any joint-space sampler on the same
    model is trying to reach, at a fraction of the effective budget;
  * as a *reference-draw factory*: the same pipeline that hand-built the
    ``dbn_lgss`` reference (Kalman collapse + FFBS), now emitted
    automatically for any conditionally Gaussian model.

Cost accounting: the inner sampler's gradient evaluations happen on the
collapsed density; on the exact-score path each costs ``2 + 3 n_colors``
target passes (a ``1 + n_colors`` probe call plus a ``1 + 2 n_colors``
gradient call), or ``~2 (n_colors + 2)`` via double-backward. Diagnostics
report both the sampler-counted collapsed evaluations and the approximate
target-equivalent passes.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch

from ..structure import GaussianBlock, SparseGaussianBlock, discover_gaussian_block
from ..core import DTYPE, TargetPosterior
from .base import InferenceAlgorithm, InferenceResult


class _CollapsedVG(torch.autograd.Function):
    """Collapsed log posterior with the exact first-order gradient of
    :meth:`GaussianBlock.collapsed_vg` -- Fisher's identity with the Gaussian
    expectation in closed form -- stitched into autograd so the inner sampler
    runs unchanged. The forward computes (value, gradient) in two batched
    target calls; the backward is a stored-gradient contraction, so no
    second-order graph is ever built."""

    @staticmethod
    def forward(ctx, etas: torch.Tensor, block):
        lp, g = block.collapsed_vg(etas.detach())
        ctx.save_for_backward(g)
        return lp

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (g,) = ctx.saved_tensors
        return grad_out.unsqueeze(-1) * g, None


class _CollapsedTarget(TargetPosterior):
    """The collapsed posterior p(eta | y) as a TargetPosterior, so any
    inference algorithm in the package can sample it unchanged. ``constrain``
    exposes the raw coordinates under a synthetic name; the wrapper
    reassembles them and never scores this target directly. With
    ``exact_grad``, gradient-bearing calls route through :class:`_CollapsedVG`;
    plain evaluations keep the value-only path. The general-graph sparse
    backend supplies the same interface using selected inversion."""

    def __init__(self, block, exact_grad: bool = True):
        self.block = block
        self.name = block.target.name + "::collapsed"
        self.dim = block.d_h
        self.exact_grad = bool(exact_grad) and not getattr(block, "banded", False)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        if self.exact_grad and torch.is_grad_enabled() and z.requires_grad:
            return _CollapsedVG.apply(z, self.block)
        return self.block.collapsed_log_post(z)

    def constrain(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"eta": z[..., 0] if self.dim == 1 else z}


class Marginalized(InferenceAlgorithm):
    """Run ``sampler`` on the collapsed posterior; draw the block exactly.

    ``g_idx`` are the indices of the conditionally Gaussian block in the
    target's unconstrained space; pass ``None`` to discover them from
    ``log_prob`` alone (:func:`conga.structure.discover_gaussian_block`),
    which also hands the discovery base points to the pattern probes.
    ``block_kwargs`` are forwarded to GaussianBlock -- e.g. ``probe_etas``
    for targets whose Hessian magnitudes are eta-dependent, or ``banded`` /
    ``block_size`` to force an assembly backend. Set ``sparse=True`` to wrap
    the discovered pattern in the general-graph sparse LDL backend.
    ``discover_kwargs`` are forwarded to the discovery pass (e.g.
    ``probe_points``).
    """

    def __init__(self, g_idx=None, sampler: Optional[InferenceAlgorithm] = None,
                 chunk: Optional[int] = 1024, block_kwargs: Optional[dict] = None,
                 discover_kwargs: Optional[dict] = None,
                 exact_grad: bool = True, sparse: bool = False):
        from .fisher_hmc import FisherHMC
        self.g_idx = None if g_idx is None else torch.as_tensor(
            g_idx, dtype=torch.long)
        self.sampler = sampler if sampler is not None else FisherHMC()
        self.chunk = chunk
        self.block_kwargs = block_kwargs or {}
        self.discover_kwargs = discover_kwargs or {}
        self.exact_grad = exact_grad
        self.sparse = bool(sparse)
        prefix = "Sparse-RB" if self.sparse else "RB"
        self.name = f"{prefix}-{self.sampler.name}"

    def fit(self, target: TargetPosterior, seed: int = 0,
            progress: object = False) -> InferenceResult:
        t0 = time.time()
        bk = dict(self.block_kwargs)
        if self.g_idx is None:
            disc = discover_gaussian_block(target, seed=seed,
                                           **self.discover_kwargs)
            g_idx = disc.g_idx
            bk.setdefault("probe_etas", disc.probe_etas)
        else:
            disc, g_idx = None, self.g_idx
        dense_block = GaussianBlock(target, g_idx, seed=seed, **bk)
        block = SparseGaussianBlock(dense_block) if self.sparse else dense_block
        collapsed = _CollapsedTarget(block, exact_grad=self.exact_grad)
        inner = self.sampler.fit(collapsed, seed=seed, progress=progress)

        C = block.n_colors
        chunk = self.chunk
        if chunk is None:
            # Bound the largest colored-probe tensor to roughly two million
            # doubles. Hub graphs can have C=d_g while bounded-degree graphs
            # retain large reconstruction batches.
            chunk = max(1, min(1024, 2_000_000 // (block.d_g * (1 + C))))
        base_seed = seed

        def draw_fn(n: int) -> Dict[str, np.ndarray]:
            named = inner.sample(n)
            if block.d_h == 1:
                etas = torch.as_tensor(np.asarray(named["eta"], float),
                                       dtype=DTYPE).unsqueeze(-1)
            else:
                etas = torch.stack(
                    [torch.as_tensor(np.asarray(named[f"eta[{i + 1}]"], float),
                                     dtype=DTYPE) for i in range(block.d_h)],
                    dim=-1)
            gen = torch.Generator().manual_seed(base_seed + 777)
            zs = []
            for lo in range(0, etas.shape[0], chunk):
                e = etas[lo:lo + chunk]
                # x ~ N(m, Lambda^{-1}), exact, via the block's own backend
                x = block.conditional_sample(e, gen=gen)
                z = torch.zeros(e.shape[0], target.dim, dtype=DTYPE)
                z[:, block.h_idx] = e
                z[:, block.g_idx] = x
                zs.append(z)
            with torch.no_grad():
                return target.constrained_draws(torch.cat(zs))

        # per collapsed evaluation-with-gradient, in K-row target passes:
        # exact path = probe call (1 + C rows) + gradient call (1 + 2C rows);
        # double-backward path = ~2 (C + 2).
        passes = (3 * C + 2) if collapsed.exact_grad else 2 * (C + 2)
        diagnostics = dict(inner.diagnostics)
        diagnostics.update({
            "d_h": block.d_h, "d_g": block.d_g, "n_colors": block.n_colors,
            "discovered": bool(disc is not None),
            "sparse": self.sparse,
            "banded": getattr(block, "banded", False),
            "bandwidth": block.bandwidth,
            "block_size": getattr(block, "nb", None),
            "reconstruction_chunk": chunk,
            "exact_grad": collapsed.exact_grad,
            "passes_per_collapsed_eval": passes,
            "target_calls_per_collapsed_eval":
                2 if collapsed.exact_grad else 1,
            # inner grad counts are collapsed evaluations; approximate
            # target-equivalent (K-row) passes for the gradient path used.
            "target_passes_approx": int(
                passes * ((inner.diagnostics.get("grad_evals") or 0)
                          + (inner.diagnostics.get("warmup_grad_evals") or 0))),
        })
        return InferenceResult(
            algorithm=self.name,
            target_name=target.name,
            _draw_fn=draw_fn,
            runtime_s=time.time() - t0,
            diagnostics=diagnostics,
        )
