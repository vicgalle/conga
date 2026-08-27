"""Structure-exact preconditioning: shared machinery.

A target with a *conditionally Gaussian block* -- a subset of coordinates whose
conditional posterior given the rest is Gaussian with closed-form moments
(latent states of a linear-Gaussian SSM, basis coefficients of an HSGP,
random effects of a linear mixed model) -- does not need its sampler metric
estimated from warmup draws: the metric can be computed. The recipe is:

  1. collapse the Gaussian block analytically, leaving a low-dimensional
     posterior over the remaining coordinates eta;
  2. draw (approximately) from that collapsed posterior -- cheap, since each
     evaluation is a small closed-form marginal likelihood;
  3. assemble the exact *marginal* joint covariance by the law of total
     covariance, Rao-Blackwellizing the Gaussian block's conditional moments
     over the eta draws (:func:`rb_joint_covariance`);
  4. whiten the joint target by the Cholesky factor of the inverse
     (:func:`whitening_cholesky`, :class:`WhitenedTarget`) and run any
     unchanged sampler on it.

The DBN prototype's ablation shows step 3 is where the value lives: the naive
conditional-precision-at-a-point metric ignores how the block's posterior
widens as the hyperparameters vary (the funnel) and whitens *worse* than a
diagonal, while the total-covariance metric sent the whitened posterior's
condition number from 116 (best diagonal) to 2.

What a target must supply is exactly what an automatic Rao-Blackwellization
pass would emit: the collapsed log-posterior and the conditional moments
``eta -> (m(eta), Cov(eta))`` -- both mechanical consequences of the model's
conditional structure. This module is the deliberately small interface those
two ingredients plug into. :class:`GaussianBlock` derives both automatically
from ``log_prob`` and the block indices alone, with no model-specific
mathematics.

Two further passes remove the remaining hand-supplied ingredients:

  * :func:`discover_gaussian_block` finds candidate block *indices* themselves
    from ``log_prob`` alone. Vanishing third derivatives with all three indices
    in G characterize quadraticity in the block; a proper Gaussian conditional
    additionally requires the resulting precision to be positive definite.
    Secant-slope probes along coordinate axes (exact for quadratics, no
    step-size limit involved) expose those third derivatives as a conflict
    graph, a greedy peel extracts a large independent set, and a joint-
    quadraticity residual checks the survivor before collapse checks its
    precision and Gaussian identity at posterior-plausible points.
  * a *banded backend* inside :class:`GaussianBlock`: when the detected
    sparsity pattern has small bandwidth under a reverse Cuthill-McKee
    reordering (chains, trees of small width), the per-draw precision is
    assembled directly into block-tridiagonal storage and factorized by a
    batched block-bidiagonal Cholesky scan -- for a state-space model this
    *is* the information-form Kalman filter -- turning the O(d_x^3) collapse
    into O(d_x b^2) time and O(d_x b) memory per draw.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from .core import DTYPE, TargetPosterior
from .sparse import SparsePrecisionBackend, SparsePrecisionFactor


class WhitenedTarget(TargetPosterior):
    """Fixed affine reparameterization ``x = center + L^{-T} z`` of a base target.

    With ``M = L L^T`` close to the posterior precision, the posterior of z is
    close to standard normal. The Jacobian is constant, so ``log_prob`` differs
    from the base by an irrelevant constant; draws are mapped back through the
    base target's ``constrain``, so scoring is unchanged.

    ``center`` (optional; default zero, the original linear map) matters for
    posteriors that sit far from the unconstrained origin -- ``rats`` lives at
    alpha ~ 240, hundreds of whitened units from the sampler's initialization,
    and a short whitened warmup is otherwise spent commuting.  The metric
    recipe estimates the Rao-Blackwellized mean along with the covariance, so
    the shift comes for free.
    """

    def __init__(self, base: TargetPosterior, L: torch.Tensor,
                 center: Optional[torch.Tensor] = None):
        self.base = base
        self.name = base.name
        self.dim = base.dim
        self._center = None if center is None else center.to(DTYPE)
        self._Linv = torch.linalg.solve_triangular(
            L, torch.eye(base.dim, dtype=DTYPE), upper=False)

    def _to_base(self, z: torch.Tensor) -> torch.Tensor:
        x = z @ self._Linv             # row-vector form of x = L^{-T} z
        return x if self._center is None else x + self._center

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.base.log_prob(self._to_base(z))

    def constrain(self, z: torch.Tensor):
        return self.base.constrain(self._to_base(z))


class GaussianBlock:
    r"""Automatic collapse of a conditionally Gaussian block from ``log_prob``.

    If the joint log density is exactly quadratic in the block ``x`` (indices
    ``g_idx``), i.e. :math:`\ell(\eta,x) = \ell_0(\eta) + g_0(\eta)^\top x -
    \tfrac12 x^\top \Lambda(\eta) x`, then everything the structure-metric
    recipe consumes follows from autodiff alone:

      * :math:`\Lambda(\eta)` column-by-column from *first-order* gradients,
        since :math:`\nabla_x \ell(\eta, v) = g_0 - \Lambda v` -- no
        second-order autograd;
      * :math:`m(\eta) = \Lambda^{-1} g_0` and the conditional covariance
        :math:`\Lambda^{-1}`;
      * the collapsed evidence, exactly (Laplace is exact for quadratics):
        :math:`\log p(y,\eta) = \ell_0 + \tfrac12 g_0^\top m
        + \tfrac{d_x}{2}\log 2\pi - \tfrac12 \log|\Lambda|`.

    The per-evaluation cost is ``1 + n_colors`` batched forward/backward
    passes: at construction the sparsity pattern of :math:`\Lambda` is
    detected from full probes at a few *random* :math:`\eta` (random, because
    a special point can zero out structurally nonzero couplings -- e.g. the
    DBN's cross terms vanish at A = 0), and a greedy Curtis--Powell--Reid
    distance-2 coloring packs structurally orthogonal columns into shared
    probe vectors. A conditionally independent block (eight schools) needs 1
    color; a chain (the DBN) a handful, independent of its length; a dense
    block degrades gracefully to ``d_x`` colors.

    Downstream of the probes, the assembly/factorization backend is chosen
    from the pattern's bandwidth under a reverse Cuthill-McKee reordering:
    small-bandwidth blocks (chains, conditionally independent structures)
    are assembled straight into block-tridiagonal storage and factorized by
    a batched block-bidiagonal Cholesky scan -- for a state-space model this
    *is* the information-form Kalman filter -- at O(d_x nb^2) work and
    O(d_x nb) memory per draw instead of the dense O(d_x^3) / O(d_x^2);
    dense patterns keep the LAPACK path. ``banded=None`` decides
    automatically, ``block_size`` trades Python scan depth against slack in
    the band.

    Construction verifies itself: the colored assembly is checked against a
    full-probe Hessian at a fresh random :math:`\eta` (catches pattern
    misses), quadraticness + symmetry are checked from gradient
    differences, and the banded backend is checked against the dense one on
    the collapsed evidence. ``joint_identity_residual`` exposes the
    end-to-end test
    :math:`\log p(y,\eta,x) - \log p(y,\eta) = \log\N(x; m, \Lambda^{-1})`.

    Rows whose evaluation is numerically infeasible (non-finite or non-PD at
    far-out :math:`\eta`) get collapsed log-posterior ``-1e30`` rather than an
    exception, so gradient-free ensemble explorers can run on top directly.
    """

    def __init__(self, target: TargetPosterior, g_idx: torch.Tensor,
                 probe_etas: Optional[torch.Tensor] = None,
                 n_pattern_probes: int = 3, pattern_tol: float = 1e-9,
                 probe_scale: float = 0.5, seed: int = 0, verify: bool = True,
                 banded: Optional[bool] = None,
                 block_size: Optional[int] = None, probe_chunk: int = 2048,
                 pattern: Optional[torch.Tensor] = None):
        # The detected pattern is the union over the probe points. When the
        # magnitudes of Lambda's entries depend strongly on eta (e.g. an HSGP
        # whose spectral density decays with the lengthscale), random probes
        # can see a numerically sparse matrix that is dense where the
        # posterior lives -- supply ``probe_etas`` near plausible parameter
        # values in that case, and check ``joint_identity_residual`` at the
        # draws you actually use downstream. A precomputed ``pattern`` (e.g.
        # the principal subpattern carried over from a cost-aware shrink)
        # skips the detection probes entirely; the construction-time
        # verification below still runs and catches a stale or wrong pattern.
        self.target = target
        self.g_idx = torch.as_tensor(g_idx, dtype=torch.long)
        mask = torch.ones(target.dim, dtype=torch.bool)
        mask[self.g_idx] = False
        self.h_idx = torch.nonzero(mask).squeeze(-1)
        self.d_g, self.d_h = len(self.g_idx), len(self.h_idx)
        self.probe_chunk = int(probe_chunk)

        gen = torch.Generator().manual_seed(seed)
        if probe_etas is not None:
            jitter = 0.05 * torch.randn(1, self.d_h, dtype=DTYPE, generator=gen)
            probes = torch.cat([probe_etas.to(DTYPE),
                                probe_etas[-1:].to(DTYPE) + jitter])
        else:
            probes = probe_scale * torch.randn(n_pattern_probes + 1, self.d_h,
                                               dtype=DTYPE, generator=gen)
        if pattern is not None:
            pattern = pattern.to(torch.bool).clone()
            assert pattern.shape == (self.d_g, self.d_g), \
                f"pattern shape {tuple(pattern.shape)} != block size {self.d_g}"
        else:
            pattern = torch.zeros(self.d_g, self.d_g, dtype=torch.bool)
            for p in probes[:-1]:                 # union, one probe at a time
                lam = self._full_lambda(p)        # (memory: one [d_g, d_g])
                pattern |= lam.abs() > pattern_tol * lam.abs().max()
        self.pattern = pattern | pattern.T
        self.colors, self.n_colors = _cpr_coloring(self.pattern)
        # probe matrix V [n_colors, d_g]: v_c = sum of unit vectors of class c
        self.V = torch.zeros(self.n_colors, self.d_g, dtype=DTYPE)
        self.V[self.colors, torch.arange(self.d_g)] = 1.0

        # bandwidth under a reverse Cuthill-McKee reordering decides the
        # assembly/factorization backend (the model's own coordinate order is
        # immaterial: the DBN's chain is nowhere near banded as written).
        # The size threshold is a measured CPU crossover: below a few hundred
        # coordinates the sequential scan's per-block Python steps cost more
        # than one dense LAPACK factorization.
        self.perm, self.pinv, self.bandwidth = _rcm(self.pattern)
        if banded is None:
            banded = self.d_g >= 512 and 4 * (self.bandwidth + 1) <= self.d_g
        self.banded = bool(banded)
        if self.banded:
            self.nb = int(block_size or max(self.bandwidth, -(-self.d_g // 64), 1))
            self.NB = -(-self.d_g // self.nb)
            self._build_banded_maps()

        self.sym_err = float("nan")
        self.pattern_err = float("nan")
        self.banded_err = float("nan")
        if verify:
            lam_full = self._full_lambda(probes[-1])
            self.sym_err = float((lam_full - lam_full.T).abs().max()
                                 / lam_full.abs().max())
            _, _, lam_col = self._assemble(probes[-1].unsqueeze(0))
            self.pattern_err = float((lam_col[0] - lam_full).abs().max()
                                     / lam_full.abs().max())
            assert self.sym_err < 1e-8, \
                f"block is not quadratic (Hessian asymmetry {self.sym_err:.1e})"
            assert self.pattern_err < 1e-8, \
                f"sparsity pattern misses entries (error {self.pattern_err:.1e})"
            if self.banded:
                e = probes[-1].unsqueeze(0)
                lp_d, lp_b = self._collapsed_dense(e), self._collapsed_banded(e)
                self.banded_err = float((lp_b - lp_d).abs().max()
                                        / lp_d.abs().max().clamp(min=1.0))
                assert self.banded_err < 1e-9, \
                    f"banded backend disagrees with dense ({self.banded_err:.1e})"

    # ---------------- internals ---------------- #
    def _grads(self, etas: torch.Tensor, X: torch.Tensor):
        """Batched (lp, grad over the x slice) at points (etas, X).

        If ``etas.requires_grad``, the returned tensors stay connected to it
        (via ``create_graph``), so a MAP optimizer can differentiate the
        collapsed log-posterior with respect to eta by double backward.
        Detached calls are chunked to ``probe_chunk`` rows so pattern probes
        on very large blocks stay within memory.
        """
        keep = etas.requires_grad and torch.is_grad_enabled()
        chunk = etas.shape[0] if keep else max(self.probe_chunk, 1)
        lps, gs = [], []
        with torch.enable_grad():
            for lo in range(0, etas.shape[0], chunk):
                e, x = etas[lo:lo + chunk], X[lo:lo + chunk]
                Xv = x.detach().requires_grad_(True)
                z = torch.zeros(e.shape[0], self.target.dim, dtype=DTYPE)
                z[:, self.h_idx] = e
                z[:, self.g_idx] = Xv
                lp = self.target.log_prob(z)
                g = torch.autograd.grad(lp.sum(), Xv, create_graph=keep)[0]
                lps.append(lp if keep else lp.detach())
                gs.append(g if keep else g.detach())
        return torch.cat(lps), torch.cat(gs)

    def _full_lambda(self, eta: torch.Tensor) -> torch.Tensor:
        """Dense Lambda(eta) from d_g + 1 gradient rows in one batch."""
        E = torch.cat([torch.zeros(1, self.d_g, dtype=DTYPE),
                       torch.eye(self.d_g, dtype=DTYPE)])
        _, G = self._grads(eta.unsqueeze(0).expand(self.d_g + 1, -1), E)
        return (G[0].unsqueeze(0) - G[1:]).T          # columns Lambda e_i

    def _probe(self, etas: torch.Tensor):
        """(lp0, g0, R) at a batch of eta draws; R_c = Lambda v_c."""
        K = etas.shape[0]
        X = torch.cat([torch.zeros(1, K, self.d_g, dtype=DTYPE),
                       self.V.unsqueeze(1).expand(-1, K, -1)])   # [C+1, K, d_g]
        eta_rep = etas.unsqueeze(0).expand(self.n_colors + 1, -1, -1)
        B = (self.n_colors + 1) * K
        lp, G = self._grads(eta_rep.reshape(B, self.d_h),
                            X.reshape(B, self.d_g))
        lp = lp.reshape(self.n_colors + 1, K)
        G = G.reshape(self.n_colors + 1, K, self.d_g)
        g0 = G[0]
        return lp[0], g0, g0.unsqueeze(0) - G[1:]

    def _assemble(self, etas: torch.Tensor):
        """(lp0, g0, dense Lambda) for a batch of eta draws."""
        lp0, g0, R = self._probe(etas)
        K = etas.shape[0]
        lam = torch.zeros(K, self.d_g, self.d_g, dtype=DTYPE)
        for c in range(self.n_colors):
            cols = torch.nonzero(self.colors == c).squeeze(-1)
            lam[:, :, cols] = (R[c].unsqueeze(-1)
                               * self.pattern[:, cols].to(DTYPE))
        lam = 0.5 * (lam + lam.transpose(-1, -2))
        return lp0, g0, lam

    # ---------------- banded backend ---------------- #
    def _build_banded_maps(self):
        """Gather/scatter maps: colored-probe values -> block-tridiag storage.

        Storage is a flat [K, NB*2*nb*nb] buffer viewed as [K, NB, 2, nb, nb]:
        slot (b, 0) holds the diagonal block D_b of the RCM-permuted Lambda,
        slot (b, 1) the sub-diagonal block Lambda[b, b-1] (slot (0, 1) stays
        zero). Every directed structural nonzero (i, j) carries the value
        R[color(j)][:, i]; same-block entries write half of it to their slot
        and half to the transposed slot, cross-block entries write half to
        the canonical lower slot (their mirror entry supplies the other
        half), which reproduces the dense path's 0.5*(lam + lam^T) exactly.
        """
        nb, d = self.nb, self.d_g
        ii, jj = torch.nonzero(self.pattern, as_tuple=True)
        pi, pj = self.pinv[ii], self.pinv[jj]
        bi, ri = pi // nb, pi % nb
        bj, rj = pj // nb, pj % nb
        db = bi - bj
        assert bool((db.abs() <= 1).all()), "block size below bandwidth"
        src = self.colors[jj] * d + ii                 # into R [K, C*d_g]

        def slot(b, t, r1, r2):
            return ((b * 2 + t) * nb + r1) * nb + r2

        same, lo, up = db == 0, db == 1, db == -1
        self._bd_src = torch.cat([src[same], src[same], src[lo], src[up]])
        self._bd_dst = torch.cat([
            slot(bi[same], 0, ri[same], rj[same]),
            slot(bi[same], 0, rj[same], ri[same]),
            slot(bi[lo], 1, ri[lo], rj[lo]),
            slot(bj[up], 1, rj[up], ri[up])])
        # identity padding for the tail rows of the last block, and a full
        # identity buffer to substitute for non-finite / non-PD draws.
        pad = torch.zeros(self.NB * 2 * nb * nb, dtype=DTYPE)
        for p in range(d, self.NB * nb):
            pad[slot(p // nb, 0, p % nb, p % nb)] = 1.0
        self._bd_pad = pad
        eye_buf = torch.zeros_like(pad)
        for p in range(self.NB * nb):
            eye_buf[slot(p // nb, 0, p % nb, p % nb)] = 1.0
        self._bd_eye = eye_buf

    def _assemble_banded(self, etas: torch.Tensor):
        """(lp0, g0_perm [K, NB, nb], blocks [K, NB, 2, nb, nb], bad [K])."""
        lp0, g0, R = self._probe(etas)
        K, nb = etas.shape[0], self.nb
        Rf = R.permute(1, 0, 2).reshape(K, -1)
        buf = torch.zeros(K, self.NB * 2 * nb * nb, dtype=DTYPE)
        buf = buf.scatter_add(1, self._bd_dst.unsqueeze(0).expand(K, -1),
                              0.5 * Rf[:, self._bd_src])
        buf = buf + self._bd_pad
        bad = (~lp0.isfinite() | ~g0.isfinite().all(-1)
               | ~buf.isfinite().all(-1))
        buf = torch.where(bad.unsqueeze(-1), self._bd_eye, buf)
        g0p = torch.zeros(K, self.NB * nb, dtype=DTYPE)
        g0p[:, :self.d_g] = g0.nan_to_num()[:, self.perm]
        return lp0, g0p.reshape(K, self.NB, nb), \
            buf.reshape(K, self.NB, 2, nb, nb), bad

    def _bt_chol(self, blocks: torch.Tensor, bad: torch.Tensor):
        """Block-bidiagonal Cholesky scan of a block-tridiagonal Lambda.

        Lambda = L L^T with L_bb = C_b, L_{b,b-1} = F_b; the recursion
        C_0 = chol(D_0), F_b = O_b C_{b-1}^{-T}, C_b = chol(D_b - F_b F_b^T)
        is the information-form Kalman/Thomas scan: O(NB) steps of nb^3 work,
        batched over draws. Non-PD draws are flagged and continued on the
        identity so the batch never raises.
        """
        D, O = blocks[:, :, 0], blocks[:, :, 1]
        eye = torch.eye(self.nb, dtype=DTYPE)
        Cs: List[torch.Tensor] = []
        Fs: List[torch.Tensor] = [torch.zeros_like(D[:, 0])]
        S = D[:, 0]
        for b in range(self.NB):
            if b > 0:
                Ft = torch.linalg.solve_triangular(
                    Cs[b - 1], O[:, b].transpose(-1, -2), upper=False)
                F = Ft.transpose(-1, -2)
                Fs.append(F)
                S = D[:, b] - F @ Ft
            ok = S.isfinite().all((-2, -1)) & ~bad
            S = torch.where(ok.unsqueeze(-1).unsqueeze(-1), S, eye)
            C, info = torch.linalg.cholesky_ex(S)
            bad = bad | ~ok | (info > 0)
            C = torch.where((info > 0).unsqueeze(-1).unsqueeze(-1), eye, C)
            Cs.append(C)
        return Cs, Fs, bad

    @staticmethod
    def _bt_solve_L(Cs, Fs, r: torch.Tensor) -> torch.Tensor:
        """Solve L y = r for block-bidiagonal L; r is [K, NB, nb]."""
        ys = []
        for b in range(len(Cs)):
            rhs = r[:, b] if b == 0 else r[:, b] - (Fs[b] @ ys[-1].unsqueeze(-1)).squeeze(-1)
            ys.append(torch.linalg.solve_triangular(
                Cs[b], rhs.unsqueeze(-1), upper=False).squeeze(-1))
        return torch.stack(ys, dim=1)

    @staticmethod
    def _bt_solve_Lt(Cs, Fs, y: torch.Tensor) -> torch.Tensor:
        """Solve L^T x = y (backward scan)."""
        NB = len(Cs)
        xs = [None] * NB
        for b in range(NB - 1, -1, -1):
            rhs = y[:, b]
            if b < NB - 1:
                rhs = rhs - (Fs[b + 1].transpose(-1, -2)
                             @ xs[b + 1].unsqueeze(-1)).squeeze(-1)
            xs[b] = torch.linalg.solve_triangular(
                Cs[b].transpose(-1, -2), rhs.unsqueeze(-1),
                upper=True).squeeze(-1)
        return torch.stack(xs, dim=1)

    @staticmethod
    def _bt_matvec(blocks: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Lambda @ x for block-tridiagonal storage; x is [K, NB, nb]."""
        D, O = blocks[:, :, 0], blocks[:, :, 1]
        out = (D @ x.unsqueeze(-1)).squeeze(-1)
        lower = (O[:, 1:] @ x[:, :-1].unsqueeze(-1)).squeeze(-1)
        upper = (O[:, 1:].transpose(-1, -2) @ x[:, 1:].unsqueeze(-1)).squeeze(-1)
        zero = torch.zeros_like(x[:, :1])
        return (out + torch.cat([zero, lower], dim=1)
                + torch.cat([upper, zero], dim=1))

    def _bt_mean_logdet(self, etas: torch.Tensor):
        """(lp0, m [K, d_g] in model order, half-logdet, blocks, Cs, Fs, bad)."""
        lp0, g0p, blocks, bad = self._assemble_banded(etas)
        Cs, Fs, bad = self._bt_chol(blocks, bad)
        y = self._bt_solve_L(Cs, Fs, g0p)
        mp = self._bt_solve_Lt(Cs, Fs, y).reshape(etas.shape[0], -1)
        m = mp[:, :self.d_g].index_select(1, self.pinv)
        half_logdet = sum(torch.log(torch.diagonal(C, dim1=-2, dim2=-1)).sum(-1)
                          for C in Cs)
        return lp0, m, half_logdet, blocks, Cs, Fs, bad

    # ---------------- public surface ---------------- #
    def conditional_moments(self, etas: torch.Tensor):
        """(m, Lambda, chol(Lambda)) at eta draws; NaN-safe rows marked in chol.

        Always dense (used by the one-time metric assembly, which needs
        E[Lambda^{-1}] in full); the banded backend serves the per-draw paths
        ``collapsed_log_post`` and ``conditional_sample``.
        """
        _, g0, lam = self._assemble(etas)
        chol, _ = _safe_cholesky(lam)
        m = torch.cholesky_solve(g0.nan_to_num().unsqueeze(-1), chol).squeeze(-1)
        return m, lam, chol

    def collapsed_log_post(self, etas: torch.Tensor) -> torch.Tensor:
        """log p(y, eta) exactly, up to the target's own additive constant.

        Evaluated as ``l(eta, m) + (d_g/2) log 2pi - 1/2 log|Lambda|`` -- the
        Laplace identity *at the conditional mean*. The equivalent expansion
        around x = 0, ``l(eta,0) + 1/2 g0' Lambda^{-1} g0 + ...``, is
        catastrophically cancellation-prone: deep in a funnel both terms grow
        like 1/tau^2 while their sum stays O(1), and the roundoff hands stray
        tail draws spuriously *high* densities (importance weights then
        degenerate on exactly those draws). At x = m every term is
        data-scale, so no cancellation occurs.
        """
        if self.banded:
            return self._collapsed_banded(etas)
        return self._collapsed_dense(etas)

    def _laplace_at_mean(self, etas, m, half_logdet, bad):
        z = torch.zeros(etas.shape[0], self.target.dim, dtype=DTYPE)
        z[:, self.h_idx] = etas
        z[:, self.g_idx] = m
        lp_m = self.target.log_prob(z)
        lp = lp_m + 0.5 * self.d_g * math.log(2 * math.pi) - half_logdet
        return torch.where(bad | ~lp.isfinite(),
                           torch.tensor(-1e30, dtype=DTYPE), lp)

    def _collapsed_dense(self, etas: torch.Tensor) -> torch.Tensor:
        lp0, g0, lam = self._assemble(etas)
        chol, bad = _safe_cholesky(lam)
        bad = bad | ~lp0.isfinite() | ~g0.isfinite().all(-1)
        m = torch.cholesky_solve(g0.nan_to_num().unsqueeze(-1), chol).squeeze(-1)
        half_logdet = torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        return self._laplace_at_mean(etas, m, half_logdet, bad)

    def _collapsed_banded(self, etas: torch.Tensor) -> torch.Tensor:
        lp0, m, half_logdet, _, _, _, bad = self._bt_mean_logdet(etas)
        return self._laplace_at_mean(etas, m, half_logdet, bad)

    # ---------------- exact first-order value-and-gradient ---------------- #
    def _trace_maps(self):
        """Cached flat maps for the coloring contraction of the trace term
        and for the flat symmetrized assembly of Lambda (same construction as
        the banded builder: each directed nonzero (i, j) carries the value
        R[color(j)][:, i], written half to (i, j) and half to (j, i))."""
        if not hasattr(self, "_tr_ii"):
            ii, jj = torch.nonzero(self.pattern, as_tuple=True)
            self._tr_ii, self._tr_jj = ii, jj
            self._tr_dst = self.colors[jj] * self.d_g + ii   # -> [C * d_g]
            self._asm_src = self.colors[jj] * self.d_g + ii  # from R [K, C*d_g]
            self._asm_dst = torch.cat([ii * self.d_g + jj, jj * self.d_g + ii])
        return self._tr_ii, self._tr_jj, self._tr_dst

    def _lam_flat(self, R: torch.Tensor) -> torch.Tensor:
        """Dense symmetrized Lambda from probe responses via flat scatter."""
        self._trace_maps()
        K = R.shape[1]
        vals = 0.5 * R.permute(1, 0, 2).reshape(K, -1)[:, self._asm_src]
        lam = torch.zeros(K, self.d_g * self.d_g, dtype=DTYPE)
        lam = lam.scatter_add(1, self._asm_dst.unsqueeze(0).expand(K, -1),
                              torch.cat([vals, vals], dim=1))
        return lam.reshape(K, self.d_g, self.d_g)

    def _full_rows(self, etas_rep: torch.Tensor, X: torch.Tensor):
        """One batched call: (lp, per-row grad over the FULL z) at (etas, X)."""
        B = etas_rep.shape[0]
        z = torch.zeros(B, self.target.dim, dtype=DTYPE)
        z[:, self.h_idx] = etas_rep
        z[:, self.g_idx] = X
        with torch.enable_grad():
            zr = z.detach().requires_grad_(True)
            lp = self.target.log_prob(zr)
            g = torch.autograd.grad(lp.nan_to_num(neginf=0.0).sum(), zr)[0]
        return lp.detach(), g.detach()

    def collapsed_vg(self, etas: torch.Tensor, delta: float = 0.5):
        r"""Exact ``(log p(y, eta), d log p / d eta)`` -- first-order only.

        Fisher's identity with the Gaussian inner expectation in closed form:
        :math:`\nabla_\eta \log p(y,\eta) = \nabla_\eta \ell(\eta, m)
        - \tfrac12\,\mathrm{tr}(\Lambda^{-1}\,\partial_\eta\Lambda)`, the
        first term by the envelope theorem (``m`` held fixed; exact because
        :math:`\nabla_x \ell(\eta, m) = 0`), the second as the
        :math:`\eta`-gradient of :math:`\langle\bar\Sigma, \Lambda(\eta)
        \rangle` with :math:`\bar\Sigma = \Lambda^{-1}` frozen, contracted
        through the same coloring as the probes and converted to log-density
        *values* by one-sided secant pairs whose quadratic bias cancels
        exactly. Two batched target calls per evaluation, no second-order
        autograd -- the double-backward alternative differentiates through
        probes, Cholesky, and solves, and costs ${\sim}2.5\times$ more wall
        clock on the DBN. Dense backend only (the banded path keeps double
        backward for now); rows whose assembly is non-finite or non-PD
        return ``(-1e30, 0)`` exactly like ``collapsed_log_post``.
        """
        assert not self.banded, "exact gradient path is dense-backend only"
        etas = etas.detach()
        K, C = etas.shape[0], self.n_colors
        # ---- call 1: probes {0, v_c}; full-z gradients ----
        X1 = torch.cat([torch.zeros(1, K, self.d_g, dtype=DTYPE),
                        self.V.unsqueeze(1).expand(-1, K, -1)])
        lp1, g1 = self._full_rows(etas.repeat(1 + C, 1),
                                  X1.reshape((1 + C) * K, self.d_g))
        lp1 = lp1.reshape(1 + C, K)
        gx = g1[:, self.g_idx].reshape(1 + C, K, self.d_g)
        ge = g1[:, self.h_idx].reshape(1 + C, K, self.d_h)
        bad = ~(lp1.isfinite().all(0)) | ~(g1.isfinite().all(-1)
                                           .reshape(1 + C, K).all(0))
        g0 = gx[0].nan_to_num()
        R = (gx[0].unsqueeze(0) - gx[1:]).nan_to_num()       # [C, K, d_g]
        lam = self._lam_flat(R)
        chol, bad_c = _safe_cholesky(lam)
        bad = bad | bad_c
        half_logdet = torch.log(
            torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        # ---- Sigma-bar = Lambda^{-1} through one triangular inverse: the
        # trace term needs its pattern entries and m = Sigma g0 comes free
        # (cholesky_inverse threads poorly; W^T W is a BLAS matmul) ----
        if not hasattr(self, "_eye_dg"):
            self._eye_dg = torch.eye(self.d_g, dtype=DTYPE)
        Wl = torch.linalg.solve_triangular(
            chol, self._eye_dg.expand(K, -1, -1), upper=False)
        sigma = Wl.transpose(-1, -2) @ Wl                    # [K, d_g, d_g]
        m = (sigma @ g0.unsqueeze(-1)).squeeze(-1)
        ii, jj, dst = self._trace_maps()
        u = torch.zeros(K, C * self.d_g, dtype=DTYPE).index_add_(
            1, dst, sigma[:, ii, jj]).reshape(K, C, self.d_g)
        nu = u.norm(dim=-1)                                  # [K, C]
        uhat = u / nu.clamp_min(1e-300).unsqueeze(-1)
        uhat = uhat.transpose(0, 1)                          # [C, K, d_g]
        # ---- call 2: {m, delta*uhat_c, v_c + delta*uhat_c}; one backward ----
        X2 = torch.cat([m.nan_to_num().unsqueeze(0), delta * uhat,
                        self.V.unsqueeze(1) + delta * uhat])
        B2 = (1 + 2 * C) * K
        z2 = torch.zeros(B2, self.target.dim, dtype=DTYPE)
        z2[:, self.h_idx] = etas.repeat(1 + 2 * C, 1)
        z2[:, self.g_idx] = X2.reshape(B2, self.d_g)
        w = torch.cat([torch.ones(1, K, dtype=DTYPE),        # the m rows
                       -nu.T / (2 * delta), nu.T / (2 * delta)])
        with torch.enable_grad():
            zr = z2.detach().requires_grad_(True)
            lp2 = self.target.log_prob(zr)
            s = (w.reshape(-1) * lp2.nan_to_num(neginf=0.0)).sum()
            g2 = torch.autograd.grad(s, zr)[0]
        lp2 = lp2.detach().reshape(1 + 2 * C, K)
        bad = bad | ~lp2[0].isfinite() | ~lp2.isfinite().all(0)
        grad = g2.detach()[:, self.h_idx].reshape(1 + 2 * C, K,
                                                  self.d_h).sum(0)
        # stored call-1 rows complete the one-sided pairs of the trace term
        grad = grad + ((nu.T / (2 * delta)).unsqueeze(-1)
                       * (ge[0].unsqueeze(0) - ge[1:])).sum(0)
        lp = lp2[0] + 0.5 * self.d_g * math.log(2 * math.pi) - half_logdet
        bad = bad | ~lp.isfinite() | ~grad.isfinite().all(-1)
        lp = torch.where(bad, torch.tensor(-1e30, dtype=DTYPE), lp)
        grad = torch.where(bad.unsqueeze(-1), torch.zeros_like(grad), grad)
        return lp, grad

    def grad_check(self, etas: torch.Tensor) -> float:
        """Max relative error of ``collapsed_vg`` against double backward."""
        lp, g = self.collapsed_vg(etas)
        e = etas.detach().requires_grad_(True)
        lp2 = self.collapsed_log_post(e)
        g2 = torch.autograd.grad(lp2.sum(), e)[0]
        scale = g2.abs().max(-1).values.clamp_min(1e-12)
        val_err = float((lp - lp2.detach()).abs().max())
        grad_err = float(((g - g2) .abs().max(-1).values / scale).max())
        return max(val_err, grad_err)

    def conditional_sample(self, etas: torch.Tensor,
                           gen: Optional[torch.Generator] = None,
                           eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Exact draws x ~ N(m(eta), Lambda(eta)^{-1}), one per eta row.

        ``eps`` (standard-normal, [K, d_g]) may be supplied for reproducible
        or antithetic sampling; the map is x = m + L^{-T} eps with
        Lambda = L L^T in the backend's own factor order, so dense and banded
        backends realize different (equally exact) transports of the same
        noise.
        """
        K = etas.shape[0]
        if eps is None:
            eps = torch.randn(K, self.d_g, dtype=DTYPE, generator=gen)
        if self.banded:
            _, m, _, _, Cs, Fs, _ = self._bt_mean_logdet(etas)
            ep = torch.zeros(K, self.NB * self.nb, dtype=DTYPE)
            ep[:, :self.d_g] = eps[:, self.perm]
            xp = self._bt_solve_Lt(Cs, Fs, ep.reshape(K, self.NB, self.nb))
            return m + xp.reshape(K, -1)[:, :self.d_g].index_select(1, self.pinv)
        m, _, chol = self.conditional_moments(etas)
        return m + torch.linalg.solve_triangular(
            chol.transpose(-1, -2), eps.unsqueeze(-1), upper=True).squeeze(-1)

    def joint_identity_residual(self, etas: torch.Tensor,
                                gen: Optional[torch.Generator] = None) -> float:
        """max | log p(y,eta,x) - log p(y,eta) - log N(x; m, Lambda^{-1}) |."""
        K = etas.shape[0]
        x = torch.randn(K, self.d_g, dtype=DTYPE, generator=gen)
        if self.banded:
            lp0, m, half_logdet, blocks, _, _, bad = self._bt_mean_logdet(etas)
            dev = torch.zeros(K, self.NB * self.nb, dtype=DTYPE)
            dev[:, :self.d_g] = (x - m)[:, self.perm]
            dev = dev.reshape(K, self.NB, self.nb)
            quad = (dev * self._bt_matvec(blocks, dev)).sum((-2, -1))
            collapsed = self._laplace_at_mean(etas, m, half_logdet, bad)
        else:
            m, lam, chol = self.conditional_moments(etas)
            dev = (x - m).unsqueeze(-1)
            quad = (dev * (lam @ dev)).sum((-2, -1))
            half_logdet = torch.log(
                torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
            collapsed = self.collapsed_log_post(etas)
        z = torch.zeros(K, self.target.dim, dtype=DTYPE)
        z[:, self.h_idx] = etas
        z[:, self.g_idx] = x
        lp_joint = self.target.log_prob(z)
        lp_cond = (-0.5 * self.d_g * math.log(2 * math.pi)
                   + half_logdet - 0.5 * quad)
        return float((lp_joint - collapsed - lp_cond).abs().max())


@dataclass
class SparseBlockEvaluation:
    """Reusable sparse factors and conditional means at a batch of cores."""

    factors: List[SparsePrecisionFactor]
    means: np.ndarray
    assembly_seconds: float

    @property
    def factor_seconds(self) -> float:
        return sum(factor.factor_seconds for factor in self.factors)


class SparseGaussianBlock:
    r"""General-graph sparse execution for a discovered :class:`GaussianBlock`.

    The block's colored first-order probes assemble only detected precision
    entries.  A cached symmetric minimum-degree ordering then supplies exact
    sparse LDL factors for solves, log determinants, conditional draws, and
    the fiberwise transport. The exact first-order collapsed score uses
    selected inversion and is exposed to samplers through the marginalized
    target's custom autograd bridge.
    """

    def __init__(self, block: GaussianBlock):
        self.block = block
        self.target = block.target
        self.g_idx, self.h_idx = block.g_idx, block.h_idx
        self.d_g, self.d_h = block.d_g, block.d_h
        pattern = block.pattern.detach().cpu().numpy()
        self.backend = SparsePrecisionBackend(pattern)
        ii, jj = np.nonzero(pattern)
        self._ii, self._jj = ii, jj
        self._entry_colors = block.colors[jj].detach().cpu().numpy()

    @property
    def n_colors(self) -> int:
        return self.block.n_colors

    @property
    def bandwidth(self) -> int:
        return self.block.bandwidth

    @property
    def banded(self) -> bool:
        return False

    def _evaluation_from_arrays(self, responses, g0, verify=False):
        from scipy import sparse

        started = time.perf_counter()
        response_np = np.asarray(responses)
        g0_np = np.asarray(g0)
        precisions = []
        for k in range(g0_np.shape[0]):
            values = response_np[self._entry_colors, k, self._ii]
            precision = sparse.coo_matrix(
                (values, (self._ii, self._jj)), shape=(self.d_g, self.d_g)
            ).tocsc()
            precisions.append(0.5 * (precision + precision.T))
        assembly_seconds = time.perf_counter() - started
        factors = [
            self.backend.factor(precision, verify=verify) for precision in precisions
        ]
        means = np.stack(
            [factor.solve(g0_np[k]) for k, factor in enumerate(factors)]
        )
        return SparseBlockEvaluation(factors, means, assembly_seconds)

    def evaluate(self, etas: torch.Tensor, verify: bool = False) -> SparseBlockEvaluation:
        """Assemble and factor one sparse precision per core row."""
        if etas.requires_grad:
            raise ValueError("sparse value backend does not support autograd cores")
        if etas.ndim != 2 or etas.shape[1] != self.d_h:
            raise ValueError(f"etas must have shape [K, {self.d_h}]")
        started = time.perf_counter()
        _, g0, responses = self.block._probe(etas.detach())
        probe_seconds = time.perf_counter() - started
        evaluation = self._evaluation_from_arrays(
            responses.detach().cpu().numpy(), g0.detach().cpu().numpy(), verify=verify
        )
        evaluation.assembly_seconds += probe_seconds
        return evaluation

    def collapsed_log_post(
        self, etas: torch.Tensor, evaluation: Optional[SparseBlockEvaluation] = None
    ) -> torch.Tensor:
        evaluation = evaluation or self.evaluate(etas)
        z = torch.zeros(etas.shape[0], self.target.dim, dtype=DTYPE)
        z[:, self.h_idx] = etas.detach()
        z[:, self.g_idx] = torch.as_tensor(evaluation.means, dtype=DTYPE)
        with torch.no_grad():
            lp_m = self.target.log_prob(z)
        half_logdet = torch.tensor(
            [factor.half_logdet for factor in evaluation.factors], dtype=DTYPE
        )
        return lp_m + 0.5 * self.d_g * math.log(2 * math.pi) - half_logdet

    def forward(
        self, etas: torch.Tensor, x: torch.Tensor,
        evaluation: Optional[SparseBlockEvaluation] = None,
    ) -> torch.Tensor:
        evaluation = evaluation or self.evaluate(etas)
        x_np = x.detach().cpu().numpy()
        out = np.stack(
            [factor.forward(x_np[k], evaluation.means[k])
             for k, factor in enumerate(evaluation.factors)]
        )
        return torch.as_tensor(out, dtype=DTYPE)

    def inverse(
        self, etas: torch.Tensor, z: torch.Tensor,
        evaluation: Optional[SparseBlockEvaluation] = None,
    ) -> torch.Tensor:
        evaluation = evaluation or self.evaluate(etas)
        z_np = z.detach().cpu().numpy()
        out = np.stack(
            [factor.inverse(z_np[k], evaluation.means[k])
             for k, factor in enumerate(evaluation.factors)]
        )
        return torch.as_tensor(out, dtype=DTYPE)

    def conditional_sample(
        self, etas: torch.Tensor, gen: Optional[torch.Generator] = None,
        eps: Optional[torch.Tensor] = None,
        evaluation: Optional[SparseBlockEvaluation] = None,
    ) -> torch.Tensor:
        if eps is None:
            eps = torch.randn(etas.shape[0], self.d_g, dtype=DTYPE, generator=gen)
        return self.inverse(etas, eps, evaluation=evaluation)

    def joint_identity_residual(
        self, etas: torch.Tensor, gen: Optional[torch.Generator] = None
    ) -> float:
        evaluation = self.evaluate(etas)
        x = torch.randn(etas.shape[0], self.d_g, dtype=DTYPE, generator=gen)
        z = torch.zeros(etas.shape[0], self.target.dim, dtype=DTYPE)
        z[:, self.h_idx] = etas
        z[:, self.g_idx] = x
        lp_joint = self.target.log_prob(z)
        collapsed = self.collapsed_log_post(etas, evaluation=evaluation)
        dev = x.detach().cpu().numpy() - evaluation.means
        quad = torch.tensor(
            [factor.quadratic(dev[k]) for k, factor in enumerate(evaluation.factors)],
            dtype=DTYPE,
        )
        half_logdet = torch.tensor(
            [factor.half_logdet for factor in evaluation.factors], dtype=DTYPE
        )
        lp_cond = (
            -0.5 * self.d_g * math.log(2 * math.pi) + half_logdet - 0.5 * quad
        )
        return float((lp_joint - collapsed - lp_cond).abs().max())

    def collapsed_vg(self, etas: torch.Tensor, delta: float = 0.5):
        r"""Exact collapsed value and core gradient using sparse selected inversion.

        This is the sparse analogue of :meth:`GaussianBlock.collapsed_vg`:
        Fisher's identity contracts only inverse entries on the detected
        precision pattern, supplied in fill-linear memory by the selected-
        inverse recursion.  The target is queried only through first-order
        value-and-gradient calls.
        """
        etas = etas.detach()
        K, C = etas.shape[0], self.block.n_colors
        X1 = torch.cat([
            torch.zeros(1, K, self.d_g, dtype=DTYPE),
            self.block.V.unsqueeze(1).expand(-1, K, -1),
        ])
        lp1, g1 = self.block._full_rows(
            etas.repeat(1 + C, 1), X1.reshape((1 + C) * K, self.d_g)
        )
        lp1 = lp1.reshape(1 + C, K)
        gx = g1[:, self.g_idx].reshape(1 + C, K, self.d_g)
        ge = g1[:, self.h_idx].reshape(1 + C, K, self.d_h)
        bad = ~(lp1.isfinite().all(0)) | ~(
            g1.isfinite().all(-1).reshape(1 + C, K).all(0)
        )
        g0 = gx[0].nan_to_num()
        responses = (gx[0].unsqueeze(0) - gx[1:]).nan_to_num()
        evaluation = self._evaluation_from_arrays(
            responses.cpu().numpy(), g0.cpu().numpy()
        )
        means = torch.as_tensor(evaluation.means, dtype=DTYPE)
        half_logdet = torch.tensor(
            [factor.half_logdet for factor in evaluation.factors], dtype=DTYPE
        )

        inverse_entries = torch.as_tensor(
            np.stack([
                factor.selected_inverse(self._ii, self._jj)
                for factor in evaluation.factors
            ]),
            dtype=DTYPE,
        )
        _, _, destinations = self.block._trace_maps()
        u = torch.zeros(K, C * self.d_g, dtype=DTYPE).index_add_(
            1, destinations, inverse_entries
        ).reshape(K, C, self.d_g)
        norms = u.norm(dim=-1)
        directions = (u / norms.clamp_min(1e-300).unsqueeze(-1)).transpose(0, 1)

        X2 = torch.cat([
            means.unsqueeze(0), delta * directions,
            self.block.V.unsqueeze(1) + delta * directions,
        ])
        rows = (1 + 2 * C) * K
        z2 = torch.zeros(rows, self.target.dim, dtype=DTYPE)
        z2[:, self.h_idx] = etas.repeat(1 + 2 * C, 1)
        z2[:, self.g_idx] = X2.reshape(rows, self.d_g)
        weights = torch.cat([
            torch.ones(1, K, dtype=DTYPE),
            -norms.T / (2 * delta), norms.T / (2 * delta),
        ])
        with torch.enable_grad():
            zr = z2.detach().requires_grad_(True)
            lp2 = self.target.log_prob(zr)
            weighted = (weights.reshape(-1) * lp2.nan_to_num(neginf=0.0)).sum()
            g2 = torch.autograd.grad(weighted, zr)[0]
        lp2 = lp2.detach().reshape(1 + 2 * C, K)
        bad = bad | ~lp2[0].isfinite() | ~lp2.isfinite().all(0)
        grad = g2.detach()[:, self.h_idx].reshape(
            1 + 2 * C, K, self.d_h
        ).sum(0)
        grad = grad + (
            (norms.T / (2 * delta)).unsqueeze(-1)
            * (ge[0].unsqueeze(0) - ge[1:])
        ).sum(0)
        lp = lp2[0] + 0.5 * self.d_g * math.log(2 * math.pi) - half_logdet
        bad = bad | ~lp.isfinite() | ~grad.isfinite().all(-1)
        lp = torch.where(bad, torch.tensor(-1e30, dtype=DTYPE), lp)
        grad = torch.where(bad.unsqueeze(-1), torch.zeros_like(grad), grad)
        return lp, grad


# --------------------------------------------------------------------------- #
# block discovery: the indices themselves, from log_prob alone
# --------------------------------------------------------------------------- #
@dataclass
class BlockDiscovery:
    """Result of :func:`discover_gaussian_block`.

    ``g_idx`` is the discovered conditionally Gaussian block;
    ``self_conflict`` marks coordinates the log density is not quadratic in
    even along their own axis (log-scales, lengthscales, non-Gaussian
    priors); ``peeled`` the coordinates removed by the conflict-graph peel
    (in order); ``joint_removed`` any further removals forced by the exact
    joint-quadraticity residual. ``residual`` keeps the normalized secant
    residual matrix for reporting; ``margins`` summarizes the separation
    between accepted (near machine precision) and rejected axes.
    """
    g_idx: torch.Tensor
    h_idx: torch.Tensor
    self_conflict: torch.Tensor
    conflict: torch.Tensor
    residual: torch.Tensor
    peeled: List[int]
    joint_removed: List[int]
    margins: dict
    base_points: torch.Tensor

    @property
    def probe_etas(self) -> torch.Tensor:
        """Base points restricted to the collapsed coordinates -- pass these
        as ``probe_etas`` to :class:`GaussianBlock` so pattern detection sees
        the same (plausible) region discovery certified."""
        return self.base_points[:, self.h_idx]


def _full_grad_rows(target: TargetPosterior, Z: torch.Tensor,
                    chunk: int = 1024) -> torch.Tensor:
    """Gradients of log_prob over the full space, chunked and detached."""
    outs = []
    for lo in range(0, Z.shape[0], chunk):
        with torch.enable_grad():
            z = Z[lo:lo + chunk].detach().requires_grad_(True)
            lp = target.log_prob(z)
            g = torch.autograd.grad(lp.sum(), z)[0]
        outs.append(g.detach())
    return torch.cat(outs)


def discover_gaussian_block(target: TargetPosterior,
                            probe_points: Optional[torch.Tensor] = None,
                            n_base: int = 2,
                            delta_pairs=((0.3, 0.9), (0.7, 2.1)),
                            probe_scale: float = 0.5, tol: float = 1e-8,
                            n_joint: int = 4, max_joint_peel: int = 32,
                            seed: int = 0, chunk: int = 1024) -> BlockDiscovery:
    r"""Find a conditionally Gaussian block from ``log_prob`` alone.

    Vanishing every third derivative
    :math:`\partial^3 \ell / \partial\theta_i \partial\theta_j
    \partial\theta_k` with i, j, k all in G characterizes quadraticity in the
    block. This routine discovers that algebraic candidate; a valid
    conditionally Gaussian block additionally requires the recovered precision
    to be positive definite wherever the conditional is defined. Collapse
    checks positive definiteness and the Gaussian identity at posterior-
    plausible points. Discovery has three stages, all built on the fact that a
    quadratic's secant slopes are exact at any step size:

    1. *Axis probes.* At each base point compare the secant slopes
       ``[g(theta + d e_k) - g(theta)] / d`` at two step sizes. A difference
       in component i witnesses the hyperedge {i, k, k}; the normalized
       residual matrix yields a conflict graph (edge i--k) whose diagonal
       marks coordinates that are non-quadratic along their own axis.
    2. *Greedy peel.* Drop self-conflicted coordinates, then remove
       max-degree nodes of the conflict graph until it is edge-free (on a
       state-space model this removes the transition-matrix entries, whose
       degree scales with T, and keeps the states).
    3. *Joint certification.* Purely trilinear hyperedges {i, j, k} with all
       indices distinct escape axis probes, so verify with the exact
       double difference ``g(t+u+v) - g(t+v) - g(t+u) + g(t)`` for random
       u, v supported on G -- identically zero iff jointly quadratic --
       peeling the argmax offender until clean.

    Like the pattern probes of :class:`GaussianBlock`, detection is
    certified *where you look*: supply ``probe_points`` near posterior-
    plausible values when curvature magnitudes are strongly
    parameter-dependent. ``tol`` is a relative threshold; the default 1e-8
    sits ~5 orders above float64 secant roundoff and catches priors that are
    only softly non-Gaussian (a wide Student-t is non-quadratic at the 1e-7
    level -- excluded, keeping the collapse exact rather than approximate).
    """
    D = target.dim
    gen = torch.Generator().manual_seed(seed)
    if probe_points is not None:
        base = probe_points.to(DTYPE)
    else:
        pts = []
        for _ in range(50 * n_base):
            if len(pts) == n_base:
                break
            th = probe_scale * torch.randn(D, dtype=DTYPE, generator=gen)
            with torch.no_grad():
                lp = target.log_prob(th.unsqueeze(0))
            if bool(lp.isfinite().all()) and bool(
                    _full_grad_rows(target, th.unsqueeze(0), chunk).isfinite().all()):
                pts.append(th)
        if len(pts) < n_base:
            raise RuntimeError("could not find finite base points; "
                               "supply probe_points")
        base = torch.stack(pts)

    # stage 1: axis-affinity residuals -> conflict graph. Axis probes stream
    # in chunks of ``chunk // 2`` axes, so the working set beyond the three
    # [D, D] accumulators (the residual matrix and, per base/step pair, the
    # slope difference and slope magnitude) is O(chunk * D) -- the pass never
    # materializes eye(D) or the [2D, D] probe batch, which is what lets
    # discovery run at the sizes the banded backend targets.
    Rn = torch.zeros(D, D, dtype=DTYPE)
    big = torch.tensor(float("inf"), dtype=DTYPE)
    A = torch.empty(D, D, dtype=DTYPE)       # |h2 - h1|          [axis, comp]
    Hm = torch.empty(D, D, dtype=DTYPE)      # max(|h1|, |h2|)    [axis, comp]
    fin = torch.empty(D, dtype=torch.bool)   # axis stayed finite
    c = max(1, chunk // 2)
    for theta in base:
        g0 = _full_grad_rows(target, theta.unsqueeze(0), chunk)[0]
        for d1, d2 in delta_pairs:
            for lo in range(0, D, c):
                ax = torch.arange(lo, min(lo + c, D))
                r = torch.arange(len(ax))
                P = theta.unsqueeze(0).repeat(2 * len(ax), 1)
                P[r, ax] += d1
                P[r + len(ax), ax] += d2
                G = _full_grad_rows(target, P, chunk)
                h1 = (G[:len(ax)] - g0) / d1     # [axis k, comp i]
                h2 = (G[len(ax):] - g0) / d2
                A[ax] = (h2 - h1).abs()
                Hm[ax] = torch.maximum(h1.abs(), h2.abs())
                fin[ax] = h1.isfinite().all(-1) & h2.isfinite().all(-1)
            gmax = torch.where(Hm.isfinite(), Hm,
                               torch.zeros((), dtype=DTYPE)).max()
            for lo in range(0, D, c):
                hi = min(lo + c, D)
                dg = Hm[lo:hi, lo:hi].diagonal().unsqueeze(-1)
                floor = torch.clamp(1e-3 * dg, min=1e-9 * gmax)
                rn = (A[lo:hi] / torch.maximum(Hm[lo:hi], floor)).nan_to_num(
                    nan=float("inf"), posinf=float("inf"))
                rn = torch.where(fin[lo:hi].unsqueeze(-1), rn, big)
                Rn[lo:hi] = torch.maximum(Rn[lo:hi], rn)
    del A, Hm
    conflict = (Rn > tol)
    conflict = conflict | conflict.T
    self_conf = conflict.diagonal().clone()

    # stage 2: greedy max-degree peel to an independent set
    cand = ~self_conf
    adj = conflict & cand.unsqueeze(0) & cand.unsqueeze(1)
    adj.fill_diagonal_(False)
    peeled: List[int] = []
    while True:
        deg = adj.sum(1)
        if int(deg.max()) == 0:
            break
        k = int(deg.argmax())
        peeled.append(k)
        cand[k] = False
        adj[k, :] = False
        adj[:, k] = False

    # stage 3: joint quadraticity certification on span(G)
    joint_removed: List[int] = []
    joint_worst = 0.0
    for _ in range(max_joint_peel + 1):
        worst_val, worst_i = 0.0, -1
        for theta in base:
            for _ in range(n_joint):
                u = 0.5 * torch.randn(D, dtype=DTYPE, generator=gen) * cand
                v = 0.5 * torch.randn(D, dtype=DTYPE, generator=gen) * cand
                G = _full_grad_rows(target, torch.stack(
                    [theta, theta + u, theta + v, theta + u + v]), chunk)
                resid = (G[3] - G[2] - G[1] + G[0]).abs()
                su, sv = (G[1] - G[0]).abs(), (G[2] - G[0]).abs()
                scale = torch.maximum(torch.maximum(su, sv),
                                      1e-3 * torch.maximum(su.max(), sv.max()))
                rn = torch.where(cand, (resid / scale).nan_to_num(
                    nan=float("inf")), torch.zeros((), dtype=DTYPE))
                if float(rn.max()) > worst_val:
                    worst_val, worst_i = float(rn.max()), int(rn.argmax())
        joint_worst = worst_val
        if worst_val <= tol:
            break
        cand[worst_i] = False
        joint_removed.append(worst_i)

    g_idx = torch.nonzero(cand).squeeze(-1)
    h_idx = torch.nonzero(~cand).squeeze(-1)
    # margins over the symmetrized residual (an edge is witnessed by either
    # orientation), extracted in row chunks so no second [D, D] is built
    kept_max, conf_min = 0.0, float("inf")
    for lo in range(0, D, c):
        hi = min(lo + c, D)
        Rs = torch.maximum(Rn[lo:hi], Rn[:, lo:hi].T)
        sub = Rs[cand[lo:hi]][:, cand]
        if sub.numel():
            kept_max = max(kept_max, float(sub.max()))
        flagged = Rs[conflict[lo:hi]]
        if flagged.numel():
            conf_min = min(conf_min, float(flagged.min()))
    margins = {
        "max_kept_axis_residual": kept_max,
        "min_conflict_residual": conf_min,
        "final_joint_residual": joint_worst,
    }
    return BlockDiscovery(g_idx=g_idx, h_idx=h_idx, self_conflict=self_conf,
                          conflict=conflict, residual=Rn, peeled=peeled,
                          joint_removed=joint_removed, margins=margins,
                          base_points=base)


def cost_aware_shrink(block: "GaussianBlock", max_moves: int = 32,
                      small_block: int = 16, min_factor: float = 2.0):
    r"""Cost-aware refinement of a discovered block: peel pattern hubs.

    The maximal conditionally Gaussian block is not always the cheapest one
    to collapse *recurrently*: a few hub coordinates (fixed effects, global
    regressors) can couple every other block coordinate, so the distance-2
    coloring -- and with it the probe cost of every collapsed evaluation --
    degrades to O(d_g) even though the rest of the block is sparse. Moving a
    hub into the core surrenders one exactly-integrated coordinate and buys
    whatever the coloring loses.

    Because the precision of a sub-block is a principal submatrix of
    :math:`\Lambda`, its sparsity pattern is the corresponding submatrix of
    the pattern already detected at construction: the shrink decision is
    graph work on that one detection, with no further model evaluations.
    The report carries the chosen subpattern (``report["pattern"]``) so the
    shrunk block can be rebuilt with ``GaussianBlock(..., pattern=...)``,
    skipping re-detection; the rebuild then pays only its usual one-probe
    self-verification.

    Greedily peel the maximum-degree vertex of the pattern graph, re-color,
    and score each prefix of moves by ``n_colors(after j moves) + j``: one
    batched gradient pass per color and one core dimension per surrendered
    coordinate, charged equally. Under this count the degenerate peel of a
    dense block (one color per coordinate -- a regression Gram) is
    cost-neutral and rejected; only hubs, whose removal collapses the
    coloring by more than themselves, can win. Among minimal-cost prefixes
    the shortest wins: at equal charged cost, keep more of the block
    exactly integrated. Two guards keep the refinement conservative:

    * blocks with ``d_g <= small_block`` are returned unchanged -- the cost
      rule marginalizes them at any coloring, and surrendering coordinates
      there trades exact integration for nothing (the crossed pilots
      block);
    * the winning prefix is accepted only if it at least halves the score
      (``min_factor``). The pathology this refinement targets is
      hub-degenerate coloring, whose wins are order-of-magnitude; a
      marginal saving does not justify giving up exactly-integrated
      coordinates (the 13-to-4-color peel on mcycle's HSGP basis is the
      measured example this rejects).

    The refinement only concerns the recurring emissions (marginalize,
    precondition). The quadrature branch evaluates all grid nodes in one
    batch, so there colors are nearly free while every core dimension is
    exponential -- it keeps the maximal block, which is why the enlarged
    eight-schools block (a 9-color hub-and-spokes with a one-dimensional
    core) must *not* be peeled.

    Returns ``(g_idx, report)``: the refined block in original coordinates
    (the input ``block.g_idx`` when no prefix wins) and a dict recording the
    peel order, the color path, and the accepted moves.
    """
    d_g = block.d_g
    report = {
        "n_moves": 0, "moved": [], "peel_order": [],
        "colors_path": [block.n_colors], "colors_before": block.n_colors,
        "colors_after": block.n_colors, "pattern": block.pattern,
    }
    if d_g <= small_block:
        return block.g_idx, report
    adj = block.pattern.clone()
    adj.fill_diagonal_(False)
    alive = torch.ones(d_g, dtype=torch.bool)
    colors_path = [block.n_colors]
    peel: List[int] = []
    for _ in range(min(max_moves, d_g - 1)):
        deg = (adj & alive.unsqueeze(0) & alive.unsqueeze(1)).sum(1)
        deg[~alive] = -1
        k = int(deg.argmax())
        if int(deg[k]) <= 0:
            break
        alive[k] = False
        peel.append(k)
        _, n_c = _cpr_coloring(block.pattern[alive][:, alive])
        colors_path.append(n_c)
    cost = [c + j for j, c in enumerate(colors_path)]
    j_star = int(np.argmin(cost))          # ties resolve to the fewest moves
    if cost[j_star] * min_factor > cost[0]:
        j_star = 0
    keep = torch.ones(d_g, dtype=torch.bool)
    if j_star:
        keep[peel[:j_star]] = False
    report.update(
        n_moves=j_star,
        moved=[int(block.g_idx[k]) for k in peel[:j_star]],
        peel_order=[int(block.g_idx[k]) for k in peel],
        colors_path=colors_path,
        colors_after=colors_path[j_star],
        pattern=block.pattern[keep][:, keep],
    )
    return block.g_idx[keep], report


def _cpr_coloring(pattern: torch.Tensor):
    """Greedy distance-2 (Curtis-Powell-Reid) coloring of Hessian columns.

    Columns i, j may share a probe vector iff no row has structural nonzeros
    in both. The conflict graph is built row-by-row from the sparsity lists
    (cost sum_r nnz_r^2, not the d^3 of a dense P^T P), so coloring stays
    cheap for the long chains the banded backend targets. Returns
    (colors [d], n_colors).
    """
    P = pattern.detach().cpu().numpy()
    d = P.shape[0]
    adj: List[set] = [set() for _ in range(d)]
    for r in range(d):
        nz = np.flatnonzero(P[r])
        for a in nz:
            adj[a].update(nz.tolist())
    colors = np.full(d, -1, dtype=np.int64)
    for i in range(d):
        used = {colors[j] for j in adj[i] if j < i}
        c = 0
        while c in used:
            c += 1
        colors[i] = c
    return torch.as_tensor(colors), int(colors.max()) + 1


def _rcm(pattern: torch.Tensor):
    """Reverse Cuthill-McKee ordering of a symmetric sparsity pattern.

    Returns (perm, pinv, half_bandwidth): position p of the reordered matrix
    holds original index perm[p], and every structural nonzero (i, j)
    satisfies |pinv[i] - pinv[j]| <= half_bandwidth.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    P = pattern.detach().cpu().numpy()
    perm = np.asarray(reverse_cuthill_mckee(csr_matrix(P), symmetric_mode=True),
                      dtype=np.int64)
    pinv = np.empty_like(perm)
    pinv[perm] = np.arange(len(perm))
    i, j = np.nonzero(P)
    bw = int(np.abs(pinv[i] - pinv[j]).max()) if len(i) else 0
    return torch.as_tensor(perm), torch.as_tensor(pinv), bw


def _safe_cholesky(lam: torch.Tensor):
    """Batched Cholesky that flags (rather than raises on) bad rows."""
    d = lam.shape[-1]
    eye = torch.eye(d, dtype=DTYPE)
    finite = lam.isfinite().all((-2, -1))
    lam_safe = torch.where(finite.unsqueeze(-1).unsqueeze(-1), lam, eye)
    chol, info = torch.linalg.cholesky_ex(lam_safe)
    bad = ~finite | (info > 0)
    chol = torch.where(bad.unsqueeze(-1).unsqueeze(-1), eye, chol)
    return chol, bad


def rb_joint_covariance(eta_draws: torch.Tensor, m_draws: torch.Tensor,
                        cond_cov_mean: torch.Tensor, h_idx: torch.Tensor,
                        g_idx: torch.Tensor) -> torch.Tensor:
    """Marginal joint covariance by the law of total covariance.

    ``eta_draws`` [K, H] are draws from the collapsed posterior, ``m_draws``
    [K, G] the conditional means of the Gaussian block at those draws, and
    ``cond_cov_mean`` [G, G] the average conditional covariance E[Cov(x|eta)].
    ``h_idx``/``g_idx`` scatter the two blocks into the model's coordinate
    order (the Gaussian block need not be contiguous or trailing). Returns the
    full [D, D] covariance:

        Cov(eta)               on the hyper block,
        Cov(eta, m(eta))       on the cross block,
        Cov(m(eta)) + E[Cov]   on the Gaussian block.
    """
    K, dim = eta_draws.shape[0], len(h_idx) + len(g_idx)
    J = torch.zeros(K, dim, dtype=DTYPE)
    J[:, h_idx] = eta_draws
    J[:, g_idx] = m_draws
    sigma = torch.cov(J.T)
    sigma[g_idx.unsqueeze(-1), g_idx.unsqueeze(0)] += cond_cov_mean
    return 0.5 * (sigma + sigma.T)


def whitening_cholesky(sigma: torch.Tensor, rel_floor: float = 1e-10) -> torch.Tensor:
    """Cholesky factor L of the precision ``M = sigma^{-1}`` (whitening map).

    Inverts through an eigendecomposition with a relative eigenvalue floor, so
    a covariance assembled from Monte-Carlo moments (possibly near-singular in
    a few directions) still yields a valid metric.
    """
    sigma = 0.5 * (sigma + sigma.T)
    evals, evecs = torch.linalg.eigh(sigma)
    evals = evals.clamp(min=rel_floor * float(evals.max()))
    M = (evecs / evals) @ evecs.T
    return torch.linalg.cholesky(0.5 * (M + M.T))


def demc(log_post, init: torch.Tensor, seed: int, n_burn: int = 3000,
         n_sample: int = 6000, thin: int = 30,
         gamma_scale: float = 1.0) -> Tuple[torch.Tensor, float, float]:
    """Differential-evolution MCMC (ter Braak 2006), two-half update scheme.

    The generic moment engine for step 2 of the recipe: each half of the
    walker ensemble proposes with scaled difference vectors of walkers from
    the *other* half (held fixed), so proposals track curved correlations a
    fixed-covariance random walk stalls on, and the kernel stays a valid
    Metropolis update. One batched collapsed-posterior evaluation per
    half-sweep. Returns (pooled post-burn-in draws, max split R-hat over the
    walker dimension, post-burn-in acceptance rate).

    Twice during burn-in (only), walkers stranded far below the ensemble's
    typical density -- overdispersed initializations on heavy-tailed or
    funnel-shaped posteriors leave a few on plateaus the difference-vector
    proposals crawl off of -- are resurrected onto jittered copies of good
    walkers. Post-burn-in sweeps are untouched Metropolis.
    """
    gen = torch.Generator().manual_seed(seed + 10)
    n, d = init.shape
    half = n // 2
    x = init.clone()
    lp = log_post(x)
    g0 = gamma_scale * 2.38 / math.sqrt(2 * d)
    kept = []
    acc_post = 0.0
    for t in range(n_burn + n_sample):
        if t in (n_burn // 3, 2 * n_burn // 3):
            stuck = lp < lp.median() - 100.0
            if bool(stuck.any()):
                good = torch.nonzero(~stuck).squeeze(-1)
                src = good[torch.randint(len(good), (int(stuck.sum()),),
                                         generator=gen)]
                x[stuck] = x[src] + 1e-3 * torch.randn(
                    int(stuck.sum()), d, dtype=DTYPE, generator=gen)
                lp[stuck] = log_post(x[stuck])
        for lo, hi, olo, ohi in ((0, half, half, n), (half, n, 0, half)):
            xa, other = x[lo:hi], x[olo:ohi]
            j = torch.randint(half, (half,), generator=gen)
            k = torch.randint(half, (half,), generator=gen)
            gam = g0 * (0.5 + torch.rand(half, 1, dtype=DTYPE, generator=gen))
            prop = (xa + gam * (other[j] - other[k])
                    + 1e-4 * torch.randn(half, d, dtype=DTYPE, generator=gen))
            lp_prop = log_post(prop)
            acc = torch.log(torch.rand(half, dtype=DTYPE, generator=gen)) < lp_prop - lp[lo:hi]
            x[lo:hi] = torch.where(acc.unsqueeze(-1), prop, xa)
            lp[lo:hi] = torch.where(acc, lp_prop, lp[lo:hi])
            if t >= n_burn:
                acc_post += float(acc.to(DTYPE).mean()) / 2
        if t >= n_burn and (t - n_burn) % thin == thin - 1:
            kept.append(x.clone())
    draws = torch.stack(kept, dim=1)                      # [n_walkers, K, d]
    K2 = draws.shape[1] // 2
    split = torch.cat([draws[:, :K2], draws[:, K2:2 * K2]], dim=0)
    cm = split.mean(1)
    W = split.var(1, correction=1).mean(0)
    Bv = cm.var(0, correction=1) * K2
    rhat = torch.sqrt(((K2 - 1) / K2 * W + Bv / K2) / W)
    return draws.reshape(-1, d), float(rhat.max()), acc_post / n_sample
