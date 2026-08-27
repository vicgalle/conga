r"""Fisher HMC: the nutpie sampler -- NUTS with Fisher-divergence adaptation.

Port of *Preconditioning Hamiltonian Monte Carlo by minimizing Fisher
Divergence* (Seyboldt, Carlson & Carpenter, 2026; arXiv:2603.18845), the
adaptation scheme behind the ``nutpie`` sampler (reference implementation
``pymc-devs/nutpie`` / ``nuts-rs``, in Rust).

The estimator
-------------
HMC preconditions the target with a mass matrix :math:`M`; every widely used
sampler (Stan, PyMC) estimates :math:`M^{-1}` from the covariance of warmup
draws, which minimizes the KL divergence from the preconditioned target to a
standard normal and conditions only the *large* eigenvalues of the posterior
covariance. The paper instead minimizes the empirical **Fisher divergence**

.. math::
    \widehat D = \tfrac1n \sum_i \|J^\top \alpha^{(i)} + J^{-1}(x^{(i)}-\mu)\|^2,

which uses the *scores* :math:`\alpha^{(i)} = \nabla\log p(x^{(i)})` -- already
computed by every leapfrog step -- alongside the draws, and penalizes large and
small eigenvalues symmetrically. Three closed-form minimizers over affine
transform families :math:`x = J y` (``mass_matrix=``):

* ``"diag"`` (nutpie's default): :math:`\sigma^{2*} =
  \sqrt{\mathrm{var}(x)/\mathrm{var}(\alpha)}` elementwise -- the geometric
  mean of the draw covariance diagonal and the *inverse* score covariance
  diagonal (Theorem 2.2). For a Gaussian this recovers
  :math:`\sqrt{\mathrm{diag}(\Sigma)/\mathrm{diag}(\Sigma^{-1})}`.
* ``"lowrank"``: low-rank-plus-diagonal (Algorithm 1). On top of the diagonal
  transform, draws and scores are projected onto their joint singular basis
  :math:`Q`, the regularized optimality condition
  :math:`\Sigma\,C^\alpha\,\Sigma = C^x` is solved by the symmetric
  positive-definite geometric mean (Algorithm 2), and only eigenpairs with
  :math:`\lambda \ge c` or :math:`\lambda \le 1/c` are kept, giving the O(kd)
  transform :math:`F(y) = D_\sigma(I + U(\mathrm{diag}(\lambda)^{1/2}-I)U^\top)y`.
  (We normalize the projected covariances by :math:`n-1`, so ``gamma``
  regularizes proper covariances; the paper's Algorithm 1 leaves them as raw
  sums of squares.)
* ``"variance"``: the industry-standard draws-only estimator
  :math:`\sigma^{2*} = \mathrm{var}(x)` with Stan's shrinkage toward unit
  scale, under the *identical* NUTS kernel, initialization and schedule. This
  is the built-in ablation: the difference between the ``"variance"`` and
  ``"diag"`` columns isolates exactly the paper's estimator, the analogue of
  its Stan comparison (our reference draws *are* Stan NUTS).

The schedule
------------
The nutpie warmup (Section 3), which adapts much faster than Stan's:

* the mass matrix is initialized from the score of the first draw,
  :math:`M = \mathrm{diag}(|\alpha^{(0)}|)` -- a gradient-outer-product
  Hessian proxy that makes the sampler scale-free (Lemma 3.1) -- instead of
  the identity;
* draws feed the estimate from the very first iteration through a
  foreground/background pair of Welford accumulators: both update every draw,
  and every ``L`` draws the background (holding the last ``L`` draws) replaces
  the foreground and restarts, so the estimation window oscillates between
  ``L`` and ``2L`` and early out-of-equilibrium draws are wiped rather than
  down-weighted. The diagonal estimate refreshes every iteration (from at
  least 3 draws); the low-rank fit, which needs the stored window, refits at
  the wipes;
* warmup phases: the first 30% of iterations use ``L = 10``, the next 55%
  ``L = 80`` (the step size is reinitialized at the phase boundary), and the
  final 15% freeze the mass matrix and only fine-tune the step size, switching
  the dual-averaging statistic from the Metropolis probability
  :math:`e^{\min(0,-\Delta H)}` to the symmetric
  :math:`2e^{\min(0,-\Delta H)}/(1+e^{-\Delta H})`, which penalizes energy
  error of either sign.

The kernel is standard multinomial NUTS (Hoffman & Gelman 2014; Betancourt
2018) as in Stan and ``nuts-rs``: trajectory doubling in a random direction
with biased progressive sampling, generalized U-turn checks on the adapted-
space momentum sums (the iterative checkpoint scheme of NumPyro), a divergence
threshold of :math:`\Delta H > 1000`, and dual-averaging to a 0.8 target
acceptance rate. All simulation happens in the adapted space, where the mass
matrix is the identity (paper Algorithm 3): positions advance by
:math:`x \mathrel{+}= \epsilon\,J v` and scores pull back by
:math:`\delta = J^\top \alpha`, so a mass-matrix update never touches the
chain state.

The coupling chart (``coupling=True``)
--------------------------------------
The affine families above are *static* preconditioners; they cannot fix
position-dependent curvature (funnels, hierarchical centered
parametrizations), which is exactly where their held-out Fisher residual
stays large. Following the
geometric view of the paper's Section B -- any diffeomorphism ``F`` may
precondition HMC, with the leapfrog run in the adapted space and the
potential extended by ``-log|det J_F|`` -- we stack **one affine-conditioner
coupling layer** on top of the affine transform, ``F = C o J``:

.. math::
    x_I = u_I, \qquad x_J = e^{(x_I - m) c^\top} \odot u_J + (x_I - m) d^\top,

a family that contains manual non-centering of hierarchical models as the
special case ``c = 1`` on a log-scale coordinate. The chart is fit during
warmup from the same draw/score window the affine estimators use: driver
coordinates ``I`` are found by *score-scale regression* (for a funnel
``x_j | x_i ~ N(., e^{2c x_i})`` the slope of ``log alpha_j^2`` on ``x_i``
is ``-2c``), activation is gated on the scale signal so Gaussian-enough
targets never pay for a chart, and the coefficients are refit at every
late-phase wipe -- as the chains descend into the funnel the window sees
more of the geometry and the fit sharpens, which is what a one-shot offline
fit cannot do. The affine estimators then see the window *through* the
chart (draws and scores pulled back with the full Jacobian, including the
``grad log|det|`` term), so ``sigma``/low-rank stay closed-form. Everything
is explicit and O(|I| |J|) per leapfrog step: no implicit integrator, no
cubic cost -- the practical middle ground between a mass matrix and
Riemannian HMC.

Fit into the benchmark
----------------------
Unlike every other sampler here (SGLD+R, RLMH, BDLS, MCLMC), Fisher HMC gets
**no mean-field ADVI warmup**: self-preconditioning from its own warmup draws
is the method's entire point, so handing it an external diagonal metric would
erase the object of study. Chains start Stan-style at ``uniform(-2, 2)`` in
the unconstrained space with the crude :math:`\mathrm{diag}(|\alpha^{(0)}|)`
metric and must find both the typical set and the geometry on their own
gradient budget -- which the diagnostics therefore report in full
(initialization + warmup + sampling). An ensemble of ``n_chains`` chains runs
vectorized in lockstep (each chain keeps its own step size, estimators and
U-turn decisions; a chain that stops doubling early just waits for the
others), and the pooled post-warmup draws are the sample that gets scored.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..core import DTYPE, TargetPosterior
from ..diagnostics import pooled_ess
from .base import InferenceAlgorithm, InferenceResult, _make_bar

_NAMES = {"diag": "FisherHMC", "lowrank": "FisherHMC-LR", "variance": "VarHMC"}


class _Coupling:
    r"""One affine-conditioner coupling layer, shared across the chain ensemble.

    The transform ``C`` acts on the output ``u`` of the affine part: driver
    coordinates ``I`` pass through unchanged, every other coordinate ``j`` is
    rescaled and shifted by an amount *affine in the drivers*,

    .. math::
        x_I = u_I, \qquad
        x_J = e^{S(x_I)} \odot u_J + t(x_I), \qquad
        S = (x_I - m)\,c^\top, \quad t = b + (x_I - m)\,d^\top,

    so ``log |det J_C| = sum_j S_j``. This family contains manual
    non-centering of a hierarchical model as the special case ``c = 1`` on
    the log-scale coordinate (e.g. ``theta = mu + e^{log tau} z`` is exactly
    ``c_{theta, log tau} = 1``, ``d_{theta, mu} = 1``). ``S`` is clamped at
    ``|S| <= SMAX`` with a matching derivative mask, so the map stays a
    consistent (piecewise-smooth) diffeomorphism and MH stays exact; the
    clamp only engages on astronomically remote proposals that the energy
    check discards anyway.
    """

    SMAX = 60.0

    def __init__(self, idx_I: torch.Tensor, idx_J: torch.Tensor,
                 m: torch.Tensor, b: torch.Tensor,
                 c: torch.Tensor, d: torch.Tensor):
        self.idx_I = idx_I          # [k]   driver coordinate indices
        self.idx_J = idx_J          # [dJ]  coupled coordinate indices
        self.m = m                  # [k]   driver centering
        self.b = b                  # [dJ]  shift intercept
        self.c = c                  # [dJ, k]  log-scale loadings
        self.d = d                  # [dJ, k]  shift loadings

    def _S_t(self, xI: torch.Tensor):
        z = xI - self.m
        S_raw = z @ self.c.T
        S = S_raw.clamp(-self.SMAX, self.SMAX)
        live = (S_raw.abs() < self.SMAX).to(z.dtype)   # [N, dJ]
        return S, self.b + z @ self.d.T, live

    def forward_u(self, u: torch.Tensor):
        """``u -> (x, logdet)``; drivers are read off ``u_I = x_I``."""
        xI = u[:, self.idx_I]
        S, t, _ = self._S_t(xI)
        x = u.clone()
        x[:, self.idx_J] = torch.exp(S) * u[:, self.idx_J] + t
        return x, S.sum(-1)

    def prep(self, x: torch.Tensor, a: torch.Tensor):
        """``(x, score) -> (u, u-space score of the pulled-back density,
        logdet)``. The u-score is ``J_C^T a + grad_u log|det J_C|``."""
        xI = x[:, self.idx_I]
        S, t, live = self._S_t(xI)
        u = x.clone()
        xmt = x[:, self.idx_J] - t                     # = e^S u_J
        u[:, self.idx_J] = xmt * torch.exp(-S)
        aJ = a[:, self.idx_J]
        g = a.clone()
        g[:, self.idx_J] = torch.exp(S) * aJ
        c_eff = live.unsqueeze(-1) * self.c            # [N, dJ, k]
        g[:, self.idx_I] = (a[:, self.idx_I]
                            + torch.einsum("nj,njk->nk", xmt * aJ, c_eff)
                            + aJ @ self.d
                            + c_eff.sum(1))
        return u, g, S.sum(-1)


def _popcount(i: int) -> int:
    return bin(i).count("1")


def _trailing_ones(i: int) -> int:
    n = 0
    while i & 1:
        n += 1
        i >>= 1
    return n


class _Welford:
    """Vectorized Welford accumulator for (draw, score) pairs, [N, d].

    Every update is synchronized across the chain ensemble (draws are always
    finite by construction), so a single scalar count suffices.
    """

    def __init__(self, N: int, d: int):
        self.mean_x = torch.zeros(N, d, dtype=DTYPE)
        self.m2_x = torch.zeros(N, d, dtype=DTYPE)
        self.mean_a = torch.zeros(N, d, dtype=DTYPE)
        self.m2_a = torch.zeros(N, d, dtype=DTYPE)
        self.count = 0

    def update(self, x: torch.Tensor, a: torch.Tensor) -> None:
        self.count += 1
        dx = x - self.mean_x
        self.mean_x = self.mean_x + dx / self.count
        self.m2_x = self.m2_x + dx * (x - self.mean_x)
        da = a - self.mean_a
        self.mean_a = self.mean_a + da / self.count
        self.m2_a = self.m2_a + da * (a - self.mean_a)

    def var(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = max(self.count - 1, 1)
        return self.m2_x / n, self.m2_a / n


class FisherHMC(InferenceAlgorithm):
    r"""NUTS with nutpie's Fisher-divergence mass-matrix adaptation
    (Seyboldt, Carlson & Carpenter 2026). ``mass_matrix`` selects the
    estimator: ``"diag"`` (Fisher diagonal, the nutpie default),
    ``"lowrank"`` (low-rank plus diagonal, the paper's strongest variant), or
    ``"variance"`` (the Stan/PyMC draws-only diagonal under the identical
    kernel and schedule -- the ablation that isolates the estimator). See the
    module docstring.
    """

    name = "FisherHMC"

    def __init__(self, mass_matrix="diag", n_chains=16, n_warmup=1000,
                 n_draws=1000, target_accept=0.8, max_treedepth=10,
                 max_energy_error=1000.0, init_step=0.1, window_early=10,
                 window_late=80, early_frac=0.3, final_frac=0.15,
                 lowrank_cutoff=2.0, lowrank_gamma=1e-5, min_window=3,
                 init_radius=2.0, da_gamma=0.05, da_t0=10.0, da_kappa=0.75,
                 coupling=False, n_drivers=6, coupling_r2=0.05,
                 coupling_clamp=2.5, coupling_tstat=4.0, coupling_polish=60):
        # ``n_warmup``/``n_draws`` are per-chain NUTS iterations (paper: 1000
        # each).  ``window_early``/``window_late`` are the wipe frequencies L
        # of the two adaptation phases; ``early_frac``/``final_frac`` split the
        # warmup 30/55/15.  ``init_step`` is nuts-rs's dual-averaging seed,
        # also used when the step size is reinitialized at phase 2.
        # ``coupling`` stacks the learned coupling chart on top of the affine
        # transform: drivers are selected once (pooled score-scale regression
        # over the first phase-2 window, activation gated on ``coupling_r2``),
        # coefficients refit at every later wipe, |c| clamped elementwise at
        # ``coupling_clamp``. ``n_drivers`` is a cap, not a target -- greedy
        # selection stops when partial signal drops below ``coupling_r2``
        # (a hierarchical model needs one driver per group scale plus one per
        # hierarchical mean, e.g. 4 for radon variable-intercept-slope).
        if mass_matrix not in _NAMES:
            raise ValueError(f"mass_matrix must be one of {sorted(_NAMES)}")
        self.mass_matrix = mass_matrix
        self.coupling = bool(coupling)
        self.n_drivers = n_drivers
        self.coupling_r2 = coupling_r2
        self.coupling_clamp = coupling_clamp
        self.coupling_tstat = coupling_tstat
        self.coupling_polish = coupling_polish
        self.name = _NAMES[mass_matrix] + ("+C" if coupling else "")
        self.n_chains = n_chains
        self.n_warmup = n_warmup
        self.n_draws = n_draws
        self.target_accept = target_accept
        self.max_treedepth = max_treedepth
        self.max_energy_error = max_energy_error
        self.init_step = init_step
        self.window_early = window_early
        self.window_late = window_late
        self.early_frac = early_frac
        self.final_frac = final_frac
        self.lowrank_cutoff = lowrank_cutoff
        self.lowrank_gamma = lowrank_gamma
        self.min_window = min_window
        self.init_radius = init_radius
        self.da_gamma = da_gamma
        self.da_t0 = da_t0
        self.da_kappa = da_kappa

    # ------------------------------------------------------------------ #
    # target-space score, guarded like the other samplers here
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_glp(target: TargetPosterior):
        def one(x: torch.Tensor):
            with torch.enable_grad():
                xr = x.detach().requires_grad_(True)
                lp = target.log_prob(xr)
                g = torch.autograd.grad(lp.sum(), xr)[0]
            g = torch.nan_to_num(g.detach(), nan=0.0, posinf=1e10, neginf=-1e10)
            lp = torch.nan_to_num(lp.detach(), nan=-1e30, neginf=-1e30,
                                  posinf=-1e30)
            return lp, g

        def glp(x: torch.Tensor):
            # Some targets raise (e.g. a GP Cholesky off the PD cone) instead
            # of returning -inf; NUTS legitimately probes such points, so fall
            # back to per-chain evaluation and mark the offenders divergent.
            try:
                return one(x)
            except Exception:
                N, d = x.shape
                lp = torch.full((N,), -1e30, dtype=DTYPE)
                g = torch.zeros(N, d, dtype=DTYPE)
                for i in range(N):
                    try:
                        lp[i], g[i] = one(x[i:i + 1])
                    except Exception:
                        pass
                return lp, g

        return glp

    # ------------------------------------------------------------------ #
    # the affine transform x = J y  (J = D_sigma (I + U diag(s) U^T))
    # ------------------------------------------------------------------ #
    @staticmethod
    def _J(v, sigma, U, s):
        """Position map ``J v`` (adapted-space velocity -> target-space step)."""
        if U is not None:
            v = v + torch.einsum("ndk,nk->nd", U,
                                 s * torch.einsum("ndk,nd->nk", U, v))
        return sigma * v

    @staticmethod
    def _Jt(a, sigma, U, s):
        """Score pullback ``J^T a`` (target-space score -> adapted space)."""
        b = sigma * a
        if U is not None:
            b = b + torch.einsum("ndk,nk->nd", U,
                                 s * torch.einsum("ndk,nd->nk", U, b))
        return b

    @staticmethod
    def _Ji(u, sigma, U, s):
        """Affine inverse ``J^{-1} u`` via Woodbury (orthonormal ``U``)."""
        w = u / sigma
        if U is not None:
            w = w + torch.einsum("ndk,nk->nd", U,
                                 (1.0 / (1.0 + s) - 1.0)
                                 * torch.einsum("ndk,nd->nk", U, w))
        return w

    def _prep_state(self, x, a, sigma, U, s, coup):
        """(x, score) -> (y, adapted-space score, logdet) under the current
        transform ``F = C o J``."""
        if coup is None:
            u, g = x, a
            ld = torch.zeros(x.shape[0], dtype=DTYPE)
        else:
            u, g, ld = coup.prep(x, a)
        return self._Ji(u, sigma, U, s), self._Jt(g, sigma, U, s), ld

    def _fwd_state(self, y, glp, sigma, U, s, coup):
        """y -> (x, x-score, x-logp, adapted-space score, logdet)."""
        u = self._J(y, sigma, U, s)
        if coup is None:
            x = u
            ld = torch.zeros(y.shape[0], dtype=DTYPE)
        else:
            x, ld = coup.forward_u(u)
        lp, a = glp(x)
        if coup is None:
            g = a
        else:
            _, g, ld = coup.prep(x, a)
        return x, a, lp, self._Jt(g, sigma, U, s), ld

    # ------------------------------------------------------------------ #
    # one vectorized multinomial-NUTS iteration for the whole ensemble
    # ------------------------------------------------------------------ #
    def _nuts_draw(self, glp, x, a, logp, sigma, U, s, coup, eps, gen,
                   sym_stat):
        """Advance every chain by one NUTS draw; all chains double in lockstep.

        Returns ``(x', a', logp', accept_stat, diverged, n_grad, depth)``, all
        per-chain. ``accept_stat`` is the mean over every evaluated leapfrog
        state of the Metropolis probability ``exp(min(0, -dE))`` (or the
        symmetric statistic when ``sym_stat``), the dual-averaging signal.
        U-turn checks use the iterative checkpoint scheme (NumPyro / Stan) on
        the adapted-space momentum sums; a divergence or in-subtree U-turn
        discards that doubling for that chain only.

        The chain position advances in the adapted space (``y += eps v``, the
        mass matrix there is the identity) and maps to the target space
        through the full transform ``F = C o J``; the potential includes the
        coupling's ``-log|det J_C|`` so the invariant density is exact.
        """
        N, d = x.shape
        D = self.max_treedepth
        y0, delta, ld0 = self._prep_state(x, a, sigma, U, s, coup)
        v0 = torch.randn(N, d, dtype=DTYPE, generator=gen)
        H0 = -(logp + ld0) + 0.5 * (v0 * v0).sum(-1)

        xp, ap, lpp = x.clone(), a.clone(), logp.clone()     # proposal
        log_w_tree = torch.zeros(N, dtype=DTYPE)

        yl, vl, dl = y0.clone(), v0.clone(), delta.clone()
        yr, vr, dr = y0.clone(), v0.clone(), delta.clone()
        rho = v0.clone()                                     # trajectory momentum sum

        stopped = torch.zeros(N, dtype=torch.bool)
        diverged = torch.zeros(N, dtype=torch.bool)
        sum_stat = torch.zeros(N, dtype=DTYPE)
        n_states = torch.zeros(N, dtype=DTYPE)
        n_grad = torch.zeros(N, dtype=torch.long)
        depth = torch.zeros(N, dtype=torch.long)

        ckpt_v = torch.zeros(D + 1, N, d, dtype=DTYPE)       # checkpoint momenta
        ckpt_cum = torch.zeros(D + 1, N, d, dtype=DTYPE)     # cumulative sums there

        for j in range(D):
            active = ~stopped
            if not bool(active.any()):
                break
            dir_pos = torch.rand(N, dtype=DTYPE, generator=gen) < 0.5
            dirf = dir_pos[:, None]
            eps_s = torch.where(dir_pos, eps, -eps)[:, None]
            yc = torch.where(dirf, yr, yl)
            vc = torch.where(dirf, vr, vl)
            dc = torch.where(dirf, dr, dl)

            log_w_sub = torch.full((N,), -math.inf, dtype=DTYPE)
            x_sp, a_sp, lp_sp = xp.clone(), ap.clone(), lpp.clone()
            cum = torch.zeros(N, d, dtype=DTYPE)             # subtree momentum sum
            sub_div = torch.zeros(N, dtype=torch.bool)
            sub_turn = torch.zeros(N, dtype=torch.bool)

            for i in range(1 << j):
                act = active & ~sub_div & ~sub_turn
                if not bool(act.any()):
                    break
                actf = act[:, None]
                # one leapfrog step in the adapted space (identity mass)
                v_half = vc + 0.5 * eps_s * dc
                y_new = yc + eps_s * v_half
                x_new, a_new, lp_new, d_new, ld_new = self._fwd_state(
                    y_new, glp, sigma, U, s, coup)
                n_grad += act.long()
                v_new = v_half + 0.5 * eps_s * d_new
                dE = (-(lp_new + ld_new)
                      + 0.5 * (v_new * v_new).sum(-1)) - H0
                bad = ~torch.isfinite(dE) | (dE > self.max_energy_error)
                # step-size statistic over every evaluated state (Stan-style)
                stat = torch.exp((-dE).clamp(max=0.0))
                if sym_stat:
                    stat = 2.0 * stat / (1.0 + torch.exp(-dE))
                sum_stat += torch.where(act, torch.nan_to_num(stat), 0.0)
                n_states += act.to(DTYPE)

                sub_div |= act & bad
                ok = act & ~bad
                okf = ok[:, None]
                yc = torch.where(okf, y_new, yc)
                vc = torch.where(okf, v_new, vc)
                dc = torch.where(okf, d_new, dc)
                # progressive multinomial sampling within the subtree
                lw = torch.where(ok, -dE, torch.full_like(dE, -math.inf))
                new_lw = torch.logaddexp(log_w_sub, lw)
                u = torch.rand(N, dtype=DTYPE, generator=gen).clamp_min(1e-300)
                take = ok & (torch.log(u) < lw - new_lw)
                log_w_sub = torch.where(ok, new_lw, log_w_sub)
                x_sp = torch.where(take[:, None], x_new, x_sp)
                a_sp = torch.where(take[:, None], a_new, a_sp)
                lp_sp = torch.where(take, lp_new, lp_sp)
                cum = cum + torch.where(okf, v_new, torch.zeros_like(v_new))
                # iterative generalized-U-turn checks inside the new subtree
                if j > 0:
                    if i % 2 == 0:
                        slot = _popcount(i >> 1)
                        ckpt_v[slot] = torch.where(okf, v_new, ckpt_v[slot])
                        ckpt_cum[slot] = torch.where(okf, cum, ckpt_cum[slot])
                    else:
                        idx_max = _popcount(i >> 1)
                        for slot in range(idx_max - _trailing_ones(i) + 1,
                                          idx_max + 1):
                            seg = cum - ckpt_cum[slot] + ckpt_v[slot]
                            turn = ((ckpt_v[slot] * seg).sum(-1) <= 0) | \
                                   ((v_new * seg).sum(-1) <= 0)
                            sub_turn |= ok & turn

            complete = active & ~sub_div & ~sub_turn
            cf = complete[:, None]
            # biased progressive merge: take the new subtree's proposal with
            # probability min(1, W_sub / W_tree)
            u = torch.rand(N, dtype=DTYPE, generator=gen).clamp_min(1e-300)
            accept_new = complete & (torch.log(u) < log_w_sub - log_w_tree)
            anf = accept_new[:, None]
            xp = torch.where(anf, x_sp, xp)
            ap = torch.where(anf, a_sp, ap)
            lpp = torch.where(accept_new, lp_sp, lpp)
            log_w_tree = torch.where(
                complete, torch.logaddexp(log_w_tree, log_w_sub), log_w_tree)
            # grow the trajectory endpoint and re-check the full trajectory
            upd_r, upd_l = cf & dirf, cf & ~dirf
            yr = torch.where(upd_r, yc, yr)
            vr = torch.where(upd_r, vc, vr)
            dr = torch.where(upd_r, dc, dr)
            yl = torch.where(upd_l, yc, yl)
            vl = torch.where(upd_l, vc, vl)
            dl = torch.where(upd_l, dc, dl)
            rho = rho + torch.where(cf, cum, torch.zeros_like(cum))
            turn_tree = ((vl * rho).sum(-1) <= 0) | ((vr * rho).sum(-1) <= 0)
            depth = torch.where(complete, depth + 1, depth)
            diverged |= sub_div
            stopped |= (active & (sub_div | sub_turn)) | (complete & turn_tree)

        accept_stat = sum_stat / n_states.clamp_min(1.0)
        return xp, ap, lpp, accept_stat, diverged, n_grad, depth

    # ------------------------------------------------------------------ #
    # mass-matrix estimators
    # ------------------------------------------------------------------ #
    def _diag_estimate(self, w: _Welford, sigma2):
        """Refresh the diagonal ``sigma^2 = M^{-1}`` from a Welford window."""
        var_x, var_a = w.var()
        if self.mass_matrix == "variance":
            n = w.count
            new = (n / (n + 5.0)) * var_x + 1e-3 * (5.0 / (n + 5.0))
        else:
            new = torch.sqrt(var_x / var_a)
        ok = torch.isfinite(new) & (new > 0)
        return torch.where(ok, new, sigma2).clamp(1e-10, 1e10)

    def _fit_lowrank(self, xs: List[torch.Tensor], as_: List[torch.Tensor],
                     sigma, U, s):
        """Algorithm 1 (low-rank-adapt) on the stored window, batched over
        chains. Falls back to the previous transform for any chain whose fit
        is degenerate."""
        X = torch.stack(xs, dim=1)                     # [N, n, d]
        A = torch.stack(as_, dim=1)
        N, n, d = X.shape
        if n < self.min_window:
            return sigma, U, s
        Xc = X - X.mean(dim=1, keepdim=True)
        Ac = A - A.mean(dim=1, keepdim=True)
        var_x = (Xc ** 2).sum(1) / (n - 1)             # [N, d]
        var_a = (Ac ** 2).sum(1) / (n - 1)
        sig2 = torch.sqrt(var_x / var_a)
        ok_d = torch.isfinite(sig2) & (sig2 > 0)
        sig2 = torch.where(ok_d, sig2, sigma ** 2).clamp(1e-10, 1e10)
        sig = torch.sqrt(sig2)
        # standardize: draws divided by sigma, scores multiplied (contravariant)
        Xs = (Xc / sig[:, None, :]).transpose(1, 2)    # [N, d, n]
        As = (Ac * sig[:, None, :]).transpose(1, 2)
        try:
            Ux = torch.linalg.svd(Xs, full_matrices=False)[0]
            Ua = torch.linalg.svd(As, full_matrices=False)[0]
            Q = torch.linalg.qr(torch.cat([Ux, Ua], dim=2)).Q   # [N, d, m]
            Px = Q.transpose(1, 2) @ Xs                # [N, m, n]
            Pa = Q.transpose(1, 2) @ As
            m = Q.shape[-1]
            eye = torch.eye(m, dtype=DTYPE).expand(N, m, m)
            g = self.lowrank_gamma
            Cx = Px @ Px.transpose(1, 2) / (n - 1) + g * eye
            Ca = Pa @ Pa.transpose(1, 2) / (n - 1) + g * eye
            # SPDM (Algorithm 2): Sig solves  Sig Ca Sig = Cx
            wa, Va = torch.linalg.eigh(Ca)
            wa = wa.clamp_min(1e-30)
            Ah = Va @ (wa.sqrt()[:, :, None] * Va.transpose(1, 2))
            Aih = Va @ (wa.rsqrt()[:, :, None] * Va.transpose(1, 2))
            wm, Vm = torch.linalg.eigh(Ah @ Cx @ Ah)
            Mh = Vm @ (wm.clamp_min(1e-30).sqrt()[:, :, None] * Vm.transpose(1, 2))
            lam, W = torch.linalg.eigh(Aih @ Mh @ Aih)  # eigenpairs of Sig
        except Exception:
            return sigma, U, s
        lam = lam.clamp_min(1e-12)
        keep = (lam >= self.lowrank_cutoff) | (lam <= 1.0 / self.lowrank_cutoff)
        lam = torch.where(keep, lam, torch.ones_like(lam))
        s_new = lam.sqrt() - 1.0                       # [N, m]
        U_new = Q @ W                                  # [N, d, m]
        # per-chain fallback: any non-finite fit keeps the previous transform
        bad = ~(torch.isfinite(sig).all(-1) & torch.isfinite(s_new).all(-1)
                & torch.isfinite(U_new.flatten(1)).all(-1))
        if bool(bad.any()):
            sig = torch.where(bad[:, None], sigma, sig)
            s_new = torch.where(bad[:, None], torch.zeros_like(s_new), s_new)
            U_new = torch.where(bad[:, None, None], torch.zeros_like(U_new), U_new)
        return sig, U_new, s_new

    def _fit_coupling(self, xs: List[torch.Tensor], as_: List[torch.Tensor],
                      prev: Optional[_Coupling]) -> Tuple[Optional[_Coupling], dict]:
        """Fit (or refit) the coupling chart from the raw ``(draw, score)``
        window, pooled across chains.

        Driver selection happens once (``prev is None``): activation is gated
        on the *score-scale* signal -- the squared correlation between a
        candidate driver ``x_i`` and ``log alpha_j^2`` of another coordinate,
        which for a funnel ``x_j | x_i ~ N(., e^{2 c x_i})`` has slope
        ``-2c`` -- and drivers are ranked by combined scale + shift explained
        variance (a location driver like a hierarchical mean carries shift
        signal only). On later calls the driver set is frozen and only
        ``m, c, d`` are refit, by pooled ridge-OLS; ``c = -slope/2`` from the
        score-scale regression, clamped elementwise. Returns ``(None, stats)``
        to signal "no change".
        """
        d = xs[0].shape[-1]
        X = torch.stack(xs, dim=1).reshape(-1, d)          # pooled [M, d]
        A = torch.stack(as_, dim=1).reshape(-1, d)
        M = X.shape[0]
        if M < max(32, 16 * (self.n_drivers + 1)):
            return None, {}
        # NO centering of the scores: E[alpha] = 0 under the target, and a
        # noisy sample mean puts a floor of log(mean^2) under the response
        # exactly where |alpha| is small, attenuating every slope.
        L = torch.log(A ** 2 + 1e-30)
        stats: dict = {}
        if prev is None:
            sel, stats = self._select_drivers(X, L)
            if not sel:
                return None, stats
            idx_I = torch.tensor(sel, dtype=torch.long)
        else:
            idx_I = prev.idx_I
        k = idx_I.numel()
        mask = torch.zeros(d, dtype=torch.bool)
        mask[idx_I] = True
        idx_J = torch.arange(d)[~mask]
        Z = X[:, idx_I]
        m = Z.mean(0)
        Zc = Z - m
        try:
            # --- scale loadings: OLS slope of log alpha^2 on the drivers,
            #     c = -slope/2, kept only where the t-statistic is decisive
            G = Zc.T @ Zc
            G = G + 1e-8 * float(G.diagonal().mean().clamp_min(1e-30)) \
                * torch.eye(k, dtype=DTYPE)
            Ginv = torch.linalg.inv(G)
            Lc = L[:, idx_J] - L[:, idx_J].mean(0)
            beta_L = Ginv @ (Zc.T @ Lc)                     # [k, dJ]
            mse_L = ((Lc - Zc @ beta_L) ** 2).mean(0)       # [dJ]
            se_L = torch.sqrt(Ginv.diagonal()[:, None]
                              * mse_L[None, :]).clamp_min(1e-30)
            c = torch.where((beta_L / se_L).abs() > self.coupling_tstat,
                            -0.5 * beta_L, torch.zeros_like(beta_L)).T
            c = self._eb_shrink(c, 0.5 * se_L.T)
            c = c.clamp(-self.coupling_clamp, self.coupling_clamp)
            # --- shift loadings: WLS of x_J on [1, drivers] with precision
            #     weights e^{-2 S} (heteroskedasticity from the scale model);
            #     slopes t-gated, the intercept b always kept
            S = Zc @ c.T                                    # [M, dJ]
            w = torch.exp((-2.0 * S).clamp(-40.0, 40.0))
            w = w / w.mean(0, keepdim=True)                 # [M, dJ]
            Zaug = torch.cat([torch.ones(M, 1, dtype=DTYPE), Zc], dim=1)
            Gw = torch.einsum("mj,mp,mq->jpq", w, Zaug, Zaug)
            Gw = Gw + 1e-8 * Gw.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] \
                * torch.eye(k + 1, dtype=DTYPE)
            XJ = X[:, idx_J]
            rhs = torch.einsum("mj,mp->jp", w * XJ, Zaug)   # [dJ, k+1]
            sol = torch.linalg.solve(Gw, rhs[..., None])[..., 0]
            fit = torch.einsum("mp,jp->mj", Zaug, sol)
            mse_w = (w * (XJ - fit) ** 2).mean(0)           # [dJ]
            Gw_inv_diag = torch.linalg.inv(Gw).diagonal(dim1=1, dim2=2)
            se_w = torch.sqrt(Gw_inv_diag * mse_w[:, None]).clamp_min(1e-30)
            t_w = sol / se_w
            b = sol[:, 0]
            dcoef = torch.where(t_w[:, 1:].abs() > self.coupling_tstat,
                                sol[:, 1:], torch.zeros_like(sol[:, 1:]))
            dcoef = self._eb_shrink(dcoef, se_w[:, 1:])
        except Exception:
            return None, stats
        if not (torch.isfinite(c).all() and torch.isfinite(dcoef).all()
                and torch.isfinite(b).all() and torch.isfinite(m).all()):
            return None, stats
        if self.coupling_polish > 0:
            c, dcoef, b = self._polish_coupling(xs, as_, idx_I, idx_J,
                                                m, c, dcoef, b)
        return _Coupling(idx_I, idx_J, m, b, c, dcoef), stats

    @staticmethod
    def _profiled_fisher(u: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Empirical Fisher divergence with the diagonal affine scale and
        location profiled out coordinate-wise (paper Thm 2.2): the minimum
        over ``sigma, mu`` of ``mean(sigma g + (u - mu)/sigma)^2`` is
        ``2 (sqrt(var u * var g) + cov(u, g))`` per coordinate."""
        uc = u - u.mean(dim=1, keepdim=True)
        gc = g - g.mean(dim=1, keepdim=True)
        vu = (uc ** 2).mean(1)
        vg = (gc ** 2).mean(1)
        cv = (uc * gc).mean(1)
        return (2.0 * (torch.sqrt(vu * vg + 1e-30) + cv)).sum(-1).mean()

    def _polish_coupling(self, xs, as_, idx_I, idx_J, m, c0, d0, b0):
        """Refine ``(c, d, b)`` by minimizing the *actual* empirical Fisher
        divergence of the transformed window (Adam, a few dozen steps,
        zero-pattern of the gated coefficients preserved, per-coordinate
        diagonal scales and locations profiled out in closed form).

        The regressions are good initializers but systematically biased
        where the score mixes prior and likelihood terms (e.g. eight
        schools: the likelihood part of ``alpha_theta`` does not scale with
        ``tau``, attenuating the fitted slope below the true non-centering
        value 1); the in-family Fisher optimum has no such bias."""
        X = torch.stack(xs, dim=1)                     # [N, n, d]
        A = torch.stack(as_, dim=1)
        xI, xJ = X[..., idx_I], X[..., idx_J]
        aI, aJ = A[..., idx_I], A[..., idx_J]
        cmask = (c0 != 0).to(DTYPE)
        dmask = (d0 != 0).to(DTYPE)
        c = c0.clone().requires_grad_(True)
        dc = d0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        opt = torch.optim.Adam([c, dc, b], lr=0.03)
        best = (None, float("inf"))
        for _ in range(self.coupling_polish):
            opt.zero_grad()
            cm, dm = c * cmask, dc * dmask
            z = xI - m
            S = (z @ cm.T).clamp(-40.0, 40.0)          # [N, n, dJ]
            t = b + z @ dm.T
            xmt = xJ - t
            u_J = xmt * torch.exp(-S)
            g_J = torch.exp(S) * aJ
            g_I = aI + (xmt * aJ) @ cm + aJ @ dm + cm.sum(0)
            loss = (self._profiled_fisher(u_J, g_J)
                    + self._profiled_fisher(xI, g_I))
            loss_val = float(loss.detach())
            if torch.isfinite(loss) and loss_val < best[1]:
                best = ((c.detach().clone(), dc.detach().clone(),
                         b.detach().clone()), loss_val)
            if not torch.isfinite(loss):
                break
            loss.backward()
            opt.step()
        if best[0] is None:
            return c0, d0, b0
        cb, db, bb = best[0]
        cb = (cb * cmask).clamp(-self.coupling_clamp, self.coupling_clamp)
        db = db * dmask
        if not (torch.isfinite(cb).all() and torch.isfinite(db).all()
                and torch.isfinite(bb).all()):
            return c0, d0, b0
        return cb, db, bb

    @staticmethod
    def _eb_shrink(coef: torch.Tensor, se: torch.Tensor) -> torch.Tensor:
        """Empirical-Bayes shrinkage of the non-gated loadings toward their
        per-driver group mean.

        Per-coordinate noise in the loadings leaves residual anharmonic
        terms ``exp((c_j - c_true) x_I)`` in the transformed potential --
        soft exponential walls a few sigma out whose curvature exceeds the
        bulk-tuned leapfrog stability limit and shows up as spurious
        divergences. Coefficients that are statistically indistinguishable
        (``tau^2 ~ 0``, e.g. the exchangeable coordinates of a funnel)
        collapse to their common value; genuinely heterogeneous ones keep
        their spread."""
        out = coef.clone()
        for i in range(coef.shape[1]):
            sel = coef[:, i] != 0.0
            if int(sel.sum()) < 3:
                continue
            cs, ss = coef[sel, i], se[sel, i]
            cbar = cs.mean()
            tau2 = (cs.var(unbiased=True) - (ss ** 2).mean()).clamp_min(0.0)
            w = tau2 / (tau2 + ss ** 2)
            out[sel, i] = cbar + w * (cs - cbar)
        return out

    def _select_drivers(self, X: torch.Tensor, L: torch.Tensor
                        ) -> Tuple[List[int], dict]:
        """Greedy driver selection with partialing-out.

        A candidate's *scale* signal is the squared correlation between
        ``x_i`` and ``log alpha_j^2`` (funnel signature), its *shift* signal
        the squared correlation with ``x_j`` itself (location drivers such
        as hierarchical means). Activation requires a decisive scale signal
        somewhere; each further driver must add partial explained variance
        beyond the ones already chosen (this is what keeps redundant funnel
        coordinates -- which all carry each other's scale -- from being
        picked twice).

        All selection statistics are computed on *rank-transformed* data:
        funnel-coupled coordinates are scale mixtures with enormous
        kurtosis, and their sample Pearson correlations are noisy enough to
        promote spurious drivers; Spearman correlations are immune while
        preserving the monotone funnel/shift signals."""
        M, d = X.shape
        ok = X.var(0, unbiased=True) > 1e-12

        def ranks(t: torch.Tensor) -> torch.Tensor:
            r = torch.argsort(torch.argsort(t, dim=0), dim=0).to(DTYPE)
            return r / (t.shape[0] - 1) - 0.5

        X, L = ranks(X), ranks(L)
        Xr = X - X.mean(0)          # responses (shift), residualized as we go
        Lr = L - L.mean(0)          # responses (scale), residualized as we go
        Zr = Xr.clone()             # candidate driver columns, residualized
        selected: List[int] = []
        stats: dict = {}
        for step in range(self.n_drivers):
            var_z = Zr.var(0, unbiased=True)
            var_L = Lr.var(0, unbiased=True)
            var_x = Xr.var(0, unbiased=True)
            C_zL = Zr.T @ Lr / (M - 1)
            r2_scale = C_zL ** 2 / (var_z[:, None] * var_L[None, :]).clamp_min(1e-30)
            C_zx = Zr.T @ Xr / (M - 1)
            r2_shift = C_zx ** 2 / (var_z[:, None] * var_x[None, :]).clamp_min(1e-30)
            r2_scale.fill_diagonal_(0.0)
            r2_shift.fill_diagonal_(0.0)
            r2_scale[~ok] = 0.0
            r2_shift[~ok] = 0.0
            if step == 0:
                scale_max = float(r2_scale.max())
                stats["scale_r2_max"] = scale_max
                if scale_max < self.coupling_r2:
                    return [], stats                        # no funnel signal
            stat = r2_scale.mean(1) + r2_shift.mean(1)
            stat[~ok] = -1.0
            if selected:
                stat[torch.tensor(selected)] = -1.0
            best = int(stat.argmax())
            if float(stat[best]) < self.coupling_r2 and step > 0:
                break
            selected.append(best)
            stats.setdefault("driver_stat", []).append(float(stat[best]))
            z = Zr[:, best:best + 1]
            z = z / z.norm().clamp_min(1e-30)               # partial out
            Lr = Lr - z @ (z.T @ Lr)
            Xr = Xr - z @ (z.T @ Xr)
            Zr = Zr - z @ (z.T @ Zr)
        return selected, stats

    # ------------------------------------------------------------------ #
    def fit(self, target: TargetPosterior, seed: int = 0,
            progress: object = False) -> InferenceResult:
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 12345)
        dim = target.dim
        N = self.n_chains
        W = self.n_warmup
        glp = self._make_glp(target)
        t0 = time.time()

        # --- Stan-style initialization: uniform(-2, 2), retried to finiteness
        x = self.init_radius * (2.0 * torch.rand(N, dim, dtype=DTYPE,
                                                 generator=gen) - 1.0)
        logp, a = glp(x)
        n_grad_warm = N
        for _ in range(100):
            fin = logp > -1e29
            if bool(fin.all()):
                break
            x2 = self.init_radius * (2.0 * torch.rand(N, dim, dtype=DTYPE,
                                                      generator=gen) - 1.0)
            lp2, a2 = glp(x2)
            n_grad_warm += N
            x = torch.where(fin[:, None], x, x2)
            a = torch.where(fin[:, None], a, a2)
            logp = torch.where(fin, logp, lp2)

        # --- init mass matrix from the first score: M = diag(|alpha^(0)|)
        a0 = a.abs()
        sigma2 = torch.where(a0 > 1e-12, 1.0 / a0,
                             torch.ones_like(a0)).clamp(1e-8, 1e8)
        sigma = sigma2.sqrt()
        U: Optional[torch.Tensor] = None               # low-rank directions
        s: Optional[torch.Tensor] = None               # sqrt(lambda) - 1
        coup: Optional[_Coupling] = None               # coupling chart
        coup_stats: dict = {}
        n_coup_refits = 0
        use_buffer = self.mass_matrix == "lowrank" or self.coupling

        # --- foreground/background estimators + low-rank window buffer
        fg, bg = _Welford(N, dim), _Welford(N, dim)
        fg.update(x, a)
        bg.update(x, a)
        buf_x, buf_a = [x.clone()], [a.clone()]

        # --- dual averaging state (per chain)
        log_eps = torch.full((N,), math.log(self.init_step), dtype=DTYPE)
        log_eps_bar = torch.zeros(N, dtype=DTYPE)
        h_bar = torch.zeros(N, dtype=DTYPE)
        mu = math.log(10.0 * self.init_step)
        t_da = 0

        p1_end = int(round(self.early_frac * W))
        p3_start = int(round((1.0 - self.final_frac) * W))
        next_wipe = self.window_early

        bar = _make_bar(progress, W + self.n_draws, self.name)
        div_warm = 0

        for it in range(1, W + 1):
            phase3 = it > p3_start
            if it == p1_end + 1:                       # phase 2: re-init step size
                log_eps = torch.full((N,), math.log(self.init_step), dtype=DTYPE)
                log_eps_bar = torch.zeros(N, dtype=DTYPE)
                h_bar = torch.zeros(N, dtype=DTYPE)
                t_da = 0
            eps = torch.exp(log_eps)
            x, a, logp, acc, div, ng, _dep = self._nuts_draw(
                glp, x, a, logp, sigma, U, s, coup, eps, gen, sym_stat=phase3)
            n_grad_warm += int(ng.sum())
            div_warm += int(div.sum())
            # dual averaging (Hoffman & Gelman) toward the target acceptance
            t_da += 1
            eta_h = 1.0 / (t_da + self.da_t0)
            h_bar = (1.0 - eta_h) * h_bar + eta_h * (self.target_accept - acc)
            log_eps = mu - math.sqrt(t_da) / self.da_gamma * h_bar
            eta = t_da ** (-self.da_kappa)
            log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar

            if not phase3:
                # the affine estimators see the window *through* the chart
                ux, ua = (x, a) if coup is None else coup.prep(x, a)[:2]
                fg.update(ux, ua)
                bg.update(ux, ua)
                if use_buffer:
                    buf_x.append(x.clone())
                    buf_a.append(a.clone())
                if self.mass_matrix != "lowrank" and fg.count >= self.min_window:
                    sigma2 = self._diag_estimate(fg, sigma2)  # diag: every draw
                    sigma = sigma2.sqrt()
                if it == next_wipe:                    # background -> foreground
                    chart_changed = False
                    if self.coupling and it > p1_end:
                        new_coup, cst = self._fit_coupling(buf_x, buf_a, coup)
                        if new_coup is not None:
                            if coup is None:           # first activation:
                                log_eps = torch.full(  # restart dual averaging
                                    (N,), math.log(self.init_step), dtype=DTYPE)
                                log_eps_bar = torch.zeros(N, dtype=DTYPE)
                                h_bar = torch.zeros(N, dtype=DTYPE)
                                t_da = 0
                            coup = new_coup
                            coup_stats.update(cst)
                            n_coup_refits += 1
                            chart_changed = True
                    if self.mass_matrix == "lowrank" or chart_changed:
                        if coup is None:
                            bu, ba = buf_x, buf_a
                        else:                          # replay window through chart
                            pairs = [coup.prep(bx, bb)[:2]
                                     for bx, bb in zip(buf_x, buf_a)]
                            bu = [p[0] for p in pairs]
                            ba = [p[1] for p in pairs]
                    if self.mass_matrix == "lowrank":
                        sigma, U, s = self._fit_lowrank(bu, ba, sigma, U, s)
                        fg, bg = bg, _Welford(N, dim)
                    elif chart_changed:
                        fresh = _Welford(N, dim)       # bg is stale under new chart
                        for uu, gg in zip(bu, ba):
                            fresh.update(uu, gg)
                        if fresh.count >= self.min_window:
                            sigma2 = self._diag_estimate(fresh, sigma2)
                            sigma = sigma2.sqrt()
                        fg, bg = fresh, _Welford(N, dim)
                    else:
                        fg, bg = bg, _Welford(N, dim)
                    if use_buffer:
                        buf_x, buf_a = [], []
                    L = self.window_early if it < p1_end else self.window_late
                    next_wipe = it + L
            if bar is not None:
                bar.update(1)
        eps = torch.exp(log_eps_bar)                   # frozen sampling step size

        # --- sampling ------------------------------------------------------ #
        pool: List[torch.Tensor] = []
        n_grad_samp = 0
        div_samp = 0
        depth_saturated = 0
        depths: List[float] = []
        accs: List[float] = []
        logp_hist: List[float] = []
        for it in range(self.n_draws):
            x, a, logp, acc, div, ng, dep = self._nuts_draw(
                glp, x, a, logp, sigma, U, s, coup, eps, gen, sym_stat=False)
            n_grad_samp += int(ng.sum())
            div_samp += int(div.sum())
            depth_saturated += int((dep >= self.max_treedepth).sum())
            depths.append(float(dep.to(DTYPE).mean()))
            accs.append(float(acc.mean()))
            logp_hist.append(float(logp.mean()))
            pool.append(x.clone())
            if bar is not None:
                bar.update(1)
                if it % 50 == 0 or it == self.n_draws - 1:
                    bar.set_postfix_str(
                        f"E[logp]={logp_hist[-1]:,.1f} acc={accs[-1]:.2f} "
                        f"depth={depths[-1]:.1f}")
        if bar is not None:
            bar.close()
        runtime = time.time() - t0

        chain_draws = torch.stack(pool, dim=0)          # [draw, chain, dim]
        self.last_chain_draws = chain_draws.detach()    # for QoI-space ESS
        Z = chain_draws.reshape(-1, dim)

        def draw_fn(n):
            M = Z.shape[0]
            if n <= M:
                idx = torch.randperm(M, generator=gen)[:n]
            else:
                idx = torch.randint(M, (n,), generator=gen)
            with torch.no_grad():
                return target.constrained_draws(Z[idx])

        ess = pooled_ess(Z, N)
        coordinate_mean = chain_draws.mean((0, 1)).cpu().numpy()
        coordinate_sd = chain_draws.flatten(0, 1).std(0).cpu().numpy()
        coordinate_bulk_ess = np.full(dim, np.nan)
        coordinate_rank_rhat = np.full(dim, np.nan)
        try:
            import arviz as az

            values = chain_draws.permute(1, 0, 2).cpu().numpy()
            dataset = az.convert_to_dataset(values)
            coordinate_bulk_ess = np.asarray(
                az.ess(dataset, method="bulk")["x"]
            ).reshape(-1)
            coordinate_rank_rhat = np.asarray(
                az.rhat(dataset, method="rank")["x"]
            ).reshape(-1)
        except (ImportError, ValueError):
            # ArviZ is optional; the sampler remains usable without rank
            # diagnostics, while experiment environments install it.
            pass
        finite_ess = coordinate_bulk_ess[np.isfinite(coordinate_bulk_ess)]
        finite_rhat = coordinate_rank_rhat[np.isfinite(coordinate_rank_rhat)]
        total_grads = n_grad_warm + n_grad_samp
        diagnostics = {
            "final_elbo": float("nan"),
            "mean_logp": float(np.mean(logp_hist[-max(1, self.n_draws // 5):])),
            "pool_size": int(Z.shape[0]),
            "n_chains": N,
            "mass_matrix": self.mass_matrix,
            "step_size": float(eps.mean()),
            "accept_rate": float(np.mean(accs)),
            "mean_treedepth": float(np.mean(depths)),
            "treedepth_saturation_rate": float(
                depth_saturated / max(1, self.n_draws * N)),
            "divergences": int(div_samp),
            "divergences_warmup": int(div_warm),
            "grad_evals": int(n_grad_samp),
            "warmup_grad_evals": int(n_grad_warm),
            "ess": float(ess),
            "ess_per_grad": float(ess / max(1, total_grads)),
            "coordinate_mean": coordinate_mean.tolist(),
            "coordinate_sd": coordinate_sd.tolist(),
            "coordinate_bulk_ess": coordinate_bulk_ess.tolist(),
            "coordinate_rank_rhat": coordinate_rank_rhat.tolist(),
            "min_bulk_ess": (float(finite_ess.min())
                             if finite_ess.size else float("nan")),
            "max_rank_rhat": (float(finite_rhat.max())
                              if finite_rhat.size else float("nan")),
        }
        if self.mass_matrix == "lowrank" and s is not None:
            diagnostics["lowrank_k"] = float((s.abs() > 1e-12).to(DTYPE)
                                             .sum(-1).mean())
        if self.coupling:
            diagnostics["coupling_active"] = coup is not None
            diagnostics["coupling_refits"] = n_coup_refits
            diagnostics["coupling_scale_r2_max"] = coup_stats.get("scale_r2_max")
            if coup is not None:
                diagnostics["coupling_drivers"] = coup.idx_I.tolist()
                diagnostics["coupling_driver_stat"] = coup_stats.get("driver_stat")
                diagnostics["coupling_c_absmax"] = float(coup.c.abs().max())
                diagnostics["coupling_d_absmax"] = float(coup.d.abs().max())
                # full per-coordinate loadings: rows follow coupling_coupled,
                # columns follow coupling_drivers
                diagnostics["coupling_coupled"] = coup.idx_J.tolist()
                diagnostics["coupling_c"] = coup.c.tolist()
                diagnostics["coupling_d"] = coup.d.tolist()
        return InferenceResult(
            algorithm=self.name,
            target_name=target.name,
            _draw_fn=draw_fn,
            runtime_s=runtime,
            diagnostics=diagnostics,
        )
