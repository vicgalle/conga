"""CONGA case study: hierarchical spatio-temporal DLM for regional electricity load.

A conditionally linear-Gaussian state-space model whose fiber is the full
hourly state trajectory (trend + harmonic seasonals + dynamic temperature
response + holiday + national factor, CAR-coupled across zones through the
transmission-grid Laplacian) and whose core collects everything that breaks
Gaussianity: three daily stochastic-volatility paths with a shared
vol-of-vol, the CAR properization kappa, the temperature mean-reversion rho,
the factor root phi_f, the factor loadings lambda_{2:R}, the observation
noise offsets c_{2:R}, and per-component innovation log-variances.

Modes
    check   contract residual certificate + routing-signal diagnostics
    sample  matched NUTS-raw vs NUTS-CONGA comparison (nutpie, JAX scores)
    full    PJM-scale accounting: build the R=14, T=8760 model, validate,
            and time the exact collapsed value and value-and-gradient

    uv run --extra experiments python -m experiments.load_dlm --mode check
    uv run --extra experiments python -m experiments.load_dlm --mode sample
    uv run --extra experiments python -m experiments.load_dlm --mode full
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
LOG2PI = math.log(2.0 * math.pi)

CONFIGS = {
    # matched-inference scale: raw joint NUTS is still runnable
    "matched": dict(R=3, D=7, H_d=3, H_w=1, H_a=0, K_b=2),
    # small instance for dense-Hessian routing diagnostics
    "small": dict(R=3, D=2, H_d=3, H_w=1, H_a=0, K_b=2),
    # the proposal's PJM-like accounting scale
    "full": dict(R=14, D=365, H_d=8, H_w=4, H_a=3, K_b=4),
}


# --------------------------------------------------------------------------
# transmission graphs
# --------------------------------------------------------------------------

def grid_laplacian(R: int) -> np.ndarray:
    if R == 3:
        edges = [(0, 1), (1, 2)]
    elif R == 14:
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
                 (7, 0), (1, 8), (3, 9), (5, 10), (7, 11), (8, 12), (9, 13),
                 (2, 9), (4, 10), (6, 11), (0, 8), (10, 13), (11, 12)]
    else:
        edges = [(i, i + 1) for i in range(R - 1)]
    Q = np.zeros((R, R))
    for i, j in edges:
        Q[i, i] += 1.0
        Q[j, j] += 1.0
        Q[i, j] -= 1.0
        Q[j, i] -= 1.0
    return Q


# --------------------------------------------------------------------------
# model container: layout, data, priors
# --------------------------------------------------------------------------

class LoadDLM:
    def __init__(self, cfg: dict, sim_seed: int = 20260808):
        self.R, self.D = cfg["R"], cfg["D"]
        self.H_d, self.H_w, self.H_a = cfg["H_d"], cfg["H_w"], cfg["H_a"]
        self.K_b = cfg["K_b"]
        self.T = 24 * self.D
        R, D, T, K_b = self.R, self.D, self.T, self.K_b

        # ------ state layout (per-zone dynamic block, then f, then bbar)
        self.n_pairs = self.H_d + self.H_w + self.H_a
        self.d_z = 2 + 2 * self.n_pairs + K_b + 1
        self.d_dyn = R * self.d_z + 1
        self.d_s = self.d_dyn + R * K_b
        self.n_fiber = T * self.d_dyn + R * K_b

        zo = np.arange(R) * self.d_z
        self.mu_idx = zo + 0
        self.beta_idx = zo + 1
        pair0 = 2
        self.a_cols = np.stack([zo[:, None] + pair0 + 2 * np.arange(self.n_pairs)
                                for _ in range(1)], 0)[0]      # (R, n_pairs)
        self.b_cols = self.a_cols + 1
        self.bcoef_cols = zo[:, None] + pair0 + 2 * self.n_pairs + np.arange(K_b)
        self.h_idx = zo + 2 + 2 * self.n_pairs + K_b
        self.f_idx = R * self.d_z
        self.bbar_cols = self.d_dyn + np.arange(R * K_b).reshape(R, K_b)

        # rotation angles and per-pair component id (0=daily,1=weekly,2=annual)
        omegas, pair_comp = [], []
        for h in range(self.H_d):
            omegas.append(2 * np.pi * (h + 1) / 24.0); pair_comp.append(0)
        for h in range(self.H_w):
            omegas.append(2 * np.pi * (h + 1) / 168.0); pair_comp.append(1)
        for h in range(self.H_a):
            omegas.append(2 * np.pi * (h + 1) / 8760.0); pair_comp.append(2)
        self.rot_cos = np.cos(np.array(omegas))
        self.rot_sin = np.sin(np.array(omegas))
        self.pair_comp = np.array(pair_comp, dtype=int)

        # component log-variance scales in core order
        self.scale_names = ["lbeta", "ld", "lw"] + (["la"] if self.H_a else []) \
            + ["lb", "lh"]
        self.scale_prior_mean = np.array(
            [-10.0, -8.0, -10.0] + ([-12.0] if self.H_a else []) + [-8.0, -8.0])
        # map each pair-component to its scale slot
        self.pair_scale_slot = np.array(
            [1, 2, 3 if self.H_a else -1])[self.pair_comp]

        self.n_scales = len(self.scale_names)
        self.k_core = 3 * D + 4 + 2 * (R - 1) + self.n_scales

        # ------ graph
        self.Q = grid_laplacian(R)

        # ------ core prior (Gaussian on the unconstrained coordinates)
        pm = np.zeros(self.k_core)
        ps = np.ones(self.k_core)
        i = 3 * D
        pm[i:i + 4] = [-2.3, -1.0, 2.2, 1.5]        # lv, lkap, arho, aphi
        ps[i:i + 4] = [0.5, 1.0, 1.0, 0.75]
        i += 4
        pm[i:i + R - 1] = 1.0; ps[i:i + R - 1] = 0.5    # lambda_{2:R}
        i += R - 1
        pm[i:i + R - 1] = 0.0; ps[i:i + R - 1] = 0.5    # c_{2:R}
        i += R - 1
        pm[i:] = self.scale_prior_mean
        ps[i:] = 1.0
        self.core_prior_mean, self.core_prior_sd = pm, ps

        # initial values of the three vol paths: mean / sd of the first coord
        self.vol0 = np.array([[-7.0, 1.0], [-5.0, 1.0], [-4.5, 1.0]])

        # ------ initial state prior (diagonal), applies to s_1
        p0 = np.ones(self.d_s)
        p0[self.mu_idx] = 2.0 ** 2
        p0[self.beta_idx] = 0.05 ** 2
        p0[self.a_cols.ravel()] = 0.7 ** 2
        p0[self.b_cols.ravel()] = 0.7 ** 2
        p0[self.bcoef_cols.ravel()] = 1.0
        p0[self.h_idx] = 0.5 ** 2
        p0[self.f_idx] = 1.0
        p0[self.bbar_cols.ravel()] = 1.0
        self.P0_diag = p0
        self.m0 = np.zeros(self.d_s)

        # ------ constant part of the transition matrix
        G0 = np.zeros((self.d_s, self.d_s))
        for r in range(R):
            G0[self.mu_idx[r], self.mu_idx[r]] = 1.0
            G0[self.mu_idx[r], self.beta_idx[r]] = 1.0
            G0[self.beta_idx[r], self.beta_idx[r]] = 1.0
            for p in range(self.n_pairs):
                a, b = self.a_cols[r, p], self.b_cols[r, p]
                G0[a, a] = self.rot_cos[p]; G0[a, b] = self.rot_sin[p]
                G0[b, a] = -self.rot_sin[p]; G0[b, b] = self.rot_cos[p]
            G0[self.h_idx[r], self.h_idx[r]] = 1.0
        for j in self.bbar_cols.ravel():
            G0[j, j] = 1.0
        self.G0 = G0

        # constant part of the observation matrix (per-eta f column, per-t rest)
        F0 = np.zeros((R, self.d_s))
        for r in range(R):
            F0[r, self.mu_idx[r]] = 1.0
            F0[r, self.a_cols[r]] = 1.0
        self.F0 = F0

        # ------ data: temperatures, holiday, day index, simulated y
        self.day = np.repeat(np.arange(D), 24)
        self._simulate(sim_seed)

        # jnp copies used by the density code
        self._j = {k: jnp.asarray(v) for k, v in dict(
            y=self.y, phi=self.phi, hol=self.hol, dayf=self.day,
            Q=self.Q, G0=self.G0, F0=self.F0, m0=self.m0, P0=np.diag(self.P0_diag),
            P0_diag=self.P0_diag, pm=self.core_prior_mean, ps=self.core_prior_sd,
        ).items()}

    # ------------------------------------------------------------------
    def temp_basis(self, temp: np.ndarray) -> np.ndarray:
        """Piecewise-linear heating/cooling arms (K_b of them)."""
        if self.K_b == 2:
            arms = [np.maximum(16.0 - temp, 0.0), np.maximum(temp - 22.0, 0.0)]
        else:
            arms = [np.maximum(16.0 - temp, 0.0), np.maximum(10.0 - temp, 0.0),
                    np.maximum(temp - 22.0, 0.0), np.maximum(temp - 28.0, 0.0)]
        return np.stack(arms, axis=-1) / 10.0

    def _simulate(self, seed: int):
        R, D, T, K_b = self.R, self.D, self.T, self.K_b
        rng = np.random.default_rng(seed)
        hours = np.arange(T)

        zone_off = np.linspace(-2.0, 2.0, R)
        temp = (20.0 + zone_off[None, :]
                + 8.0 * np.sin(2 * np.pi * (hours[:, None] - 9.0) / 24.0))
        ar = np.zeros(R)
        noise = rng.normal(0, 0.6, size=(T, R))
        rows = []
        for t in range(T):
            ar = 0.98 * ar + noise[t]
            rows.append(ar.copy())
        temp = temp + np.array(rows)
        # heat wave in zones 0 and 1 during days 4-5 (or mid-sample when long)
        hw_days = {4, 5} if D <= 14 else set(range(D // 2, D // 2 + 4))
        hw = np.isin(self.day, list(hw_days)).astype(float)
        temp[:, 0] += 9.0 * hw
        if R > 1:
            temp[:, 1] += 6.0 * hw
        self.temp = temp
        self.phi = self.temp_basis(temp)                       # (T, R, K_b)

        hol_day = 5 if D <= 14 else D - 40
        self.hol = (self.day == hol_day).astype(float)         # (T,)

        # --- true core
        sigv = 0.15
        truth = dict(sigma_v=sigv, kappa=0.3, rho=0.9, phi_f=0.95)
        lam = np.ones(R); lam[1:] = np.linspace(1.3, 0.7, R - 1)
        cof = np.zeros(R); cof[1:] = np.linspace(0.3, -0.4, R - 1)
        truth["lam"] = lam.tolist(); truth["c"] = cof.tolist()
        scal = dict(zip(self.scale_names, self.scale_prior_mean))
        z = -7.0 + np.cumsum(np.r_[0.0, sigv * rng.normal(size=D - 1)])
        z += 1.5 * np.isin(np.arange(D), list(hw_days))        # vol spike
        zf = -5.0 + np.cumsum(np.r_[0.0, sigv * rng.normal(size=D - 1)])
        xv = -4.5 + np.cumsum(np.r_[0.0, sigv * rng.normal(size=D - 1)])
        truth["z"], truth["zf"], truth["xv"] = z.tolist(), zf.tolist(), xv.tolist()
        self.truth = truth

        # --- transition pieces at the truth
        G = self.G0.copy()
        rho, phif = truth["rho"], truth["phi_f"]
        for r in range(R):
            for k in range(K_b):
                G[self.bcoef_cols[r, k], self.bcoef_cols[r, k]] = rho
                G[self.bcoef_cols[r, k], self.bbar_cols[r, k]] = 1.0 - rho
        G[self.f_idx, self.f_idx] = phif
        car_cov = np.linalg.inv(self.Q + truth["kappa"] * np.eye(R))
        car_chol = np.linalg.cholesky(car_cov)

        bbar = 0.4 + 0.3 * rng.uniform(size=(R, K_b))
        s = np.zeros(self.d_s)
        s[self.mu_idx] = rng.normal(0, 0.4, R)
        s[self.a_cols[:, 0]] = rng.normal(0.8, 0.1, R)
        if self.n_pairs > 1:
            s[self.a_cols[:, 1]] = rng.normal(0.3, 0.05, R)
        s[self.b_cols[:, 0]] = rng.normal(0, 0.2, R)
        s[self.bcoef_cols.ravel()] = (bbar + rng.normal(0, 0.1, (R, K_b))).ravel()
        s[self.h_idx] = rng.normal(-0.4, 0.1, R)
        s[self.f_idx] = rng.normal(0, 0.3)
        s[self.bbar_cols.ravel()] = bbar.ravel()
        self.truth["bbar"] = bbar.tolist()

        sd_beta = math.exp(0.5 * scal["lbeta"])
        sd_h = math.exp(0.5 * scal["lh"])
        sd_b = math.exp(0.5 * scal["lb"])
        pair_sd = np.array([math.exp(0.5 * scal["ld"]),
                            math.exp(0.5 * scal["lw"]),
                            math.exp(0.5 * scal.get("la", -12.0))])[self.pair_comp]

        y = np.zeros((T, R))
        states = np.zeros((T, self.d_dyn))
        for t in range(T):
            if t > 0:
                d = self.day[t]
                s = G @ s
                s[self.mu_idx] += math.exp(0.5 * z[d]) * (car_chol @ rng.normal(size=R))
                s[self.beta_idx] += sd_beta * rng.normal(size=R)
                pn = pair_sd[None, :] * rng.normal(size=(R, self.n_pairs))
                s[self.a_cols.ravel()] += pn.ravel()
                pn = pair_sd[None, :] * rng.normal(size=(R, self.n_pairs))
                s[self.b_cols.ravel()] += pn.ravel()
                s[self.bcoef_cols.ravel()] += sd_b * rng.normal(size=R * K_b)
                s[self.h_idx] += sd_h * rng.normal(size=R)
                s[self.f_idx] += math.exp(0.5 * zf[self.day[t]]) * rng.normal()
            F = self.F0.copy()
            F[:, self.f_idx] = lam
            F[np.arange(R)[:, None], self.bcoef_cols] = self.phi[t]
            F[np.arange(R), self.h_idx] = self.hol[t]
            mean = F @ s
            obs_sd = np.exp(0.5 * (xv[self.day[t]] + cof))
            y[t] = mean + obs_sd * rng.normal(size=R)
            states[t] = s[:self.d_dyn]
        self.y = y
        self.true_states = states

    # ------------------------------------------------------------------
    # core unpacking (jax)
    # ------------------------------------------------------------------
    def unpack_core(self, eta):
        D, R = self.D, self.R
        uz, uzf, uxv = eta[:D], eta[D:2 * D], eta[2 * D:3 * D]
        i = 3 * D
        lv, lkap, arho, aphi = eta[i], eta[i + 1], eta[i + 2], eta[i + 3]
        i += 4
        lam = jnp.concatenate([jnp.ones(1), eta[i:i + R - 1]]); i += R - 1
        cof = jnp.concatenate([jnp.zeros(1), eta[i:i + R - 1]]); i += R - 1
        scales = eta[i:]
        sigv = jnp.exp(lv)
        kappa = jnp.exp(lkap)
        rho = jax.nn.sigmoid(arho)
        phif = jnp.tanh(aphi)

        def path(u, m0, s0):
            z0 = m0 + s0 * u[0]
            return jnp.concatenate([z0[None], z0 + jnp.cumsum(sigv * u[1:])])

        z = path(uz, *self.vol0[0])
        zf = path(uzf, *self.vol0[1])
        xv = path(uxv, *self.vol0[2])
        return dict(z=z, zf=zf, xv=xv, sigv=sigv, kappa=kappa, rho=rho,
                    phif=phif, lam=lam, cof=cof, scales=scales)

    def core_logprior(self, eta):
        j = self._j
        u = (eta - j["pm"]) / j["ps"]
        return (-0.5 * jnp.sum(u ** 2) - jnp.sum(jnp.log(j["ps"]))
                - 0.5 * self.k_core * LOG2PI)

    def _scale_vec_pairs(self, scales):
        """log-variance per harmonic pair, from the component scales."""
        table = [scales[1], scales[2]] + ([scales[3]] if self.H_a else [])
        return jnp.stack(table)[jnp.asarray(self.pair_comp)]

    def _G(self, c):
        G = self._j["G0"]
        bc = self.bcoef_cols.ravel()
        G = G.at[bc, bc].set(c["rho"])
        G = G.at[self.bcoef_cols.ravel(), self.bbar_cols.ravel()].set(1.0 - c["rho"])
        G = G.at[self.f_idx, self.f_idx].set(c["phif"])
        return G

    def _obs_var(self, c):
        """(T, R) observation noise variances."""
        return jnp.exp(c["xv"][jnp.asarray(self.day)][:, None] + c["cof"][None, :])

    # ------------------------------------------------------------------
    # raw joint log density over q = [states (T*d_dyn), bbar (R*K_b), core]
    # ------------------------------------------------------------------
    def split_raw(self, q):
        T, d = self.T, self.d_dyn
        x = q[:T * d].reshape(T, d)
        bbar = q[T * d:T * d + self.R * self.K_b].reshape(self.R, self.K_b)
        eta = q[T * d + self.R * self.K_b:]
        return x, bbar, eta

    def raw_logp(self, q):
        j = self._j
        R, T, K_b = self.R, self.T, self.K_b
        x, bbar, eta = self.split_raw(q)
        c = self.unpack_core(eta)
        day = jnp.asarray(self.day)

        # observation
        yhat = (x[:, self.mu_idx]
                + x[:, self.a_cols.reshape(-1)].reshape(T, R, self.n_pairs).sum(-1)
                + (x[:, self.bcoef_cols.reshape(-1)].reshape(T, R, K_b)
                   * j["phi"]).sum(-1)
                + j["hol"][:, None] * x[:, self.h_idx]
                + c["lam"][None, :] * x[:, self.f_idx][:, None])
        ovar = self._obs_var(c)
        resid = j["y"] - yhat
        ll = -0.5 * jnp.sum(LOG2PI + jnp.log(ovar) + resid ** 2 / ovar)

        # transitions t=2..T
        xp, xc = x[:-1], x[1:]
        dcur = day[1:]
        mu_p = xp[:, self.mu_idx] + xp[:, self.beta_idx]
        w_mu = xc[:, self.mu_idx] - mu_p
        A = j["Q"] + c["kappa"] * jnp.eye(R)
        sign, logdetA = jnp.linalg.slogdet(A)
        quad = jnp.einsum("ti,ij,tj->t", w_mu, A, w_mu)
        z_d = c["z"][dcur]
        ll += jnp.sum(-0.5 * (R * LOG2PI + R * z_d - logdetA)
                      - 0.5 * jnp.exp(-z_d) * quad)

        def diag_trans(w, logvar):
            return jnp.sum(-0.5 * (LOG2PI + logvar + w ** 2 * jnp.exp(-logvar)))

        s = c["scales"]
        ll += diag_trans(xc[:, self.beta_idx] - xp[:, self.beta_idx], s[0])
        # harmonics
        a_p = xp[:, self.a_cols.reshape(-1)].reshape(-1, R, self.n_pairs)
        b_p = xp[:, self.b_cols.reshape(-1)].reshape(-1, R, self.n_pairs)
        a_c = xc[:, self.a_cols.reshape(-1)].reshape(-1, R, self.n_pairs)
        b_c = xc[:, self.b_cols.reshape(-1)].reshape(-1, R, self.n_pairs)
        cosw = jnp.asarray(self.rot_cos)[None, None, :]
        sinw = jnp.asarray(self.rot_sin)[None, None, :]
        w_a = a_c - (cosw * a_p + sinw * b_p)
        w_b = b_c - (-sinw * a_p + cosw * b_p)
        lv_pairs = self._scale_vec_pairs(s)[None, None, :]
        ll += jnp.sum(-0.5 * (LOG2PI + lv_pairs + w_a ** 2 * jnp.exp(-lv_pairs)))
        ll += jnp.sum(-0.5 * (LOG2PI + lv_pairs + w_b ** 2 * jnp.exp(-lv_pairs)))
        # temperature coefficients, mean reverting to bbar
        b_pc = xp[:, self.bcoef_cols.reshape(-1)].reshape(-1, R, K_b)
        b_cc = xc[:, self.bcoef_cols.reshape(-1)].reshape(-1, R, K_b)
        w_bc = b_cc - (c["rho"] * b_pc + (1 - c["rho"]) * bbar[None])
        ll += diag_trans(w_bc, s[-2])
        # holiday RW
        ll += diag_trans(xc[:, self.h_idx] - xp[:, self.h_idx], s[-1])
        # factor AR(1) with its own SV path
        w_f = xc[:, self.f_idx] - c["phif"] * xp[:, self.f_idx]
        zf_d = c["zf"][dcur]
        ll += jnp.sum(-0.5 * (LOG2PI + zf_d + w_f ** 2 * jnp.exp(-zf_d)))

        # initial state prior (dynamic block + statics)
        p0 = j["P0_diag"]
        s1 = jnp.concatenate([x[0], bbar.reshape(-1)])
        u0 = (s1 - j["m0"]) / jnp.sqrt(p0)
        ll += jnp.sum(-0.5 * (LOG2PI + jnp.log(p0) + u0 ** 2))

        return ll + self.core_logprior(eta)

    # ------------------------------------------------------------------
    # collapsed core density: Kalman prediction-error decomposition
    # ------------------------------------------------------------------
    def _filter_pieces(self, eta):
        j = self._j
        R = self.R
        c = self.unpack_core(eta)
        G = self._G(c)
        A = j["Q"] + c["kappa"] * jnp.eye(R)
        car_cov = jnp.linalg.inv(A)
        car_full = jnp.zeros((self.d_s, self.d_s)).at[
            jnp.ix_(jnp.asarray(self.mu_idx), jnp.asarray(self.mu_idx))].set(car_cov)
        eff = jnp.zeros((self.d_s, self.d_s)).at[self.f_idx, self.f_idx].set(1.0)

        wdiag = jnp.zeros(self.d_s)
        s = c["scales"]
        wdiag = wdiag.at[jnp.asarray(self.beta_idx)].set(jnp.exp(s[0]))
        lv_pairs = self._scale_vec_pairs(s)
        pv = jnp.exp(lv_pairs)
        wdiag = wdiag.at[jnp.asarray(self.a_cols.reshape(-1))].set(
            jnp.tile(pv, R))
        wdiag = wdiag.at[jnp.asarray(self.b_cols.reshape(-1))].set(
            jnp.tile(pv, R))
        wdiag = wdiag.at[jnp.asarray(self.bcoef_cols.reshape(-1))].set(jnp.exp(s[-2]))
        wdiag = wdiag.at[jnp.asarray(self.h_idx)].set(jnp.exp(s[-1]))
        Wdiag = jnp.diag(wdiag)

        day = jnp.asarray(self.day)
        ez = jnp.exp(c["z"])[day]
        ezf = jnp.exp(c["zf"])[day]
        ovar = self._obs_var(c)

        F0 = j["F0"].at[:, self.f_idx].set(c["lam"])
        zone_rows = jnp.arange(R)[:, None]
        bc_cols = jnp.asarray(self.bcoef_cols)
        h_cols = jnp.asarray(self.h_idx)

        def F_t(phi_t, hol_t):
            F = F0.at[zone_rows, bc_cols].set(phi_t)
            F = F.at[jnp.arange(R), h_cols].set(hol_t * jnp.ones(R))
            return F

        def W_t(ez_t, ezf_t):
            return Wdiag + ez_t * car_full + ezf_t * eff

        return c, G, F_t, W_t, ez, ezf, ovar

    def _update(self, mpred, Ppred, F, ov, y_t):
        R = self.R
        e = y_t - F @ mpred
        S = F @ Ppred @ F.T + jnp.diag(ov)
        L = jnp.linalg.cholesky(S)
        alpha = jax.scipy.linalg.solve_triangular(L, e, lower=True)
        ll_t = (-0.5 * R * LOG2PI - jnp.sum(jnp.log(jnp.diagonal(L)))
                - 0.5 * jnp.sum(alpha ** 2))
        K = jax.scipy.linalg.cho_solve((L, True), F @ Ppred).T
        m = mpred + K @ e
        Aj = jnp.eye(self.d_s) - K @ F
        P = Aj @ Ppred @ Aj.T + K @ jnp.diag(ov) @ K.T
        return m, P, ll_t

    def collapsed_logp(self, eta, remat_days: bool = True):
        j = self._j
        c, G, F_t, W_t, ez, ezf, ovar = self._filter_pieces(eta)
        phi, hol, y = j["phi"], j["hol"], j["y"]

        def hour_step(carry, inp):
            mpred, Ppred, ll = carry
            phi_t, hol_t, y_t, ez_t, ezf_t, ov_t = inp
            F = F_t(phi_t, hol_t)
            m, P, ll_t = self._update(mpred, Ppred, F, ov_t, y_t)
            mnext = G @ m
            Pnext = G @ P @ G.T + W_t(ez_t, ezf_t)
            return (mnext, Pnext, ll + ll_t), None

        # inputs at hour t: (phi, hol, y, W-day-scale for the step INTO t+1)
        ez_next = jnp.concatenate([ez[1:], ez[-1:]])
        ezf_next = jnp.concatenate([ezf[1:], ezf[-1:]])
        inputs = (phi, hol, y, ez_next, ezf_next, ovar)
        D = self.D

        def day_step(carry, day_inp):
            step = jax.checkpoint(
                lambda cy, di: jax.lax.scan(hour_step, cy, di)) if remat_days \
                else (lambda cy, di: jax.lax.scan(hour_step, cy, di))
            carry, _ = step(carry, day_inp)
            return carry, None

        day_inputs = jax.tree.map(
            lambda a: a.reshape((D, 24) + a.shape[1:]), inputs)
        init = (j["m0"], j["P0"], 0.0)
        (mlast, Plast, ll), _ = jax.lax.scan(day_step, init, day_inputs)
        return ll + self.core_logprior(eta)

    # ------------------------------------------------------------------
    # stored filter + FFBS (reconstruction, conditional log density)
    # ------------------------------------------------------------------
    def _filter_stored(self, eta):
        j = self._j
        c, G, F_t, W_t, ez, ezf, ovar = self._filter_pieces(eta)
        phi, hol, y = j["phi"], j["hol"], j["y"]

        def step(carry, inp):
            mpred, Ppred, ll = carry
            phi_t, hol_t, y_t, ez_t, ezf_t, ov_t = inp
            F = F_t(phi_t, hol_t)
            m, P, ll_t = self._update(mpred, Ppred, F, ov_t, y_t)
            mnext = G @ m
            Pnext = G @ P @ G.T + W_t(ez_t, ezf_t)
            return (mnext, Pnext, ll + ll_t), (m, P, mpred, Ppred)

        ez_next = jnp.concatenate([ez[1:], ez[-1:]])
        ezf_next = jnp.concatenate([ezf[1:], ezf[-1:]])
        inputs = (phi, hol, y, ez_next, ezf_next, ovar)
        init = (j["m0"], j["P0"], 0.0)
        (ml, Pl, ll), (m, P, mpred, Ppred) = jax.lax.scan(step, init, inputs)
        return ll, m, P, mpred, Ppred, G

    def _backward(self, eta, eps=None, xs=None):
        """FFBS backward pass over the non-redundant fiber coordinates.

        The statics (bbar) are single coordinates that ride along in every
        Kalman state, so the exact conditional factorizes as one full-state
        Gaussian at t=T followed by dynamic-subvector conditionals for t<T
        (the statics at t are copies of those at t+1, not new coordinates).
        Returns (trajectory, conditional logpdf).
        """
        _, m, P, mpred, Ppred, G = self._filter_stored(eta)
        d, dd = self.d_s, self.d_dyn
        sample = xs is None

        def logn(x, mean, L, k):
            r = jax.scipy.linalg.solve_triangular(L, x - mean, lower=True)
            return (-0.5 * k * LOG2PI - jnp.sum(jnp.log(jnp.diagonal(L)))
                    - 0.5 * jnp.sum(r ** 2))

        PT = 0.5 * (P[-1] + P[-1].T) + 1e-13 * jnp.eye(d)
        LT = jnp.linalg.cholesky(PT)
        if sample:
            sT = m[-1] + LT @ eps[-1]
        else:
            sT = xs[-1]
        lp = logn(sT, m[-1], LT, d)

        def step(carry, inp):
            s_next, lp = carry
            m_t, P_t, mpred_n, Ppred_n, e_t = inp
            Jt = jnp.linalg.solve(
                Ppred_n + 1e-13 * jnp.eye(d), G @ P_t).T
            mean = m_t + Jt @ (s_next - mpred_n)
            C = P_t - Jt @ Ppred_n @ Jt.T
            Cd = 0.5 * (C[:dd, :dd] + C[:dd, :dd].T) + 1e-14 * jnp.eye(dd)
            L = jnp.linalg.cholesky(Cd)
            if sample:
                dyn_t = mean[:dd] + L @ e_t[:dd]
            else:
                dyn_t = e_t[:dd]
            s_t = jnp.concatenate([dyn_t, s_next[dd:]])
            lp = lp + logn(dyn_t, mean[:dd], L, dd)
            return (s_t, lp), s_t

        seq = eps[:-1] if sample else xs[:-1]
        inps = (m[:-1][::-1], P[:-1][::-1], mpred[1:][::-1],
                Ppred[1:][::-1], seq[::-1])
        (s1, lp), traj_rev = jax.lax.scan(step, (sT, lp), inps)
        traj = jnp.concatenate([traj_rev[::-1], sT[None]], axis=0)
        return traj, lp

    def ffbs_draw(self, eta, eps):
        traj, _ = self._backward(eta, eps=eps)
        return traj

    def fiber_logpdf(self, eta, xs):
        _, lp = self._backward(eta, xs=xs)
        return lp

    # ------------------------------------------------------------------
    # common coordinates for matched diagnostics
    # ------------------------------------------------------------------
    def common_names_static(self):
        names = (["sigma_v", "kappa", "rho", "phi_f"]
                 + [f"lam[{r+1}]" for r in range(1, self.R)]
                 + [f"c[{r+1}]" for r in range(1, self.R)]
                 + self.scale_names)
        return names

    def core_common(self, eta):
        c = self.unpack_core(eta)
        stat = jnp.concatenate([
            jnp.stack([c["sigv"], c["kappa"], c["rho"], c["phif"]]),
            c["lam"][1:], c["cof"][1:], c["scales"]])
        return jnp.concatenate([c["z"], c["zf"], c["xv"], stat])

    def common_from_raw(self, q):
        x, bbar, eta = self.split_raw(q)
        return jnp.concatenate([
            self.core_common(eta), x.reshape(-1), bbar.reshape(-1)])

    def common_from_core(self, eta, eps):
        traj = self.ffbs_draw(eta, eps)
        x = traj[:, :self.d_dyn]
        bbar = traj[0, self.d_dyn:]
        return jnp.concatenate([
            self.core_common(eta), x.reshape(-1), bbar])


# --------------------------------------------------------------------------
# nutpie plumbing shared in spirit with the Stock--Watson driver
# --------------------------------------------------------------------------

class JaxScore:
    def __init__(self, logp, dim: int):
        self.vg = jax.jit(jax.value_and_grad(logp))
        self.dim = int(dim)
        self.calls = 0
        self.guard_hits = 0
        self._lock = threading.Lock()

    def compile(self):
        x = jnp.zeros(self.dim, dtype=jnp.float64)
        t0 = time.perf_counter()
        val, grad = self.vg(x)
        np.asarray(grad); float(val)
        return time.perf_counter() - t0

    def time_grad(self, n=20):
        x = jnp.zeros(self.dim, dtype=jnp.float64)
        val, grad = self.vg(x); jax.block_until_ready(grad)
        t0 = time.perf_counter()
        for _ in range(n):
            val, grad = self.vg(x)
        jax.block_until_ready(grad)
        return (time.perf_counter() - t0) / n

    def reset(self):
        with self._lock:
            self.calls = 0
            self.guard_hits = 0

    def __call__(self, x):
        with self._lock:
            self.calls += 1
        val, grad = self.vg(jnp.asarray(x, dtype=jnp.float64))
        value = float(val)
        gradient = np.asarray(grad, dtype=np.float64, order="C")
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            with self._lock:
                self.guard_hits += 1
            return -1e30, np.zeros(self.dim, dtype=np.float64)
        return value, gradient


def make_nutpie_model(score: JaxScore, initial: np.ndarray):
    from nutpie.compiled_pyfunc import from_pyfunc
    dim = score.dim

    def make_expand(seed1, seed2, chain):
        del seed1, seed2, chain
        return lambda x: {"theta": np.asarray(x, dtype=np.float64, order="C")}

    def make_init(seed):
        rng = np.random.default_rng(seed)
        return initial + 0.01 * rng.standard_normal(dim)

    return from_pyfunc(
        ndim=dim, make_logp_fn=lambda: score, make_expand_fn=make_expand,
        expanded_dtypes=[np.dtype("float64")], expanded_shapes=[(dim,)],
        expanded_names=["theta"], make_initial_point_fn=make_init,
        raw_logp_fn=None)


def diagnostics(common, k_core):
    import arviz as az
    dataset = az.convert_to_dataset(common)
    bulk = np.asarray(az.ess(dataset, method="bulk")["x"]).reshape(-1)
    rhat = np.asarray(az.rhat(dataset, method="rank")["x"]).reshape(-1)

    def summ(mask):
        b, r = bulk[mask], rhat[mask]
        b, r = b[np.isfinite(b)], r[np.isfinite(r)]
        return (float(b.min()) if b.size else None,
                float(r.max()) if r.size else None)

    n = bulk.shape[0]
    core_mask = np.zeros(n, dtype=bool); core_mask[:k_core] = True
    e_all, r_all = summ(np.ones(n, dtype=bool))
    e_core, r_core = summ(core_mask)
    e_fib, r_fib = summ(~core_mask)
    return {"min_bulk_ess": e_all, "max_rank_rhat": r_all,
            "min_bulk_ess_core": e_core, "max_rank_rhat_core": r_core,
            "min_bulk_ess_fiber": e_fib, "max_rank_rhat_fiber": r_fib}


def chunked_common(fn, positions, extra=None, batch=16):
    """Apply a vmapped jitted fn over draws in memory-bounded batches."""
    outs = []
    n = positions.shape[0]
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        if extra is None:
            outs.append(np.asarray(fn(positions[sl])))
        else:
            outs.append(np.asarray(fn(positions[sl], extra[sl])))
    return np.concatenate(outs, axis=0)


# --------------------------------------------------------------------------
# mode: check
# --------------------------------------------------------------------------

def run_check(args):
    out = {}
    model = LoadDLM(CONFIGS["matched"])
    out["matched_dims"] = dict(
        R=model.R, T=model.T, states_per_hour=model.d_dyn,
        kalman_state=model.d_s, fiber=model.n_fiber, core=model.k_core,
        joint=model.n_fiber + model.k_core)
    print("dims", json.dumps(out["matched_dims"]))

    # ---- residual certificate: r(x, eta) at dispersed probe points
    rng = np.random.default_rng(7)
    raw_logp = jax.jit(model.raw_logp)
    coll_logp = jax.jit(lambda e: model.collapsed_logp(e))
    fib_logpdf = jax.jit(model.fiber_logpdf)
    ffbs = jax.jit(model.ffbs_draw)

    resid, rel, dr_x = [], [], []
    for i in range(args.probes):
        eta = 0.3 * rng.standard_normal(model.k_core)
        l_coll = float(coll_logp(jnp.asarray(eta)))
        rs = []
        for _ in range(2):
            # dispersed but typical x: overdispersed conditional draws
            # (the fiber noise is scaled 1.5x through the FFBS innovations)
            eps = 1.5 * rng.standard_normal((model.T, model.d_s))
            traj = np.asarray(ffbs(jnp.asarray(eta), jnp.asarray(eps)))
            bbar = traj[0, model.d_dyn:]
            q = np.concatenate([traj[:, :model.d_dyn].reshape(-1), bbar, eta])
            l_joint = float(raw_logp(jnp.asarray(q)))
            l_cond = float(fib_logpdf(jnp.asarray(eta), jnp.asarray(traj)))
            r = l_joint - l_coll - l_cond
            rs.append(r)
            resid.append(r); rel.append(abs(r) / abs(l_joint))
            print(f"probe {i}: joint={l_joint:.6f} coll={l_coll:.6f} "
                  f"cond={l_cond:.6f} r={r:.3e}")
        # r(x, eta) must also be constant in x for each eta
        dr_x.append(abs(rs[0] - rs[1]))
    out["residual_certificate"] = dict(
        max_abs=float(np.max(np.abs(resid))), max_rel=float(np.max(rel)),
        max_dr_across_x=float(np.max(dr_x)), probes=args.probes)

    # ---- score timings at matched scale
    raw_score = JaxScore(model.raw_logp, model.n_fiber + model.k_core)
    core_score = JaxScore(
        lambda e: model.collapsed_logp(e, remat_days=False), model.k_core)
    out["compile_s"] = dict(raw=raw_score.compile(), core=core_score.compile())
    out["grad_seconds"] = dict(raw=raw_score.time_grad(),
                               core=core_score.time_grad())
    print("grad_seconds", json.dumps(out["grad_seconds"]))

    # ---- routing signals on the small instance (dense fiber Hessian)
    small = LoadDLM(CONFIGS["small"])
    nfib = small.n_fiber
    rng2 = np.random.default_rng(11)
    etas = [0.4 * rng2.standard_normal(small.k_core) for _ in range(2)]
    x0 = 0.3 * rng2.standard_normal(nfib)

    def joint_of_x(x, eta):
        return small.raw_logp(jnp.concatenate([x, jnp.asarray(eta)]))

    hess = jax.jit(lambda x, eta: jax.jacfwd(jax.grad(joint_of_x))(x, eta))
    lam1 = -np.asarray(hess(jnp.asarray(x0), etas[0]))
    lam2 = -np.asarray(hess(jnp.asarray(x0), etas[1]))
    x1 = 0.3 * rng2.standard_normal(nfib)
    lam1b = -np.asarray(hess(jnp.asarray(x1), etas[0]))
    hess_x_dev = float(np.max(np.abs(lam1 - lam1b))
                       / np.max(np.abs(lam1)))

    def comm_norm(a, b):
        return float(np.linalg.norm(a @ b - b @ a)
                     / (np.linalg.norm(a) * np.linalg.norm(b)))

    evals1, evecs1 = np.linalg.eigh(lam1)
    evals2, evecs2 = np.linalg.eigh(lam2)
    k = 10
    v1, v2 = evecs1[:, -k:], evecs2[:, -k:]
    sv = np.linalg.svd(v1.T @ v2, compute_uv=False)
    max_angle_deg = float(np.degrees(np.arccos(np.clip(sv.min(), -1, 1))))

    # CAR pencil contrast (paper's spectral branch: exactly commuting)
    Qp = grid_laplacian(64)
    pen1 = math.exp(-2 * 0.3) * (Qp + 0.5 * np.eye(64)) + math.exp(-2 * 0.1) * np.eye(64)
    pen2 = math.exp(2 * 0.4) * (Qp + 0.5 * np.eye(64)) + math.exp(-2 * 1.0) * np.eye(64)
    pencil_comm = comm_norm(pen1, pen2)

    # non-Toeplitz: diagonal block at two interior hours
    dz = small.d_dyn
    t1, t2 = 10, 30
    blk1 = lam1[t1 * dz:(t1 + 1) * dz, t1 * dz:(t1 + 1) * dz]
    blk2 = lam1[t2 * dz:(t2 + 1) * dz, t2 * dz:(t2 + 1) * dz]
    toeplitz_dev = float(np.max(np.abs(blk1 - blk2)) / np.max(np.abs(blk1)))

    # temporal band structure (dynamic block only; statics are an arrow)
    tmax = 0.0
    for dt in range(2, 5):
        for t in range(0, small.T - dt, 7):
            blk = lam1[t * dz:(t + 1) * dz, (t + dt) * dz:(t + dt + 1) * dz]
            tmax = max(tmax, float(np.max(np.abs(blk))))
    arrow = float(np.max(np.abs(lam1[:small.T * dz, small.T * dz:])))

    out["routing_signals"] = dict(
        fiber_hessian_x_invariance=hess_x_dev,
        commutator_rel_norm=comm_norm(lam1, lam2),
        car_pencil_commutator=pencil_comm,
        top10_eigvec_max_principal_angle_deg=max_angle_deg,
        toeplitz_block_rel_dev=toeplitz_dev,
        offband_max_abs=tmax,
        static_arrow_max_abs=arrow,
        small_fiber_dim=int(nfib))
    print("routing", json.dumps(out["routing_signals"], indent=2))

    path = ROOT / "results" / "runs" / "load_dlm_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path}")


# --------------------------------------------------------------------------
# mode: sample
# --------------------------------------------------------------------------

def summaries(model, common):
    flat = common.reshape(-1, common.shape[-1])
    names = model.common_names_static()
    idx = {}
    D = model.D
    for d in (0, D // 2, D - 1):
        idx[f"z[{d}]"] = d
        idx[f"zf[{d}]"] = D + d
        idx[f"xv[{d}]"] = 2 * D + d
    for i, nm in enumerate(names):
        idx[nm] = 3 * D + i
    n_core_common = 3 * D + len(names)
    # a few fiber coordinates: zone-0 trend and the factor at mid-sample
    tmid = model.T // 2
    idx["mu[0,mid]"] = n_core_common + tmid * model.d_dyn + model.mu_idx[0]
    idx["f[mid]"] = n_core_common + tmid * model.d_dyn + model.f_idx
    out = {}
    for nm, i in idx.items():
        v = flat[:, i]
        out[nm] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                       q05=float(np.quantile(v, 0.05)),
                       q95=float(np.quantile(v, 0.95)))
    return out


def run_sample(args):
    import nutpie
    model = LoadDLM(CONFIGS[args.config])
    k_common_core = 3 * model.D + len(model.common_names_static())
    raw_dim = model.n_fiber + model.k_core

    raw_score = JaxScore(model.raw_logp, raw_dim)
    core_score = JaxScore(
        lambda e: model.collapsed_logp(e, remat_days=False), model.k_core)
    tc_raw = raw_score.compile()
    tc_core = core_score.compile()
    g_raw = raw_score.time_grad(10)
    g_core = core_score.time_grad(10)
    print(f"compile raw={tc_raw:.1f}s core={tc_core:.1f}s  "
          f"grad raw={g_raw*1e3:.2f}ms core={g_core*1e3:.2f}ms", flush=True)

    # initial positions: core at zero, raw states at the conditional mean
    core_init = np.zeros(model.k_core)
    mean_traj = np.asarray(jax.jit(model.ffbs_draw)(
        jnp.asarray(core_init), jnp.zeros((model.T, model.d_s))))
    bbar0 = mean_traj[0, model.d_dyn:]
    raw_init = np.concatenate(
        [mean_traj[:, :model.d_dyn].reshape(-1), bbar0, core_init])

    models = {"raw": make_nutpie_model(raw_score, raw_init),
              "core": make_nutpie_model(core_score, core_init)}
    common_raw = jax.jit(jax.vmap(model.common_from_raw))
    common_core = jax.jit(jax.vmap(model.common_from_core))

    # the seed list is a run parameter, not part of the statistical protocol:
    # resume compares everything else and unions the seeds
    config = dict(chains=args.chains, warmup=args.warmup, draws=args.draws,
                  maxdepth=args.maxdepth,
                  target_accept=args.nuts_target_accept, config=args.config)
    output = ROOT / "results" / "runs" / "load_dlm_sample.json"
    if args.resume and output.exists():
        payload = json.loads(output.read_text())
        stored = {k: v for k, v in payload.get("config", {}).items()
                  if k != "seeds"}
        if stored != config:
            raise RuntimeError("resume config mismatch")
        payload["config"]["seeds"] = sorted(
            set(payload["config"].get("seeds", [])) | set(args.seeds))
    else:
        payload = dict(
            model="load-DLM", dims=dict(
                R=model.R, T=model.T, d_dyn=model.d_dyn, d_s=model.d_s,
                fiber=model.n_fiber, core=model.k_core, raw=raw_dim),
            truth={k: v for k, v in model.truth.items()},
            config={**config, "seeds": list(args.seeds)},
            grad_seconds=dict(raw=g_raw, core=g_core),
            compile_seconds=dict(raw=tc_raw, core=tc_core), rows=[])

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, output)

    save()
    done = {(r["arm"], r["seed"]) for r in payload["rows"] if "error" not in r}
    arms = args.arms
    for seed in args.seeds:
        for arm in arms:
            if (arm, seed) in done:
                print(f"skip {arm} seed={seed}", flush=True)
                continue
            print(f"run {arm} seed={seed}", flush=True)
            collapsed = arm.endswith("CONGA")
            score = core_score if collapsed else raw_score
            score.reset()
            t0 = time.perf_counter()
            trace = nutpie.sample(
                models["core" if collapsed else "raw"],
                chains=args.chains, draws=args.draws, tune=args.warmup,
                seed=seed, progress_bar=False, save_warmup=True,
                maxdepth=args.maxdepth,
                max_energy_error=1000.0,
                target_accept=args.nuts_target_accept)
            seconds = time.perf_counter() - t0
            positions = np.asarray(trace.posterior["theta"].values)
            N, S, _ = positions.shape
            r0 = time.perf_counter()
            flat = positions.reshape(N * S, -1)
            if collapsed:
                eps = np.random.default_rng(seed + 777).standard_normal(
                    (N * S, model.T, model.d_s))
                common = chunked_common(common_core, flat, extra=eps, batch=8)
            else:
                common = chunked_common(common_raw, flat, batch=64)
            common = common.reshape(N, S, -1)
            rec_s = time.perf_counter() - r0
            total_steps = int(trace.sample_stats["n_steps"].sum())
            row = dict(
                arm=arm, seed=seed, sampler="nutpie NUTS",
                sampled_dim=int(positions.shape[-1]), seconds=seconds,
                reconstruction_s=rec_s, score_calls=score.calls,
                guard_hits=score.guard_hits,
                mean_steps=float(trace.sample_stats["n_steps"].mean()),
                total_steps=total_steps,
                divergences=int(trace.sample_stats["diverging"].sum()),
                mean_accept=float(trace.sample_stats["mean_tree_accept"].mean()),
                **diagnostics(common, k_common_core))
            row["summary"] = summaries(model, common)
            ess = row["min_bulk_ess"]
            row["ess_per_s"] = (ess / (seconds + rec_s)) if ess else None
            payload["rows"] = [r for r in payload["rows"]
                               if (r["arm"], r["seed"]) != (arm, seed)]
            payload["rows"].append(row)
            save()
            print(json.dumps({k: v for k, v in row.items() if k != "summary"},
                             sort_keys=True), flush=True)
    print(f"saved -> {output}")


# --------------------------------------------------------------------------
# mode: full  (PJM-scale accounting)
# --------------------------------------------------------------------------

def run_full(args):
    t_build0 = time.perf_counter()
    model = LoadDLM(CONFIGS["full"])
    build_s = time.perf_counter() - t_build0
    n = model.n_fiber
    out = dict(dims=dict(
        R=model.R, T=model.T, states_per_hour=model.d_dyn,
        kalman_state=model.d_s, fiber=n, core=model.k_core,
        joint=n + model.k_core), build_and_simulate_s=build_s)
    print("dims", json.dumps(out["dims"]), flush=True)

    # oracle-discovery accounting (dense conflict graph is quadratic)
    out["discovery_accounting"] = dict(
        conflict_graph_pairs=float(n) * (n - 1) / 2.0,
        conflict_graph_bytes_at_1bit=float(n) * n / 8.0,
        note="dense conflict graph at 4.5M fiber coordinates needs ~2.6 TB "
             "at one bit per pair; the model-algebra contract path is the "
             "only realistic front door")
    print(json.dumps(out["discovery_accounting"]), flush=True)

    eta0 = jnp.asarray(0.1 * np.random.default_rng(3).standard_normal(model.k_core))

    # value only
    val_fn = jax.jit(lambda e: model.collapsed_logp(e, remat_days=False))
    t0 = time.perf_counter(); v = float(val_fn(eta0))
    compile_val_s = time.perf_counter() - t0
    t0 = time.perf_counter(); v = float(val_fn(eta0))
    value_s = time.perf_counter() - t0
    out["collapsed_value"] = dict(compile_s=compile_val_s, seconds=value_s,
                                  value=v)
    print("value", json.dumps(out["collapsed_value"]), flush=True)

    # value and gradient with per-day rematerialization
    vg_fn = jax.jit(jax.value_and_grad(lambda e: model.collapsed_logp(e)))
    t0 = time.perf_counter()
    val, grad = vg_fn(eta0); jax.block_until_ready(grad)
    compile_vg_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    val, grad = vg_fn(eta0); jax.block_until_ready(grad)
    grad_s = time.perf_counter() - t0
    finite = bool(np.all(np.isfinite(np.asarray(grad))))
    out["collapsed_value_and_grad"] = dict(
        compile_s=compile_vg_s, seconds=grad_s, value=float(val),
        grad_finite=finite,
        grad_norm=float(np.linalg.norm(np.asarray(grad))))
    print("value_and_grad", json.dumps(out["collapsed_value_and_grad"]),
          flush=True)

    path = ROOT / "results" / "runs" / "load_dlm_full.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {path}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["check", "sample", "full"],
                    default="check")
    ap.add_argument("--config", default="matched")
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arms", nargs="+",
                    default=["NUTS-CONGA", "NUTS-raw"])
    ap.add_argument("--maxdepth", type=int, default=10)
    ap.add_argument("--nuts-target-accept", type=float, default=0.9)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="small contract check or short sampling run")
    args = ap.parse_args()
    if args.quick:
        args.probes = min(args.probes, 1)
        args.chains = 2
        args.warmup = 20
        args.draws = 20
        args.seeds = [0]
    if args.mode == "check":
        run_check(args)
    elif args.mode == "sample":
        run_sample(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
