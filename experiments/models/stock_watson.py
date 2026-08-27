"""Stock--Watson inflation model used in the WALNUTS evaluation.

The target follows Section 4.4 of Bou-Rabee et al. (2025), including its
innovation parameterization.  The original centered model and the data are
from the public replication code for Kleppe (2019), which the WALNUTS paper
cites as its data source.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch

from conga.core import DTYPE, TargetPosterior


class StockWatson(TargetPosterior):
    r"""Three coupled random walks for quarterly US inflation.

    The unconstrained state is exactly the innovation parameterization used
    in the WALNUTS paper::

        (z[1], diff(z) / sigma,
         x[1], diff(x) / sigma,
         tau[1], diff(tau) * exp(-z[:-1] / 2),
         log(sigma**2)).

    The first element of each process has the improper flat prior used by the
    original Stan program.  The posterior is proper after conditioning on the
    observations.  Constants independent of the state are omitted.
    """

    def __init__(self, data: dict):
        self.y = torch.as_tensor(np.asarray(data["y"], dtype=float), dtype=DTYPE)
        self.T = int(data.get("T", self.y.numel()))
        if self.T != self.y.numel() or self.T < 2:
            raise ValueError("Stock--Watson data require T == len(y) >= 2")
        self.alpha = float(data.get("alpha", 5.0))
        self.beta = float(data.get("beta", 0.5))
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError("Gamma shape and rate must be positive")
        self.name = "stock_watson-stock_watson"
        self.dim = 3 * self.T

    def _split(self, q: torch.Tensor):
        T = self.T
        z_bar = q[..., :T - 1]
        x_bar = q[..., T - 1:2 * T - 1]
        tau_bar = q[..., 2 * T - 1:3 * T - 1]
        log_sigma2 = q[..., -1]
        return z_bar, x_bar, tau_bar, log_sigma2

    @staticmethod
    def _random_walk(initial: torch.Tensor, innovations: torch.Tensor,
                     scale: torch.Tensor) -> torch.Tensor:
        if scale.ndim == initial.ndim:
            scale = scale.unsqueeze(-1)
        tail = initial.unsqueeze(-1) + (scale * innovations).cumsum(-1)
        return torch.cat([initial.unsqueeze(-1), tail], dim=-1)

    def constrain(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_bar, x_bar, tau_bar, log_sigma2 = self._split(q)
        sigma = torch.exp(0.5 * log_sigma2)
        z = self._random_walk(z_bar[..., 0], z_bar[..., 1:], sigma)
        x = self._random_walk(x_bar[..., 0], x_bar[..., 1:], sigma)
        tau_scale = torch.exp(0.5 * z)
        tau = self._random_walk(tau_bar[..., 0], tau_bar[..., 1:], tau_scale)
        return {
            "z": z,
            "x": x,
            "tau": tau,
            "sigma": sigma,
            "log_sigma2": log_sigma2,
        }

    def log_prob(self, q: torch.Tensor) -> torch.Tensor:
        z_bar, x_bar, tau_bar, log_sigma2 = self._split(q)
        con = self.constrain(q)

        # With precision sigma^-2 ~ Gamma(alpha, beta), transforming to
        # log(sigma^2) gives -alpha*l - beta*exp(-l).  The random-walk
        # normalizers cancel the Jacobian of the innovation map.
        lp = -self.alpha * log_sigma2 - self.beta * torch.exp(-log_sigma2)
        lp = lp - 0.5 * (
            (z_bar[..., 1:] ** 2).sum(-1)
            + (x_bar[..., 1:] ** 2).sum(-1)
            + (tau_bar[..., 1:] ** 2).sum(-1)
        )

        x, tau = con["x"], con["tau"]
        scaled_resid = (self.y - tau) * torch.exp(-0.5 * x)
        return lp - 0.5 * x.sum(-1) - 0.5 * (scaled_resid ** 2).sum(-1)

    def numpy_logp_grad(self, q: np.ndarray):
        """Single-point density and analytic score for callback-based samplers."""
        q = np.asarray(q, dtype=np.float64)
        T = self.T
        z_bar = q[:T - 1]
        x_bar = q[T - 1:2 * T - 1]
        tau_bar = q[2 * T - 1:3 * T - 1]
        log_sigma2 = q[-1]

        with np.errstate(over="ignore", invalid="ignore"):
            sigma = np.exp(0.5 * log_sigma2)
            z = np.concatenate(([z_bar[0]],
                                z_bar[0] + sigma * np.cumsum(z_bar[1:])))
            x = np.concatenate(([x_bar[0]],
                                x_bar[0] + sigma * np.cumsum(x_bar[1:])))
            tau_scale = np.exp(0.5 * z)
            tau = np.concatenate((
                [tau_bar[0]],
                tau_bar[0] + np.cumsum(tau_scale * tau_bar[1:]),
            ))

            resid = self.y.numpy() - tau
            obs_precision = np.exp(-x)
            lp = (-self.alpha * log_sigma2
                  - self.beta * np.exp(-log_sigma2)
                  - 0.5 * np.dot(z_bar[1:], z_bar[1:])
                  - 0.5 * np.dot(x_bar[1:], x_bar[1:])
                  - 0.5 * np.dot(tau_bar[1:], tau_bar[1:])
                  - 0.5 * x.sum()
                  - 0.5 * np.dot(resid * resid, obs_precision))

            # Reverse the three cumulative-sum reconstructions.  Each suffix
            # sum is the adjoint of one random-walk prefix sum.
            adj_tau = resid * obs_precision
            suffix_tau = np.cumsum(adj_tau[::-1])[::-1]
            grad_tau_bar = np.empty(T)
            grad_tau_bar[0] = suffix_tau[0]
            grad_tau_bar[1:] = tau_scale * suffix_tau[1:] - tau_bar[1:]
            adj_z = 0.5 * tau_scale * tau_bar[1:] * suffix_tau[1:]

            adj_x = -0.5 + 0.5 * resid * resid * obs_precision
            suffix_x = np.cumsum(adj_x[::-1])[::-1]
            grad_x_bar = np.empty(T)
            grad_x_bar[0] = suffix_x[0]
            grad_x_bar[1:] = sigma * suffix_x[1:] - x_bar[1:]

            suffix_z = np.cumsum(adj_z[::-1])[::-1]
            grad_z_bar = np.empty(T - 1)
            grad_z_bar[0] = suffix_z[0]
            grad_z_bar[1:] = sigma * suffix_z[1:] - z_bar[1:]

            adj_sigma = (np.dot(x_bar[1:], suffix_x[1:])
                         + np.dot(z_bar[1:], suffix_z[1:]))
            grad_log_sigma2 = (-self.alpha
                               + self.beta * np.exp(-log_sigma2)
                               + 0.5 * sigma * adj_sigma)

        grad = np.concatenate((grad_z_bar, grad_x_bar, grad_tau_bar,
                               [grad_log_sigma2]))
        return float(lp), grad


class StockWatsonCenteredTau(StockWatson):
    """Stock--Watson with the level path in centered coordinates.

    The persistent and transient volatility paths retain the WALNUTS
    innovation coordinates, while ``tau`` is represented directly.  This is
    the exact temporal fiber exposed by the Stock--Watson model contract:
    conditional on ``(z, x, log_sigma2)``, the level path has tridiagonal
    Gaussian precision.
    """

    def __init__(self, data: dict):
        super().__init__(data)
        self.name = "stock_watson-stock_watson_centered_tau"

    def constrain(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_bar, x_bar, tau, log_sigma2 = self._split(q)
        sigma = torch.exp(0.5 * log_sigma2)
        z = self._random_walk(z_bar[..., 0], z_bar[..., 1:], sigma)
        x = self._random_walk(x_bar[..., 0], x_bar[..., 1:], sigma)
        return {
            "z": z,
            "x": x,
            "tau": tau,
            "sigma": sigma,
            "log_sigma2": log_sigma2,
        }

    def log_prob(self, q: torch.Tensor) -> torch.Tensor:
        z_bar, x_bar, tau, log_sigma2 = self._split(q)
        con = self.constrain(q)
        z, x = con["z"], con["x"]

        # The innovation density for tau_bar, multiplied by its exact
        # Jacobian to centered tau, is the centered random-walk density.
        lp = -self.alpha * log_sigma2 - self.beta * torch.exp(-log_sigma2)
        lp = lp - 0.5 * (
            (z_bar[..., 1:] ** 2).sum(-1)
            + (x_bar[..., 1:] ** 2).sum(-1)
        )
        tau_increments = tau[..., 1:] - tau[..., :-1]
        tau_scale = torch.exp(0.5 * z)
        lp = lp - 0.5 * z.sum(-1)
        lp = lp - 0.5 * ((tau_increments / tau_scale) ** 2).sum(-1)

        scaled_resid = (self.y - tau) * torch.exp(-0.5 * x)
        return lp - 0.5 * x.sum(-1) - 0.5 * (scaled_resid ** 2).sum(-1)


def builders() -> Dict[str, Callable[[dict], TargetPosterior]]:
    return {"stock_watson-stock_watson": StockWatson}
