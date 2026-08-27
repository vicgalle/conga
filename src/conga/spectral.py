"""Exact spectral and sparse transports for scalar-scale graph GMRFs.

For a fixed graph precision ``Q0`` and core ``eta = (u, v)``, the conditional
precision is ``exp(-2u) Q0 + exp(-2v) I``.  Its matrices commute over the core,
so one cached eigenbasis supplies the symmetric (fiberwise Brenier) transport.
Path and square-lattice Laplacians use orthonormal cosine transforms instead of
materializing that basis.  The sparse backend below is the exact-contract
Cholesky/Knothe--Rosenblatt comparator; unlike ``SparseGaussianBlock`` it does
not charge oracle recovery on every score call.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy import fft, sparse

from .sparse import SparsePrecisionBackend, SparsePrecisionFactor


def _eta_pair(eta) -> tuple[float, float]:
    value = np.asarray(eta, dtype=np.float64)
    if value.shape != (2,):
        raise ValueError(f"eta must have shape (2,), got {value.shape}")
    return float(value[0]), float(value[1])


def _collapsed_value_gradient(eigenvalues, y_coefficients, eta):
    """Collapsed target used by ``GraphGMRFTarget``, including its core prior."""
    values = np.asarray(eta, dtype=np.float64)
    scalar = values.ndim == 1
    values = np.atleast_2d(values)
    if values.shape[1] != 2:
        raise ValueError(f"eta must have last dimension 2, got {values.shape}")

    u, v = values[:, 0], values[:, 1]
    y2 = np.square(y_coefficients)[None, :]
    n = eigenvalues.size
    y_norm2 = float(np.square(y_coefficients).sum())
    # NUTS can probe far outside the typical set. The caller turns resulting
    # non-finite values into rejected proposals; avoid emitting noisy warnings.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        a, b = np.exp(-2.0 * u), np.exp(-2.0 * v)
        spectrum = a[:, None] * eigenvalues[None, :] + b[:, None]
        information_quad = np.sum(
            np.square(b[:, None]) * y2 / spectrum, axis=1
        )
        value = (
            -n * u
            - n * v
            - 0.5 * b * y_norm2
            + 0.5 * information_quad
            + 0.5 * n * math.log(2.0 * math.pi)
            - 0.5 * np.log(spectrum).sum(axis=1)
            - 0.5 * np.square((u + 1.0) / 0.7)
            - 0.5 * np.square((v + 1.5) / 0.7)
        )

        state_term = a[:, None] * eigenvalues[None, :]
        obs_term = b[:, None]
        grad_u = (
            -n
            + np.sum(state_term / spectrum, axis=1)
            + np.sum(
                state_term * np.square(obs_term) * y2 / np.square(spectrum),
                axis=1,
            )
            - (u + 1.0) / (0.7 ** 2)
        )
        grad_v = (
            -n
            + b * y_norm2
            - 2.0 * np.sum(np.square(obs_term) * y2 / spectrum, axis=1)
            + np.sum(np.power(obs_term, 3) * y2 / np.square(spectrum), axis=1)
            + np.sum(obs_term / spectrum, axis=1)
            - (v + 1.5) / (0.7 ** 2)
        )
    gradient = np.column_stack([grad_u, grad_v])
    if scalar:
        return float(value[0]), gradient[0]
    return value, gradient


@dataclass
class SpectralGMRFEvaluation:
    """Numerical spectral quantities at one or a batch of core points."""

    eta: np.ndarray
    precision_eigenvalues: np.ndarray
    mean_coefficients: np.ndarray
    mean: np.ndarray


class GMRFSpectralBackend:
    """Symmetric Gaussian transport in a dense or fast graph eigenbasis."""

    def __init__(
        self,
        eigenvalues,
        y,
        *,
        basis_kind: str,
        eigenvectors: np.ndarray | None = None,
        side: int | None = None,
        build_seconds: float = 0.0,
    ):
        self.eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
        self.y = np.asarray(y, dtype=np.float64).reshape(-1)
        if self.eigenvalues.shape != self.y.shape:
            raise ValueError("eigenvalues and y must have the same length")
        if np.any(self.eigenvalues <= 0.0):
            raise ValueError("base precision eigenvalues must be positive")
        self.n = self.eigenvalues.size
        self.basis_kind = basis_kind
        self.eigenvectors = eigenvectors
        self.side = side
        self.build_seconds = float(build_seconds)
        self.y_coefficients = self.analysis(self.y)

    @classmethod
    def dense(cls, precision, y):
        started = time.perf_counter()
        matrix = sparse.csc_matrix(precision, dtype=np.float64).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        backend = cls(
            eigenvalues,
            y,
            basis_kind="dense_eigen",
            eigenvectors=eigenvectors,
        )
        backend.build_seconds = time.perf_counter() - started
        return backend

    @classmethod
    def path(cls, n: int, kappa: float, y):
        started = time.perf_counter()
        frequency = np.arange(n, dtype=np.float64)
        eigenvalues = kappa + 2.0 - 2.0 * np.cos(math.pi * frequency / n)
        backend = cls(eigenvalues, y, basis_kind="path_dct")
        backend.build_seconds = time.perf_counter() - started
        return backend

    @classmethod
    def lattice(cls, side: int, kappa: float, y):
        started = time.perf_counter()
        frequency = np.arange(side, dtype=np.float64)
        one_dimensional = 2.0 - 2.0 * np.cos(math.pi * frequency / side)
        eigenvalues = (
            kappa + one_dimensional[:, None] + one_dimensional[None, :]
        ).reshape(-1)
        backend = cls(
            eigenvalues,
            y,
            basis_kind="lattice_dct",
            side=side,
        )
        backend.build_seconds = time.perf_counter() - started
        return backend

    @property
    def basis_bytes(self) -> int:
        total = self.eigenvalues.nbytes + self.y_coefficients.nbytes
        if self.eigenvectors is not None:
            total += self.eigenvectors.nbytes
        return int(total)

    def analysis(self, value) -> np.ndarray:
        rows = np.asarray(value, dtype=np.float64)
        if rows.shape[-1] != self.n:
            raise ValueError(f"last dimension {rows.shape[-1]} != {self.n}")
        if self.basis_kind == "dense_eigen":
            return rows @ self.eigenvectors
        if self.basis_kind == "path_dct":
            return fft.dct(rows, type=2, norm="ortho", axis=-1, workers=1)
        if self.basis_kind == "lattice_dct":
            shape = rows.shape
            grid = rows.reshape(shape[:-1] + (self.side, self.side))
            transformed = fft.dctn(
                grid, type=2, norm="ortho", axes=(-2, -1), workers=1
            )
            return transformed.reshape(shape)
        raise ValueError(f"unknown basis kind {self.basis_kind!r}")

    def synthesis(self, coefficients) -> np.ndarray:
        rows = np.asarray(coefficients, dtype=np.float64)
        if rows.shape[-1] != self.n:
            raise ValueError(f"last dimension {rows.shape[-1]} != {self.n}")
        if self.basis_kind == "dense_eigen":
            return rows @ self.eigenvectors.T
        if self.basis_kind == "path_dct":
            return fft.idct(rows, type=2, norm="ortho", axis=-1, workers=1)
        if self.basis_kind == "lattice_dct":
            shape = rows.shape
            grid = rows.reshape(shape[:-1] + (self.side, self.side))
            transformed = fft.idctn(
                grid, type=2, norm="ortho", axes=(-2, -1), workers=1
            )
            return transformed.reshape(shape)
        raise ValueError(f"unknown basis kind {self.basis_kind!r}")

    def precision_eigenvalues(self, eta) -> np.ndarray:
        values = np.asarray(eta, dtype=np.float64)
        scalar = values.ndim == 1
        values = np.atleast_2d(values)
        if values.shape[1] != 2:
            raise ValueError(f"eta must have last dimension 2, got {values.shape}")
        u, v = values[:, 0], values[:, 1]
        spectrum = (
            np.exp(-2.0 * u[:, None]) * self.eigenvalues[None, :]
            + np.exp(-2.0 * v[:, None])
        )
        return spectrum[0] if scalar else spectrum

    def evaluate(self, eta) -> SpectralGMRFEvaluation:
        eta = np.asarray(eta, dtype=np.float64)
        scalar = eta.ndim == 1
        values = np.atleast_2d(eta)
        if values.shape[1] != 2:
            raise ValueError(f"eta must have last dimension 2, got {values.shape}")
        spectrum = self.precision_eigenvalues(values)
        mean_coefficients = (
            np.exp(-2.0 * values[:, 1, None])
            * self.y_coefficients[None, :]
            / spectrum
        )
        mean = self.synthesis(mean_coefficients)
        if scalar:
            spectrum = spectrum[0]
            mean_coefficients = mean_coefficients[0]
            mean = mean[0]
        return SpectralGMRFEvaluation(
            eta=eta,
            precision_eigenvalues=spectrum,
            mean_coefficients=mean_coefficients,
            mean=mean,
        )

    def collapsed_vg(self, eta):
        return _collapsed_value_gradient(self.eigenvalues, self.y_coefficients, eta)

    def forward(self, value, evaluation: SpectralGMRFEvaluation) -> np.ndarray:
        centered = np.asarray(value, dtype=np.float64) - evaluation.mean
        coefficients = self.analysis(centered)
        return self.synthesis(
            coefficients * np.sqrt(evaluation.precision_eigenvalues)
        )

    def inverse(self, value, evaluation: SpectralGMRFEvaluation) -> np.ndarray:
        coefficients = self.analysis(value)
        centered = self.synthesis(
            coefficients / np.sqrt(evaluation.precision_eigenvalues)
        )
        return centered + evaluation.mean

    def centered_quadratic_cost(self, eta) -> float:
        """Exact Brenier coupling cost after removing the common mean shift."""
        spectrum = self.precision_eigenvalues(eta)
        return float(
            np.reciprocal(spectrum).sum()
            + self.n
            - 2.0 * np.reciprocal(np.sqrt(spectrum)).sum()
        )


@dataclass
class SparseGMRFEvaluation:
    """Numerical sparse factor and conditional mean at one core point."""

    eta: np.ndarray
    factor: SparsePrecisionFactor
    mean: np.ndarray


class GMRFSparseCholeskyBackend:
    """Exact-contract sparse triangular comparator for the same GMRF family."""

    def __init__(self, base_precision, y):
        started = time.perf_counter()
        self.base_precision = sparse.csc_matrix(base_precision, dtype=np.float64)
        self.base_precision.sum_duplicates()
        self.base_precision.sort_indices()
        if self.base_precision.shape[0] != self.base_precision.shape[1]:
            raise ValueError("base precision must be square")
        self.n = self.base_precision.shape[0]
        self.y = np.asarray(y, dtype=np.float64).reshape(-1)
        if self.y.size != self.n:
            raise ValueError("base precision and y dimensions differ")
        self.backend = SparsePrecisionBackend(self.base_precision != 0)
        coo = self.base_precision.tocoo()
        self._rows = coo.row.astype(np.int64)
        self._cols = coo.col.astype(np.int64)
        self._values = coo.data
        self._diagonal = np.arange(self.n, dtype=np.int64)
        self.build_seconds = time.perf_counter() - started

    def evaluate(self, eta) -> SparseGMRFEvaluation:
        eta = np.asarray(eta, dtype=np.float64)
        u, v = _eta_pair(eta)
        a, b = math.exp(-2.0 * u), math.exp(-2.0 * v)
        precision = a * self.base_precision + b * sparse.eye(
            self.n, format="csc"
        )
        factor = self.backend.factor(precision)
        mean = factor.solve(b * self.y)
        return SparseGMRFEvaluation(eta=eta, factor=factor, mean=mean)

    def collapsed_vg(self, eta):
        evaluation = self.evaluate(eta)
        u, v = _eta_pair(evaluation.eta)
        a, b = math.exp(-2.0 * u), math.exp(-2.0 * v)
        mean = evaluation.mean
        residual = self.y - mean
        state_quad = float(mean @ (self.base_precision @ mean))
        obs_quad = float(residual @ residual)
        value = (
            -self.n * u
            - 0.5 * a * state_quad
            - self.n * v
            - 0.5 * b * obs_quad
            - 0.5 * ((u + 1.0) / 0.7) ** 2
            - 0.5 * ((v + 1.5) / 0.7) ** 2
            + 0.5 * self.n * math.log(2.0 * math.pi)
            - evaluation.factor.half_logdet
        )

        inverse_on_q = evaluation.factor.selected_inverse(self._rows, self._cols)
        trace_q = float(self._values @ inverse_on_q)
        inverse_diagonal = evaluation.factor.selected_inverse(
            self._diagonal, self._diagonal
        )
        trace_i = float(inverse_diagonal.sum())
        gradient = np.asarray([
            -self.n + a * (state_quad + trace_q) - (u + 1.0) / (0.7 ** 2),
            -self.n + b * (obs_quad + trace_i) - (v + 1.5) / (0.7 ** 2),
        ])
        return float(value), gradient

    def forward(self, value, evaluation: SparseGMRFEvaluation) -> np.ndarray:
        return evaluation.factor.forward(value, evaluation.mean)

    def inverse(self, value, evaluation: SparseGMRFEvaluation) -> np.ndarray:
        return evaluation.factor.inverse(value, evaluation.mean)

    def centered_quadratic_cost(self, evaluation: SparseGMRFEvaluation) -> float:
        """Exact cost of the triangular coupling, excluding the mean shift."""
        inverse_trace = evaluation.factor.selected_inverse(
            self._diagonal, self._diagonal
        ).sum()
        cross_trace = np.reciprocal(np.sqrt(evaluation.factor.diagonal)).sum()
        return float(inverse_trace + self.n - 2.0 * cross_trace)
