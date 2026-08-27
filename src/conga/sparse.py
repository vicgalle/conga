"""Sparse precision factorization and transport for general graph blocks.

The symbolic graph is fixed while the numerical precision may vary with a
small conditioning core.  ``SparsePrecisionBackend`` therefore computes a
fill-reducing symmetric ordering once, then reuses it for numerical LDL^T
factorizations.  The factor exposes the exact whitening map as well as solves
and log determinants without materializing a dense matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Tuple

import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve_triangular


def _to_csc(matrix) -> sparse.csc_matrix:
    out = sparse.csc_matrix(matrix, dtype=np.float64)
    out.sum_duplicates()
    out.sort_indices()
    return out


@njit(cache=True)
def _find_row(indices, start, stop, row):
    lo, hi = start, stop
    while lo < hi:
        mid = (lo + hi) // 2
        if indices[mid] < row:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < stop and indices[lo] == row else -1


@njit(cache=True)
def _takahashi_values(indptr, indices, lower_values, diagonal):
    """Selected inverse values aligned with the CSC entries of unit-lower L."""
    n = len(diagonal)
    selected = np.zeros_like(lower_values)
    for i in range(n - 1, -1, -1):
        start, stop = indptr[i], indptr[i + 1]
        diagonal_position = _find_row(indices, start, stop, i)
        for position_j in range(start, stop):
            j = indices[position_j]
            if j <= i:
                continue
            total = 0.0
            for position_k in range(start, stop):
                k = indices[position_k]
                if k <= i:
                    continue
                column = min(k, j)
                row = max(k, j)
                position = _find_row(
                    indices, indptr[column], indptr[column + 1], row
                )
                total += lower_values[position_k] * selected[position]
            selected[position_j] = -total
        value = 1.0 / diagonal[i]
        for position_k in range(start, stop):
            k = indices[position_k]
            if k > i:
                value -= lower_values[position_k] * selected[position_k]
        selected[diagonal_position] = value
    return selected


@njit(cache=True)
def _selected_lookup(indptr, indices, selected, rows, cols):
    values = np.empty(len(rows), dtype=np.float64)
    for p in range(len(rows)):
        column = min(rows[p], cols[p])
        row = max(rows[p], cols[p])
        position = _find_row(indices, indptr[column], indptr[column + 1], row)
        values[p] = np.nan if position < 0 else selected[position]
    return values


def _permutation_from_superlu(pattern: sparse.csc_matrix) -> np.ndarray:
    """Symmetric minimum-degree ordering, returned in elimination order."""
    n = pattern.shape[0]
    off = pattern.copy().astype(np.float64)
    off.setdiag(0.0)
    off.eliminate_zeros()
    off.data.fill(-1.0)
    degree = np.asarray((off != 0).sum(axis=1)).reshape(-1)
    probe = off + sparse.diags(degree + 1.0, format="csc")
    factor = splu(
        probe,
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options={"SymmetricMode": True},
    )
    if not np.array_equal(factor.perm_r, factor.perm_c):
        raise RuntimeError("symmetric ordering unexpectedly required row pivots")
    # SuperLU stores old -> new positions; indexing needs new -> old order.
    order = np.argsort(factor.perm_c)
    if sorted(order.tolist()) != list(range(n)):
        raise RuntimeError("invalid sparse ordering")
    return order


@dataclass
class SparsePrecisionFactor:
    """One numerical ``A = P^T L D L^T P`` factorization."""

    order: np.ndarray
    inverse_order: np.ndarray
    lower: sparse.csc_matrix
    diagonal: np.ndarray
    lu: object
    factor_seconds: float
    original_lower_nnz: int

    @property
    def dim(self) -> int:
        return len(self.order)

    @property
    def logdet(self) -> float:
        return float(np.log(self.diagonal).sum())

    @property
    def half_logdet(self) -> float:
        return 0.5 * self.logdet

    @property
    def factor_nnz(self) -> int:
        return int(self.lower.nnz)

    @property
    def fill_ratio(self) -> float:
        return self.factor_nnz / self.original_lower_nnz

    def _rows(self, value) -> Tuple[np.ndarray, Tuple[int, ...]]:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape[-1] != self.dim:
            raise ValueError(f"last dimension {arr.shape[-1]} != {self.dim}")
        return arr.reshape(-1, self.dim), arr.shape

    def solve(self, rhs) -> np.ndarray:
        """Solve ``A x = rhs`` with vectors stored along the last axis."""
        rows, shape = self._rows(rhs)
        solved_p = self.lu.solve(rows[:, self.order].T).T
        solved = np.empty_like(solved_p)
        solved[:, self.order] = solved_p
        return solved.reshape(shape)

    def forward(self, value, mean=0.0) -> np.ndarray:
        """Whiten ``x`` to ``z`` in model order."""
        rows, shape = self._rows(np.asarray(value) - np.asarray(mean))
        rows_p = rows[:, self.order]
        z_p = (self.lower.T @ rows_p.T).T * np.sqrt(self.diagonal)
        z = np.empty_like(z_p)
        z[:, self.order] = z_p
        return z.reshape(shape)

    def inverse(self, value, mean=0.0) -> np.ndarray:
        """Map standard-normal coordinates to the original model order."""
        rows, shape = self._rows(value)
        rhs = (rows[:, self.order] / np.sqrt(self.diagonal)).T
        dev_p = spsolve_triangular(
            self.lower.T, rhs, lower=False, unit_diagonal=True
        ).T
        dev = np.empty_like(dev_p)
        dev[:, self.order] = dev_p
        return (dev + np.asarray(mean)).reshape(shape)

    def quadratic(self, value) -> np.ndarray:
        z = self.forward(value)
        return np.square(z).sum(axis=-1)

    def selected_inverse(self, rows, cols) -> np.ndarray:
        """Entries of ``A^{-1}`` on the filled graph via Takahashi recursion.

        The recursion stores one value per nonzero of the lower factor, rather
        than materializing the generally dense inverse.  Requested coordinates
        are in the original model order and must belong to the filled graph.
        """
        if not hasattr(self, "_selected_inverse_values"):
            lower = self.lower.tocsc()
            self._selected_inverse_values = _takahashi_values(
                lower.indptr, lower.indices, lower.data, self.diagonal
            )

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if rows.shape != cols.shape:
            raise ValueError("selected inverse row and column shapes differ")
        row_p = self.inverse_order[rows]
        col_p = self.inverse_order[cols]
        values = _selected_lookup(
            self.lower.indptr,
            self.lower.indices,
            self._selected_inverse_values,
            row_p.reshape(-1),
            col_p.reshape(-1),
        )
        if np.isnan(values).any():
            raise ValueError("requested inverse entry is outside the filled graph")
        return values.reshape(rows.shape)


class SparsePrecisionBackend:
    """Cached symbolic analysis for a family of sparse SPD precisions."""

    def __init__(self, pattern):
        started = time.perf_counter()
        pat = _to_csc(pattern).astype(bool)
        if pat.shape[0] != pat.shape[1]:
            raise ValueError("precision pattern must be square")
        pat = (pat + pat.T).astype(bool).tocsc()
        pat.setdiag(True)
        pat.eliminate_zeros()
        self.pattern = pat
        self.dim = pat.shape[0]
        self.order = _permutation_from_superlu(pat)
        self.inverse_order = np.empty_like(self.order)
        self.inverse_order[self.order] = np.arange(self.dim)
        self.original_lower_nnz = int(sparse.tril(pat).nnz)
        self.symbolic_seconds = time.perf_counter() - started

    def factor(self, precision, verify: bool = False) -> SparsePrecisionFactor:
        matrix = _to_csc(precision)
        if matrix.shape != (self.dim, self.dim):
            raise ValueError(
                f"precision shape {matrix.shape} != {(self.dim, self.dim)}"
            )
        asym = matrix - matrix.T
        asym_scale = np.max(np.abs(asym.data)) if asym.nnz else 0.0
        matrix_scale = np.max(np.abs(matrix.data)) if matrix.nnz else 1.0
        if asym_scale > 1e-10 * max(matrix_scale, 1.0):
            raise ValueError("precision is not numerically symmetric")
        matrix = 0.5 * (matrix + matrix.T)
        matrix_p = matrix[self.order][:, self.order].tocsc()

        started = time.perf_counter()
        lu = splu(
            matrix_p,
            permc_spec="NATURAL",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
        elapsed = time.perf_counter() - started
        identity = np.arange(self.dim)
        if not np.array_equal(lu.perm_r, identity) or not np.array_equal(
            lu.perm_c, identity
        ):
            raise np.linalg.LinAlgError("precision required numerical pivoting")
        diagonal = np.asarray(lu.U.diagonal(), dtype=np.float64)
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
            raise np.linalg.LinAlgError("precision is not positive definite")

        lower = lu.L.tocsc()
        lower.sort_indices()
        if verify:
            residual = lu.U - sparse.diags(diagonal) @ lower.T
            error = np.max(np.abs(residual.data)) if residual.nnz else 0.0
            if error > 1e-9 * max(matrix_scale, 1.0):
                raise RuntimeError(f"sparse LDL symmetry error {error:.2e}")

        return SparsePrecisionFactor(
            order=self.order,
            inverse_order=self.inverse_order,
            lower=lower,
            diagonal=diagonal,
            lu=lu,
            factor_seconds=elapsed,
            original_lower_nnz=self.original_lower_nnz,
        )
