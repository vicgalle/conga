"""General-graph GMRF discovery and sparse transport suite.

The graph is used only to define and independently score each target.  The
reported precision pattern, coloring, and sparse transport are recovered from
the target's value-and-gradient interface through ``GaussianBlock``.

Reproduce the paper artifacts with::

    uv run --extra experiments python -m experiments.graph_gmrf

Use ``--quick`` for a small regression run and ``--plot-only`` to redraw the
figure from the checked-in JSON results.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay
import torch

from conga.core import DTYPE, TargetPosterior
from conga.sparse import SparsePrecisionBackend
from conga.structure import GaussianBlock, SparseGaussianBlock

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_PATH = os.path.join(ROOT, "results", "runs", "graph_gmrf_suite.json")
FIGURE_PATH = os.path.join(ROOT, "results", "runs", "graph_gmrf_suite.pdf")
PNG_PATH = os.path.join(ROOT, "results", "runs", "graph_gmrf_suite.png")

FAMILIES = ("tree", "lattice", "geometric", "expander", "star")
DISPLAY = {
    "tree": "binary tree",
    "lattice": "2D lattice",
    "geometric": "planar geometric",
    "expander": "3-regular expander",
    "star": "star",
}
COLORS = {
    "tree": "#2A9D8F",
    "lattice": "#3B6EA8",
    "geometric": "#7A5195",
    "expander": "#D55E00",
    "star": "#5B5B5B",
}
MARKERS = {"tree": "o", "lattice": "s", "geometric": "D", "expander": "^", "star": "v"}


def _adjacency(n: int, edges: Iterable[Tuple[int, int]]) -> sparse.csc_matrix:
    edge_array = np.asarray(list(edges), dtype=np.int64)
    if edge_array.size == 0:
        return sparse.csc_matrix((n, n), dtype=np.float64)
    rows = np.r_[edge_array[:, 0], edge_array[:, 1]]
    cols = np.r_[edge_array[:, 1], edge_array[:, 0]]
    out = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsc()
    out.data.fill(1.0)
    out.setdiag(0.0)
    out.eliminate_zeros()
    return out


def make_graph(family: str, n: int, seed: int) -> sparse.csc_matrix:
    if family == "tree":
        return _adjacency(n, ((i, (i - 1) // 2) for i in range(1, n)))
    if family == "lattice":
        side = round(math.sqrt(n))
        if side * side != n:
            raise ValueError("lattice sizes must be perfect squares")
        edges = []
        for i in range(side):
            for j in range(side):
                node = i * side + j
                if i + 1 < side:
                    edges.append((node, node + side))
                if j + 1 < side:
                    edges.append((node, node + 1))
        return _adjacency(n, edges)
    if family == "geometric":
        rng = np.random.default_rng(seed)
        points = rng.random((n, 2))
        simplices = Delaunay(points).simplices
        edges = set()
        for tri in simplices:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
                edges.add(tuple(sorted((int(a), int(b)))))
        return _adjacency(n, sorted(edges))
    if family == "expander":
        if n % 2:
            raise ValueError("expander sizes must be even")
        rng = np.random.default_rng(seed)
        cycle = {tuple(sorted((i, (i + 1) % n))) for i in range(n)}
        for _ in range(1000):
            order = rng.permutation(n)
            matching = {
                tuple(sorted((int(order[i]), int(order[i + 1]))))
                for i in range(0, n, 2)
            }
            if len(matching) == n // 2 and matching.isdisjoint(cycle):
                return _adjacency(n, sorted(cycle | matching))
        raise RuntimeError("could not construct disjoint random matching")
    if family == "star":
        return _adjacency(n, ((0, i) for i in range(1, n)))
    raise ValueError(f"unknown graph family {family!r}")


def graph_precision(adjacency: sparse.csc_matrix, kappa: float = 0.2):
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return sparse.diags(degree + kappa, format="csc") - adjacency


class GraphGMRFTarget(TargetPosterior):
    """Properized ICAR smoother with two log-scale core coordinates."""

    def __init__(self, adjacency: sparse.csc_matrix, seed: int, kappa: float = 0.2):
        self.name = "graph_gmrf"
        self.n = adjacency.shape[0]
        self.dim = self.n + 2
        self.kappa = float(kappa)
        upper = sparse.triu(adjacency, k=1).tocoo()
        self.edge_i = torch.as_tensor(upper.row.copy(), dtype=torch.long)
        self.edge_j = torch.as_tensor(upper.col.copy(), dtype=torch.long)

        q0 = graph_precision(adjacency, kappa)
        factor = SparsePrecisionBackend(q0 != 0).factor(q0)
        rng = np.random.default_rng(seed + 1009)
        state_sd, obs_sd = 0.35, 0.20
        latent = state_sd * factor.inverse(rng.standard_normal(self.n))
        self.y = torch.as_tensor(
            latent + obs_sd * rng.standard_normal(self.n), dtype=DTYPE
        )

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        state_log_sd, obs_log_sd, x = z[..., 0], z[..., 1], z[..., 2:]
        edge_delta = x[..., self.edge_i] - x[..., self.edge_j]
        state_quad = edge_delta.square().sum(-1) + self.kappa * x.square().sum(-1)
        obs_quad = (self.y - x).square().sum(-1)
        return (
            -self.n * state_log_sd
            - 0.5 * torch.exp(-2 * state_log_sd) * state_quad
            - self.n * obs_log_sd
            - 0.5 * torch.exp(-2 * obs_log_sd) * obs_quad
            - 0.5 * ((state_log_sd + 1.0) / 0.7).square()
            - 0.5 * ((obs_log_sd + 1.5) / 0.7).square()
        )

    def constrain(self, z: torch.Tensor):
        return {"state_sd": z[:, 0].exp(), "obs_sd": z[:, 1].exp(), "x": z[:, 2:]}


def _median_timing(function, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        values.append(time.perf_counter() - started)
    return float(np.median(values))


def run_cell(family: str, n: int, graph_seed: int, repeats: int, dense_max: int) -> Dict:
    adjacency = make_graph(family, n, graph_seed)
    expected = (adjacency + sparse.eye(n, format="csc")).astype(bool)
    target = GraphGMRFTarget(adjacency, seed=graph_seed)
    g_idx = torch.arange(2, target.dim)

    started = time.perf_counter()
    block = GaussianBlock(target, g_idx, banded=False)
    discovery_seconds = time.perf_counter() - started
    detected = block.pattern.detach().cpu().numpy()
    expected_dense = expected.toarray()
    tp = int(np.logical_and(detected, expected_dense).sum())
    precision = tp / max(int(detected.sum()), 1)
    recall = tp / int(expected_dense.sum())

    sparse_block = SparseGaussianBlock(block)
    eta = torch.tensor([[math.log(0.35), math.log(0.20)]], dtype=DTYPE)
    sparse_block.evaluate(eta, verify=True)
    sparse_block.collapsed_vg(eta)  # warm the selected-inverse kernel
    evaluations = []
    totals = []
    for _ in range(repeats):
        started = time.perf_counter()
        evaluation = sparse_block.evaluate(eta)
        totals.append(time.perf_counter() - started)
        evaluations.append(evaluation)
    total_index = int(np.argsort(totals)[len(totals) // 2])
    evaluation = evaluations[total_index]
    factor = evaluation.factors[0]

    rng = np.random.default_rng(701 + graph_seed)
    eps = torch.as_tensor(rng.standard_normal((1, n)), dtype=DTYPE)
    x = sparse_block.inverse(eta, eps, evaluation=evaluation)
    recovered = sparse_block.forward(eta, x, evaluation=evaluation)
    roundtrip_error = float((recovered - eps).abs().max())
    certificate_gen = torch.Generator().manual_seed(1901 + n + graph_seed)
    identity_residual = sparse_block.joint_identity_residual(eta, gen=certificate_gen)

    map_seconds = _median_timing(
        lambda: (
            sparse_block.forward(eta, x, evaluation=evaluation),
            sparse_block.inverse(eta, eps, evaluation=evaluation),
        ),
        repeats=max(repeats, 3),
    )
    score_seconds = _median_timing(
        lambda: sparse_block.collapsed_vg(eta), repeats=max(repeats, 3)
    )

    q0 = graph_precision(adjacency, target.kappa)
    numerical_precision = (
        math.exp(-2 * float(eta[0, 0])) * q0
        + math.exp(-2 * float(eta[0, 1])) * sparse.eye(n, format="csc")
    )
    dense_seconds = None
    dense_logdet_error = None
    if n <= dense_max:
        dense_matrix = numerical_precision.toarray()
        dense_seconds = _median_timing(lambda: np.linalg.cholesky(dense_matrix), repeats)
        dense_logdet = 2 * np.log(np.diag(np.linalg.cholesky(dense_matrix))).sum()
        dense_logdet_error = abs(float(dense_logdet) - factor.logdet)

    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return {
        "family": family,
        "n": n,
        "graph_seed": graph_seed,
        "edges": int(adjacency.nnz // 2),
        "max_degree": int(degree.max()),
        "n_colors": block.n_colors,
        "bandwidth_rcm": block.bandwidth,
        "pattern_precision": precision,
        "pattern_recall": recall,
        "pattern_error": block.pattern_err,
        "certificate_residual": identity_residual,
        "roundtrip_error": roundtrip_error,
        "discovery_seconds": discovery_seconds,
        "symbolic_seconds": sparse_block.backend.symbolic_seconds,
        "assembly_seconds": evaluation.assembly_seconds,
        "factor_seconds": evaluation.factor_seconds,
        "recurring_seconds": float(np.median(totals)),
        "map_pair_seconds": map_seconds,
        "score_seconds": score_seconds,
        "factor_nnz": factor.factor_nnz,
        "precision_lower_nnz": factor.original_lower_nnz,
        "fill_ratio": factor.fill_ratio,
        "sparse_factor_bytes": int(
            factor.lower.data.nbytes
            + factor.lower.indices.nbytes
            + factor.lower.indptr.nbytes
            + factor.diagonal.nbytes
        ),
        "dense_bytes": int(8 * n * n),
        "dense_factor_seconds": dense_seconds,
        "dense_logdet_error": dense_logdet_error,
    }


def _aggregate(records, family, n, key):
    values = [row[key] for row in records if row["family"] == family and row["n"] == n]
    return float(np.median(values))


def plot(results: Dict):
    records = results["records"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25))
    specs = (
        ("n_colors", "colored probes", "A"),
        ("fill_ratio", r"factor fill  $\mathrm{nnz}(L)/\mathrm{nnz}(Q_\triangle)$", "B"),
        ("score_seconds", "value + score (ms)", "C"),
    )
    for ax, (key, ylabel, panel) in zip(axes, specs):
        for family in FAMILIES:
            sizes = sorted({r["n"] for r in records if r["family"] == family})
            values = [_aggregate(records, family, n, key) for n in sizes]
            if key == "score_seconds":
                values = [1000 * value for value in values]
            ax.plot(
                sizes, values, color=COLORS[family], marker=MARKERS[family],
                lw=1.35, ms=3.8, label=DISPLAY[family],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("latent nodes  $n$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", color="#D9D9D9", lw=0.45)
        ax.text(-0.18, 1.04, panel, transform=ax.transAxes, fontweight="bold")
    axes[0].legend(frameon=False, loc="upper left", ncol=1)
    fig.tight_layout(w_pad=1.25)
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(results: Dict):
    records = results["records"]
    print("family               n  colors  fill   score(ms)  cert       pattern")
    for row in records:
        if row["graph_seed"] != 0:
            continue
        print(
            f"{DISPLAY[row['family']]:20s} {row['n']:4d} {row['n_colors']:7d} "
            f"{row['fill_ratio']:5.2f} {1000 * row['score_seconds']:10.2f} "
            f"{row['certificate_residual']:.1e} "
            f"{row['pattern_precision']:.3f}/{row['pattern_recall']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dense-max", type=int, default=576)
    args = parser.parse_args()

    if args.plot_only:
        with open(RESULT_PATH) as handle:
            results = json.load(handle)
        plot(results)
        summarize(results)
        return

    sizes = (64, 256) if args.quick else (64, 256, 576, 1024)
    families = ("tree", "lattice", "expander", "star") if args.quick else FAMILIES
    suite_started = time.perf_counter()
    records = []
    for family in families:
        graph_seeds = (0, 1, 2) if family in ("geometric", "expander") and not args.quick else (0,)
        for n in sizes:
            for graph_seed in graph_seeds:
                print(f"[{family}] n={n} graph_seed={graph_seed}", flush=True)
                row = run_cell(family, n, graph_seed, args.repeats, args.dense_max)
                records.append(row)
                print(
                    f"  colors={row['n_colors']} fill={row['fill_ratio']:.2f} "
                    f"score={1000 * row['score_seconds']:.2f}ms "
                    f"certificate={row['certificate_residual']:.1e}",
                    flush=True,
                )

    results = {
        "experiment": "graph_gmrf_sparse_transport",
        "model": "properized ICAR Gaussian graph smoother",
        "sizes": list(sizes),
        "families": list(families),
        "repeats": args.repeats,
        "total_seconds": time.perf_counter() - suite_started,
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "records": records,
    }
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as handle:
        json.dump(results, handle, indent=2)
    plot(results)
    summarize(results)
    print(f"wrote {RESULT_PATH}\nwrote {FIGURE_PATH}\nwrote {PNG_PATH}")


if __name__ == "__main__":
    main()
