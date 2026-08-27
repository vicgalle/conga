"""End-to-end raw versus sparse Rao--Blackwellized NUTS on graph GMRFs.

The structural graph suite lives in ``certified_flow_gmrf.py``. This companion
driver asks the downstream inference question: does the recovered sparse
Gaussian block improve NUTS after accounting for colored collapsed gradients
and exact reconstruction of the latent field?

The run checkpoints after every arm and is safe to resume::

    uv run --extra experiments python -m experiments.graph_gmrf_nuts --resume

Use ``--quick`` for a small integration run and ``--plot-only`` to redraw the
figure from existing results.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp
import torch

from conga.inference import FisherHMC, Marginalized
from experiments.graph_gmrf import (
    COLORS,
    DISPLAY,
    FAMILIES,
    GraphGMRFTarget,
    graph_precision,
    make_graph,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_PATH = os.path.join(ROOT, "results", "runs", "graph_gmrf_nuts.json")
FIGURE_PATH = os.path.join(ROOT, "results", "runs", "graph_gmrf_nuts.pdf")
PNG_PATH = os.path.join(ROOT, "results", "runs", "graph_gmrf_nuts.png")
ARMS = ("raw", "sparse_rb")
# ``flow`` rows come from certified_flow_gmrf_flow.py and are merged for plotting
ARM_LABEL = {"raw": "raw NUTS", "sparse_rb": "sparse RB-NUTS",
             "flow": "learned flow"}
ARM_HATCH = {"raw": "", "sparse_rb": "///", "flow": "..."}
PLOT_ARMS = ("raw", "flow", "sparse_rb")


def _core_reference(adjacency: sparse.csc_matrix, target: GraphGMRFTarget,
                    order: int = 80) -> Dict:
    """Mode-centered deterministic quadrature for posterior moments of u,v."""
    eigenvalues, eigenvectors = np.linalg.eigh(
        graph_precision(adjacency, target.kappa).toarray()
    )
    projected_y2 = np.square(eigenvectors.T @ target.y.detach().cpu().numpy())
    prior_mean = np.array([-1.0, -1.5])
    prior_variance = 0.7 ** 2

    def value_gradient_hessian(eta):
        u, v = eta
        state = np.exp(2 * u) / eigenvalues
        observation = math.exp(2 * v)
        covariance = state + observation
        q = (projected_y2 - covariance) / np.square(covariance)
        dq = (covariance - 2 * projected_y2) / np.power(covariance, 3)
        value = (
            -0.5 * np.sum(np.log(covariance) + projected_y2 / covariance)
            -0.5 * np.sum(np.square((eta - prior_mean) / 0.7))
        )
        gradient = np.array([
            np.sum(state * q) - (u - prior_mean[0]) / prior_variance,
            np.sum(observation * q) - (v - prior_mean[1]) / prior_variance,
        ])
        hessian = np.array([
            [np.sum(2 * state * q + 2 * state * state * dq) - 1 / prior_variance,
             np.sum(2 * state * observation * dq)],
            [np.sum(2 * state * observation * dq),
             np.sum(2 * observation * q + 2 * observation * observation * dq)
             - 1 / prior_variance],
        ])
        return float(value), gradient, hessian

    starts = (
        prior_mean,
        np.array([-0.5, -2.0]),
        np.array([-1.5, -1.0]),
        np.array([-2.0, -0.5]),
    )
    optimizations = [
        minimize(
            lambda eta: -value_gradient_hessian(eta)[0],
            start,
            jac=lambda eta: -value_gradient_hessian(eta)[1],
            method="L-BFGS-B",
            bounds=((-6.0, 3.0), (-6.0, 3.0)),
            options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 1000},
        )
        for start in starts
    ]
    optimum = min(optimizations, key=lambda result: result.fun)
    if not optimum.success:
        raise RuntimeError(f"core reference optimization failed: {optimum.message}")
    mode = optimum.x
    _, mode_gradient, mode_hessian = value_gradient_hessian(mode)
    laplace_covariance = -np.linalg.inv(mode_hessian)
    transform = np.linalg.cholesky(laplace_covariance)

    raw_nodes, raw_weights = leggauss(order)
    radius = 10.0
    nodes = radius * raw_nodes
    log_weights = np.log(radius * raw_weights)
    log_mass = np.empty((order, order))
    values = np.empty((order, order, 2))
    for i, first in enumerate(nodes):
        standard = np.column_stack([
            np.full(order, first), nodes,
        ])
        eta = mode + standard @ transform.T
        values[i] = eta
        u, v = eta[:, 0], eta[:, 1]
        covariance = (
            np.exp(2 * u[:, None]) / eigenvalues[None, :]
            + np.exp(2 * v[:, None])
        )
        log_likelihood = -0.5 * (
            np.log(covariance).sum(1)
            + (projected_y2[None, :] / covariance).sum(1)
        )
        log_prior = (
            -0.5 * ((u - prior_mean[0]) / 0.7) ** 2
            -0.5 * ((v - prior_mean[1]) / 0.7) ** 2
        )
        log_mass[i] = log_likelihood + log_prior + log_weights[i] + log_weights
    log_mass -= logsumexp(log_mass)
    weights = np.exp(log_mass)
    mean = (weights[..., None] * values).sum((0, 1))
    centered = values - mean
    covariance = np.einsum("ij,ija,ijb->ab", weights, centered, centered)
    boundary_mass = float(
        weights[[0, -1], :].sum() + weights[:, [0, -1]].sum()
    )
    return {
        "mean": mean.tolist(),
        "sd": np.sqrt(np.diag(covariance)).tolist(),
        "correlation": float(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])),
        "method": "mode-centered Gauss-Legendre quadrature",
        "quadrature_order": order,
        "standardized_radius": radius,
        "mode": mode.tolist(),
        "mode_gradient_max": float(np.max(np.abs(mode_gradient))),
        "boundary_mass": boundary_mass,
    }


def _finite_extreme(values: Iterable[float], which: str) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not finite.size:
        return float("nan")
    return float(finite.min() if which == "min" else finite.max())


def run_arm(family: str, n: int, graph_seed: int, sampler_seed: int,
            arm: str, chains: int, warmup: int, draws: int,
            max_treedepth: int, reference: Dict,
            show_progress: bool = False) -> Dict:
    adjacency = make_graph(family, n, graph_seed)
    target = GraphGMRFTarget(adjacency, seed=graph_seed)
    sampler = FisherHMC(
        n_chains=chains,
        n_warmup=warmup,
        n_draws=draws,
        max_treedepth=max_treedepth,
    )
    if arm == "raw":
        algorithm = sampler
    elif arm == "sparse_rb":
        algorithm = Marginalized(
            g_idx=torch.arange(2, target.dim),
            sampler=sampler,
            sparse=True,
            chunk=None,
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")

    fit = algorithm.fit(
        target, seed=sampler_seed,
        progress=(f"{DISPLAY[family]} n={n} {ARM_LABEL[arm]} s={sampler_seed}"
                  if show_progress else False),
    )
    started = time.perf_counter()
    fit.sample(chains * draws)
    reconstruction_seconds = time.perf_counter() - started
    diagnostics = fit.diagnostics
    core_mean = np.asarray(diagnostics["coordinate_mean"][:2])
    core_sd = np.asarray(diagnostics["coordinate_sd"][:2])
    reference_mean = np.asarray(reference["mean"])
    reference_sd = np.asarray(reference["sd"])
    core_ess = _finite_extreme(diagnostics["coordinate_bulk_ess"][:2], "min")
    core_rhat = _finite_extreme(diagnostics["coordinate_rank_rhat"][:2], "max")
    passes = int(diagnostics.get("passes_per_collapsed_eval", 1))
    total_gradients = int(
        diagnostics["grad_evals"] + diagnostics["warmup_grad_evals"]
    )
    n_colors = int(diagnostics.get("n_colors", 0))
    reconstruction_rows = (
        chains * draws * (1 + n_colors) if arm == "sparse_rb" else 0
    )
    target_rows = total_gradients * passes + reconstruction_rows
    total_seconds = fit.runtime_s + reconstruction_seconds
    return {
        "family": family,
        "n": n,
        "graph_seed": graph_seed,
        "sampler_seed": sampler_seed,
        "arm": arm,
        "chains": chains,
        "warmup": warmup,
        "draws": draws,
        "max_treedepth": max_treedepth,
        "n_colors": n_colors,
        "runtime_seconds": fit.runtime_s,
        "reconstruction_seconds": reconstruction_seconds,
        "total_seconds": total_seconds,
        "grad_evals": int(diagnostics["grad_evals"]),
        "warmup_grad_evals": int(diagnostics["warmup_grad_evals"]),
        "passes_per_gradient": passes,
        "target_rows": target_rows,
        "core_min_bulk_ess": core_ess,
        "core_max_rank_rhat": core_rhat,
        "core_ess_per_second": core_ess / total_seconds,
        "target_rows_per_core_ess": target_rows / core_ess,
        "divergences": int(diagnostics["divergences"]),
        "warmup_divergences": int(diagnostics["divergences_warmup"]),
        "accept_rate": float(diagnostics["accept_rate"]),
        "mean_treedepth": float(diagnostics["mean_treedepth"]),
        "core_mean": core_mean.tolist(),
        "core_sd": core_sd.tolist(),
        "max_mean_z_error": float(np.max(np.abs(core_mean - reference_mean) / reference_sd)),
        "max_abs_log_scale_error": float(np.max(np.abs(np.log(core_sd / reference_sd)))),
    }


def _aggregate(records, family: str, n: int, arm: str, key: str) -> float:
    values = [
        row[key] for row in records
        if row["family"] == family and row["n"] == n and row["arm"] == arm
    ]
    return float(np.median(values))


def _passes_rhat_gate(records, family: str, n: int, arm: str,
                      threshold: float = 1.05) -> bool:
    values = [
        row["core_max_rank_rhat"] for row in records
        if row["family"] == family and row["n"] == n and row["arm"] == arm
    ]
    return bool(values) and all(np.isfinite(value) and value <= threshold
                                for value in values)


def plot(results: Dict, extra_records: Iterable[Dict] = ()) -> None:
    records = list(results["records"]) + list(extra_records)
    arms = tuple(arm for arm in PLOT_ARMS
                 if any(row["arm"] == arm for row in records))
    sizes = sorted({row["n"] for row in records})
    n = max(sizes)
    families = [family for family in FAMILIES if any(
        row["family"] == family and row["n"] == n for row in records
    )]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.25))
    specs = (
        ("core_ess_per_second", "minimum core ESS / s", "A", True),
        ("target_rows_per_core_ess", "target rows / core ESS", "B", True),
        ("core_max_rank_rhat", r"worst core rank-$\widehat R$", "C", False),
    )
    positions = np.arange(len(families))
    width = 0.8 / len(arms)
    for ax, (key, ylabel, panel, log_scale) in zip(axes, specs):
        for arm_index, arm in enumerate(arms):
            if key == "core_max_rank_rhat":
                values = [max(
                    row[key] for row in records
                    if row["family"] == family and row["n"] == n
                    and row["arm"] == arm
                ) for family in families]
            else:
                values = [
                    (_aggregate(records, family, n, arm, key)
                     if _passes_rhat_gate(records, family, n, arm)
                     else np.nan)
                    for family in families
                ]
            offset = (arm_index - (len(arms) - 1) / 2) * width
            ax.bar(
                positions + offset, values, width,
                color=[COLORS[family] for family in families],
                edgecolor="#333333", linewidth=0.45,
                hatch=ARM_HATCH[arm], label=ARM_LABEL[arm],
            )
            if key != "core_max_rank_rhat":
                for position, family, value in zip(positions + offset, families, values):
                    if not np.isfinite(value):
                        ax.text(position, 0.03, "fail", transform=ax.get_xaxis_transform(),
                                rotation=90, ha="center", va="bottom", fontsize=7,
                                color="#8B1E1E")
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_xticks(positions)
        ax.set_xticklabels([DISPLAY[family].replace("binary ", "").replace("3-regular ", "")
                            for family in families], rotation=28, ha="right")
        ax.grid(True, axis="y", which="major", color="#D9D9D9", lw=0.45)
        ax.text(-0.18, 1.04, panel, transform=ax.transAxes, fontweight="bold")
        if key == "core_max_rank_rhat":
            ax.axhline(1.05, color="#8B1E1E", lw=0.8, ls="--")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:len(arms)], labels[:len(arms)], frameon=False,
               loc="upper center", ncol=len(arms), bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write(results: Dict, output: str) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as handle:
        json.dump(results, handle, indent=2)


def _load_extra(path: str) -> list:
    """Learned-flow rows written by certified_flow_gmrf_flow.py, if present."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as handle:
        return json.load(handle)["records"]


def summarize(results: Dict, extra_records: Iterable[Dict] = ()) -> None:
    print("family               n  arm          gate  ESS/s  rows/ESS  Rhat  div  mean-z  seconds")
    for row in list(results["records"]) + list(extra_records):
        gate = "pass" if row["core_max_rank_rhat"] <= 1.05 else "fail"
        print(
            f"{DISPLAY[row['family']]:20s} {row['n']:4d} {row['arm']:10s} "
            f"{gate:4s} {row['core_ess_per_second']:6.2f} "
            f"{row['target_rows_per_core_ess']:9.0f} "
            f"{row['core_max_rank_rhat']:5.3f} {row['divergences']:4d} "
            f"{row['max_mean_z_error']:7.3f} {row['total_seconds']:8.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=RESULT_PATH)
    parser.add_argument("--flow-results",
                        default=os.path.join(ROOT, "results",
                                             "runs", "graph_gmrf_flow.json"),
                        help="learned-flow rows to merge into the figure")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--sizes", nargs="+", type=int, default=[256])
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--max-treedepth", type=int, default=6)
    parser.add_argument("--quadrature-order", type=int, default=80)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        with open(args.output) as handle:
            results = json.load(handle)
        extra = _load_extra(args.flow_results)
        plot(results, extra)
        summarize(results, extra)
        return

    if args.quick:
        args.sizes = [64]
        args.families = ["tree", "star"]
        args.seeds = [0]
        args.chains = 2
        args.warmup = 30
        args.draws = 30
        args.max_treedepth = 5
        args.quadrature_order = 40

    if args.resume and os.path.exists(args.output):
        with open(args.output) as handle:
            results = json.load(handle)
    else:
        results = {
            "experiment": "graph_gmrf_sparse_rb_nuts",
            "hardware": {"platform": platform.platform(), "processor": platform.processor()},
            "protocol": {
                "sizes": args.sizes,
                "families": args.families,
                "seeds": args.seeds,
                "chains": args.chains,
                "warmup": args.warmup,
                "draws": args.draws,
                "max_treedepth": args.max_treedepth,
                "quadrature_order": args.quadrature_order,
            },
            "references": {},
            "records": [],
        }

    completed = {
        (row["family"], row["n"], row["graph_seed"], row["sampler_seed"], row["arm"])
        for row in results["records"]
    }
    for family in args.families:
        for n in args.sizes:
            graph_seed = 0
            reference_key = f"{family}:{n}:{graph_seed}"
            if reference_key not in results["references"]:
                adjacency = make_graph(family, n, graph_seed)
                target = GraphGMRFTarget(adjacency, seed=graph_seed)
                print(f"[{family}] n={n} deterministic core reference", flush=True)
                results["references"][reference_key] = _core_reference(
                    adjacency, target, args.quadrature_order
                )
                _write(results, args.output)
            reference = results["references"][reference_key]
            for sampler_seed in args.seeds:
                for arm in ARMS:
                    key = (family, n, graph_seed, sampler_seed, arm)
                    if key in completed:
                        continue
                    print(
                        f"[{family}] n={n} seed={sampler_seed} {ARM_LABEL[arm]}",
                        flush=True,
                    )
                    row = run_arm(
                        family, n, graph_seed, sampler_seed, arm,
                        args.chains, args.warmup, args.draws,
                        args.max_treedepth, reference, args.progress,
                    )
                    results["records"].append(row)
                    completed.add(key)
                    _write(results, args.output)
                    print(
                        f"  ESS/s={row['core_ess_per_second']:.2f} "
                        f"rows/ESS={row['target_rows_per_core_ess']:.0f} "
                        f"Rhat={row['core_max_rank_rhat']:.3f} "
                        f"div={row['divergences']} time={row['total_seconds']:.1f}s",
                        flush=True,
                    )

    extra = _load_extra(args.flow_results)
    plot(results, extra)
    summarize(results, extra)
    print(f"wrote {args.output}\nwrote {FIGURE_PATH}\nwrote {PNG_PATH}")


if __name__ == "__main__":
    main()
