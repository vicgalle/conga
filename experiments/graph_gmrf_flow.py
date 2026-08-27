"""Learned-flow comparator for the graph GMRF suite.

``certified_flow_gmrf_nuts.py`` compares raw NUTS with sparse
Rao--Blackwellized NUTS on the graph-indexed GMRFs.  Neither arm learns
anything, so the suite lacks a learned baseline.  This driver adds one: nutpie's
online Fisher-divergence flow adaptation (and optionally its low-rank metric)
fitted on the *raw* ``(u, v, x)`` target, at the same graph, size, chains,
draws and tree depth as the other two arms.

The raw target is retranscribed in JAX so that nutpie can differentiate it, and
checked against the PyTorch target of ``certified_flow_gmrf.py`` at random
points before any sampling; the dataset ``y`` is taken from the PyTorch target
itself, so all three arms see identical data.  Core moments are scored against
the same mode-centered Gauss--Legendre reference used by the NUTS driver.

    uv run --extra experiments python -m experiments.graph_gmrf_flow \
        --families lattice --seeds 0 1 2

Use ``--quick`` for a small integration run; ``--resume`` skips completed
cells.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from typing import Dict

import arviz as az
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import nutpie
from nutpie.compiled_pyfunc import from_pyfunc
from scipy import sparse
import torch

from conga.core import DTYPE
from experiments.graph_gmrf import (
    DISPLAY,
    FAMILIES,
    GraphGMRFTarget,
    make_graph,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_PATH = os.path.join(ROOT, "results", "runs", "graph_gmrf_flow.json")
NUTS_PATH = os.path.join(ROOT, "results", "runs", "graph_gmrf_nuts.json")
ARM_LABEL = {"flow": "nutpie flow", "low_rank": "nutpie low-rank"}


def _jax_log_prob(target: GraphGMRFTarget, adjacency: sparse.csc_matrix):
    """Independent JAX transcription of ``GraphGMRFTarget.log_prob``."""
    upper = sparse.triu(adjacency, k=1).tocoo()
    edge_i = jnp.asarray(upper.row.copy())
    edge_j = jnp.asarray(upper.col.copy())
    y = jnp.asarray(target.y.detach().cpu().numpy(), dtype=jnp.float64)
    n = float(target.n)
    kappa = float(target.kappa)

    def log_prob(z):
        state_log_sd, obs_log_sd, x = z[0], z[1], z[2:]
        edge_delta = x[edge_i] - x[edge_j]
        state_quad = jnp.sum(edge_delta ** 2) + kappa * jnp.sum(x ** 2)
        obs_quad = jnp.sum((y - x) ** 2)
        return (
            -n * state_log_sd
            - 0.5 * jnp.exp(-2 * state_log_sd) * state_quad
            - n * obs_log_sd
            - 0.5 * jnp.exp(-2 * obs_log_sd) * obs_quad
            - 0.5 * ((state_log_sd + 1.0) / 0.7) ** 2
            - 0.5 * ((obs_log_sd + 1.5) / 0.7) ** 2
        )

    return log_prob


def _transcription_error(target: GraphGMRFTarget, log_prob, seed: int,
                         probes: int = 4) -> float:
    rng = np.random.default_rng(seed + 77)
    worst = 0.0
    for _ in range(probes):
        z = np.r_[
            rng.normal(-1.0, 0.3, 1), rng.normal(-1.5, 0.3, 1),
            0.4 * rng.standard_normal(target.n),
        ]
        reference = float(target.log_prob(torch.as_tensor(z, dtype=DTYPE)).item())
        candidate = float(log_prob(jnp.asarray(z, dtype=jnp.float64)))
        worst = max(worst, abs(reference - candidate) / max(abs(reference), 1.0))
    return worst


def _compile(target: GraphGMRFTarget, log_prob):
    value_and_grad = jax.jit(jax.value_and_grad(log_prob))
    dim = target.dim

    def make_logp():
        def fn(z):
            value, gradient = value_and_grad(jnp.asarray(z, dtype=jnp.float64))
            return float(value), np.asarray(gradient, dtype=np.float64, order="C")
        return fn

    def make_expand(seed1, seed2, chain):
        del seed1, seed2, chain

        def expand(z):
            return {"theta": np.asarray(z, dtype=np.float64, order="C")}
        return expand

    initial = np.r_[-1.0, -1.5, np.zeros(target.n)]
    return from_pyfunc(
        ndim=dim, make_logp_fn=make_logp, make_expand_fn=make_expand,
        expanded_dtypes=[np.dtype("float64")], expanded_shapes=[(dim,)],
        expanded_names=["theta"],
        make_initial_point_fn=lambda seed: initial.copy(),
        # nutpie's flow adaptation trains against the raw JAX log density
        raw_logp_fn=lambda z: (log_prob(z),),
    )


def _rows(trace) -> int:
    total = int(trace.sample_stats["n_steps"].sum())
    if "warmup_sample_stats" in trace.children:
        total += int(trace["warmup_sample_stats"]["n_steps"].sum())
    return total


def run_arm(family: str, n: int, graph_seed: int, sampler_seed: int, arm: str,
            chains: int, warmup: int, draws: int, max_treedepth: int,
            reference: Dict) -> Dict:
    adjacency = make_graph(family, n, graph_seed)
    target = GraphGMRFTarget(adjacency, seed=graph_seed)
    log_prob = _jax_log_prob(target, adjacency)
    transcription_error = _transcription_error(target, log_prob, graph_seed)
    model = _compile(target, log_prob)

    started = time.perf_counter()
    trace = nutpie.sample(
        model, chains=chains, draws=draws, tune=warmup, seed=sampler_seed,
        progress_bar=False, save_warmup=True, adaptation=arm,
        max_energy_error=1000.0, maxdepth=max_treedepth,
    )
    total_seconds = time.perf_counter() - started

    values = np.asarray(trace.posterior["theta"].values)
    dataset = az.convert_to_dataset(values[:, :, :2])
    ess = np.asarray(az.ess(dataset, method="bulk")["x"])
    rhat = np.asarray(az.rhat(dataset, method="rank")["x"])
    core = values[:, :, :2].reshape(-1, 2)
    core_mean, core_sd = core.mean(0), core.std(0, ddof=1)
    reference_mean = np.asarray(reference["mean"])
    reference_sd = np.asarray(reference["sd"])
    core_ess = float(np.nanmin(ess))
    target_rows = _rows(trace)
    return {
        "family": family, "n": n, "graph_seed": graph_seed,
        "sampler_seed": sampler_seed, "arm": arm, "chains": chains,
        "warmup": warmup, "draws": draws, "max_treedepth": max_treedepth,
        "transcription_rel_error": transcription_error,
        "total_seconds": total_seconds,
        "target_rows": target_rows,
        "core_min_bulk_ess": core_ess,
        "core_max_rank_rhat": float(np.nanmax(rhat)),
        "core_ess_per_second": core_ess / total_seconds,
        "target_rows_per_core_ess": target_rows / max(core_ess, 1e-12),
        "divergences": int(trace.sample_stats["diverging"].sum()),
        "warmup_divergences": (
            int(trace["warmup_sample_stats"]["diverging"].sum())
            if "warmup_sample_stats" in trace.children else 0
        ),
        "accept_rate": float(trace.sample_stats["mean_tree_accept"].mean()),
        "mean_steps": float(trace.sample_stats["n_steps"].mean()),
        "core_mean": core_mean.tolist(), "core_sd": core_sd.tolist(),
        "max_mean_z_error": float(
            np.max(np.abs(core_mean - reference_mean) / reference_sd)
        ),
        "max_abs_log_scale_error": float(
            np.max(np.abs(np.log(core_sd / reference_sd)))
        ),
    }


def _write(results: Dict, output: str) -> None:
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as handle:
        json.dump(results, handle, indent=2)


def summarize(results: Dict) -> None:
    print("family               n  arm             gate  ESS/s  rows/ESS  Rhat  div  mean-z  seconds")
    for row in results["records"]:
        gate = "pass" if row["core_max_rank_rhat"] <= 1.05 else "fail"
        print(
            f"{DISPLAY[row['family']]:20s} {row['n']:4d} {row['arm']:13s} "
            f"{gate:4s} {row['core_ess_per_second']:6.2f} "
            f"{row['target_rows_per_core_ess']:9.0f} "
            f"{row['core_max_rank_rhat']:5.3f} {row['divergences']:4d} "
            f"{row['max_mean_z_error']:7.3f} {row['total_seconds']:8.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=RESULT_PATH)
    parser.add_argument("--reference-source", default=NUTS_PATH)
    parser.add_argument("--families", nargs="+", choices=FAMILIES,
                        default=["lattice"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[256])
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", choices=sorted(ARM_LABEL),
                        default=["flow"])
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=800)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--max-treedepth", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.sizes = [256]
        args.families = ["lattice"]
        args.seeds = [0]
        args.chains = 2
        args.warmup = 40
        args.draws = 40
        args.max_treedepth = 5

    with open(args.reference_source) as handle:
        references = json.load(handle)["references"]

    if args.resume and os.path.exists(args.output):
        with open(args.output) as handle:
            results = json.load(handle)
    else:
        results = {
            "experiment": "graph_gmrf_learned_flow",
            "hardware": {"platform": platform.platform(),
                         "processor": platform.processor()},
            "protocol": {
                "sizes": args.sizes, "families": args.families,
                "seeds": args.seeds, "arms": args.arms, "chains": args.chains,
                "warmup": args.warmup, "draws": args.draws,
                "max_treedepth": args.max_treedepth,
                "reference_source": os.path.basename(args.reference_source),
            },
            "records": [],
        }

    completed = {
        (row["family"], row["n"], row["sampler_seed"], row["arm"])
        for row in results["records"]
    }
    for family in args.families:
        for n in args.sizes:
            key = f"{family}:{n}:{args.graph_seed}"
            if key not in references:
                raise KeyError(f"no cached core reference for {key}")
            reference = references[key]
            for sampler_seed in args.seeds:
                for arm in args.arms:
                    if (family, n, sampler_seed, arm) in completed:
                        continue
                    print(f"[{family}] n={n} seed={sampler_seed} "
                          f"{ARM_LABEL[arm]}", flush=True)
                    row = run_arm(
                        family, n, args.graph_seed, sampler_seed, arm,
                        args.chains, args.warmup, args.draws,
                        args.max_treedepth, reference,
                    )
                    results["records"].append(row)
                    completed.add((family, n, sampler_seed, arm))
                    _write(results, args.output)
                    print(
                        f"  ESS/s={row['core_ess_per_second']:.2f} "
                        f"rows/ESS={row['target_rows_per_core_ess']:.0f} "
                        f"Rhat={row['core_max_rank_rhat']:.3f} "
                        f"div={row['divergences']} "
                        f"mean-z={row['max_mean_z_error']:.3f} "
                        f"time={row['total_seconds']:.1f}s",
                        flush=True,
                    )

    summarize(results)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
