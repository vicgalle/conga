"""Paper experiment: NUTS/WALNUTS before and after CONGA compilation.

All four arms use JAX-compiled value-and-gradient callbacks.  NUTS is the
released nutpie sampler; WALNUTS is the released walnutpie sampler.  Each raw
pair shares one score implementation, and each CONGA pair shares one compiled
collapsed-core score implementation.

The model uses the WALNUTS innovation coordinates for z and x.  CONGA exposes
tau in centered coordinates, integrates that Gaussian path with an O(T)
tridiagonal Cholesky scan, samples the 504-dimensional core, and reconstructs
tau exactly after sampling.

The default protocol is the paper protocol: eight chains, 800 warmup and 800
retained draws, seeds 0--2.  Results are checkpointed after every arm and the
command is resumable.

    uv run --extra experiments python -m experiments.stock_watson --quick
    uv run --extra experiments python -m experiments.stock_watson --resume
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

import arviz as az
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import nutpie
import torch
from nutpie.compiled_pyfunc import from_pyfunc

from conga.structure import GaussianBlock
from experiments.models.stock_watson import StockWatson, StockWatsonCenteredTau
from experiments.walnuts import run_walnuts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "runs" / "stock_watson.json"
DATA_PATH = ROOT / "data" / "stock_watson.json"
LOG2PI = math.log(2.0 * math.pi)
ARMS = ("NUTS-raw", "WALNUTS-raw", "NUTS-CONGA", "WALNUTS-CONGA")


def _split_raw(q, T):
    return (q[..., :T - 1], q[..., T - 1:2 * T - 1],
            q[..., 2 * T - 1:3 * T - 1], q[..., -1])


def _split_core(eta, T):
    return eta[..., :T - 1], eta[..., T - 1:2 * T - 1], eta[..., -1]


def _random_walk(initial, innovations, scale):
    return jnp.concatenate(
        [initial[None], initial + jnp.cumsum(scale * innovations)])


def build_functions(y_np: np.ndarray):
    """Return the raw density, compiled core density, and reconstruction maps."""
    y = jnp.asarray(y_np, dtype=jnp.float64)
    T = int(y.shape[0])

    def core_processes(eta):
        zbar, xbar, ell = _split_core(eta, T)
        sigma = jnp.exp(0.5 * ell)
        z = _random_walk(zbar[0], zbar[1:], sigma)
        x = _random_walk(xbar[0], xbar[1:], sigma)
        return zbar, xbar, ell, z, x

    def raw_processes(q):
        zbar, xbar, taubar, ell = _split_raw(q, T)
        sigma = jnp.exp(0.5 * ell)
        z = _random_walk(zbar[0], zbar[1:], sigma)
        x = _random_walk(xbar[0], xbar[1:], sigma)
        tau = _random_walk(taubar[0], taubar[1:], jnp.exp(0.5 * z))
        return zbar, xbar, taubar, ell, z, x, tau

    def raw_logp(q):
        zbar, xbar, taubar, ell, _, x, tau = raw_processes(q)
        resid = (y - tau) * jnp.exp(-0.5 * x)
        return (-5.0 * ell - 0.5 * jnp.exp(-ell)
                - 0.5 * (jnp.sum(zbar[1:] ** 2)
                         + jnp.sum(xbar[1:] ** 2)
                         + jnp.sum(taubar[1:] ** 2))
                - 0.5 * jnp.sum(x) - 0.5 * jnp.sum(resid ** 2))

    def moments(eta):
        _, _, _, z, x = core_processes(eta)
        rw_precision = jnp.exp(-z)
        obs_precision = jnp.exp(-x)
        diagonal = obs_precision
        diagonal = diagonal.at[:-1].add(rw_precision)
        diagonal = diagonal.at[1:].add(rw_precision)
        off_diagonal = -rw_precision
        shift = obs_precision * y

        c0 = jnp.sqrt(diagonal[0])

        def factor_step(c_prev, inputs):
            diagonal_i, off_i = inputs
            f_i = off_i / c_prev
            c_i = jnp.sqrt(diagonal_i - f_i ** 2)
            return c_i, (c_i, f_i)

        _, (c_rest, f_rest) = jax.lax.scan(
            factor_step, c0, (diagonal[1:], off_diagonal))
        c = jnp.concatenate([c0[None], c_rest])
        f = jnp.concatenate([jnp.zeros(1, dtype=eta.dtype), f_rest])

        u0 = shift[0] / c[0]

        def forward_step(u_prev, inputs):
            shift_i, c_i, f_i = inputs
            u_i = (shift_i - f_i * u_prev) / c_i
            return u_i, u_i

        _, u_rest = jax.lax.scan(
            forward_step, u0, (shift[1:], c[1:], f[1:]))
        u = jnp.concatenate([u0[None], u_rest])

        mean_last = u[-1] / c[-1]

        def backward_step(mean_next, inputs):
            u_i, c_i, f_next = inputs
            mean_i = (u_i - f_next * mean_next) / c_i
            return mean_i, mean_i

        _, mean_rev = jax.lax.scan(
            backward_step, mean_last,
            (u[:-1][::-1], c[:-1][::-1], f[1:][::-1]))
        mean = jnp.concatenate([mean_rev[::-1], mean_last[None]])
        return mean, c, f, obs_precision, shift

    def collapsed_logp(eta):
        zbar, xbar, ell, z, x = core_processes(eta)
        mean, c, _, obs_precision, shift = moments(eta)
        base = (-5.0 * ell - 0.5 * jnp.exp(-ell)
                - 0.5 * (jnp.sum(zbar[1:] ** 2)
                         + jnp.sum(xbar[1:] ** 2))
                - 0.5 * jnp.sum(z) - 0.5 * jnp.sum(x)
                - 0.5 * jnp.sum(obs_precision * y ** 2))
        return (base + 0.5 * jnp.dot(shift, mean)
                + 0.5 * T * LOG2PI - jnp.sum(jnp.log(c)))

    def reconstruct_one(eta, eps):
        _, _, ell, z, x = core_processes(eta)
        mean, c, f, _, _ = moments(eta)
        noise_last = eps[-1] / c[-1]

        def backward_step(noise_next, inputs):
            eps_i, c_i, f_next = inputs
            noise_i = (eps_i - f_next * noise_next) / c_i
            return noise_i, noise_i

        _, noise_rev = jax.lax.scan(
            backward_step, noise_last,
            (eps[:-1][::-1], c[:-1][::-1], f[1:][::-1]))
        noise = jnp.concatenate([noise_rev[::-1], noise_last[None]])
        tau = mean + noise
        return jnp.concatenate([z, x, tau, ell[None]])

    def common_from_raw(q):
        *_, ell, z, x, tau = raw_processes(q)
        return jnp.concatenate([z, x, tau, ell[None]])

    return {
        "T": T,
        "raw_dim": 3 * T,
        "core_dim": 2 * T,
        "raw_logp": raw_logp,
        "collapsed_logp": collapsed_logp,
        "reconstruct": jax.jit(jax.vmap(reconstruct_one)),
        "common_raw": jax.jit(jax.vmap(common_from_raw)),
    }


class JaxScore:
    """Thread-safe guarded callback shared by nutpie and walnutpie.

    The finite fallback keeps both samplers on the same numerical target when
    a proposal leaves the stable floating-point domain of the compiled score.
    """

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
        np.asarray(grad)
        float(val)
        return time.perf_counter() - t0

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


def initial_positions(funcs):
    """Same deterministic core point, with raw tau at its conditional mean."""
    core = np.zeros(funcs["core_dim"], dtype=np.float64)
    common = np.asarray(funcs["reconstruct"](
        jnp.asarray(core[None]), jnp.zeros((1, funcs["T"]))))[0]
    tau = common[2 * funcs["T"] - 1:3 * funcs["T"] - 1]
    raw = np.zeros(funcs["raw_dim"], dtype=np.float64)
    raw[2 * funcs["T"] - 1:3 * funcs["T"] - 1] = np.concatenate(
        [tau[:1], np.diff(tau)])
    return {"raw": raw, "core": core}


def make_nutpie_model(score: JaxScore, initial: np.ndarray):
    dim = score.dim

    def make_expand(seed1, seed2, chain):
        del seed1, seed2, chain
        return lambda x: {"theta": np.asarray(x, dtype=np.float64, order="C")}

    return from_pyfunc(
        ndim=dim,
        make_logp_fn=lambda: score,
        make_expand_fn=make_expand,
        expanded_dtypes=[np.dtype("float64")],
        expanded_shapes=[(dim,)],
        expanded_names=["theta"],
        make_initial_point_fn=lambda seed: initial.copy(),
        raw_logp_fn=None,
    )


def _nuts_rows(trace):
    total = int(trace.sample_stats["n_steps"].sum())
    if "warmup_sample_stats" in trace.children:
        total += int(trace["warmup_sample_stats"]["n_steps"].sum())
    return total


def common_chains(position_chains, *, collapsed, funcs, seed):
    N, S, _ = position_chains.shape
    flat = position_chains.reshape(-1, position_chains.shape[-1])
    if collapsed:
        eps = np.random.default_rng(seed + 777).standard_normal((N * S, funcs["T"]))
        common = funcs["reconstruct"](
            jnp.asarray(flat, dtype=jnp.float64),
            jnp.asarray(eps, dtype=jnp.float64))
    else:
        common = funcs["common_raw"](jnp.asarray(flat, dtype=jnp.float64))
    return np.asarray(common).reshape(N, S, -1)


def diagnostics(common):
    dataset = az.convert_to_dataset(common)
    bulk = np.asarray(az.ess(dataset, method="bulk")["x"]).reshape(-1)
    rhat = np.asarray(az.rhat(dataset, method="rank")["x"]).reshape(-1)
    finite_ess = bulk[np.isfinite(bulk)]
    finite_rhat = rhat[np.isfinite(rhat)]
    return {
        "min_bulk_ess": float(finite_ess.min()) if finite_ess.size else None,
        "max_rank_rhat": float(finite_rhat.max()) if finite_rhat.size else None,
    }


def summaries(common, T):
    names = {
        "z[1]": 0,
        f"z[{(T - 1) // 2}]": max(0, (T - 1) // 2 - 1),
        f"z[{T - 1}]": T - 2,
        "x[1]": T - 1,
        f"x[{T // 2}]": T - 1 + T // 2 - 1,
        f"x[{T}]": 2 * T - 2,
        "tau[1]": 2 * T - 1,
        f"tau[{T // 2}]": 2 * T - 1 + T // 2 - 1,
        f"tau[{T}]": 3 * T - 2,
        "log_sigma2": 3 * T - 1,
    }
    flat = common.reshape(-1, common.shape[-1])
    out = {}
    for name, index in names.items():
        values = flat[:, index]
        out[name] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "q05": float(np.quantile(values, 0.05)),
            "q50": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return out


def run_nuts(arm, score, model, funcs, args, seed):
    collapsed = arm.endswith("CONGA")
    score.reset()
    t0 = time.perf_counter()
    trace = nutpie.sample(
        model, chains=args.chains, draws=args.draws, tune=args.warmup,
        seed=seed, progress_bar=False, save_warmup=True, adaptation="diag",
        max_energy_error=1000.0, maxdepth=args.maxdepth,
        target_accept=args.nuts_target_accept)
    seconds = time.perf_counter() - t0
    positions = np.asarray(trace.posterior["theta"].values)
    r0 = time.perf_counter()
    common = common_chains(positions, collapsed=collapsed, funcs=funcs, seed=seed)
    reconstruct_s = time.perf_counter() - r0
    row = {
        "arm": arm, "seed": seed, "sampler": "nutpie NUTS",
        "score": "JAX compiled", "sampled_dim": positions.shape[-1],
        "seconds": seconds, "reconstruction_s": reconstruct_s,
        "score_calls": score.calls, "gradient_rows": _nuts_rows(trace),
        "guard_hits": score.guard_hits,
        "mean_steps": float(trace.sample_stats["n_steps"].mean()),
        "divergences": int(trace.sample_stats["diverging"].sum()),
        "mean_accept": float(trace.sample_stats["mean_tree_accept"].mean()),
        **diagnostics(common), "summary": summaries(common, funcs["T"]),
    }
    return row, common


def run_walnuts_arm(arm, score, funcs, initial, args, seed):
    collapsed = arm.endswith("CONGA")
    score.reset()
    inits = np.tile(initial, (args.chains, 1))
    kwargs = {
        "step_size_init": args.walnuts_step_size,
        "max_hamiltonian_error": args.walnuts_max_hamiltonian_error,
        "min_micro_steps": args.walnuts_min_micro_steps,
    }
    if args.max_step_halvings is not None:
        kwargs["max_step_halvings"] = args.max_step_halvings
    t0 = time.perf_counter()
    positions, _ = run_walnuts(
        score, score.dim, args.chains, args.warmup, args.draws, seed,
        inits=inits, max_doublings=args.maxdepth, sampler_kwargs=kwargs)
    seconds = time.perf_counter() - t0
    r0 = time.perf_counter()
    common = common_chains(positions, collapsed=collapsed, funcs=funcs, seed=seed)
    reconstruct_s = time.perf_counter() - r0
    row = {
        "arm": arm, "seed": seed, "sampler": "walnutpie WALNUTS",
        "score": "JAX compiled", "sampled_dim": positions.shape[-1],
        "seconds": seconds, "reconstruction_s": reconstruct_s,
        "score_calls": score.calls, "gradient_rows": score.calls,
        "guard_hits": score.guard_hits,
        "mean_steps": score.calls / (args.chains * (args.warmup + args.draws)),
        "divergences": None,
        **diagnostics(common), "summary": summaries(common, funcs["T"]),
    }
    return row, common


def contract_check(funcs, raw_target):
    """Cross-check the independent JAX raw and collapsed implementations."""
    rng = np.random.default_rng(42)
    raw_vg = jax.jit(jax.value_and_grad(funcs["raw_logp"]))
    raw_value_error = raw_grad_error = 0.0
    for _ in range(3):
        q = 0.1 * rng.standard_normal(funcs["raw_dim"])
        value, grad = raw_vg(jnp.asarray(q))
        qt = torch.as_tensor(q, dtype=torch.float64).requires_grad_()
        ref = raw_target.log_prob(qt)
        ref_grad = torch.autograd.grad(ref, qt)[0]
        raw_value_error = max(
            raw_value_error, abs(float(value) - float(ref.detach())))
        raw_grad_error = max(raw_grad_error,
                             float(np.max(np.abs(np.asarray(grad)
                                                 - ref_grad.detach().numpy()))))

    centered = StockWatsonCenteredTau(
        {"T": raw_target.T, "y": raw_target.y.detach().numpy()})
    T = raw_target.T
    g_idx = torch.arange(2 * T - 1, 3 * T - 1)
    pattern = torch.eye(T, dtype=torch.bool)
    pattern[:-1, 1:] |= torch.eye(T - 1, dtype=torch.bool)
    pattern[1:, :-1] |= torch.eye(T - 1, dtype=torch.bool)
    block = GaussianBlock(
        centered, g_idx, probe_etas=torch.zeros(2, funcs["core_dim"]),
        pattern=pattern, banded=False)
    core_vg = jax.jit(jax.value_and_grad(funcs["collapsed_logp"]))
    collapsed_value_error = collapsed_grad_error = 0.0
    for _ in range(2):
        eta = 0.05 * rng.standard_normal(funcs["core_dim"])
        value, grad = core_vg(jnp.asarray(eta))
        et = torch.as_tensor(eta, dtype=torch.float64).requires_grad_()
        ref = block.collapsed_log_post(et.unsqueeze(0))[0]
        ref_grad = torch.autograd.grad(ref, et)[0]
        collapsed_value_error = max(collapsed_value_error,
                                    abs(float(value) - float(ref.detach())))
        collapsed_grad_error = max(
            collapsed_grad_error,
            float(np.max(np.abs(np.asarray(grad) - ref_grad.detach().numpy()))))
    identity = block.joint_identity_residual(
        0.05 * torch.randn(4, funcs["core_dim"], dtype=torch.float64))
    return {
        "raw_value_abs": raw_value_error,
        "raw_grad_abs": raw_grad_error,
        "collapsed_value_abs": collapsed_value_error,
        "collapsed_grad_abs": collapsed_grad_error,
        "conditional_identity_abs": identity,
        "d_g": T, "d_h": funcs["core_dim"], "colors": 3,
        "bandwidth": 1, "nnz": 3 * T - 2,
    }


def save_payload(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--maxdepth", type=int, default=10)
    ap.add_argument("--max-step-halvings", type=int, default=None)
    ap.add_argument("--nuts-target-accept", type=float, default=0.95)
    ap.add_argument("--walnuts-step-size", type=float, default=0.1,
                    help="WALNUTS macro step size h used to initialize adaptation")
    ap.add_argument("--walnuts-max-hamiltonian-error", type=float, default=0.3,
                    help="WALNUTS local energy threshold delta")
    ap.add_argument("--walnuts-min-micro-steps", type=int, default=8,
                    help="minimum WALNUTS micro steps per macro step")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-draws", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="two chains, 20 warmup/draws, seed 0")
    args = ap.parse_args()
    if args.quick:
        args.chains = 2
        args.warmup = 20
        args.draws = 20
        args.seeds = [0]
    if not args.output.is_absolute():
        args.output = ROOT / args.output

    with DATA_PATH.open() as handle:
        raw_target = StockWatson(json.load(handle))
    funcs = build_functions(raw_target.y.detach().numpy())
    check = contract_check(funcs, raw_target)
    print("contract " + json.dumps(check, sort_keys=True), flush=True)
    if (check["raw_value_abs"] > 1e-9 or check["raw_grad_abs"] > 1e-8
            or check["collapsed_value_abs"] > 1e-8
            or check["collapsed_grad_abs"] > 1e-7):
        raise RuntimeError("Stock-Watson JAX transcription failed its cross-check")

    raw_score = JaxScore(funcs["raw_logp"], funcs["raw_dim"])
    core_score = JaxScore(funcs["collapsed_logp"], funcs["core_dim"])
    raw_compile_s = raw_score.compile()
    core_compile_s = core_score.compile()
    initials = initial_positions(funcs)
    models = {"raw": make_nutpie_model(raw_score, initials["raw"]),
              "core": make_nutpie_model(core_score, initials["core"])}

    config = {
        "chains": args.chains, "warmup": args.warmup, "draws": args.draws,
        "seeds": args.seeds, "maxdepth": args.maxdepth,
        "max_step_halvings": args.max_step_halvings,
        "nuts_target_accept": args.nuts_target_accept,
        "walnuts_step_size": args.walnuts_step_size,
        "walnuts_max_hamiltonian_error": args.walnuts_max_hamiltonian_error,
        "walnuts_min_micro_steps": args.walnuts_min_micro_steps,
    }
    if args.resume and args.output.exists():
        with args.output.open() as handle:
            payload = json.load(handle)
        if payload.get("config") != config:
            raise RuntimeError("resume output has a different experiment config")
    else:
        payload = {
            "model": "Stock-Watson", "T": funcs["T"],
            "raw_dim": funcs["raw_dim"], "core_dim": funcs["core_dim"],
            "config": config, "contract": check,
            "compile_seconds": {"raw": raw_compile_s, "core": core_compile_s},
            "rows": [],
        }
        save_payload(args.output, payload)

    done = {(r["arm"], r["seed"]) for r in payload["rows"]
            if "error" not in r}
    draw_dir = args.output.with_suffix("").with_name(
        args.output.stem + "_draws")
    for seed in args.seeds:
        for arm in args.arms:
            if (arm, seed) in done:
                print(f"skip {arm} seed={seed}", flush=True)
                continue
            print(f"run {arm} seed={seed}", flush=True)
            score = core_score if arm.endswith("CONGA") else raw_score
            try:
                if arm.startswith("NUTS"):
                    model = models["core" if arm.endswith("CONGA") else "raw"]
                    row, common = run_nuts(arm, score, model, funcs, args, seed)
                else:
                    initial = initials["core" if arm.endswith("CONGA") else "raw"]
                    row, common = run_walnuts_arm(
                        arm, score, funcs, initial, args, seed)
            except Exception as exc:
                row = {"arm": arm, "seed": seed,
                       "error": f"{type(exc).__name__}: {exc}"}
                common = None
            if args.save_draws and common is not None:
                try:
                    draw_dir.mkdir(parents=True, exist_ok=True)
                    draw_path = draw_dir / f"{arm.lower()}_seed{seed}.npz"
                    np.savez_compressed(draw_path, common=common.astype(np.float32))
                    row["draw_file"] = str(draw_path.relative_to(ROOT))
                except Exception as exc:
                    row["draw_error"] = f"{type(exc).__name__}: {exc}"
            payload["rows"] = [r for r in payload["rows"]
                               if (r["arm"], r["seed"]) != (arm, seed)]
            payload["rows"].append(row)
            save_payload(args.output, payload)
            print(json.dumps({k: v for k, v in row.items()
                              if k != "summary"}, sort_keys=True), flush=True)

    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
