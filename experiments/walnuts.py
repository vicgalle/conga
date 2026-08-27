"""Thin wrapper around the released walnutpie sampler."""
from __future__ import annotations

import numpy as np
import walnutpie as wp


def run_walnuts(
    logp,
    dim: int,
    n_chains: int,
    warmup: int,
    draws: int,
    seed: int,
    *,
    save_warmup: bool = False,
    max_doublings: int = 10,
    inits=None,
    sampler_kwargs=None,
):
    """Run WALNUTS and return a ``[chain, draw, dimension]`` array."""
    kwargs = dict(sampler_kwargs or {})
    max_doublings = kwargs.pop("max_trajectory_doublings", max_doublings)
    if inits is not None:
        kwargs["inits"] = np.ascontiguousarray(inits, dtype=np.float64)
    fits = wp.walnuts_pyfunc(
        logp,
        num_params=dim,
        num_chains=n_chains,
        seed=seed,
        min_warmup_iter=warmup,
        max_warmup_iter=warmup,
        min_sampling_iter=draws,
        max_sampling_iter=draws,
        max_trajectory_doublings=max_doublings,
        save_warmup=save_warmup,
        **kwargs,
    )
    chains = np.stack([np.asarray(chain) for chain in fits], axis=0)
    warmup_draws = (
        np.stack([np.asarray(chain.warmup.warmup_draws) for chain in fits], axis=0)
        if save_warmup
        else None
    )
    return chains, warmup_draws
