"""Lightweight diagnostics shared by the released sampler implementation."""
from __future__ import annotations

import torch


def pooled_ess(draws: torch.Tensor, n_chains: int) -> float:
    """Estimate ESS for the mean coordinate of a pooled chain ensemble."""
    total = draws.shape[0]
    if total < 4 or n_chains <= 0 or total % n_chains != 0:
        return float(total)
    kept = total // n_chains
    if kept < 4:
        return float(total)
    values = draws.reshape(kept, n_chains, -1).mean(-1)
    values = values - values.mean(0, keepdim=True)
    variance = (values**2).mean(0)
    ess_total = 0.0
    max_lag = min(kept - 1, 200)
    for chain in range(n_chains):
        scale = float(variance[chain])
        if scale <= 1e-30:
            ess_total += kept
            continue
        correlation_sum, previous = 0.0, 1.0
        for lag in range(1, max_lag + 1):
            correlation = float(
                (values[:-lag, chain] * values[lag:, chain]).mean()
            ) / scale
            if correlation + previous < 0.0:
                break
            correlation_sum += correlation
            previous = correlation
        ess_total += kept / max(1.0 + 2.0 * correlation_sum, 1.0)
    return float(ess_total)
