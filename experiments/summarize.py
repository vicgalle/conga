"""Print a compact Section 4.2 summary from the checked-in paper results."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results" / "paper"


def _read(name: str):
    with (PAPER / name).open() as handle:
        return json.load(handle)


def _stock_watson() -> None:
    rows = _read("stock_watson.json")["rows"]
    grouped = defaultdict(list)
    for row in rows:
        if "error" not in row:
            grouped[row["arm"]].append(row)

    print("Stock--Watson (median over three seeds)")
    print("  arm                 seconds   min ESS   worst R-hat   divergences")
    for arm in ("NUTS-raw", "WALNUTS-raw", "NUTS-CONGA", "WALNUTS-CONGA"):
        group = grouped[arm]
        divergences = [row["divergences"] for row in group]
        divergence_text = "--" if divergences[0] is None else str(sum(divergences))
        print(
            f"  {arm:19s} {median(r['seconds'] for r in group):8.1f} "
            f"{median(r['min_bulk_ess'] for r in group):9.0f} "
            f"{max(r['max_rank_rhat'] for r in group):13.3f} "
            f"{divergence_text:>13s}"
        )
    for sampler in ("NUTS", "WALNUTS"):
        raw = {row["seed"]: row for row in grouped[f"{sampler}-raw"]}
        compiled = {row["seed"]: row for row in grouped[f"{sampler}-CONGA"]}
        gains = []
        for seed in sorted(raw.keys() & compiled.keys()):
            raw_efficiency = raw[seed]["min_bulk_ess"] / raw[seed]["seconds"]
            compiled_efficiency = (
                compiled[seed]["min_bulk_ess"] / compiled[seed]["seconds"]
            )
            gains.append(compiled_efficiency / raw_efficiency)
        print(f"  paired {sampler} minimum-ESS/s gain: {median(gains):.1f}x")


def _load_dlm() -> None:
    payload = _read("load_dlm_sample.json")
    grouped = defaultdict(list)
    for row in payload["rows"]:
        if "error" not in row:
            grouped[row["arm"]].append(row)
    print("\nElectricity-load DLM")
    for arm in ("NUTS-raw", "NUTS-CONGA"):
        group = grouped[arm]
        print(
            f"  {arm:12s}: median {median(r['seconds'] for r in group):.1f} s, "
            f"minimum ESS/s {median(r['ess_per_s'] for r in group):.3f}, "
            f"worst R-hat {max(r['max_rank_rhat'] for r in group):.3f}"
        )
    raw = {row["seed"]: row for row in grouped["NUTS-raw"]}
    compiled = {row["seed"]: row for row in grouped["NUTS-CONGA"]}
    gain = (
        median(row["ess_per_s"] for row in compiled.values())
        / median(row["ess_per_s"] for row in raw.values())
    )
    print(f"  minimum-ESS/s gain from medians: {gain:.0f}x")


def _graph_gmrf() -> None:
    fixed = _read("graph_gmrf_nuts.json")["records"]
    learned = _read("graph_gmrf_flow.json")["records"]
    rows = fixed + learned
    print("\nGraph-indexed fields (median over three seeds)")
    print("  family              raw NUTS   learned flow   sparse RB-NUTS")
    for family in ("tree", "lattice", "geometric", "expander", "star"):
        values = {}
        for arm in ("raw", "flow", "sparse_rb"):
            group = [row for row in rows if row["family"] == family and row["arm"] == arm]
            passed = group and all(row["core_max_rank_rhat"] <= 1.05 for row in group)
            values[arm] = (
                f"{median(row['core_ess_per_second'] for row in group):.2f}"
                if passed
                else "fail"
            )
        print(
            f"  {family:18s} {values['raw']:>8s} "
            f"{values['flow']:>14s} {values['sparse_rb']:>16s}"
        )


def main() -> None:
    _stock_watson()
    _load_dlm()
    _graph_gmrf()


if __name__ == "__main__":
    main()
