"""Paired-bootstrap significance test between adjacent rows in results_final.csv.

Used to defend per-architecture or per-intervention deltas with a
statistically meaningful p-value (audit fix #6 reserved item from
HPC_LAUNCH_GUIDE.md §3 G8).

Methodology — paired bootstrap on per-seed Top-1 differences:
  For two experiment groups A and B (with N seeds each, paired by seed ID),
  compute Δ = mean(A) − mean(B). Resample seeds with replacement B times
  and compute Δ_b for each resample. p-value = fraction of resamples where
  Δ_b * sign(Δ_observed) ≤ 0 (one-sided null: B is at least as good as A).

This avoids the assumption of normal error distribution that a paired
t-test makes and is the standard ML-paper choice for small-N seed sweeps.

Usage:

    # Compare every pair of experiments via summarize_seeds.py groupings:
    python tools/paired_bootstrap.py --csv results_final.csv \\
        --output results/significance.md

    # Compare just two specific experiments:
    python tools/paired_bootstrap.py --csv results_final.csv \\
        --a bdsl_block_gcn_shubert_bdsl_asl \\
        --b bdsl_block_gcn_shubert_bdsl_only \\
        --n-resamples 10000
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _per_run_top1(df):
    """Reuse summarize_seeds.py's per-run aggregation; lazy import to avoid
    coupling at module-import time."""
    from tools.summarize_seeds import _metrics_per_run, _base_name, _seed_id

    best = _metrics_per_run(df)
    best["_base"] = best["Experiment"].map(_base_name)
    best["_seed"] = best["Experiment"].map(_seed_id)
    by_group = defaultdict(dict)   # group -> {seed_id: top1}
    for _, row in best.iterrows():
        if row["_base"] is None or row["_seed"] is None:
            continue
        by_group[row["_base"]][int(row["_seed"])] = float(row["Top1_Acc"])
    return by_group


def paired_bootstrap(a_per_seed, b_per_seed, n_resamples=10000, seed=1234,
                     one_sided=True):
    """Compare two groups paired by seed ID.

    Returns: {delta, p_value, ci95, n_paired_seeds}
    """
    # Pair on shared seed IDs only.
    shared = sorted(set(a_per_seed) & set(b_per_seed))
    if len(shared) < 2:
        return {"delta": float("nan"), "p_value": float("nan"),
                "ci95": (float("nan"), float("nan")),
                "n_paired_seeds": len(shared)}
    diffs = np.array([a_per_seed[s] - b_per_seed[s] for s in shared], dtype=np.float64)
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)
    samples = rng.choice(diffs, size=(n_resamples, len(diffs)), replace=True).mean(axis=1)
    if one_sided:
        if observed >= 0:
            p_value = float((samples <= 0).mean())
        else:
            p_value = float((samples >= 0).mean())
    else:
        p_value = float((np.abs(samples) >= abs(observed)).mean())
    ci_lo = float(np.percentile(samples, 2.5))
    ci_hi = float(np.percentile(samples, 97.5))
    return {
        "delta": observed,
        "p_value": p_value,
        "ci95": (ci_lo, ci_hi),
        "n_paired_seeds": len(shared),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results_final.csv")
    ap.add_argument("--output", default=None,
                    help="optional markdown report")
    ap.add_argument("--a", default=None, help="experiment-A base name")
    ap.add_argument("--b", default=None, help="experiment-B base name")
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--one-sided", action="store_true", default=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    by_group = _per_run_top1(df)

    if args.a and args.b:
        if args.a not in by_group:
            raise SystemExit(f"experiment-A {args.a!r} not in CSV "
                             f"(known: {sorted(by_group)[:10]} ...)")
        if args.b not in by_group:
            raise SystemExit(f"experiment-B {args.b!r} not in CSV")
        res = paired_bootstrap(by_group[args.a], by_group[args.b],
                               n_resamples=args.n_resamples,
                               one_sided=args.one_sided)
        print(f"\n=== Paired bootstrap: {args.a} vs {args.b} ===")
        print(f"  Δ (A - B) Top-1: {res['delta']*100:+.3f} pp")
        print(f"  paired seeds:    {res['n_paired_seeds']}")
        print(f"  95% CI on Δ:     [{res['ci95'][0]*100:+.3f}, "
              f"{res['ci95'][1]*100:+.3f}] pp")
        print(f"  p-value:         {res['p_value']:.4f}")
        if res["p_value"] < 0.05:
            print(f"  Significant at α=0.05 ({'A > B' if res['delta'] > 0 else 'B > A'})")
        return

    # All pairs.
    bases = sorted(by_group)
    if len(bases) < 2:
        raise SystemExit("need at least 2 experiment groups in the CSV")

    rows = []
    for i, a in enumerate(bases):
        for b in bases[i + 1:]:
            res = paired_bootstrap(by_group[a], by_group[b],
                                   n_resamples=args.n_resamples,
                                   one_sided=args.one_sided)
            rows.append({"a": a, "b": b, **res})

    print(f"\n=== Pairwise bootstrap (one-sided, n={args.n_resamples}) ===")
    print(f"{'A':<40} {'B':<40} {'Δ (pp)':>8} {'p':>8}")
    for r in rows:
        d = r["delta"] * 100 if not np.isnan(r["delta"]) else float("nan")
        print(f"{r['a']:<40} {r['b']:<40} {d:>+8.3f} {r['p_value']:>8.4f}")

    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write("# Paired-bootstrap significance pairs\n\n")
            f.write(f"Method: one-sided paired bootstrap on per-seed Top-1 diffs, "
                    f"{args.n_resamples} resamples.\n\n")
            f.write("| A | B | Δ Top-1 (pp) | 95% CI on Δ | p | n_seeds |\n")
            f.write("|---|---|---:|:---:|---:|---:|\n")
            for r in rows:
                d = r["delta"] * 100 if not np.isnan(r["delta"]) else float("nan")
                lo = r["ci95"][0] * 100
                hi = r["ci95"][1] * 100
                p = r["p_value"]
                f.write(f"| {r['a']} | {r['b']} | "
                        f"{d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                        f"{p:.4f} | {r['n_paired_seeds']} |\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
