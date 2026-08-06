"""Aggregate multi-seed runs from results_final.csv into a comparison table.

For each base experiment (rows whose Experiment_name starts with a given
config stem), collapses all `_seed<N>` runs using the per-run best Top-1
result and reports mean, standard deviation, and a bootstrap 95% CI.

Usage:
    python tools/summarize_seeds.py
    python tools/summarize_seeds.py --csv results_final.csv --seeds 0 1 2
    python tools/summarize_seeds.py --markdown          # pretty markdown table
    python tools/summarize_seeds.py --out summary.csv   # also emit CSV
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


# _seed<N> is normally the trailing token, but test-set rows carry it in the
# middle: '<stem>_seed<N>_testset'. Match it anywhere so both collapse.
_SEED_TOKEN = re.compile(r"_seed(\d+)", re.IGNORECASE)
_LOSO_SUFFIX = re.compile(r"_loso_test(\d+)$", re.IGNORECASE)


def _base_name(experiment_name):
    """'bdsl_block_gcn_si_seed1'                          -> 'bdsl_block_gcn_si'.
    'bdsl_block_gcn_si_loso_test02_seed0'                  -> 'bdsl_block_gcn_si'.
    'bdsl_block_gcn_si_loso_test02'                        -> 'bdsl_block_gcn_si'.
    'rgb_s3d_bdsl_si_seed2_testset'                        -> 'rgb_s3d_bdsl_si_testset'.
    Anything unsuffixed is kept as-is.
    """
    if not isinstance(experiment_name, str):
        return None
    # Remove the _seed<N> token wherever it sits. For plain runs it is trailing
    # ('..._seed2' -> '...'); for the per-seed test rows it precedes a trailing
    # marker ('..._seed2_testset' -> '..._testset'), so seeds collapse into one
    # group per split.
    name = _SEED_TOKEN.sub("", experiment_name, count=1)
    # Then strip _loso_test<XX> if present — so LOSO runs collapse to the
    # base config across both seed and held-out-signer axes.
    m = _LOSO_SUFFIX.search(name)
    if m:
        name = name[: m.start()]
    return name


def _seed_id(experiment_name):
    m = _SEED_TOKEN.search(experiment_name or "")
    return int(m.group(1)) if m else None


def _loso_test_signer(experiment_name):
    """Extract the LOSO test signer ID from an experiment name, or None."""
    if not isinstance(experiment_name, str):
        return None
    # Strip _seed<N> first so the LOSO regex can match in the middle.
    name = _SEED_TOKEN.sub("", experiment_name, count=1)
    m = _LOSO_SUFFIX.search(name)
    return int(m.group(1)) if m else None


def _bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=1234):
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return (lo, hi)


def _metrics_per_run(df):
    """Collapse all epochs of a given (Experiment, WorkDir) to per-run metrics.

    Returns one row per (Experiment, WorkDir) with:
      Top1_Acc           - best Top-1 over epochs
      Top5_AtTop1        - Top-5 from the SAME epoch that achieved best Top-1
                           (audit fix #6: the historical default that goes in
                            results_final.csv via the `same_epoch_as_logged_top1`
                            Top5_Policy).
      Top5_Best          - best Top-5 INDEPENDENTLY over epochs (a different
                            epoch may win Top-5 vs Top-1). Reviewers will compare
                            these — we report both.
    """
    if "Top1_Acc" not in df.columns:
        raise ValueError("results_final.csv must contain a Top1_Acc column")
    df = df.copy()
    df["Top1_Acc"] = pd.to_numeric(df["Top1_Acc"], errors="coerce")
    df["Top5_Acc"] = pd.to_numeric(df.get("Top5_Acc", np.nan), errors="coerce")
    df = df.dropna(subset=["Top1_Acc"])

    grouped = df.groupby(["Experiment", "WorkDir"])
    # Same-epoch-as-best-Top-1 row (legacy policy).
    idx = grouped["Top1_Acc"].idxmax()
    best = df.loc[idx].reset_index(drop=True)
    best = best.rename(columns={"Top5_Acc": "Top5_AtTop1"})
    # Independent best Top-5 across all epochs of the same run.
    best_top5 = grouped["Top5_Acc"].max().rename("Top5_Best").reset_index()
    best = best.merge(best_top5, on=["Experiment", "WorkDir"], how="left")
    return best


# Backwards-compat shim — older callers may import _best_per_run.
def _best_per_run(df):
    return _metrics_per_run(df).rename(columns={"Top5_AtTop1": "Top5_Acc"})


def summarize(csv_path, only_seeds=None):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "Experiment" not in df.columns:
        raise ValueError("Experiment column missing from CSV")
    best = _metrics_per_run(df)
    best["_base"] = best["Experiment"].map(_base_name)
    best["_seed"] = best["Experiment"].map(_seed_id)

    groups = defaultdict(list)
    for _, row in best.iterrows():
        base = row["_base"]
        seed = row["_seed"]
        if only_seeds is not None and seed is not None and seed not in only_seeds:
            continue
        groups[base].append({
            "seed": seed,
            "top1": float(row["Top1_Acc"]),
            "top5_at_top1": (float(row["Top5_AtTop1"])
                              if not pd.isna(row.get("Top5_AtTop1")) else None),
            "top5_best": (float(row["Top5_Best"])
                           if not pd.isna(row.get("Top5_Best")) else None),
            "workdir": row.get("WorkDir", ""),
        })

    rows = []
    for base, entries in sorted(groups.items()):
        seeds = [e["seed"] for e in entries if e["seed"] is not None]
        top1 = np.asarray([e["top1"] for e in entries], dtype=np.float64)
        top5_at = np.asarray(
            [e["top5_at_top1"] for e in entries if e["top5_at_top1"] is not None],
            dtype=np.float64,
        )
        top5_best = np.asarray(
            [e["top5_best"] for e in entries if e["top5_best"] is not None],
            dtype=np.float64,
        )
        lo, hi = _bootstrap_ci(top1)
        rows.append({
            "experiment": base,
            "n_runs": len(entries),
            "seeds": sorted(set(seeds)) if seeds else [None],
            "top1_mean": float(top1.mean()) if len(top1) else float("nan"),
            "top1_std": float(top1.std(ddof=1)) if len(top1) > 1 else 0.0,
            "top1_ci95_lo": lo,
            "top1_ci95_hi": hi,
            "top5_at_top1_mean": float(top5_at.mean()) if len(top5_at) else float("nan"),
            "top5_best_mean": float(top5_best.mean()) if len(top5_best) else float("nan"),
        })
    return rows


def _print_markdown(rows):
    # Audit fix #6: surface both Top-5 policies so reviewers don't pick
    # whichever is worse for us.
    header = (
        "| Experiment | N | Top-1 mean +/- std | Top-1 95% CI | "
        "Top-5 @ best Top-1 | Top-5 best (indep) |"
    )
    sep = "|---|---:|---:|:---:|---:|---:|"
    print(header)
    print(sep)
    for r in rows:
        mean = r["top1_mean"] * 100
        std = r["top1_std"] * 100
        lo = r["top1_ci95_lo"] * 100
        hi = r["top1_ci95_hi"] * 100
        t5_at = r["top5_at_top1_mean"] * 100
        t5_best = r["top5_best_mean"] * 100
        ci = f"[{lo:.2f}, {hi:.2f}]" if not np.isnan(lo) else "-"
        print(
            f"| {r['experiment']} | {r['n_runs']} | "
            f"{mean:.2f} +/- {std:.2f} | {ci} | "
            f"{t5_at:.2f} | {t5_best:.2f} |"
        )


def _print_plain(rows):
    for r in rows:
        mean = r["top1_mean"] * 100
        std = r["top1_std"] * 100
        lo = r["top1_ci95_lo"] * 100
        hi = r["top1_ci95_hi"] * 100
        ci = f"[{lo:6.2f}, {hi:6.2f}]" if not np.isnan(lo) else "   --"
        t5_at = r["top5_at_top1_mean"] * 100
        t5_best = r["top5_best_mean"] * 100
        print(
            f"{r['experiment']:<38} n={r['n_runs']}  "
            f"Top1={mean:6.2f} +/- {std:4.2f}  95%CI={ci}  "
            f"Top5@T1={t5_at:6.2f}  Top5best={t5_best:6.2f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results_final.csv")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="if set, restrict to rows with these seed ids")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--out", default=None, help="optional output CSV path")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"{args.csv} not found", file=sys.stderr)
        sys.exit(1)

    rows = summarize(args.csv, only_seeds=args.seeds)
    if args.markdown:
        _print_markdown(rows)
    else:
        _print_plain(rows)

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
