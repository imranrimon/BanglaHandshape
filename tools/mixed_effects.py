"""Mixed-effects analysis for the §A factorial (reviewer major-revision).

Consumes the per-(cond, signer, seed) CSV that
``path3_handshape_benchmark/factorial_signer_session.py`` writes and produces the
coefficient / variance-component table that *replaces* the manuscript's Eq. 1
"estimated separately" decomposition with identified numbers:

  beta1 (signer-seen)  -- controlled signer-exposure effect (cond B - cond A)
  beta2 (dup / burst)  -- extra lift from adjacent near-dup frames (cond D - cond B)
  sigma^2_signer       -- between-held-out-signer variance (random intercept)

Two estimators, reported side by side:
  * statsmodels MixedLM:  top1 ~ signer_seen + dup + (1 | signer)
  * bootstrap-over-signers: resample the 10 signers with replacement and take the
    paired within-signer deltas -- the honest CI given only ~10 signers, where the
    MixedLM asymptotic SEs are optimistic.

Condition -> factor coding (nested; cond "C" = signer-unseen+dup is absent by design,
see the factorial module's docstring, so no full 2x2 interaction is fit):
    A: signer_seen=0 dup=0     B: signer_seen=1 dup=0     D: signer_seen=1 dup=1

Usage:
  python tools/mixed_effects.py --csv results/A_factorial_letters.csv \\
      --output results/A_mixed_effects.md
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

_CODE = {"A": (0, 0), "B": (1, 0), "D": (1, 1)}


def _load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            r["signer"] = int(r["signer"]); r["seed"] = int(r["seed"])
            r["top1"] = float(r["top1"])
            rows.append(r)
    return rows


def _per_signer_cond_mean(rows):
    """(signer -> {cond -> mean top1 over seeds}) for one source."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["signer"]][r["cond"]].append(r["top1"])
    return {s: {c: float(np.mean(v)) for c, v in cd.items()} for s, cd in acc.items()}


def _bootstrap_deltas(psc, n_boot=5000, seed=0):
    """Bootstrap over signers: beta1 = mean_s(B-A), beta2 = mean_s(D-B)."""
    rng = np.random.default_rng(seed)
    signers = [s for s, cd in psc.items() if {"A", "B", "D"}.issubset(cd)]
    if len(signers) < 2:
        return None
    b1 = np.array([psc[s]["B"] - psc[s]["A"] for s in signers])
    b2 = np.array([psc[s]["D"] - psc[s]["B"] for s in signers])
    n = len(signers)
    boot1, boot2 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot1.append(b1[idx].mean()); boot2.append(b2[idx].mean())
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return dict(n_signers=n,
                beta1=float(b1.mean()), beta1_ci=ci(boot1),
                beta2=float(b2.mean()), beta2_ci=ci(boot2))


def _mixedlm(rows):
    """statsmodels MixedLM on the per-(cond,signer,seed) points."""
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}
    data = pd.DataFrame([{"top1": r["top1"], "signer": r["signer"],
                          "signer_seen": _CODE[r["cond"]][0], "dup": _CODE[r["cond"]][1]}
                         for r in rows if r["cond"] in _CODE])
    try:
        m = smf.mixedlm("top1 ~ signer_seen + dup", data, groups=data["signer"]).fit(reml=False)
        ci = m.conf_int()
        return dict(beta1=float(m.params["signer_seen"]),
                    beta1_ci=(float(ci.loc["signer_seen", 0]), float(ci.loc["signer_seen", 1])),
                    beta2=float(m.params["dup"]),
                    beta2_ci=(float(ci.loc["dup", 0]), float(ci.loc["dup", 1])),
                    sigma2_signer=float(m.cov_re.iloc[0, 0]))
    except Exception as e:  # singular fits happen with tiny n
        return {"error": str(e)}


def run(args):
    rows = _load(args.csv)
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)

    lines = ["# §A factorial: identified signer / session-proxy effects\n",
             "Percentage-point effects on Top-1 (frozen-feature head). "
             "beta1 = signer exposure (B-A); beta2 = near-dup/burst lift (D-B); "
             "sigma^2_signer = between-held-out-signer variance.\n",
             "| source | #signers | mean A | mean B | mean D | "
             "beta1 (MixedLM) | beta1 (boot 95% CI) | beta2 (MixedLM) | "
             "beta2 (boot 95% CI) | sigma^2_signer |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    for src, srows in by_src.items():
        psc = _per_signer_cond_mean(srows)
        means = {c: float(np.mean([psc[s][c] for s in psc if c in psc[s]]))
                 for c in ("A", "B", "D")}
        boot = _bootstrap_deltas(psc, seed=args.seed)
        mlm = _mixedlm(srows)
        b1m = f"{mlm['beta1']:+.2f}" if "beta1" in mlm else "n/a"
        b2m = f"{mlm['beta2']:+.2f}" if "beta2" in mlm else "n/a"
        s2 = f"{mlm['sigma2_signer']:.2f}" if "sigma2_signer" in mlm else "n/a"
        if boot:
            b1c = f"[{boot['beta1_ci'][0]:+.2f}, {boot['beta1_ci'][1]:+.2f}]"
            b2c = f"[{boot['beta2_ci'][0]:+.2f}, {boot['beta2_ci'][1]:+.2f}]"
            ns = boot["n_signers"]
        else:
            b1c = b2c = "n/a"; ns = len(psc)
        lines.append(f"| {src} | {ns} | {means['A']:.2f} | {means['B']:.2f} | "
                     f"{means['D']:.2f} | {b1m} | {b1c} | {b2m} | {b2c} | {s2} |")
        if "error" in mlm:
            lines.append(f"<!-- MixedLM({src}) failed: {mlm['error']} -->")

    out = "\n".join(lines) + "\n"
    print(out)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            fh.write(out)
        print(f"[written] {args.output}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="factorial_signer_session.py output CSV")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument("--output", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
