"""E15 — turn the expert handshape feature-coding sheet into the paper result.

Pipeline (all steps are optional/guarded so partial inputs still produce output):

  1. Read the filled `handshape_feature_coding_sheet.csv` for one source and build a
     feature-DERIVED pairwise handshape-similarity matrix S in [0, 1] (weighted
     articulatory-feature agreement — no O(N^2) human rating needed).
  2. If an SI confusion matrix is given, correlate S with the SI **confusability**
     C_ij = (P(pred=j|true=i) + P(pred=i|true=j)) / 2  over all class pairs.
     Hypothesis (E15): rho > 0 — signer-shift errors fall along handshape similarity.
  3. If a per-class Delta_SI gap CSV is given, correlate each class's mean similarity
     to its top-k neighbours (a "crowdedness" score) with its SD->SI accuracy drop.
  4. Emit a round-2 `pairwise_validation_sheet.csv`: a stratified sample of pairs for
     the expert to also rate holistically (validates that S tracks human judgment).
  5. Write `<out>/E15_correlation.md` and a scatter PNG.

Class ordering: the confusion matrix is assumed to be in `discover_source` order —
i.e. class folders sorted LEXICOGRAPHICALLY ("Sign 0","Sign 1","Sign 10",...), which
is exactly what `class_alignment._list_class_folders*` and the eval code produce. The
feature sheet is aligned to that same order, so rows need not be pre-sorted.

Usage:
    python -m tools.handshape_similarity_analysis \
        --source bdsl47_letters \
        --feature-csv docs/E15_handshape_annotation/handshape_feature_coding_sheet.csv \
        --confusion-npz work_dir/bhc_lora/confusion_bdsl47_letters_seed0.npz \
        --per-class-gap results/T3_per_class_gap_bdsl47_letters.csv \
        --out results/E15
"""

from __future__ import annotations

import argparse
import csv
import os
from itertools import combinations

import numpy as np

# Weight each articulatory dimension by how much it defines the gross handshape.
# Reported in the paper so the derivation is transparent; sums to 1.0.
FEATURE_WEIGHTS = {
    "selected_fingers": 0.35,
    "aperture":         0.25,
    "thumb":            0.15,
    "flexion":          0.12,
    "spread":           0.08,
    "contact":          0.05,
}
# Ordinal dimensions get partial credit (1 - normalized rank distance); the rest are
# exact-match (1.0 / 0.0).
ORDINAL = {
    "aperture": ["open", "flat-o", "tight-o", "closed"],
    "flexion":  ["straight", "bent", "curved", "hooked", "fist"],
}
EXACT = ["thumb", "spread", "contact"]


def _norm(s):
    return (s or "").strip().lower()


def _finger_set(v):
    v = _norm(v)
    if v in ("", "none", "na"):
        return frozenset()
    return frozenset(tok for tok in v.replace(",", " ").split() if tok)


def _dim_sim(dim, a, b):
    """Per-dimension similarity in [0,1], or None if either side is uncoded."""
    if dim == "selected_fingers":
        sa, sb = _finger_set(a), _finger_set(b)
        if not a.strip() or not b.strip():
            return None
        if not sa and not sb:
            return 1.0
        u = sa | sb
        return (len(sa & sb) / len(u)) if u else 1.0
    va, vb = _norm(a), _norm(b)
    if va in ("", "na") or vb in ("", "na"):
        return None
    if dim in ORDINAL:
        order = ORDINAL[dim]
        if va in order and vb in order:
            span = max(1, len(order) - 1)
            return 1.0 - abs(order.index(va) - order.index(vb)) / span
        return 1.0 if va == vb else 0.0
    return 1.0 if va == vb else 0.0


def pairwise_similarity(rows):
    """rows: list of dicts (feature sheet). Returns (labels, class_folders, S)."""
    folders = sorted({r["class_folder"] for r in rows})            # discover_source order
    by_folder = {r["class_folder"]: r for r in rows}
    n = len(folders)
    S = np.eye(n, dtype=np.float64)
    beng = [by_folder[f].get("confirmed_bengali") or by_folder[f].get("proposed_bengali") or f
            for f in folders]
    for i, j in combinations(range(n), 2):
        ri, rj = by_folder[folders[i]], by_folder[folders[j]]
        num = den = 0.0
        for dim, w in FEATURE_WEIGHTS.items():
            s = _dim_sim(dim, ri.get(dim, ""), rj.get(dim, ""))
            if s is not None:
                num += w * s
                den += w
        S[i, j] = S[j, i] = (num / den) if den > 0 else np.nan
    return beng, folders, S


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 4:
        return float("nan"), 0
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(x, y).correlation)
    except Exception:                                              # rank-corr fallback
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        rx = (rx - rx.mean()); ry = (ry - ry.mean())
        rho = float((rx @ ry) / (np.sqrt((rx @ rx) * (ry @ ry)) + 1e-12))
    return rho, int(x.size)


def _load_confusion(path):
    if path is None or not os.path.exists(path):
        return None
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        for k in ("confusion", "cm", "matrix", "arr_0"):
            if k in z:
                return np.asarray(z[k], float)
        return np.asarray(z[list(z.keys())[0]], float)
    if path.endswith(".npy"):
        return np.asarray(np.load(path), float)
    return np.loadtxt(path, delimiter=",")


def _confusability(cm):
    """Row-normalize to P(pred|true), then symmetric off-diagonal confusability."""
    cm = np.asarray(cm, float)
    P = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    C = 0.5 * (P + P.T)
    np.fill_diagonal(C, 0.0)
    return C


def _load_gap_csv(path):
    """Flexible: return {class_folder: gap}. Accepts 'gap' or (top1_sd - top1_si)."""
    if path is None or not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = r.get("class_folder") or r.get("class") or r.get("folder")
            if not key:
                continue
            if r.get("gap") not in (None, ""):
                out[key] = float(r["gap"])
            elif r.get("top1_sd") and r.get("top1_si"):
                out[key] = float(r["top1_sd"]) - float(r["top1_si"])
    return out


def _stratified_pairs(folders, beng, S, k=30, bins=5, source=""):
    rows, n = [], len(folders)
    pairs = [(i, j) for i, j in combinations(range(n), 2) if np.isfinite(S[i, j])]
    if not pairs:
        return rows
    vals = np.array([S[i, j] for i, j in pairs])
    edges = np.linspace(vals.min(), vals.max() + 1e-9, bins + 1)
    per = max(1, k // bins)
    for b in range(bins):
        idx = [p for p, v in zip(pairs, vals) if edges[b] <= v < edges[b + 1]]
        idx.sort(key=lambda ij: S[ij[0], ij[1]])
        for i, j in idx[:: max(1, len(idx) // per)][:per]:
            rows.append([source, folders[i], beng[i], folders[j], beng[j],
                         f"{S[i, j]:.4f}", "", ""])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="bdsl47_letters")
    ap.add_argument("--feature-csv", required=True)
    ap.add_argument("--confusion-npz", default=None,
                    help="SI confusion matrix (.npz/.npy/.csv) in discover_source order")
    ap.add_argument("--per-class-gap", default=None,
                    help="CSV with per-class Delta_SI (cols: class_folder, gap)")
    ap.add_argument("--out", default="results/E15")
    ap.add_argument("--neighbours", type=int, default=3,
                    help="k for the per-class crowdedness score")
    args = ap.parse_args()

    with open(args.feature_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("source") == args.source]
    coded = [r for r in rows if any(_norm(r.get(d, "")) not in ("", "na")
                                    for d in FEATURE_WEIGHTS)]
    os.makedirs(args.out, exist_ok=True)

    if len(coded) < 2:
        md = (f"# E15 — handshape-similarity analysis ({args.source})\n\n"
              f"**Awaiting annotation.** Only {len(coded)}/{len(rows)} rows in "
              f"`{args.feature_csv}` are coded. Fill the feature columns "
              f"(see `docs/E15_handshape_annotation/README.md`) and re-run.\n")
        open(os.path.join(args.out, "E15_correlation.md"), "w", encoding="utf-8").write(md)
        print(md)
        return

    beng, folders, S = pairwise_similarity(coded)
    n = len(folders)
    print(f"[E15] {n} coded handshapes for {args.source}; "
          f"{np.isfinite(S[np.triu_indices(n, 1)]).sum()} defined pairs")

    lines = [f"# E15 — handshape-similarity analysis ({args.source})", "",
             f"{n} expert-coded handshapes. Similarity is feature-derived "
             f"(weights: {FEATURE_WEIGHTS}).", ""]

    # (2) similarity vs SI confusability
    cm = _load_confusion(args.confusion_npz)
    if cm is not None and cm.shape[0] == n:
        C = _confusability(cm)
        iu = np.triu_indices(n, 1)
        rho, npair = _spearman(S[iu], C[iu])
        # magnitude-aware interpretation (a near-zero rho does NOT support E15;
        # with a low-confidence DRAFT feature coding it most likely reflects coding
        # coarseness rather than the linguistic hypothesis).
        a = abs(rho)
        if a < 0.10:
            verdict = (f"|rho|={a:.3f} is negligible: this does NOT support E15. With "
                       "the current low-confidence machine-draft feature coding, a "
                       "near-zero rho most likely reflects coding coarseness (identical "
                       "codings collapse similarity), so a verified expert coding is "
                       "needed before drawing any conclusion.")
        elif rho <= -0.10:
            verdict = (f"rho={rho:+.3f} is NEGATIVE: this contradicts E15 as coded "
                       "(errors do not fall on articulatorily-similar handshapes).")
        elif rho < 0.30:
            verdict = (f"rho={rho:+.3f} is a WEAK positive: mild support for E15, but "
                       "not conclusive; verify with expert coding.")
        else:
            verdict = (f"rho={rho:+.3f} is a MODERATE-or-stronger positive: supports "
                       "E15 — signer-shift errors concentrate on articulatorily-similar "
                       "handshapes.")
        lines += ["## Similarity vs SI confusability", "",
                  f"Spearman rho(S, confusability) = **{rho:+.3f}** over {npair} pairs.",
                  "", verdict, ""]
        print(f"[E15] rho(similarity, confusability) = {rho:+.3f} (n={npair}) -> {verdict.split(':')[0]}")
    elif cm is not None:
        lines += [f"> [warn] confusion matrix is {cm.shape}, expected {n}x{n} — "
                  "check it is in discover_source (lexicographic-folder) order.", ""]

    # (3) per-class crowdedness vs Delta_SI
    gap = _load_gap_csv(args.per_class_gap)
    if gap:
        k = min(args.neighbours, n - 1)
        crowd, gaps, names = [], [], []
        for i, fol in enumerate(folders):
            if fol not in gap:
                continue
            nb = np.sort(S[i][np.isfinite(S[i])])[::-1]
            nb = nb[1:k + 1] if nb.size > 1 else nb            # drop self
            crowd.append(float(nb.mean()) if nb.size else np.nan)
            gaps.append(gap[fol]); names.append(fol)
        rho2, npt = _spearman(crowd, gaps)
        lines += ["## Handshape crowdedness vs per-class Delta_SI", "",
                  f"Spearman rho(mean top-{k} similarity, Delta_SI) = **{rho2:+.3f}** "
                  f"over {npt} classes.", "",
                  "Positive => handshapes with more similar neighbours drop more "
                  "under a signer shift.", ""]
        print(f"[E15] rho(crowdedness, Delta_SI) = {rho2:+.3f} (n={npt})")

    open(os.path.join(args.out, "E15_correlation.md"), "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")

    # (1b) dump the similarity matrix for reuse / the SF figure
    np.savez(os.path.join(args.out, "similarity_matrix.npz"),
             S=S, folders=np.array(folders), bengali=np.array(beng))

    # (4) round-2 validation sheet
    vpath = os.path.join(args.out, "pairwise_validation_sheet.csv")
    with open(vpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "class_a", "bengali_a", "class_b", "bengali_b",
                    "feature_derived_sim", "expert_holistic_0_4", "notes"])
        w.writerows(_stratified_pairs(folders, beng, S, source=args.source))
    print(f"[E15] wrote {vpath} (round-2 holistic-rating sheet)")

    # scatter (guarded)
    if cm is not None and cm.shape[0] == n:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            C = _confusability(cm); iu = np.triu_indices(n, 1)
            plt.figure(figsize=(4.2, 4.0))
            plt.scatter(S[iu], C[iu], s=10, alpha=0.5)
            plt.xlabel("expert handshape similarity"); plt.ylabel("SI confusability")
            plt.title(f"{args.source}: similarity vs SI confusability")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "E15_scatter.png"), dpi=150)
            print(f"[E15] wrote {os.path.join(args.out, 'E15_scatter.png')}")
        except Exception as e:
            print(f"[E15] scatter skipped: {e}")

    print("=== E15 analysis DONE ===")


if __name__ == "__main__":
    main()
