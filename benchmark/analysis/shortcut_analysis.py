"""Tier-1 shortcut analyses for the sister paper — per-class SD-SI delta + calibration.

Two analyses that harden the identity-shortcut story on BdSL47 (the only source
with real user IDs). Both reuse the SAME encoder-load + fresh-head-refit-on-frozen
-features pattern as `plot_confusion.py` and `eval_cross_dataset.py`: `train_baseline.py`
checkpoints ONLY the backbone (`model.backbone.state_dict()`), never the per-source
classification heads, so we load the LoRA-adapted backbone, fit a fresh classifier on
the target source's TRAIN features, and predict on its eval split. The confusion and
calibration *structure* is driven by the encoder — this is faithful to the trained
representation, not a literal replay of the trained head.

(1) PER-CLASS SD-SI DELTA (T_shortcut_perclass.md + SF3_perclass_delta.png)
    For each class c, delta[c] = Top1_SD[c] - Top1_SI[c], sorted descending.
    Thesis: the signer-identity shortcut concentrates on visually-similar handshape
    pairs — the classes with the largest SD>>SI gap are exactly the ones the SD model
    "solves" by memorizing the signer rather than the handshape.

(2) CALIBRATION UNDER SIGNER SHIFT (T_calibration.md + SF4_reliability.png)
    15-bin Expected Calibration Error (ECE) on the SD and SI test predictions per
    source. SI models — evaluated on unseen signers — are typically MORE overconfident
    (higher ECE) than SD models evaluated on seen signers.

SPLIT MATCHING (critical for comparability with the T3 table): mirrors
`train_baseline._train_one_seed` exactly. The SI model is evaluated on the VAL
partition of the user-disjoint split (val user 4); the SD model on the VAL partition
of the force_random split (seed 0, val_frac=test_frac=0.10).

Usage (bdsl_graph env):
    HF_HUB_OFFLINE=1 python -m benchmark.analysis.shortcut_analysis \
        --si-dir work_dir/bhc_bdsl47_si --sd-dir work_dir/bhc_bdsl47_sd \
        --seed 0 --epoch 0 --sources bdsl47_digits bdsl47_letters
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# Force offline so timm/HF never phones home for the DINOv2 weights on the cluster.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from banglahandshape.class_alignment import discover_source, SourceSpec
from banglahandshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, split_user_disjoint, split_random,
)
from banglahandshape.dinov2_lora import build_dinov2_lora
from benchmark.baselines.train_baseline import (
    _provided_test_dir, _build_transforms,
)
from benchmark.baselines.train_probe_cached import _extract

# Source roots (mirrors DEFAULT_ROOTS in plot_confusion.py / class_alignment).
DEFAULT_ROOTS = {
    "bdsl47_digits":  "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}

# Fixed BdSL47 SI split (matches bdsl47_si.yaml: val user 4, test user 5). We
# evaluate on the VAL partition, exactly like train_baseline.
SI_VAL_USERS = {4}
SI_TEST_USERS = {5}
# Fixed BdSL47 SD split (matches bdsl47_sd.yaml: force_random, seed 0, 80/10/10).
SD_SEED = 0
SD_VAL_FRAC = 0.10
SD_TEST_FRAC = 0.10

N_BINS = 15  # 15-bin equal-width ECE, per the spec.


def _latest_epoch(encoder_dir, seed):
    """Return the highest available epoch for `encoder_seed<seed>_epoch<E>.pt`,
    or None if the dir has no such checkpoint."""
    if not os.path.isdir(encoder_dir):
        return None
    pat = re.compile(rf"^encoder_seed{seed}_epoch(\d+)\.pt$")
    epochs = []
    for fn in os.listdir(encoder_dir):
        m = pat.match(fn)
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


def _si_split(spec):
    """SI (user-disjoint) split for BdSL47; return (train, eval=val-partition)."""
    items = enumerate_source(spec)
    tr, va, _te = split_user_disjoint(items, SI_VAL_USERS, SI_TEST_USERS)
    return tr, va


def _sd_split(spec):
    """SD (force_random) split for BdSL47; return (train, eval=val-partition)."""
    items = enumerate_source(spec)
    tr, va, _te = split_random(items, seed=SD_SEED,
                               val_frac=SD_VAL_FRAC, test_frac=SD_TEST_FRAC)
    return tr, va


def _loader(spec, entries, transform, bs, nw):
    ds = HandshapeDataset([(spec, entries)], transform=transform)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                      pin_memory=torch.cuda.is_available())


def _load_encoder(encoder_dir, seed, epoch, num_classes, timm_name,
                  lora_rank, lora_alpha, lora_dropout, lora_targets, device):
    """Build a LoRA DINOv2 (single-source head) and load the backbone-only ckpt."""
    model = build_dinov2_lora(
        num_classes_per_source=[num_classes],
        timm_name=timm_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
        pretrained=True,
        full_finetune=False,
    )
    ckpt = os.path.join(encoder_dir, f"encoder_seed{seed}_epoch{epoch}.pt")
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    print(f"    loaded {os.path.basename(ckpt)} from {encoder_dir} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    return model.to(device)


def _refit_head_predict(model, spec, train_entries, eval_entries, transform,
                        bs, nw, device):
    """Fit a fresh logistic-regression head on FROZEN train features, then predict
    on eval features. Returns (y_true, y_pred, probs) where probs is (N, C) softmax
    over the FULL [0, num_classes) label space (sklearn-observed classes are mapped
    back; unobserved classes get probability 0)."""
    from sklearn.linear_model import LogisticRegression

    Xtr, ytr = _extract(model, _loader(spec, train_entries, transform, bs, nw), device)
    Xva, yva = _extract(model, _loader(spec, eval_entries, transform, bs, nw), device)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xva)  # already in the original label space (clf.classes_)

    # Map sklearn's per-column probabilities back into the full [0, C) label space.
    proba = clf.predict_proba(Xva)               # (N, len(clf.classes_))
    full = np.zeros((proba.shape[0], spec.num_classes), dtype=np.float64)
    for col, cls in enumerate(clf.classes_):
        full[:, int(cls)] = proba[:, col]
    return yva.astype(np.int64), preds.astype(np.int64), full


def _per_class_top1(y_true, y_pred, num_classes):
    """Per-class Top-1 accuracy; NaN for classes absent from the eval set."""
    acc = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        mask = (y_true == c)
        if mask.any():
            acc[c] = float((y_pred[mask] == c).mean())
    return acc


def _expected_calibration_error(probs, y_true, n_bins=N_BINS):
    """Standard equal-width ECE. conf = max softmax prob; ECE = sum_b (n_b/N)*
    |acc_b - conf_b|. Returns (ece, per-bin dict for the reliability diagram)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    n = len(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = {"lo": [], "hi": [], "count": [], "acc": [], "conf": []}
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # Last bin is closed on the right so conf == 1.0 lands somewhere.
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        cnt = int(mask.sum())
        bins["lo"].append(lo); bins["hi"].append(hi); bins["count"].append(cnt)
        if cnt > 0:
            acc_b = float(correct[mask].mean())
            conf_b = float(conf[mask].mean())
            ece += (cnt / n) * abs(acc_b - conf_b)
            bins["acc"].append(acc_b); bins["conf"].append(conf_b)
        else:
            bins["acc"].append(np.nan); bins["conf"].append(np.nan)
    return float(ece), bins


def _idx_to_name(spec):
    """Invert class_to_idx -> {idx: class_name}."""
    return {i: c for c, i in spec.class_to_idx.items()}


# --------------------------------------------------------------------------- #
#  Analysis (1): per-class SD-SI delta
# --------------------------------------------------------------------------- #
def _run_perclass(rows, out_md, out_png):
    """rows: list of (source, class_name, acc_sd, acc_si, delta). Write markdown
    table (sorted desc by delta) + top-15 bar plot."""
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)

    ordered = sorted(rows, key=lambda r: (-(r[4] if not np.isnan(r[4]) else -1e9)))
    lines = []
    lines.append("# T_shortcut_perclass — per-class SD-SI Top-1 delta (BdSL47)\n")
    lines.append("Per-class Top-1 for the signer-DEPENDENT (SD) and "
                 "signer-INDEPENDENT (SI) BdSL47 encoders, and their difference\n"
                 "`delta = acc_SD - acc_SI` (sorted descending). A large positive "
                 "delta means the class is much easier when the same signer's frames\n"
                 "appear in train AND test — i.e. the identity shortcut is doing the "
                 "work. The shortcut concentrates on visually-similar handshape pairs.\n")
    lines.append("*Heads are re-fit on FROZEN backbone features (only the backbone is "
                 "checkpointed) — faithful to the encoder's structure, not a literal "
                 "replay of the trained head.*\n")
    lines.append("| Source | Class | Top1 SD | Top1 SI | delta (SD-SI) |")
    lines.append("|---|---|---:|---:|---:|")
    for src, cname, a_sd, a_si, d in ordered:
        f = lambda v: ("n/a" if (v is None or np.isnan(v)) else f"{v*100:.2f}")
        lines.append(f"| {src} | {cname} | {f(a_sd)} | {f(a_si)} | {f(d)} |")
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {out_md}  ({len(ordered)} classes)", flush=True)

    # Bar plot: top-15 classes by delta.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in ordered if not np.isnan(r[4])]
    top = valid[:15]
    if not top:
        print("  [warn] no finite deltas — skipping SF3 bar plot", flush=True)
        return
    labels = [f"{src}:{cname}" for src, cname, _a, _b, _d in top]
    deltas = [d * 100 for _s, _c, _a, _b, d in top]
    fig, ax = plt.subplots(figsize=(max(6, len(top) * 0.55), 5))
    colors = ["#c0392b" if d >= 0 else "#2980b9" for d in deltas]
    ax.bar(range(len(top)), deltas, color=colors)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Top-1 delta  (SD - SI)  [pp]")
    ax.set_title("SF3 — top-15 classes by signer-identity shortcut (BdSL47)")
    ax.axhline(0.0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


# --------------------------------------------------------------------------- #
#  Analysis (2): calibration under signer shift
# --------------------------------------------------------------------------- #
def _run_calibration(cal_rows, out_md, out_png):
    """cal_rows: list of (source, ece_sd, ece_si, bins_sd, bins_si). Write markdown
    + optional reliability diagram."""
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)

    lines = []
    lines.append("# T_calibration — Expected Calibration Error under signer shift "
                 "(BdSL47)\n")
    lines.append(f"{N_BINS}-bin equal-width Expected Calibration Error (ECE) of the "
                 "signer-DEPENDENT (SD) and signer-INDEPENDENT (SI) BdSL47 encoders,\n"
                 "per source. `conf = max softmax prob`; "
                 "`ECE = sum_b (n_b/N)*|acc_b - conf_b|`. Lower is better-calibrated.\n")
    lines.append("*Heads are re-fit on FROZEN backbone features (only the backbone is "
                 "checkpointed).*\n")
    lines.append("| Source | ECE SD | ECE SI | ECE SI - ECE SD |")
    lines.append("|---|---:|---:|---:|")
    worse_si = 0
    n_src = 0
    for src, ece_sd, ece_si, _bsd, _bsi in cal_rows:
        lines.append(f"| {src} | {ece_sd*100:.2f} | {ece_si*100:.2f} | "
                     f"{(ece_si - ece_sd)*100:+.2f} |")
        n_src += 1
        if ece_si > ece_sd:
            worse_si += 1
    if n_src:
        mean_sd = float(np.mean([r[1] for r in cal_rows]))
        mean_si = float(np.mean([r[2] for r in cal_rows]))
        lines.append(f"| **mean** | **{mean_sd*100:.2f}** | **{mean_si*100:.2f}** | "
                     f"**{(mean_si - mean_sd)*100:+.2f}** |")
    lines.append("")
    if n_src and worse_si == n_src:
        takeaway = ("**Takeaway:** on every source the SI model has HIGHER ECE than "
                    "the SD model — the signer-independent model is more overconfident "
                    "under signer shift (evaluated on unseen signers), as predicted.")
    elif n_src and worse_si > 0:
        takeaway = (f"**Takeaway:** the SI model has higher ECE (is more overconfident "
                    f"under signer shift) on {worse_si}/{n_src} source(s).")
    else:
        takeaway = ("**Takeaway:** the SI model is NOT more overconfident here — ECE "
                    "does not increase under signer shift on these sources.")
    lines.append(takeaway)
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {out_md}", flush=True)

    # Reliability diagram (one subplot column per source; SD vs SI overlaid).
    if not cal_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = len(cal_rows)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False)
    for ax, (src, ece_sd, ece_si, bsd, bsi) in zip(axes[0], cal_rows):
        centers = [(lo + hi) / 2 for lo, hi in zip(bsd["lo"], bsd["hi"])]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
        ax.plot(centers, bsd["acc"], "o-", color="#2980b9",
                label=f"SD (ECE={ece_sd*100:.1f})")
        ax.plot(centers, bsi["acc"], "s-", color="#c0392b",
                label=f"SI (ECE={ece_si*100:.1f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_title(src, fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("SF4 — reliability diagrams (BdSL47, SD vs SI)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--si-dir", default="work_dir/bhc_bdsl47_si",
                    help="dir with the SI encoder_seed<N>_epoch<E>.pt checkpoints")
    ap.add_argument("--sd-dir", default="work_dir/bhc_bdsl47_sd",
                    help="dir with the SD encoder_seed<N>_epoch<E>.pt checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=0,
                    help="checkpoint epoch (0 = auto-detect the latest available)")
    ap.add_argument("--sources", nargs="+",
                    default=["bdsl47_digits", "bdsl47_letters"])
    ap.add_argument("--timm-name", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", nargs="+",
                    default=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--perclass-md", default="results/T_shortcut_perclass.md")
    ap.add_argument("--perclass-png", default="results/SF3_perclass_delta.png")
    ap.add_argument("--calibration-md", default="results/T_calibration.md")
    ap.add_argument("--reliability-png", default="results/SF4_reliability.png")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    # Resolve epochs (auto-detect latest if --epoch 0). Bail gracefully (exit 0)
    # if either encoder dir has no checkpoint for this seed — still a valid script.
    si_epoch = args.epoch or _latest_epoch(args.si_dir, args.seed)
    sd_epoch = args.epoch or _latest_epoch(args.sd_dir, args.seed)
    if si_epoch is None:
        print(f"[skip] no SI checkpoint (encoder_seed{args.seed}_epoch*.pt) in "
              f"{args.si_dir!r} — nothing to analyze. Train bdsl47_si.yaml first.")
        return
    if sd_epoch is None:
        print(f"[skip] no SD checkpoint (encoder_seed{args.seed}_epoch*.pt) in "
              f"{args.sd_dir!r} — nothing to analyze. Train bdsl47_sd.yaml first.")
        return
    print(f"SI epoch={si_epoch} ({args.si_dir})   SD epoch={sd_epoch} ({args.sd_dir})",
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = _build_transforms(args.image_size)
    lora_kwargs = dict(timm_name=args.timm_name, lora_rank=args.lora_rank,
                       lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                       lora_targets=args.lora_targets, device=device)

    perclass_rows = []   # (source, class_name, acc_sd, acc_si, delta)
    calibration_rows = []  # (source, ece_sd, ece_si, bins_sd, bins_si)

    for name in args.sources:
        root = DEFAULT_ROOTS.get(name)
        if not root or not os.path.isdir(root):
            print(f"[skip source] {name}: root not found ({root!r})", flush=True)
            continue
        spec = discover_source(name, root)
        idx2name = _idx_to_name(spec)
        print(f"\n=== source {name}: {spec.num_classes} classes  root={root} ===",
              flush=True)

        # --- SI model on the SI (user-disjoint) eval set ---
        si_tr, si_va = _si_split(spec)
        print(f"  [SI] train={len(si_tr)} eval={len(si_va)} (user-disjoint, val user 4)",
              flush=True)
        si_model = _load_encoder(args.si_dir, args.seed, si_epoch, spec.num_classes,
                                 **lora_kwargs)
        yt_si, yp_si, pr_si = _refit_head_predict(
            si_model, spec, si_tr, si_va, transform,
            args.batch_size, args.num_workers, device)
        del si_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        acc_si = _per_class_top1(yt_si, yp_si, spec.num_classes)
        ece_si, bins_si = _expected_calibration_error(pr_si, yt_si)
        print(f"  [SI] overall Top-1={float((yp_si == yt_si).mean())*100:.2f}%  "
              f"ECE={ece_si*100:.2f}", flush=True)

        # --- SD model on the SD (force_random) eval set ---
        sd_tr, sd_va = _sd_split(spec)
        print(f"  [SD] train={len(sd_tr)} eval={len(sd_va)} (force_random seed 0)",
              flush=True)
        sd_model = _load_encoder(args.sd_dir, args.seed, sd_epoch, spec.num_classes,
                                 **lora_kwargs)
        yt_sd, yp_sd, pr_sd = _refit_head_predict(
            sd_model, spec, sd_tr, sd_va, transform,
            args.batch_size, args.num_workers, device)
        del sd_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        acc_sd = _per_class_top1(yt_sd, yp_sd, spec.num_classes)
        ece_sd, bins_sd = _expected_calibration_error(pr_sd, yt_sd)
        print(f"  [SD] overall Top-1={float((yp_sd == yt_sd).mean())*100:.2f}%  "
              f"ECE={ece_sd*100:.2f}", flush=True)

        for c in range(spec.num_classes):
            a_sd, a_si = acc_sd[c], acc_si[c]
            delta = (a_sd - a_si) if (not np.isnan(a_sd) and not np.isnan(a_si)) \
                else np.nan
            perclass_rows.append((name, idx2name.get(c, str(c)), a_sd, a_si, delta))
        calibration_rows.append((name, ece_sd, ece_si, bins_sd, bins_si))

    if not perclass_rows:
        print("[skip] no sources produced results — nothing written.")
        return

    print("\n--- writing outputs ---", flush=True)
    _run_perclass(perclass_rows, args.perclass_md, args.perclass_png)
    _run_calibration(calibration_rows, args.calibration_md, args.reliability_png)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
