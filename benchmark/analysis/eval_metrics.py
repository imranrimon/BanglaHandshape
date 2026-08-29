"""Phase-1 journal metrics — imbalance-aware + calibration + temperature scaling.

Beyond Top-1, published Bangla handshape papers rarely report the metrics that
matter under class imbalance and distribution (signer) shift. This computes, for
the BdSL47 signer-DEPENDENT (SD) and signer-INDEPENDENT (SI) encoders:

  * IMBALANCE-AWARE:  macro-F1, balanced accuracy (mean per-class recall), Top-1,
    and the full per-class recall vector.
  * CALIBRATION:      15-bin Expected Calibration Error (ECE), Maximum Calibration
    Error (MCE), one-vs-rest mean Brier score, and NLL.
  * TEMPERATURE SCALING recalibration: a single scalar T > 0 that rescales the
    logits (`logits / T`) is fit by minimizing NLL on the eval logits (LBFGS over
    [0.5, 5] with a grid fallback); ECE and NLL are reported before and after.

REFIT-HEAD CAVEAT (mirrors shortcut_analysis.py / eval_cross_dataset.py):
`train_baseline.py` checkpoints ONLY the backbone (`model.backbone.state_dict()`),
never the per-source classification heads. So we load the LoRA-adapted backbone,
fit a FRESH classifier head on the target source's TRAIN features, and predict on
its eval split. Every number here is faithful to the encoder's representation
structure, NOT a literal replay of the trained head's predictions. The "logits"
used for calibration/temperature-scaling are the log-probabilities of that refit
head (a linear/logistic head over frozen features), consistent across SD and SI.

TEMPERATURE-SCALING NOTE: the scalar T is fit transductively on the SAME eval set
whose ECE/NLL we then report. For this analysis (comparing the *achievable*
calibration of SD vs SI encoders) that is acceptable and is flagged in the output;
it is NOT a held-out recalibration claim.

SPLIT MATCHING (critical for comparability with the T3 table): mirrors
`train_baseline._train_one_seed` / `shortcut_analysis.py` exactly. The SI model is
evaluated on the VAL partition of the user-disjoint split (val user 4); the SD
model on the VAL partition of the force_random split (seed 0, val=test=0.10).

Usage (bdsl_graph env):
    HF_HUB_OFFLINE=1 python -m benchmark.analysis.eval_metrics \
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

from banglahandshape.class_alignment import discover_source
from banglahandshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, split_user_disjoint, split_random,
)
from banglahandshape.dinov2_lora import build_dinov2_lora
from benchmark.baselines.train_baseline import _build_transforms
from benchmark.baselines.train_probe_cached import _extract

# Source roots (mirrors DEFAULT_ROOTS in shortcut_analysis.py / plot_confusion.py).
DEFAULT_ROOTS = {
    "bdsl47_digits":  "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}

# Fixed BdSL47 SI split (matches bdsl47_si.yaml: val user 4, test user 5). We
# evaluate on the VAL partition, exactly like train_baseline / shortcut_analysis.
SI_VAL_USERS = {4}
SI_TEST_USERS = {5}
# Fixed BdSL47 SD split (matches bdsl47_sd.yaml: force_random, seed 0, 80/10/10).
SD_SEED = 0
SD_VAL_FRAC = 0.10
SD_TEST_FRAC = 0.10

N_BINS = 15  # 15-bin equal-width ECE / MCE, per the spec.
EPS = 1e-12  # softmax-prob floor for log-based metrics (NLL).


# --------------------------------------------------------------------------- #
#  Checkpoint / split helpers (mirror shortcut_analysis.py exactly)
# --------------------------------------------------------------------------- #
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
    on eval features. Returns (y_true, y_pred, probs, logits) where probs/logits
    are (N, num_classes) over the FULL [0, num_classes) label space (sklearn-
    observed classes are mapped back; unobserved classes get 0 probability, and a
    very-negative logit so temperature scaling never resurrects them)."""
    from sklearn.linear_model import LogisticRegression

    Xtr, ytr = _extract(model, _loader(spec, train_entries, transform, bs, nw), device)
    Xva, yva = _extract(model, _loader(spec, eval_entries, transform, bs, nw), device)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xva)  # already in the original label space (clf.classes_)

    # Map sklearn's per-column probabilities AND decision scores (logits) back into
    # the full [0, C) label space. decision_function gives the pre-softmax linear
    # scores we treat as logits for calibration + temperature scaling.
    proba = clf.predict_proba(Xva)               # (N, len(clf.classes_))
    dec = clf.decision_function(Xva)             # (N, len(clf.classes_)) or (N,) if binary
    if dec.ndim == 1:  # sklearn returns 1-D scores for the 2-class case
        dec = np.stack([-dec, dec], axis=1)
    full_p = np.zeros((proba.shape[0], spec.num_classes), dtype=np.float64)
    full_l = np.full((proba.shape[0], spec.num_classes), -1e9, dtype=np.float64)
    for col, cls in enumerate(clf.classes_):
        full_p[:, int(cls)] = proba[:, col]
        full_l[:, int(cls)] = dec[:, col]
    return (yva.astype(np.int64), preds.astype(np.int64), full_p, full_l)


# --------------------------------------------------------------------------- #
#  Imbalance-aware metrics (sklearn)
# --------------------------------------------------------------------------- #
def _classification_metrics(y_true, y_pred, num_classes):
    """macro-F1, balanced accuracy (mean per-class recall), Top-1, per-class recall.
    Per-class recall is over the FULL [0, C) label space; classes absent from the
    eval set report NaN and are dropped from the balanced-accuracy mean."""
    from sklearn.metrics import f1_score, recall_score, accuracy_score

    labels = list(range(num_classes))
    top1 = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels,
                              average="macro", zero_division=0))
    # Per-class recall over the full label space (0 for classes sklearn didn't see).
    rec_all = recall_score(y_true, y_pred, labels=labels,
                           average=None, zero_division=0)
    # Balanced accuracy = mean recall over classes actually PRESENT in y_true.
    present = np.unique(y_true)
    bal_acc = float(np.mean([rec_all[c] for c in present])) if len(present) else float("nan")
    # NaN for classes absent from the eval set (reported as such in the per-class list).
    per_class = []
    for c in range(num_classes):
        per_class.append(float(rec_all[c]) if c in present else float("nan"))
    return {"top1": top1, "macro_f1": macro_f1, "balanced_acc": bal_acc,
            "per_class_recall": per_class}


# --------------------------------------------------------------------------- #
#  Calibration metrics
# --------------------------------------------------------------------------- #
def _ece_mce(probs, y_true, n_bins=N_BINS):
    """15-bin equal-width ECE and MCE from max-softmax confidence.
    ECE = sum_b (n_b/N)*|acc_b - conf_b|;  MCE = max_b |acc_b - conf_b|."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    n = len(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:      # last bin closed on the right so conf == 1.0 lands
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            gap = abs(float(correct[mask].mean()) - float(conf[mask].mean()))
            ece += (cnt / n) * gap
            mce = max(mce, gap)
    return float(ece), float(mce)


def _brier(probs, y_true, num_classes):
    """One-vs-rest mean Brier score: mean over classes of the per-class
    Brier score (mean squared error of the class-c probability vs its one-hot)."""
    onehot = np.zeros((len(y_true), num_classes), dtype=np.float64)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    # Mean over samples AND classes = one-vs-rest mean Brier.
    return float(np.mean((probs - onehot) ** 2))


def _nll(probs, y_true):
    """Mean negative log-likelihood of the true class (natural log)."""
    p = np.clip(probs[np.arange(len(y_true)), y_true], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def _softmax(logits):
    """Row-wise softmax on a (N, C) logit array (numpy, numerically stable)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_temperature(logits, y_true, lo=0.5, hi=5.0):
    """Fit a single scalar T>0 minimizing NLL of softmax(logits / T) on the eval
    set. LBFGS on log-T (keeps T>0) constrained to [lo, hi], with a coarse grid
    fallback if LBFGS misbehaves (e.g. the -1e9 unobserved-class logits)."""
    lg = torch.tensor(logits, dtype=torch.float64)
    yt = torch.tensor(y_true, dtype=torch.long)
    lossf = torch.nn.CrossEntropyLoss()

    # --- LBFGS on log-T so T stays positive ---
    log_t = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))  # T = 1.0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100,
                            line_search_fn="strong_wolfe")

    def _closure():
        opt.zero_grad()
        T = log_t.exp().clamp(lo, hi)
        loss = lossf(lg / T, yt)
        loss.backward()
        return loss

    try:
        opt.step(_closure)
        T_lbfgs = float(log_t.exp().clamp(lo, hi).item())
    except Exception as e:  # pragma: no cover - numerical guard
        print(f"    [temp-scale] LBFGS failed ({e}); using grid only", flush=True)
        T_lbfgs = None

    # --- coarse+fine grid fallback / cross-check ---
    grid = np.linspace(lo, hi, 91)  # step 0.05
    with torch.no_grad():
        nlls = [float(lossf(lg / float(t), yt).item()) for t in grid]
    T_grid = float(grid[int(np.argmin(nlls))])

    # Prefer whichever candidate gives the lower NLL.
    def _nll_at(T):
        with torch.no_grad():
            return float(lossf(lg / float(T), yt).item())
    candidates = [T for T in (T_lbfgs, T_grid) if T is not None]
    T_best = min(candidates, key=_nll_at)
    return float(np.clip(T_best, lo, hi))


# --------------------------------------------------------------------------- #
#  Per (source, split) evaluation
# --------------------------------------------------------------------------- #
def _eval_one(model, spec, tr, va, transform, bs, nw, device):
    """Refit head, then compute the full metric bundle + temperature scaling.
    Returns a dict with classification, calibration (pre), and recalibration."""
    y_true, y_pred, probs, logits = _refit_head_predict(
        model, spec, tr, va, transform, bs, nw, device)

    clf = _classification_metrics(y_true, y_pred, spec.num_classes)
    ece, mce = _ece_mce(probs, y_true)
    brier = _brier(probs, y_true, spec.num_classes)
    nll = _nll(probs, y_true)

    # Temperature scaling on the eval logits (transductive; flagged in output).
    T = _fit_temperature(logits, y_true)
    probs_T = _softmax(logits / T)
    ece_T, mce_T = _ece_mce(probs_T, y_true)
    nll_T = _nll(probs_T, y_true)

    return {
        "n_eval": int(len(y_true)),
        "top1": clf["top1"], "macro_f1": clf["macro_f1"],
        "balanced_acc": clf["balanced_acc"],
        "per_class_recall": clf["per_class_recall"],
        "ece": ece, "mce": mce, "brier": brier, "nll": nll,
        "T": T, "ece_T": ece_T, "mce_T": mce_T, "nll_T": nll_T,
    }


# --------------------------------------------------------------------------- #
#  Markdown writers
# --------------------------------------------------------------------------- #
def _fmt_pct(v):
    return "n/a" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v*100:.2f}"


def _write_metrics_md(rows, out_md):
    """rows: list of dicts with source, split, and the metric bundle."""
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    lines = []
    lines.append("# T_metrics — imbalance-aware classification metrics (BdSL47)\n")
    lines.append("Macro-F1, balanced accuracy (mean per-class recall), and Top-1 for "
                 "the signer-DEPENDENT (SD) and signer-INDEPENDENT (SI) BdSL47 "
                 "encoders,\nper source. SI is evaluated on the user-disjoint VAL "
                 "partition (val user 4); SD on the force_random VAL partition "
                 "(seed 0). All values are percentages.\n")
    lines.append("*REFIT-HEAD CAVEAT: only the backbone is checkpointed by "
                 "`train_baseline.py`, so a FRESH logistic head is fit on FROZEN "
                 "backbone features and evaluated on the eval split. These numbers "
                 "are faithful to the encoder's representation structure, NOT a "
                 "literal replay of the trained classification head.*\n")
    lines.append("| Source | Split | N eval | Top-1 | macro-F1 | balanced-acc |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(f"| {r['source']} | {r['split']} | {r['n_eval']} | "
                     f"{_fmt_pct(r['top1'])} | {_fmt_pct(r['macro_f1'])} | "
                     f"{_fmt_pct(r['balanced_acc'])} |")
    lines.append("")
    # Per-class recall dump (one block per source x split).
    lines.append("## Per-class recall\n")
    lines.append("Recall for every class in `[0, num_classes)`; `n/a` = class "
                 "absent from that eval split. Percentages.\n")
    for r in rows:
        vals = ", ".join(_fmt_pct(v) for v in r["per_class_recall"])
        lines.append(f"- **{r['source']} / {r['split']}** "
                     f"({len(r['per_class_recall'])} classes): {vals}")
    lines.append("")
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {out_md}  ({len(rows)} source x split rows)", flush=True)


def _write_recalibration_md(rows, out_md):
    """rows: list of dicts with source, split, and the calibration bundle."""
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    lines = []
    lines.append("# T_recalibration — calibration & temperature scaling (BdSL47)\n")
    lines.append(f"{N_BINS}-bin Expected/Maximum Calibration Error (ECE/MCE), "
                 "one-vs-rest mean Brier score, and NLL of the SD and SI BdSL47 "
                 "encoders,\nplus a single-scalar TEMPERATURE-SCALING recalibration "
                 "(`logits / T`). ECE/MCE/Brier are percentages (x100); NLL is nats; "
                 "T is unitless.\n")
    lines.append("*TEMPERATURE-SCALING NOTE: the scalar T is fit transductively on "
                 "the SAME eval set whose ECE/NLL is then reported (NLL-minimizing "
                 "LBFGS over T in [0.5, 5], grid fallback). This measures the "
                 "ACHIEVABLE calibration of each encoder, not a held-out "
                 "recalibration.*\n")
    lines.append("*REFIT-HEAD CAVEAT: logits are the decision scores of a fresh "
                 "logistic head over FROZEN backbone features (only the backbone is "
                 "checkpointed) — faithful to the encoder, not the trained head.*\n")
    lines.append("| Source | Split | T | ECE before | ECE after | MCE before | "
                 "MCE after | Brier | NLL before | NLL after |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['source']} | {r['split']} | {r['T']:.3f} | "
            f"{r['ece']*100:.2f} | {r['ece_T']*100:.2f} | "
            f"{r['mce']*100:.2f} | {r['mce_T']*100:.2f} | "
            f"{r['brier']*100:.2f} | {r['nll']:.4f} | {r['nll_T']:.4f} |")
    lines.append("")
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {out_md}  ({len(rows)} source x split rows)", flush=True)


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
    ap.add_argument("--metrics-md", default="results/T_metrics.md")
    ap.add_argument("--recalibration-md", default="results/T_recalibration.md")
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

    metric_rows = []   # dicts with source, split, classification bundle
    recal_rows = []    # dicts with source, split, calibration bundle

    for name in args.sources:
        root = DEFAULT_ROOTS.get(name)
        if not root or not os.path.isdir(root):
            print(f"[skip source] {name}: root not found ({root!r})", flush=True)
            continue
        spec = discover_source(name, root)
        print(f"\n=== source {name}: {spec.num_classes} classes  root={root} ===",
              flush=True)

        # --- SI model on the SI (user-disjoint) eval set ---
        si_tr, si_va = _si_split(spec)
        print(f"  [SI] train={len(si_tr)} eval={len(si_va)} (user-disjoint, val user 4)",
              flush=True)
        si_model = _load_encoder(args.si_dir, args.seed, si_epoch, spec.num_classes,
                                 **lora_kwargs)
        si = _eval_one(si_model, spec, si_tr, si_va, transform,
                       args.batch_size, args.num_workers, device)
        del si_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [SI] Top-1={si['top1']*100:.2f}  macroF1={si['macro_f1']*100:.2f}  "
              f"balAcc={si['balanced_acc']*100:.2f}  ECE={si['ece']*100:.2f}->"
              f"{si['ece_T']*100:.2f} (T={si['T']:.3f})", flush=True)

        # --- SD model on the SD (force_random) eval set ---
        sd_tr, sd_va = _sd_split(spec)
        print(f"  [SD] train={len(sd_tr)} eval={len(sd_va)} (force_random seed 0)",
              flush=True)
        sd_model = _load_encoder(args.sd_dir, args.seed, sd_epoch, spec.num_classes,
                                 **lora_kwargs)
        sd = _eval_one(sd_model, spec, sd_tr, sd_va, transform,
                       args.batch_size, args.num_workers, device)
        del sd_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [SD] Top-1={sd['top1']*100:.2f}  macroF1={sd['macro_f1']*100:.2f}  "
              f"balAcc={sd['balanced_acc']*100:.2f}  ECE={sd['ece']*100:.2f}->"
              f"{sd['ece_T']*100:.2f} (T={sd['T']:.3f})", flush=True)

        for split_tag, bundle in (("SD", sd), ("SI", si)):
            row = {"source": name, "split": split_tag}
            row.update(bundle)
            metric_rows.append(row)
            recal_rows.append(row)

    if not metric_rows:
        print("[skip] no sources produced results — nothing written.")
        return

    print("\n--- writing outputs ---", flush=True)
    _write_metrics_md(metric_rows, args.metrics_md)
    _write_recalibration_md(recal_rows, args.recalibration_md)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
