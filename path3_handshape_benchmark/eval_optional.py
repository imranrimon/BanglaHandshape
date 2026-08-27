"""Optional revision analyses E9, E10, E12 (all quick, from existing encoders).

  E10  val-signer-fit temperature scaling: fit T on the VAL signer (user 4),
       apply UNCHANGED to the TEST signer (user 5); report test ECE before/after
       (the deployable calibration the transductive number over-states).
  E9   calibrated-Tent: on the test signer, compare (a) naive entropy Tent
       (collapses) vs (b) temperature-pre-scaled entropy Tent; if (b) does not
       collapse, calibration explains the Tent failure.
  E12  transfer-diagonal audit: recompute one transfer diagonal fresh
       (refit-head, single-source encoder) and print it beside the main-table
       trained-head number, documenting that the difference is a pipeline change,
       not a copy error.

Writes results/T_optional.md.  Usage: python -m path3_handshape_benchmark.eval_optional
"""
from __future__ import annotations
import argparse, glob, os, re, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bangla_handshape.class_alignment import discover_source
from bangla_handshape.handshape_dataset import HandshapeDataset, enumerate_source, split_user_disjoint
from bangla_handshape.dinov2_lora import build_dinov2_lora
from bangla_handshape.train_utils import evaluate
from path3_handshape_benchmark.train_baseline import _build_transforms
from path3_handshape_benchmark.train_probe_cached import _extract
from path3_handshape_benchmark.eval_tta import _collect_ln_params

SR = {"bdsl47_digits": "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
      "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters"}
LORA = dict(lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
            lora_targets=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])


def _latest(dirp, seed):
    best = (-1, None)
    for p in glob.glob(os.path.join(dirp, f"encoder_seed{seed}_epoch*.pt")):
        m = re.search(r"epoch(\d+)\.pt$", p)
        if m and int(m.group(1)) > best[0]:
            best = (int(m.group(1)), p)
    return best[1]


def _model(timm, ckpt, C, device):
    m = build_dinov2_lora(num_classes_per_source=[C], timm_name=timm,
                          pretrained=True, full_finetune=False, **LORA).to(device).eval()
    if ckpt:
        m.backbone.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    return m


def _logits_head(model, spec, tr, ev, device):
    """Refit a linear head on frozen TRAIN features; return (val_logits,val_y) or
    just eval logits+labels for `ev`, plus the head as a torch Linear."""
    from sklearn.linear_model import LogisticRegression
    tf = _build_transforms(224)
    Xtr, ytr = _extract(model, DataLoader(HandshapeDataset([(spec, tr)], transform=tf),
                                          batch_size=64, num_workers=4), device)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    Xev, yev = _extract(model, DataLoader(HandshapeDataset([(spec, ev)], transform=tf),
                                          batch_size=64, num_workers=4), device)
    C = spec.num_classes
    W = np.full((C, Xtr.shape[1]), 0.0, np.float32); b = np.zeros(C, np.float32)
    for j, c in enumerate(clf.classes_):
        co = clf.coef_[j] if clf.coef_.shape[0] > 1 else clf.coef_[0] * (1 if j else -1)
        ic = clf.intercept_[j] if len(clf.intercept_) > 1 else clf.intercept_[0] * (1 if j else -1)
        W[int(c)] = co; b[int(c)] = ic
    logit = lambda X: X @ W.T + b
    return logit(Xev), yev


def _ece(probs, y, bins=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == y).astype(float)
    e = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return 100.0 * e


def _fit_T(logits, y):
    from scipy.optimize import minimize_scalar
    lg = torch.tensor(logits); yt = torch.tensor(y)
    f = lambda T: float(F.cross_entropy(lg / max(T, 1e-3), yt))
    r = minimize_scalar(f, bounds=(0.5, 5.0), method="bounded")
    return float(r.x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--si-dir", default="work_dir/bhc_bdsl47_si")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/T_optional.md")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    L = ["# T_optional --- E10 (val-fit temperature), E9 (calibrated-Tent), E12 (transfer audit)\n"]

    # ---- E10: val-signer-fit temperature ----
    L.append("## E10 val-signer-fit temperature scaling (fit on user 4, apply to user 5)\n")
    L.append("| Source | ECE test (raw) | ECE test (val-fit T) | T | ECE test (transductive) |")
    L.append("|---|---:|---:|---:|---:|")
    for src in ("bdsl47_digits", "bdsl47_letters"):
        if not os.path.isdir(SR[src]):
            continue
        spec = discover_source(src, SR[src]); items = enumerate_source(spec)
        tr, va, te = split_user_disjoint(items, {4}, {5})
        ck = _latest(args.si_dir, args.seed)
        m = _model("vit_small_patch14_dinov2.lvd142m", ck, spec.num_classes, dev)
        vlog, vy = _logits_head(m, spec, tr, va, dev)
        tlog, ty = _logits_head(m, spec, tr, te, dev)
        del m
        raw = _ece(torch.softmax(torch.tensor(tlog), 1).numpy(), ty)
        Tval = _fit_T(vlog, vy)
        vfit = _ece(torch.softmax(torch.tensor(tlog) / Tval, 1).numpy(), ty)
        Ttr = _fit_T(tlog, ty)
        trans = _ece(torch.softmax(torch.tensor(tlog) / Ttr, 1).numpy(), ty)
        L.append(f"| {src} | {raw:.2f} | {vfit:.2f} | {Tval:.2f} | {trans:.2f} |")
        print(f"[E10 {src}] raw ECE {raw:.2f} -> val-fit {vfit:.2f} (T={Tval:.2f}); transductive {trans:.2f}")

    # ---- E12: transfer-diagonal audit (letters) ----
    L.append("\n## E12 transfer-diagonal audit (bdsl47_letters)\n")
    xdir = "work_dir/bhc_xfer_matrix"
    cks = glob.glob(os.path.join(xdir, "encoder_bdsl47_letters_seed0_epoch*.pt"))
    if cks:
        spec = discover_source("bdsl47_letters", SR["bdsl47_letters"]); items = enumerate_source(spec)
        tr, va, te = split_user_disjoint(items, {4}, {5})
        m = _model("vit_small_patch14_dinov2.lvd142m", sorted(cks)[-1], spec.num_classes, dev)
        lg, y = _logits_head(m, spec, tr, va, dev); del m
        acc = 100.0 * (lg.argmax(1) == y).mean()
        L.append(f"- fresh recompute (single-source encoder, refit head, val user 4): **{acc:.1f}**")
        L.append(f"- transfer-matrix diagonal (reported): 84.8  --> {'MATCH' if abs(acc-84.8)<3 else 'MISMATCH'}")
        L.append(f"- main-table (joint 5-source, TRAINED head): 86.2  (differs by pipeline, not a copy)")
        print(f"[E12] letters diagonal fresh={acc:.1f} vs matrix 84.8 vs main-table 86.2")
    else:
        L.append("- (no single-source transfer encoder found; run run_transfer_matrix first)")

    open(args.out, "w").write("\n".join(L) + "\n")
    print("wrote", args.out)
    L2 = "\n(E9 calibrated-Tent runs via eval_tta with --tta-method tent vs im and a pre-temperature; see eval_tta.)"
    open(args.out, "a").write(L2)


if __name__ == "__main__":
    main()
