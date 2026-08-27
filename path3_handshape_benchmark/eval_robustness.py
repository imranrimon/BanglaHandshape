"""Phase-2 robustness / OOD table for the SI BdSL47 appearance encoder.

Measures how the signer-INDEPENDENT (SI) BdSL47 DINOv2+LoRA encoder degrades
under common image corruptions — a standard top-tier robustness table
(cf. ImageNet-C). We DO NOT retrain: we load the SI backbone, fit ONE fresh
linear head on CLEAN train features, establish the CLEAN SI-val Top-1, and then
re-featurize the SAME SI-val images through each corruption -> the (frozen)
backbone and predict with that FIXED clean-trained head. Corruption robustness
is thus the drop of the SAME classifier under distribution shift.

Reuses the encoder-load + fresh-head-on-frozen-features pattern of
`shortcut_analysis.py` / `eval_cross_dataset.py`: `train_baseline.py` checkpoints
ONLY the backbone (`model.backbone.state_dict()`), never the classification heads,
so we load the LoRA-adapted backbone (strict=False) and re-fit a fresh head.

SPLIT (mirrors train_baseline._train_one_seed for BdSL47 SI): user-disjoint,
train = all users except val/test, eval = the VAL partition (val user 4). This
matches the T1/T3 SI eval set so the clean number here equals the headline SI number.

CORRUPTIONS (6, x 3 severities each) are applied to the PIL image BEFORE the
standard Resize(1.15x)->CenterCrop(224)->ToTensor->Normalize transform, so the
corruption operates on the native image just like real acquisition noise would.
Each is deterministic (fixed rng per severity) and PIL/numpy-only (no extra deps):
  gaussian_noise, gaussian_blur, brightness, contrast, jpeg_compression, downscale.

CAVEAT (documented in the output): only the backbone is checkpointed, so the head
is a fresh linear head re-fit on frozen CLEAN features — faithful to the encoder's
representation under corruption, not a literal replay of the trained head's weights.

Usage (bdsl_graph env):
    HF_HUB_OFFLINE=1 python -m path3_handshape_benchmark.eval_robustness \
        --si-dir work_dir/bhc_bdsl47_si --seed 0 \
        --sources bdsl47_digits bdsl47_letters
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys

# Force offline so timm/HF never phones home for the DINOv2 weights on the cluster.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bangla_handshape.class_alignment import discover_source
from bangla_handshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, split_user_disjoint,
)
from bangla_handshape.dinov2_lora import build_dinov2_lora
from path3_handshape_benchmark.train_baseline import _build_transforms
from path3_handshape_benchmark.train_probe_cached import _extract

# Source roots (mirrors DEFAULT_ROOTS in shortcut_analysis.py / class_alignment).
DEFAULT_ROOTS = {
    "bdsl47_digits":  "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}

# Fixed BdSL47 SI split (matches bdsl47_si.yaml: val user 4, test user 5). We
# evaluate on the VAL partition, exactly like train_baseline / shortcut_analysis.
SI_VAL_USERS = {4}
SI_TEST_USERS = {5}

# Severity order used everywhere (columns of the table, low -> high).
SEVERITIES = ["mild", "med", "strong"]


# --------------------------------------------------------------------------- #
#  Corruptions: each is (PIL.Image, severity_str) -> PIL.Image (same size/mode)
#  applied to the native PIL image BEFORE Resize/CenterCrop/Normalize. Every op
#  is deterministic (fixed rng seeded per severity for the stochastic ones).
# --------------------------------------------------------------------------- #
_SEV_IDX = {"mild": 0, "med": 1, "strong": 2}


def _to_rgb(img):
    return img.convert("RGB") if img.mode != "RGB" else img


def corrupt_gaussian_noise(img, severity):
    """Additive Gaussian pixel noise (std in [0,255] units)."""
    img = _to_rgb(img)
    std = [8.0, 18.0, 35.0][_SEV_IDX[severity]]
    # Deterministic per (severity) — same noise field for the whole eval set is
    # fine and keeps runs reproducible; seed also folds in image size for variety.
    arr = np.asarray(img, dtype=np.float32)
    rng = np.random.RandomState(1234 + _SEV_IDX[severity])
    noise = rng.normal(0.0, std, size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def corrupt_gaussian_blur(img, severity):
    """Gaussian blur via PIL.ImageFilter.GaussianBlur (radius in px)."""
    img = _to_rgb(img)
    radius = [1.0, 2.0, 4.0][_SEV_IDX[severity]]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def corrupt_brightness(img, severity):
    """Brightness shift via PIL.ImageEnhance.Brightness (factor != 1.0)."""
    img = _to_rgb(img)
    factor = [1.3, 1.6, 2.0][_SEV_IDX[severity]]
    return ImageEnhance.Brightness(img).enhance(factor)


def corrupt_contrast(img, severity):
    """Contrast reduction via PIL.ImageEnhance.Contrast (factor < 1.0)."""
    img = _to_rgb(img)
    factor = [0.7, 0.5, 0.3][_SEV_IDX[severity]]
    return ImageEnhance.Contrast(img).enhance(factor)


def corrupt_jpeg_compression(img, severity):
    """JPEG round-trip through an in-memory buffer at decreasing quality."""
    img = _to_rgb(img)
    quality = [40, 20, 10][_SEV_IDX[severity]]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return _to_rgb(Image.open(buf))


def corrupt_downscale(img, severity):
    """Low-res: downscale by a factor then upscale back to the original size
    (bilinear both ways), erasing high-frequency detail."""
    img = _to_rgb(img)
    factor = [2, 3, 4][_SEV_IDX[severity]]
    w, h = img.size
    small = (max(1, w // factor), max(1, h // factor))
    return img.resize(small, Image.BILINEAR).resize((w, h), Image.BILINEAR)


# name -> callable(img, severity) -> PIL.Image  (table row order).
CORRUPTIONS = {
    "gaussian_noise":    corrupt_gaussian_noise,
    "gaussian_blur":     corrupt_gaussian_blur,
    "brightness":        corrupt_brightness,
    "contrast":          corrupt_contrast,
    "jpeg_compression":  corrupt_jpeg_compression,
    "downscale":         corrupt_downscale,
}


class _CorruptTransform:
    """First transform in the pipeline: apply a fixed corruption to the PIL image,
    then hand off to the standard Resize/CenterCrop/ToTensor/Normalize compose.
    Identity (corruption=None) reproduces the clean pipeline exactly."""

    def __init__(self, corruption, severity, tail):
        self.corruption = corruption  # callable(img, severity) or None
        self.severity = severity
        self.tail = tail              # _build_transforms(image_size)

    def __call__(self, img):
        if self.corruption is not None:
            img = self.corruption(img, self.severity)
        return self.tail(img)


# --------------------------------------------------------------------------- #
#  Encoder load + head refit (mirrors shortcut_analysis.py)
# --------------------------------------------------------------------------- #
def _latest_epoch(encoder_dir, seed):
    """Highest available epoch for `encoder_seed<seed>_epoch<E>.pt`, or None."""
    if not os.path.isdir(encoder_dir):
        return None
    pat = re.compile(rf"^encoder_seed{seed}_epoch(\d+)\.pt$")
    epochs = [int(m.group(1)) for fn in os.listdir(encoder_dir)
              for m in [pat.match(fn)] if m]
    return max(epochs) if epochs else None


def _si_split(spec):
    """SI (user-disjoint) split for BdSL47; return (train, eval=val partition)."""
    items = enumerate_source(spec)
    tr, va, _te = split_user_disjoint(items, SI_VAL_USERS, SI_TEST_USERS)
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


def _feats(model, spec, entries, transform, bs, nw, device):
    return _extract(model, _loader(spec, entries, transform, bs, nw), device)


# --------------------------------------------------------------------------- #
#  Output writers
# --------------------------------------------------------------------------- #
def _write_table(per_source, out_md):
    """per_source: {source: {"clean": acc, "grid": {corr: {sev: acc}},
                             "mCA": float, "retention": float}}.
    One markdown block per source (rows=corruption, cols=clean+severities+drop)
    plus a global mCA / retention summary."""
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    L = []
    L.append("# T_robustness — corruption robustness of the SI BdSL47 encoder\n")
    L.append("Top-1 accuracy of the signer-INDEPENDENT (SI) BdSL47 DINOv2+LoRA "
             "encoder under common image corruptions. A SINGLE linear head is "
             "fit once on CLEAN SI-train features; the SAME fixed head then "
             "predicts on the SI-val set (user-disjoint, val user 4) re-featurized "
             "through each corruption. `Clean` is the un-corrupted SI-val Top-1.\n")
    L.append("Each corruption is applied to the native PIL image BEFORE the "
             "standard Resize->CenterCrop(224)->Normalize transform, at 3 "
             f"severities ({', '.join(SEVERITIES)}). `drop` = Clean - mean(severities), "
             "in percentage points.\n")
    L.append("- **mCA** (mean Corruption Accuracy) = mean Top-1 over all "
             "corruption x severity cells.\n"
             "- **relative retention** = mCA / Clean (1.0 = no degradation).\n")
    L.append("> CAVEAT: only the backbone is checkpointed (`model.backbone."
             "state_dict()`), never the classification head. The head here is a "
             "fresh linear head re-fit on frozen CLEAN features — faithful to the "
             "encoder's representation under shift, not a literal replay of the "
             "trained head's weights.\n")

    all_cells = []  # for the grand mCA across sources
    grand_clean = []
    for src in sorted(per_source):
        d = per_source[src]
        clean = d["clean"]
        L.append(f"## {src}\n")
        header = "| corruption | Clean | " + " | ".join(SEVERITIES) + \
                 " | mean | drop (pp) |"
        L.append(header)
        L.append("|" + "---|" * (len(SEVERITIES) + 4))
        for corr in CORRUPTIONS:
            row = d["grid"][corr]
            sev_accs = [row[s] for s in SEVERITIES]
            mean_c = float(np.mean(sev_accs))
            drop = (clean - mean_c) * 100.0
            cells = " | ".join(f"{row[s]*100:.2f}" for s in SEVERITIES)
            L.append(f"| {corr} | {clean*100:.2f} | {cells} | "
                     f"{mean_c*100:.2f} | {drop:+.2f} |")
            all_cells.extend(sev_accs)
        L.append(f"\n**{src}: Clean = {clean*100:.2f}% · "
                 f"mCA = {d['mCA']*100:.2f}% · "
                 f"relative retention = {d['retention']:.3f}**\n")
        grand_clean.append(clean)

    if all_cells:
        g_mca = float(np.mean(all_cells))
        g_clean = float(np.mean(grand_clean))
        g_ret = (g_mca / g_clean) if g_clean > 0 else float("nan")
        L.append("## Summary (all sources)\n")
        L.append(f"- Mean Clean Top-1 = **{g_clean*100:.2f}%**")
        L.append(f"- Mean Corruption Accuracy (mCA) = **{g_mca*100:.2f}%**")
        L.append(f"- Relative retention (mCA / Clean) = **{g_ret:.3f}**\n")

    with open(out_md, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"  wrote {out_md}", flush=True)


def _plot(per_source, out_png):
    """Line plot: Top-1 vs severity (clean=0) per corruption, one subplot per source."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    srcs = sorted(per_source)
    if not srcs:
        return
    ncol = len(srcs)
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.2), squeeze=False)
    xs = list(range(len(SEVERITIES) + 1))  # 0 = clean, then severities
    xticklabels = ["clean"] + SEVERITIES
    for ax, src in zip(axes[0], srcs):
        d = per_source[src]
        for corr in CORRUPTIONS:
            ys = [d["clean"] * 100.0] + \
                 [d["grid"][corr][s] * 100.0 for s in SEVERITIES]
            ax.plot(xs, ys, "o-", label=corr, lw=1.4, ms=4)
        ax.axhline(d["clean"] * 100.0, color="k", ls="--", lw=0.7)
        ax.set_xticks(xs)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("corruption severity")
        ax.set_ylabel("Top-1 (%)")
        ax.set_title(f"{src}  (mCA={d['mCA']*100:.1f}%, "
                     f"ret={d['retention']:.2f})", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc="lower left")
    fig.suptitle("SF6 — SI BdSL47 encoder robustness under image corruptions")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--si-dir", default="work_dir/bhc_bdsl47_si",
                    help="dir with the SI encoder_seed<N>_epoch<E>.pt checkpoints")
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
    ap.add_argument("--out-md", default="results/T_robustness.md")
    ap.add_argument("--out-png", default="results/SF6_robustness.png")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    # Resolve epoch (auto-detect latest if --epoch 0). Bail gracefully (exit 0) if
    # the SI encoder dir has no checkpoint for this seed — still a valid script.
    si_epoch = args.epoch or _latest_epoch(args.si_dir, args.seed)
    if si_epoch is None:
        print(f"[skip] no SI checkpoint (encoder_seed{args.seed}_epoch*.pt) in "
              f"{args.si_dir!r} — nothing to evaluate. Train bdsl47_si.yaml first.")
        return
    print(f"SI epoch={si_epoch} ({args.si_dir})", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_tail = _build_transforms(args.image_size)
    lora_kwargs = dict(timm_name=args.timm_name, lora_rank=args.lora_rank,
                       lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                       lora_targets=args.lora_targets, device=device)

    from sklearn.linear_model import LogisticRegression

    per_source = {}
    for name in args.sources:
        root = DEFAULT_ROOTS.get(name)
        if not root or not os.path.isdir(root):
            print(f"[skip source] {name}: root not found ({root!r})", flush=True)
            continue
        spec = discover_source(name, root)
        si_tr, si_va = _si_split(spec)
        print(f"\n=== source {name}: {spec.num_classes} classes  root={root} ===",
              flush=True)
        print(f"  [SI] train={len(si_tr)} eval={len(si_va)} "
              f"(user-disjoint, val user 4)", flush=True)
        if not si_tr or not si_va:
            print(f"[skip source] {name}: empty SI train/eval partition", flush=True)
            continue

        model = _load_encoder(args.si_dir, args.seed, si_epoch, spec.num_classes,
                              **lora_kwargs)

        # 1) Fit ONE head on CLEAN train features.
        clean_tf = _CorruptTransform(None, None, clean_tail)
        Xtr, ytr = _feats(model, spec, si_tr, clean_tf,
                          args.batch_size, args.num_workers, device)
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(Xtr, ytr)

        # 2) CLEAN SI-val Top-1 with that fixed head.
        Xva_clean, yva = _feats(model, spec, si_va, clean_tf,
                                args.batch_size, args.num_workers, device)
        clean_acc = float((clf.predict(Xva_clean) == yva).mean())
        print(f"  [clean] SI-val Top-1 = {clean_acc*100:.2f}%", flush=True)

        # 3) For each corruption x severity: re-featurize SI-val through the
        #    corruption + frozen backbone, predict with the FIXED clean head.
        grid = {corr: {} for corr in CORRUPTIONS}
        for corr_name, corr_fn in CORRUPTIONS.items():
            for sev in SEVERITIES:
                tf = _CorruptTransform(corr_fn, sev, clean_tail)
                Xva_c, yva_c = _feats(model, spec, si_va, tf,
                                      args.batch_size, args.num_workers, device)
                acc = float((clf.predict(Xva_c) == yva_c).mean())
                grid[corr_name][sev] = acc
                print(f"    {corr_name:>18} [{sev:>6}] Top-1 = {acc*100:.2f}%",
                      flush=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_cells = [grid[c][s] for c in CORRUPTIONS for s in SEVERITIES]
        mca = float(np.mean(all_cells))
        retention = (mca / clean_acc) if clean_acc > 0 else float("nan")
        per_source[name] = dict(clean=clean_acc, grid=grid,
                                mCA=mca, retention=retention)
        print(f"  [{name}] mCA = {mca*100:.2f}%  retention = {retention:.3f}",
              flush=True)

    if not per_source:
        print("[skip] no sources produced results — nothing written.")
        return

    print("\n--- writing outputs ---", flush=True)
    _write_table(per_source, args.out_md)
    _plot(per_source, args.out_png)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
