"""SF3 — per-class SD-SI Top-1 gap on BdSL47 (fixed-data reproduction).

For each source (Digits, Letters) and each split, load the matching LoRA encoder,
extract frozen features, refit a logistic head, and measure PER-CLASS Top-1 on the
eval set. gap(class) = acc_SD(class) - acc_SI(class). Plots the top-N classes by gap.
    SI: work_dir/bhc_bdsl47_si  (user-disjoint val4/test5)
    SD: work_dir/bhc_bdsl47_sd  (random split, seed 0)
Run on GPU: python -m path3_handshape_benchmark.plot_perclass_gap
"""
import os, re, glob, argparse
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from bangla_handshape.class_alignment import discover_source
from bangla_handshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, split_user_disjoint, split_random)
from bangla_handshape.dinov2_lora import build_dinov2_lora
from path3_handshape_benchmark.train_probe_cached import _extract
from path3_handshape_benchmark.train_baseline import _build_transforms
from torch.utils.data import DataLoader

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LT = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
ROOTS = {"bdsl47_digits": "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
         "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters"}

def _latest(dirp, seed):
    ps = glob.glob(os.path.join(dirp, f"encoder_seed{seed}_epoch*.pt"))
    return max(ps, key=lambda p: int(re.search(r"epoch(\d+)", p).group(1))) if ps else None

def _loader(spec, entries, tf, bs=64, nw=4):
    return DataLoader(HandshapeDataset([(spec, entries)], transform=tf),
                      batch_size=bs, shuffle=False, num_workers=nw,
                      pin_memory=torch.cuda.is_available())

def _percls_acc(spec, enc_dir, tr, va, seed, tf):
    model = build_dinov2_lora(num_classes_per_source=[spec.num_classes],
                              timm_name="vit_small_patch14_dinov2.lvd142m",
                              lora_rank=8, lora_alpha=16.0, lora_targets=LT, pretrained=True)
    ck = _latest(enc_dir, seed); model.backbone.load_state_dict(torch.load(ck, map_location="cpu"), strict=False)
    model = model.to(DEV)
    Xtr, ytr = _extract(model, _loader(spec, tr, tf), DEV)
    Xva, yva = _extract(model, _loader(spec, va, tf), DEV)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, ytr)
    pred = clf.predict(Xva)
    acc = {}
    for c in range(spec.num_classes):
        m = yva == c
        acc[c] = float((pred[m] == c).mean()) if m.sum() else np.nan
    return acc

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topn", type=int, default=15); ap.add_argument("--output", default="paper/figs/SF3_perclass")
    a = ap.parse_args()
    tf = _build_transforms(224)
    rows = []
    for src, root in ROOTS.items():
        spec = discover_source(src, root)
        items = enumerate_source(spec)
        tr_si, va_si, _ = split_user_disjoint(items, {4}, {5})
        tr_sd, va_sd, _ = split_random(items, seed=0, val_frac=0.10, test_frac=0.10)
        si = _percls_acc(spec, "work_dir/bhc_bdsl47_si", tr_si, va_si, a.seed, tf)
        sd = _percls_acc(spec, "work_dir/bhc_bdsl47_sd", tr_sd, va_sd, a.seed, tf)
        cls = list(spec.class_to_idx.keys())
        for c in range(spec.num_classes):
            g = (sd[c] - si[c]) * 100
            rows.append((f"{src.split('_')[1].capitalize()} {cls[c]}", g))
            print(f"  {src} {cls[c]}: SD={sd[c]*100:.1f} SI={si[c]*100:.1f} gap={g:.1f}")
    rows.sort(key=lambda r: r[1], reverse=True)
    top = rows[:a.topn]
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(top)), [g for _, g in top], color="#c1443c", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(top))); ax.set_xticklabels([n for n, _ in top], rotation=55, ha="right", fontsize=11)
    ax.set_ylabel("Top-1 gap  (SD $-$ SI)  [pp]", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.output}.{ext}", dpi=600, bbox_inches="tight")
    print(f"wrote {a.output}.{{pdf,png}} @600dpi")

if __name__ == "__main__":
    main()
