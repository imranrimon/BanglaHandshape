"""Appearance LOUO — leave-one-user-out for the FROZEN DINOv2-B probe on BdSL47.

The symmetric counterpart to `louo_keypoint.py`: pose is "frozen keypoint
features + small head", so the apples-to-apples appearance baseline is "frozen
DINOv2-B features + linear head" under the SAME leave-one-signer-out protocol.
Makes the pose-vs-appearance modality claim symmetric (both frozen, both LOUO).

Extracts DINOv2-B features once per image (same backbone/transform as
probe_dinov2_b, so numbers are comparable), caches them with user IDs, then for
each held-out signer trains a linear head on the other 9 and evaluates on the
held-out one. FINAL-epoch accuracy (no epoch-peeking). Reports Top-1 mean±std
across the 10 signers.

Usage (bdsl_graph):
    python -m path3_handshape_benchmark.louo_appearance --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bangla_handshape.class_alignment import discover_source
from bangla_handshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, load_img_cache,
)
from bangla_handshape.dinov2_lora import build_dinov2_lora
from path3_handshape_benchmark.train_baseline import (
    _build_transforms, _build_cached_transforms, _append_csv_row,
)
from path3_handshape_benchmark.train_probe_cached import _extract

SOURCES = {
    "bdsl47_digits":  "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}
USERS = list(range(1, 11))
TIMM = "vit_base_patch14_dinov2.lvd142m"
FEAT_DIR = "work_dir/_feat_cache_louo"


def _extract_source(model, spec, transform, cached_kw, device, bs=64):
    """Features aligned to enumerate order -> (feats[N,D], labels[N], users[N])."""
    items = enumerate_source(spec)
    ds = HandshapeDataset([(spec, items)], transform=transform, **cached_kw)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4,
                        persistent_workers=True, prefetch_factor=2,
                        pin_memory=torch.cuda.is_available())
    feats, labels = _extract(model, loader, device)   # order == items order
    users = np.array([u for _p, _l, u in items], dtype=np.int64)
    return feats, labels.astype(np.int64), users


def _load_or_extract(seeds_unused, device):
    specs = {n: discover_source(n, r) for n, r in SOURCES.items()}
    ncps = [specs[n].num_classes for n in SOURCES]
    os.makedirs(FEAT_DIR, exist_ok=True)
    transform = _build_transforms(224)
    idx, cdir = load_img_cache()
    cached_kw = dict(img_cache_index=idx, img_cache_dir=cdir,
                     cached_transform=_build_cached_transforms())
    if idx is not None:
        print(f"[img-cache] using {len(idx)} cached entries for extraction", flush=True)
    model = None
    data = {}
    for name in SOURCES:
        f = os.path.join(FEAT_DIR, f"{name}.npz")
        if os.path.exists(f):
            d = np.load(f)
            data[name] = (d["feats"], d["labels"], d["users"], specs[name].num_classes)
            print(f"[cache-hit] {name}: {d['feats'].shape}", flush=True)
            continue
        if model is None:
            model = build_dinov2_lora(ncps, timm_name=TIMM, lora_rank=0,
                                      lora_targets=[], pretrained=True).to(device)
        feats, labels, users = _extract_source(model, specs[name], transform,
                                               cached_kw, device)
        np.savez(f, feats=feats, labels=labels, users=users)
        data[name] = (feats, labels, users, specs[name].num_classes)
        print(f"[extracted] {name}: {feats.shape}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return data


def _head_final_acc(Xtr, ytr, Xva, yva, ncls, seed, device, epochs=50, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, device=device); yva_t = torch.tensor(yva, dtype=torch.long, device=device)
    head = nn.Linear(Xtr_t.shape[1], ncls).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    n, bs = Xtr_t.shape[0], 256
    acc = 0.0
    for _ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); lossf(head(Xtr_t[idx]), ytr_t[idx]).backward(); opt.step()
        head.eval()
        with torch.no_grad():
            acc = (head(Xva_t).argmax(1) == yva_t).float().mean().item()  # final epoch
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--results-csv", default="results_final.csv")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _load_or_extract(args.seeds, device)

    per_source = {name: [] for name in SOURCES}
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for seed in args.seeds:
        for u in USERS:
            line = []
            for name in SOURCES:
                feats, labels, users, ncls = data[name]
                tr = users != u
                va = users == u
                acc = _head_final_acc(feats[tr], labels[tr], feats[va], labels[va],
                                      ncls, seed, device, epochs=args.epochs)
                per_source[name].append(acc)
                _append_csv_row(args.results_csv, {
                    "Timestamp": ts,
                    "Experiment": f"bhc_probe_dinov2_b_louo_{name}_testuser{u:02d}_seed{seed}",
                    "Epoch": args.epochs, "Top1_Acc": f"{acc:.6f}", "Top5_Acc": "",
                    "Top5_Policy": "final_epoch", "WorkDir": "./work_dir/bhc_probe_dinov2_b_louo"})
                line.append(f"{name}={acc*100:5.1f}")
            print(f"[seed {seed}] held user {u:2d}: " + " ".join(line), flush=True)

    print("\n=== APPEARANCE (DINOv2-B frozen) LOUO — Top-1 mean±std across signers ===")
    for name in per_source:
        v = np.array(per_source[name]) * 100
        print(f"  {name:16s} {v.mean():5.2f} +/- {v.std():4.2f}  "
              f"(min {v.min():.1f}, max {v.max():.1f}, folds={len(v)})")


if __name__ == "__main__":
    main()
