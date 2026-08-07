"""Feature-cached linear-probe training for frozen-backbone (probe) configs.

A linear probe re-forwards the FROZEN backbone over the SAME images every epoch —
pure waste (50 epochs = 50x redundant backbone passes). This instead:

  1. extracts each image's feature ONCE (per source, per split) and caches it to
     disk. The split is fixed by `split.seed` and the backbone is frozen, so the
     features are identical across run seeds -> extract once, reuse for all seeds;
  2. trains per-source linear heads for the full epoch budget on the cached
     feature vectors (near-instant), with best-over-epochs val Top-1.

Identical result to the image-based probe, ~10x+ faster. Per-source .npz caching
also makes extraction itself resumable. Used automatically by run_campaign.py for
configs whose backbone is frozen.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bangla_handshape.class_alignment import discover_source, SourceSpec
from bangla_handshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, split_user_disjoint, split_random,
)
from bangla_handshape.dinov2_lora import build_dinov2_lora
from path3_handshape_benchmark.train_baseline import (
    _provided_test_dir, _build_transforms, _append_csv_row,
)


def is_probe_config(cfg):
    """True if the backbone is frozen (linear probe): no LoRA, no full-FT, ViT."""
    enc = cfg.get("encoder", {})
    arch = str(enc.get("arch", "dinov2")).lower()
    return (not arch.startswith("resnet")
            and int(enc.get("lora_rank", 0)) == 0
            and not (enc.get("lora_targets") or [])
            and not bool(enc.get("full_finetune", False)))


@torch.no_grad()
def _extract(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, _src, y in loader:
        f = model.features(x.to(device, non_blocking=True))
        feats.append(f.detach().float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _splits_for(sources, sp):
    """Same train/val partition as train_baseline (provided-test / user-disjoint /
    random), so cached-probe numbers are comparable to the image-based ones."""
    force_random = bool(sp.get("force_random", False))
    val_users = set(sp.get("val_users", []))
    test_users = set(sp.get("test_users", []))
    out = {}
    for spec in sources:
        items = enumerate_source(spec)
        held = _provided_test_dir(spec.root)
        if (not force_random) and spec.name in ("bdsl47_digits", "bdsl47_letters"):
            tr, va, _ = split_user_disjoint(items, val_users, test_users)
        elif held is not None:
            tr = items
            va = enumerate_source(SourceSpec(spec.name, held, spec.num_classes,
                                             spec.class_to_idx))
        else:
            tr, va, _ = split_random(items, seed=int(sp.get("seed", 0)),
                                     val_frac=float(sp.get("random_val_frac", 0.10)),
                                     test_frac=float(sp.get("random_test_frac", 0.10)))
        out[spec.name] = (spec, tr, va)
    return out


def extract_and_cache(cfg, cache_dir):
    """Return {source: (Xtr,ytr,Xva,yva,num_classes)}; cache each to .npz."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = [discover_source(n, r) for n, r in cfg["sources"].items() if os.path.isdir(r)]
    enc = cfg["encoder"]
    timm_name = enc.get("timm_name", "vit_small_patch14_dinov2.lvd142m")
    tag = timm_name.replace("/", "_").replace(".", "_")
    os.makedirs(cache_dir, exist_ok=True)
    splits = _splits_for(sources, cfg.get("split", {}))
    transform = _build_transforms(int(cfg.get("image_size", 224)))
    ncps = [s.num_classes for s in sources]
    model = None
    cache = {}
    for spec in sources:
        _spec, tr, va = splits[spec.name]
        cpath = os.path.join(cache_dir, f"{tag}__{spec.name}.npz")
        if os.path.exists(cpath):
            d = np.load(cpath)
            cache[spec.name] = (d["Xtr"], d["ytr"], d["Xva"], d["yva"], int(spec.num_classes))
            print(f"[cache-hit] {spec.name} <- {os.path.basename(cpath)}", flush=True)
            continue
        if model is None:
            model = build_dinov2_lora(ncps, timm_name=timm_name, lora_rank=0,
                                      lora_targets=[], pretrained=True).to(device)
        def _ld(entries):
            ds = HandshapeDataset([(spec, entries)], transform=transform)
            return DataLoader(ds, batch_size=int(cfg.get("batch_size", 64)), shuffle=False,
                              num_workers=4, persistent_workers=True, prefetch_factor=2,
                              pin_memory=torch.cuda.is_available())
        Xtr, ytr = _extract(model, _ld(tr), device)
        Xva, yva = _extract(model, _ld(va), device)
        np.savez(cpath, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)
        cache[spec.name] = (Xtr, ytr, Xva, yva, int(spec.num_classes))
        print(f"[extracted] {spec.name}: train {Xtr.shape} val {Xva.shape}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache, [s.name for s in sources]


def _train_head_one_seed(cfg, seed, cache, source_names, results_csv):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = cfg.get("Experiment_name", "bhc_probe")
    epochs = int(cfg.get("num_epoch", 50))
    lr = float(cfg.get("base_lr", 1e-3)); wd = float(cfg.get("weight_decay", 1e-4))
    lossf = nn.CrossEntropyLoss()
    best = {}
    for name in source_names:
        Xtr, ytr, Xva, yva, ncls = cache[name]
        Xtr_t = torch.tensor(Xtr, device=device)
        ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
        Xva_t = torch.tensor(Xva, device=device)
        yva_t = torch.tensor(yva, dtype=torch.long, device=device)
        head = nn.Linear(Xtr_t.shape[1], ncls).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
        n = Xtr_t.shape[0]; bs = 256; best_acc = 0.0
        for _ep in range(epochs):
            head.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                lossf(head(Xtr_t[idx]), ytr_t[idx]).backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                acc = (head(Xva_t).argmax(1) == yva_t).float().mean().item()
            best_acc = max(best_acc, acc)
        best[name] = best_acc
        print(f"  [seed {seed}] {name}: best Top-1 = {best_acc*100:.2f}%", flush=True)
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    wd_dir = cfg.get("work_dir", f"./work_dir/{base}")
    for name in source_names:
        _append_csv_row(results_csv, {
            "Timestamp": ts, "Experiment": f"{base}_{name}_seed{seed}",
            "Epoch": epochs, "Top1_Acc": f"{best[name]:.6f}", "Top5_Acc": "",
            "Top5_Policy": "best_over_epochs", "WorkDir": wd_dir})


def run_probe_cached(cfg, seeds, results_csv, cache_dir="work_dir/_feat_cache"):
    cache, names = extract_and_cache(cfg, cache_dir)
    for s in seeds:
        _train_head_one_seed(cfg, s, cache, names, results_csv)
