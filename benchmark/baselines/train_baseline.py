"""Path 3 entry point — train per-source baselines (linear probe + LoRA).

Reuses the same training loop as Path 1 (multi-head DINOv2 with optional LoRA
adapters; LoRA rank 0 = linear-probe). Reports per-source Top-1 and writes one
row to results_final.csv per (config, source, seed) for the multi-seed
aggregator.

Usage:
    python -m benchmark.baselines.train_baseline ^
        --config benchmark/configs/lora.yaml --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from banglahandshape.class_alignment import discover_source, write_inventory, SourceSpec
from banglahandshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, split_user_disjoint, split_random,
    cap_per_class, load_img_cache,
)
from banglahandshape.dinov2_lora import build_dinov2_lora
from banglahandshape.train_utils import train_one_epoch, evaluate

# Reuse main.py's CSV columns.
RESULT_COLUMNS = ["Timestamp", "Experiment", "Epoch", "Top1_Acc",
                  "Top5_Acc", "Top5_Policy", "WorkDir"]


def _init_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _build_transforms(image_size):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _build_cached_transforms():
    """Tensor-only tail of _build_transforms, for the pre-resize cache (input is
    an already Resize+CenterCrop'd 224x224 uint8 array). ToTensor on a uint8 HWC
    ndarray yields the same CHW float[0,1] as on the equivalent PIL image, so the
    result is bit-identical to the full transform."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _append_csv_row(csv_path, row):
    new_file = not (os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _provided_test_dir(root):
    """If `root` is a '.../train' folder with a sibling 'test'/'Test' folder,
    return that sibling (the authors' held-out set); else None. Lets us prefer a
    real held-out split over a leaky random split of an augmented train set
    (e.g. BSLD_45, BDSL49, which ship their own test set)."""
    norm = os.path.normpath(root)
    if os.path.basename(norm).lower() != "train":
        return None
    parent = os.path.dirname(norm)
    for name in ("test", "Test", "TEST"):
        cand = os.path.join(parent, name)
        if os.path.isdir(cand):
            return cand
    return None


def _load_crop_index(cache_dir, source_names):
    """Hand-crop ablation (A-mechanism): load MediaPipe hand bboxes
    (data_prep/extract_hand_bbox.py) into {abs_path -> (x0,y0,x1,y1)}. Only
    detected boxes are included; undetected images fall through to the full frame
    (no crop). Returns None if nothing loaded (a safe no-op)."""
    import numpy as np
    idx = {}
    for name in source_names:
        f = os.path.join(cache_dir, f"{name}.npz")
        if not os.path.exists(f):
            print(f"[hand-crop] WARN: no bbox cache for {name} at {f}")
            continue
        d = np.load(f, allow_pickle=True)
        for p, b, det in zip(d["paths"], d["bbox"], d["detected"]):
            if int(det) == 1:
                idx[os.path.normpath(os.path.abspath(str(p)))] = \
                    tuple(int(v) for v in b)
    print(f"[hand-crop] loaded {len(idx)} hand boxes from {cache_dir}")
    return idx or None


def _train_one_seed(cfg, seed, results_csv="results_final.csv"):
    _init_seed(seed)
    base_exp = cfg.get("Experiment_name", "bhc_run")
    work_dir = cfg.get("work_dir", f"./work_dir/{base_exp}")
    os.makedirs(work_dir, exist_ok=True)

    sources = []
    for name, root in cfg["sources"].items():
        if not os.path.isdir(root):
            print(f"[WARN] missing source: {name} at {root}; skipping")
            continue
        sources.append(discover_source(name, root))
    if not sources:
        raise RuntimeError("no sources on disk")
    write_inventory(sources, os.path.join(work_dir, "source_inventory.json"))

    sp = cfg.get("split", {})
    val_users = set(sp.get("val_users", []))
    test_users = set(sp.get("test_users", []))
    # force_random overrides BdSL47's user-disjoint (SI) split with a random (SD)
    # split — this is what makes the T3 signer-dependent measurement possible;
    # without it BdSL47 is ALWAYS user-disjoint regardless of config.
    force_random = bool(sp.get("force_random", False))
    max_train_per_class = int(sp.get("max_train_per_class", 0))  # A4 few-shot; 0 = all
    train_pairs, val_pairs = [], []
    for spec in sources:
        items = enumerate_source(spec)
        held_out = _provided_test_dir(spec.root)
        if (not force_random) and spec.name in ("bdsl47_digits", "bdsl47_letters"):
            tr, va, _te = split_user_disjoint(items, val_users, test_users)
        elif held_out is not None:
            # Author-provided held-out split: train on the whole train/ folder,
            # evaluate on the sibling test/ folder. Avoids the augmentation /
            # random-split leak on datasets that ship their own test set (BSLD_45,
            # BDSL49). NOTE: still signer-dependent when the source has no user IDs.
            tr = items
            eval_spec = SourceSpec(name=spec.name, root=held_out,
                                   num_classes=spec.num_classes,
                                   class_to_idx=spec.class_to_idx)
            va = enumerate_source(eval_spec)
            print(f"[provided-test] {spec.name}: train={len(tr)} eval={len(va)} "
                  f"(held-out {os.path.basename(held_out)})")
        else:
            tr, va, _te = split_random(items, seed=int(sp.get("seed", 0)),
                                       val_frac=float(sp.get("random_val_frac", 0.10)),
                                       test_frac=float(sp.get("random_test_frac", 0.10)))
        tr = cap_per_class(tr, max_train_per_class, seed=int(sp.get("seed", 0)))
        train_pairs.append((spec, tr))
        val_pairs.append((spec, va))

    transform = _build_transforms(int(cfg.get("image_size", 224)))
    cached_tf = _build_cached_transforms()
    img_cache_index, img_cache_dir = load_img_cache()
    if img_cache_index is not None:
        hits = sum(1 for p, _s, _l in
                   [(pp, 0, 0) for _spec, ents in train_pairs for pp, _l, _u in ents]
                   if os.path.normpath(os.path.abspath(p)) in img_cache_index)
        print(f"[img-cache] {len(img_cache_index)} cached entries; "
              f"train hits {hits}/{sum(len(e) for _s, e in train_pairs)}")
    crop_index = None
    if cfg.get("hand_crop_cache"):
        crop_index = _load_crop_index(cfg["hand_crop_cache"],
                                      [s.name for s in sources])
    _ck = dict(img_cache_index=img_cache_index, img_cache_dir=img_cache_dir,
               cached_transform=cached_tf, crop_index=crop_index)
    ds_train = HandshapeDataset(train_pairs, transform=transform, **_ck)
    ds_val = HandshapeDataset(val_pairs, transform=transform, **_ck)

    nw = int(cfg.get("num_workers", 0))
    # persistent_workers keeps the worker pool alive across epochs (avoids the
    # per-epoch respawn that aggravates Windows' shared-memory commitment limit).
    extra = dict(persistent_workers=True, prefetch_factor=2) if nw > 0 else {}
    loader_train = DataLoader(
        ds_train, batch_size=int(cfg.get("batch_size", 64)),
        shuffle=True, num_workers=nw,
        drop_last=True, pin_memory=torch.cuda.is_available(), **extra,
    )
    loader_val = DataLoader(
        ds_val, batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False, num_workers=nw,
        pin_memory=torch.cuda.is_available(), **extra,
    )

    enc_cfg = cfg["encoder"]
    arch = str(enc_cfg.get("arch", "dinov2")).lower()
    if arch.startswith("resnet"):
        from banglahandshape.cnn_baseline import build_cnn
        model = build_cnn(
            num_classes_per_source=ds_train.num_classes_per_source(),
            arch=arch,
            pretrained=bool(enc_cfg.get("pretrained", False)),
            freeze_backbone=bool(enc_cfg.get("freeze_backbone", False)),
        )
    else:
        model = build_dinov2_lora(
            num_classes_per_source=ds_train.num_classes_per_source(),
            timm_name=enc_cfg.get("timm_name", "vit_small_patch14_dinov2.lvd142m"),
            lora_rank=int(enc_cfg.get("lora_rank", 0)),
            lora_alpha=float(enc_cfg.get("lora_alpha", 0.0)),
            lora_dropout=float(enc_cfg.get("lora_dropout", 0.0)),
            lora_targets=enc_cfg.get("lora_targets") or [],
            pretrained=bool(enc_cfg.get("pretrained", True)),
            full_finetune=bool(enc_cfg.get("full_finetune", False)),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[seed {seed}] LoRA replacements={model.num_lora_replacements}; "
          f"train items={len(ds_train)}; val items={len(ds_val)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("base_lr", 5e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )
    num_epoch = int(cfg.get("num_epoch", 5))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_per_source = {i: 0.0 for i in range(len(sources))}
    # Resume from the last epoch checkpoint if this (experiment, seed) was
    # interrupted, so a long run picks up instead of restarting at epoch 0.
    resume_path = os.path.join(work_dir, f"resume_seed{seed}.pt")
    start_epoch = 0
    if os.path.exists(resume_path):
        try:
            ck = torch.load(resume_path, map_location=device)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = int(ck["epoch"])
            best_per_source = ck["best_per_source"]
            print(f"[seed {seed}] resumed from epoch {start_epoch}/{num_epoch}")
        except Exception as e:  # corrupt/incompatible checkpoint -> start fresh
            print(f"[seed {seed}] resume failed ({e}); starting fresh")
    for epoch in range(start_epoch, num_epoch):
        print(f"[seed {seed}] epoch {epoch+1}/{num_epoch}")
        train_one_epoch(model, loader_train, optimizer, device,
                        log_every=int(cfg.get("log_every", 50)),
                        grad_clip=float(cfg.get("grad_clip", 1.0)), scaler=scaler)
        accs = evaluate(model, loader_val, device)
        for src_i, acc in accs.items():
            name = ds_train.source_names()[src_i]
            print(f"  val {name}: Top-1 = {acc*100:.2f}%")
            best_per_source[src_i] = max(best_per_source[src_i], acc)
        # Full-state resume checkpoint (overwritten each epoch).
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "best_per_source": best_per_source}, resume_path)
        if (epoch + 1) % int(cfg.get("save_interval", 1)) == 0 or epoch == num_epoch - 1:
            ckpt = os.path.join(work_dir, f"encoder_seed{seed}_epoch{epoch+1}.pt")
            torch.save(model.backbone.state_dict(), ckpt)
            print(f"  saved {ckpt}")

    # Write per-source rows to results_final.csv (one row per source).
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for src_i, acc in best_per_source.items():
        name = ds_train.source_names()[src_i]
        exp = f"{base_exp}_{name}_seed{seed}"
        _append_csv_row(results_csv, {
            "Timestamp": timestamp,
            "Experiment": exp,
            "Epoch": num_epoch,
            "Top1_Acc": f"{acc:.6f}",
            "Top5_Acc": "",
            "Top5_Policy": "best_over_epochs",
            "WorkDir": work_dir,
        })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--results-csv", default="results_final.csv")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for seed in args.seeds:
        _train_one_seed(cfg, seed, results_csv=args.results_csv)


if __name__ == "__main__":
    main()
