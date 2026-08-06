"""Path 1 entry point — LoRA fine-tune DINOv2 on the aggregated Bangla handshape corpus.

Usage:
    python -m path1_bangla_dinov2.train --config path1_bangla_dinov2/configs/train_lora.yaml --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bangla_handshape.class_alignment import discover_source, write_inventory
from bangla_handshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, split_user_disjoint, split_random,
)
from bangla_handshape.dinov2_lora import build_dinov2_lora
from bangla_handshape.train_utils import train_one_epoch, evaluate


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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _init_seed(args.seed)
    work_dir = cfg.get("work_dir", "./work_dir/bdino_lora")
    os.makedirs(work_dir, exist_ok=True)

    # 1) Discover sources and inventory them.
    sources = []
    for name, root in cfg["sources"].items():
        if not os.path.isdir(root):
            print(f"[WARN] missing source: {name} at {root}; skipping")
            continue
        sources.append(discover_source(name, root))
    if not sources:
        raise RuntimeError("no sources found on disk")
    write_inventory(sources, os.path.join(work_dir, "source_inventory.json"))
    for s in sources:
        print(f"source {s.name}: {s.num_classes} classes at {s.root}")

    # 2) Enumerate items and split per-source.
    sp = cfg.get("split", {})
    val_users = set(sp.get("val_users", []))
    test_users = set(sp.get("test_users", []))
    train_pairs, val_pairs = [], []
    for spec in sources:
        items = enumerate_source(spec)
        if spec.name in ("bdsl47_digits", "bdsl47_letters"):
            tr, va, te = split_user_disjoint(items, val_users, test_users)
            print(f"  {spec.name}: train={len(tr)} val={len(va)} test={len(te)} (user-disjoint)")
        else:
            tr, va, te = split_random(items, seed=int(sp.get("seed", 0)),
                                      val_frac=float(sp.get("random_val_frac", 0.10)),
                                      test_frac=float(sp.get("random_test_frac", 0.10)))
            print(f"  {spec.name}: train={len(tr)} val={len(va)} test={len(te)} (random)")
        train_pairs.append((spec, tr))
        val_pairs.append((spec, va))

    # 3) Build datasets and loaders.
    transform = _build_transforms(int(cfg.get("image_size", 224)))
    ds_train = HandshapeDataset(train_pairs, transform=transform,
                                image_size=int(cfg.get("image_size", 224)))
    ds_val = HandshapeDataset(val_pairs, transform=transform,
                              image_size=int(cfg.get("image_size", 224)))
    print(f"train items: {len(ds_train)}  val items: {len(ds_val)}")

    loader_train = DataLoader(
        ds_train, batch_size=int(cfg.get("batch_size", 64)),
        shuffle=True, num_workers=int(cfg.get("num_workers", 0)),
        drop_last=True, pin_memory=torch.cuda.is_available(),
    )
    loader_val = DataLoader(
        ds_val, batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False, num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    # 4) Build model.
    enc_cfg = cfg["encoder"]
    model = build_dinov2_lora(
        num_classes_per_source=ds_train.num_classes_per_source(),
        timm_name=enc_cfg.get("timm_name", "vit_small_patch14_dinov2.lvd142m"),
        lora_rank=int(enc_cfg.get("lora_rank", 8)),
        lora_alpha=float(enc_cfg.get("lora_alpha", 16.0)),
        lora_dropout=float(enc_cfg.get("lora_dropout", 0.0)),
        lora_targets=enc_cfg.get("lora_targets"),
        pretrained=bool(enc_cfg.get("pretrained", True)),
    )
    print(f"LoRA replacements: {model.num_lora_replacements}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train/1e6:.2f}M / {n_total/1e6:.2f}M total ({n_train/n_total:.1%})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 5) Optimizer + cosine scheduler.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("base_lr", 5e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )
    num_epoch = int(cfg.get("num_epoch", 10))

    # 6) Train.
    for epoch in range(num_epoch):
        print(f"epoch {epoch+1}/{num_epoch}")
        train_one_epoch(model, loader_train, optimizer, device,
                        log_every=int(cfg.get("log_every", 50)),
                        grad_clip=float(cfg.get("grad_clip", 1.0)))
        accs = evaluate(model, loader_val, device)
        for src_i, acc in accs.items():
            name = ds_train.source_names()[src_i]
            print(f"  val {name}: Top-1 = {acc*100:.2f}%")
        if (epoch + 1) % int(cfg.get("save_interval", 2)) == 0 or epoch == num_epoch - 1:
            ckpt = os.path.join(work_dir, f"encoder_epoch{epoch+1}.pt")
            # Save backbone-only for downstream feature extraction.
            torch.save(model.backbone.state_dict(), ckpt)
            print(f"  saved {ckpt}")


if __name__ == "__main__":
    main()
