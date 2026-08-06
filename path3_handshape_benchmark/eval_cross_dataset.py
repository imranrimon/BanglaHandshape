"""Audit fix #3: cross-dataset transfer matrix for the sister paper (S2).

Per-source label spaces in our four handshape datasets are DISJOINT (37 vs
49 vs 45 vs 10 vs 37 classes; the folder-name "0" in source A is not
guaranteed to mean the same thing as "0" in source B). "Train on A, classify
B" without re-fitting a head is therefore not a well-defined operation.

Protocol we will actually defend in the paper:

  1. For each *encoder* trained on source A (or jointly on all sources),
     freeze the encoder.
  2. For every target source B (including A=B as upper bound):
     a. Extract pooled features on B's TRAIN and VAL sets using the frozen encoder.
     b. Fit a fresh scikit-learn LogisticRegression on B's TRAIN features.
     c. Eval B's VAL Top-1 with the fitted head.
  3. Write an N x N markdown matrix.

This decouples encoder-transfer quality (what the figure is about) from head
re-training (which is necessary because label spaces are disjoint).

Usage:
    python -m path3_handshape_benchmark.eval_cross_dataset ^
        --encoder-dir work_dir/bhc_lora ^
        --epoch 10 --seed 0 ^
        --output results/S2_transfer_matrix.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bangla_handshape.class_alignment import discover_default
from bangla_handshape.dinov2_lora import build_dinov2_lora
from bangla_handshape.handshape_dataset import (
    HandshapeDataset,
    enumerate_source,
    split_user_disjoint,
    split_random,
)


def _build_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])


def _per_source_splits(sources, val_users, test_users, seed):
    """Return {source_name: (spec, train_entries, val_entries)}.

    Mirrors the protocol used in path3_handshape_benchmark.train_baseline so
    the train/val partition is identical (otherwise S1 and S2 would not be
    comparable).
    """
    out = {}
    for spec in sources:
        items = enumerate_source(spec)
        if spec.name in ("bdsl47_digits", "bdsl47_letters"):
            tr, va, _te = split_user_disjoint(items, val_users, test_users)
        else:
            tr, va, _te = split_random(
                items, seed=seed, val_frac=0.10, test_frac=0.10
            )
        out[spec.name] = (spec, tr, va)
    return out


@torch.no_grad()
def _extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    for batch in loader:
        x, _src_idx, y = batch
        x = x.to(device, non_blocking=True)
        f = model.features(x)
        feats.append(f.detach().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def _eval_one_encoder(model, src_to_loaders, device):
    """For one frozen encoder, return {target_source: Top-1}."""
    from sklearn.linear_model import LogisticRegression

    per_target = {}
    for target_name, (train_loader, val_loader) in src_to_loaders.items():
        tr_x, tr_y = _extract_features(model, train_loader, device)
        va_x, va_y = _extract_features(model, val_loader, device)
        # n_jobs=-1 only honoured if the underlying BLAS is parallel; fine on Windows.
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(tr_x, tr_y)
        per_target[target_name] = float(clf.score(va_x, va_y))
    return per_target


def _resolve_checkpoint(encoder_dir, src_a, seed, epoch):
    """Try (in order) source-specific then combined checkpoint filenames."""
    candidates = [
        os.path.join(encoder_dir, f"encoder_{src_a}_seed{seed}_epoch{epoch}.pt"),
        os.path.join(encoder_dir, f"encoder_seed{seed}_epoch{epoch}.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--encoder-dir",
        required=True,
        help="directory holding encoder_*.pt checkpoints "
             "(either per-source: encoder_<name>_seed<N>_epoch<E>.pt, "
             "or combined: encoder_seed<N>_epoch<E>.pt)",
    )
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-users", nargs="+", type=int, default=[4])
    ap.add_argument("--test-users", nargs="+", type=int, default=[5])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--lora-rank", type=int, default=4,
        help="must match the LoRA rank the encoder was trained with",
    )
    ap.add_argument("--lora-alpha", type=float, default=8.0)
    ap.add_argument(
        "--lora-targets", nargs="+",
        default=["attn.qkv", "attn.proj"],
        help="must match the LoRA targets used during encoder training",
    )
    ap.add_argument("--timm-name", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--output", default="results/S2_transfer_matrix.md")
    args = ap.parse_args()

    sources = discover_default(repo_root=".")
    if not sources:
        raise RuntimeError("no sources discovered on disk")
    source_names = [s.name for s in sources]
    print(f"sources discovered: {source_names}")

    splits = _per_source_splits(
        sources, set(args.val_users), set(args.test_users), args.seed
    )

    transform = _build_transforms(args.image_size)
    src_to_loaders = {}
    for name, (spec, tr, va) in splits.items():
        ds_tr = HandshapeDataset([(spec, tr)], transform=transform)
        ds_va = HandshapeDataset([(spec, va)], transform=transform)
        ld_tr = DataLoader(
            ds_tr, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        ld_va = DataLoader(
            ds_va, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        src_to_loaders[name] = (ld_tr, ld_va)
        print(f"  {name}: train={len(ds_tr)} val={len(ds_va)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes_per_source = [s.num_classes for s in sources]

    matrix = {}
    for src_a in source_names:
        ckpt = _resolve_checkpoint(args.encoder_dir, src_a, args.seed, args.epoch)
        if ckpt is None:
            print(f"[skip] no checkpoint for src={src_a}")
            continue
        # Build a fresh backbone with the same LoRA shape and load the state.
        model = build_dinov2_lora(
            num_classes_per_source=num_classes_per_source,
            timm_name=args.timm_name,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_targets=args.lora_targets,
            pretrained=True,
        )
        state = torch.load(ckpt, map_location="cpu")
        # train_baseline saves model.backbone.state_dict(); load_state_dict
        # tolerates the missing head keys via strict=False.
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        if unexpected:
            print(f"  [WARN] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
        model = model.to(device)
        print(f"\n=== encoder trained on A={src_a} (ckpt: {os.path.basename(ckpt)}) ===")

        per_target = _eval_one_encoder(model, src_to_loaders, device)
        matrix[src_a] = per_target
        for b, acc in per_target.items():
            mark = " (diag)" if b == src_a else ""
            print(f"  -> {b}: Top-1 = {acc*100:.2f}%{mark}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write markdown
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(f"# S2 — Cross-dataset handshape transfer matrix\n\n")
        f.write(f"Encoder: DINOv2-S/14 + LoRA(rank={args.lora_rank}). ")
        f.write(f"Seed {args.seed}, epoch {args.epoch}. Per-cell value is Top-1 (%).\n\n")
        f.write(
            "Rows = encoder source A (frozen). "
            "Columns = target source B (fresh logistic-regression head fit "
            "on B's train features, evaluated on B's val). "
            "Diagonal A=B is the upper bound — same source.\n\n"
        )
        f.write("| A \\ B | " + " | ".join(source_names) + " |\n")
        f.write("|---|" + "|".join(["---:"] * len(source_names)) + "|\n")
        for a in source_names:
            row = []
            for b in source_names:
                v = matrix.get(a, {}).get(b)
                row.append(f"{v*100:.2f}" if v is not None else "—")
            f.write(f"| {a} | " + " | ".join(row) + " |\n")
        # Optional: row/column averages so the reader can compare "how transferable"
        # vs "how easy to target" at a glance.
        if matrix:
            f.write("\n## Marginals\n\n")
            f.write("| source | row-mean (off-diag) | col-mean (off-diag) |\n")
            f.write("|---|---:|---:|\n")
            for name in source_names:
                row_vals = [matrix.get(name, {}).get(b) for b in source_names if b != name]
                col_vals = [matrix.get(a, {}).get(name) for a in source_names if a != name]
                row_vals = [v for v in row_vals if v is not None]
                col_vals = [v for v in col_vals if v is not None]
                rm = sum(row_vals) / len(row_vals) if row_vals else None
                cm = sum(col_vals) / len(col_vals) if col_vals else None
                f.write(
                    f"| {name} | "
                    f"{rm*100:.2f} | {cm*100:.2f} |\n"
                    if rm is not None and cm is not None
                    else f"| {name} | — | — |\n"
                )

    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
