"""SF1 — per-class confusion heatmap for the sister paper.

Renders an N×N confusion matrix on one source's evaluation split using a
trained encoder. Default target is BDSL49 (49×49), the runbook's SF1 figure.

Why a re-fit head: `train_baseline.py` checkpoints ONLY the backbone
(`model.backbone.state_dict()`), not the per-source classification heads — so we
cannot replay the exact trained head. Consistent with `eval_cross_dataset.py`,
we load the (LoRA-adapted or frozen) backbone, fit a fresh logistic-regression
head on the target source's TRAIN features, and predict its eval split. The
confusion *structure* is driven by the encoder, so this faithfully shows which
handshapes the representation confuses. Pass --encoder-dir to use a trained
encoder, or omit it to profile the raw pretrained backbone (no fine-tuning).

Usage (bdsl_graph):
    python -m path3_handshape_benchmark.plot_confusion \
        --source bdsl49_recognition \
        --encoder-dir work_dir/bhc_lora --seed 0 --epoch 50 \
        --lora-rank 8 --lora-targets attn.qkv attn.proj mlp.fc1 mlp.fc2 \
        --output results/SF1_confusion_bdsl49.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bangla_handshape.class_alignment import discover_source, SourceSpec
from bangla_handshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, split_user_disjoint, split_random,
)
from bangla_handshape.dinov2_lora import build_dinov2_lora
from path3_handshape_benchmark.train_baseline import (
    _provided_test_dir, _build_transforms,
)
from path3_handshape_benchmark.train_probe_cached import _extract

DEFAULT_ROOTS = {
    "bdsl_mnist":         "data/BdSL-MNIST",
    "bdsl47_digits":      "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters":     "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
    "bsld_45":            "data/BSLD_45/Train",
    "bdsl49_recognition": "data/bdsl49_extracted/Recognition_1/Recognition_1/train",
}


def _split_for(spec, val_users, test_users, seed):
    """Same train/eval partition as train_baseline, so the figure matches the table."""
    items = enumerate_source(spec)
    held = _provided_test_dir(spec.root)
    if spec.name in ("bdsl47_digits", "bdsl47_letters"):
        tr, va, _te = split_user_disjoint(items, val_users, test_users)
    elif held is not None:
        tr = items
        va = enumerate_source(SourceSpec(spec.name, held, spec.num_classes,
                                         spec.class_to_idx))
    else:
        tr, va, _te = split_random(items, seed=seed, val_frac=0.10, test_frac=0.10)
    return tr, va


def _loader(spec, entries, transform, bs, nw):
    ds = HandshapeDataset([(spec, entries)], transform=transform)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                      pin_memory=torch.cuda.is_available())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="bdsl49_recognition",
                    help="which source to plot (default: bdsl49_recognition -> 49x49)")
    ap.add_argument("--root", default=None, help="override the source root dir")
    ap.add_argument("--encoder-dir", default=None,
                    help="dir with encoder_seed<N>_epoch<E>.pt; omit to use raw pretrained backbone")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--timm-name", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-targets", nargs="+",
                    default=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
    ap.add_argument("--val-users", nargs="+", type=int, default=[4])
    ap.add_argument("--test-users", nargs="+", type=int, default=[5])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--normalize", action="store_true",
                    help="row-normalize (per true-class) so the diagonal reads as recall")
    ap.add_argument("--output", default=None,
                    help="PNG path (default results/SF1_confusion_<source>.png)")
    args = ap.parse_args()

    root = args.root or DEFAULT_ROOTS.get(args.source)
    if not root or not os.path.isdir(root):
        sys.exit(f"source root not found: {root!r} — pass --root or re-fetch data")
    spec = discover_source(args.source, root)
    print(f"source {spec.name}: {spec.num_classes} classes  root={root}")

    tr, va = _split_for(spec, set(args.val_users), set(args.test_users), args.seed)
    print(f"  train={len(tr)}  eval={len(va)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_lora = args.encoder_dir is not None
    model = build_dinov2_lora(
        num_classes_per_source=[spec.num_classes],
        timm_name=args.timm_name,
        lora_rank=(args.lora_rank if use_lora else 0),
        lora_alpha=args.lora_alpha,
        lora_targets=(args.lora_targets if use_lora else []),
        pretrained=True,
    )
    if use_lora:
        ckpt = os.path.join(args.encoder_dir,
                            f"encoder_seed{args.seed}_epoch{args.epoch}.pt")
        if not os.path.isfile(ckpt):
            sys.exit(f"checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location="cpu")
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        print(f"  loaded {os.path.basename(ckpt)} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    model = model.to(device)

    transform = _build_transforms(args.image_size)
    Xtr, ytr = _extract(model, _loader(spec, tr, transform, args.batch_size, args.num_workers), device)
    Xva, yva = _extract(model, _loader(spec, va, transform, args.batch_size, args.num_workers), device)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(Xtr, ytr)
    preds = clf.predict(Xva)
    acc = float((preds == yva).mean())
    print(f"  re-fit head eval Top-1 = {acc*100:.2f}%")

    n = spec.num_classes
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(yva, preds):
        cm[int(t), int(p)] += 1

    disp = cm.astype(np.float64)
    if args.normalize:
        row = disp.sum(axis=1, keepdims=True)
        disp = np.divide(disp, row, out=np.zeros_like(disp), where=row > 0)

    out = args.output or f"results/SF1_confusion_{spec.name}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axm = plt.subplots(figsize=(max(6, n * 0.22), max(5, n * 0.22)))
    im = axm.imshow(disp, cmap="viridis", vmin=0,
                    vmax=(1.0 if args.normalize else None))
    axm.set_xlabel("predicted class")
    axm.set_ylabel("true class")
    axm.set_title(f"{spec.name}  ({n}×{n})  Top-1={acc*100:.1f}%"
                  + ("  [row-normalized]" if args.normalize else ""))
    fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04)
    ticks = np.arange(n)
    if n <= 50:
        axm.set_xticks(ticks); axm.set_yticks(ticks)
        labels = list(spec.class_to_idx.keys())
        axm.set_xticklabels(labels, rotation=90, fontsize=5)
        axm.set_yticklabels(labels, fontsize=5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    npy = os.path.splitext(out)[0] + "_matrix.npy"
    np.save(npy, cm)
    print(f"wrote {out}\nwrote {npy}  (raw counts)")


if __name__ == "__main__":
    main()
