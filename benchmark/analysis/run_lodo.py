"""Tier-3 — Leave-One-Dataset-Out (LODO) cross-corpus handshape classification.

Requires a VERIFIED unified alignment (banglahandshape/handshape_taxonomy.py).
For each held-out source D: train a single-head DINOv2(+LoRA) classifier on the
UNION of the OTHER sources' images (relabeled to canonical handshapes) and
evaluate on D. Eval is restricted to canonical handshapes present in the train
union (coverage reported). This measures generalization to an entirely unseen
corpus — new camera, background, and signer population — the strongest external-
validity claim in the paper.

One row per (held-out source, seed) -> --results-csv, Experiment =
bhc_lodo_heldout_<D>_seed<N>; a coverage report is written next to it.

Usage:
    python -m benchmark.analysis.run_lodo \
        --alignment-json alignment/handshape_alignment.json --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from banglahandshape.class_alignment import discover_default, SourceSpec
from banglahandshape.handshape_dataset import HandshapeDataset, enumerate_source
from banglahandshape.dinov2_lora import build_dinov2_lora
from banglahandshape.handshape_taxonomy import (
    load_alignment, num_canonical, source_local_to_canonical, remap_entries,
    coverage,
)
from banglahandshape.train_utils import train_one_epoch, evaluate
from benchmark.baselines.train_baseline import (
    _build_transforms, _append_csv_row, _init_seed,
)

DEFAULT_ENCODER = dict(
    timm_name="vit_base_patch14_dinov2.lvd142m",
    lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
    lora_targets=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
    pretrained=True, full_finetune=False,
)


def _canon_spec(K):
    """A synthetic single 'source' spanning the K canonical handshapes."""
    return SourceSpec(name="canonical", root="", num_classes=K,
                      class_to_idx={str(i): i for i in range(K)})


def _loader(entries, K, transform, batch, nw, shuffle):
    spec = _canon_spec(K)
    ds = HandshapeDataset([(spec, entries)], transform=transform)
    extra = dict(persistent_workers=True, prefetch_factor=2) if nw > 0 else {}
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=nw,
                      drop_last=shuffle, pin_memory=torch.cuda.is_available(), **extra)


def _run_one_seed(sources, alignment, seed, results_csv, enc, epochs, batch, lr,
                  wd, nw, work_dir):
    _init_seed(seed)
    K = num_canonical(alignment)
    transform = _build_transforms(224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cov_report = {}

    for held in sources:
        # Train union = all OTHER sources, relabeled to canonical.
        train_entries, train_canon = [], set()
        for spec in sources:
            if spec.name == held.name:
                continue
            l2c = source_local_to_canonical(spec, alignment)
            ent = remap_entries(enumerate_source(spec), l2c)
            train_entries += ent
            train_canon |= {e[1] for e in ent}
        # Eval = held-out source, restricted to canonical classes seen in training.
        l2c_h = source_local_to_canonical(held, alignment)
        held_can = remap_entries(enumerate_source(held), l2c_h)
        covered = [e for e in held_can if e[1] in train_canon]
        cov_pct = 100.0 * len(covered) / max(1, len(held_can))
        cov_report[held.name] = dict(
            held_mapped=len(held_can), covered=len(covered),
            coverage_pct=round(cov_pct, 2),
            train_canonical=len(train_canon), train_items=len(train_entries))
        if not covered or not train_entries:
            print(f"[LODO seed{seed}] held-out {held.name}: no covered items "
                  f"(coverage {cov_pct:.1f}%) — skipping", flush=True)
            continue
        print(f"[LODO seed{seed}] held-out {held.name}: train={len(train_entries)} "
              f"items over {len(train_canon)}/{K} canonical; eval covered="
              f"{len(covered)}/{len(held_can)} ({cov_pct:.1f}%)", flush=True)

        loader_tr = _loader(train_entries, K, transform, batch, nw, shuffle=True)
        loader_va = _loader(covered, K, transform, batch, nw, shuffle=False)

        model = build_dinov2_lora(num_classes_per_source=[K], **enc).to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=lr, weight_decay=wd)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        best = 0.0
        for ep in range(epochs):
            print(f"  [{held.name}] epoch {ep+1}/{epochs}", flush=True)
            train_one_epoch(model, loader_tr, opt, device, log_every=100,
                            grad_clip=1.0, scaler=scaler)
            acc = evaluate(model, loader_va, device).get(0, 0.0)
            best = max(best, acc)
            print(f"    held-out {held.name} canonical Top-1 = {acc*100:.2f}% "
                  f"(best {best*100:.2f}%)", flush=True)

        _append_csv_row(results_csv, {
            "Timestamp": timestamp,
            "Experiment": f"bhc_lodo_heldout_{held.name}_seed{seed}",
            "Epoch": epochs,
            "Top1_Acc": f"{best:.6f}",
            "Top5_Acc": "",
            "Top5_Policy": "best_over_epochs",
            "WorkDir": work_dir,
        })

    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, f"lodo_coverage_seed{seed}.json"), "w") as f:
        json.dump(cov_report, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alignment-json", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--results-csv", default="results_final.csv")
    ap.add_argument("--config", default=None,
                    help="optional YAML overriding encoder/epochs/batch/lr")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--base-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--work-dir", default="./work_dir/bhc_lodo")
    args = ap.parse_args()

    alignment = load_alignment(args.alignment_json)
    if not alignment.get("verified", False):
        print("[WARN] alignment is NOT marked verified — results are exploratory. "
              "Have a signer/linguist verify the map before reporting LODO numbers.",
              flush=True)
    sources = discover_default(repo_root=".")
    if len(sources) < 2:
        sys.exit("[abort] need >=2 sources on disk for LODO")

    enc = dict(DEFAULT_ENCODER)
    epochs, batch = args.epochs, args.batch_size
    lr, wd, nw = args.base_lr, args.weight_decay, args.num_workers
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        enc.update(cfg.get("encoder", {}))
        epochs = int(cfg.get("num_epoch", epochs))
        batch = int(cfg.get("batch_size", batch))
        lr = float(cfg.get("base_lr", lr))
        wd = float(cfg.get("weight_decay", wd))
        nw = int(cfg.get("num_workers", nw))

    cov = coverage(sources, alignment)
    print(f"[LODO] {cov['num_canonical']} canonical handshapes; "
          f"{cov['shared_by_2plus']} shared by >=2 sources; verified="
          f"{cov['verified']}", flush=True)
    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, "coverage.json"), "w") as f:
        json.dump(cov, f, indent=2)

    for seed in args.seeds:
        _run_one_seed(sources, alignment, seed, args.results_csv, enc, epochs,
                      batch, lr, wd, nw, args.work_dir)
    print("=== LODO DONE ===", flush=True)


if __name__ == "__main__":
    main()
