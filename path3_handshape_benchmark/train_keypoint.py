"""A6 modality control — train the multi-head keypoint MLP on cached MediaPipe
hand keypoints, under the SAME signer-independent protocol as the appearance
encoders (so keypoint_mlp is directly comparable to bdsl47_si).

Reads `work_dir/_kp_cache/<source>.npz` (from `extract_keypoints.py`), applies
the identical split logic as `train_baseline._train_one_seed`, drops
non-detected images, trains a small MLP, and writes one row per (config, source,
seed) to results_final.csv with best-over-epochs Top-1.

Sources whose train split has too few detected keypoints (e.g. BSLD_45 = 0%
detection, drawn-skeleton overlay) are skipped with a logged reason.

Usage (bdsl_graph):
    python -m path3_handshape_benchmark.train_keypoint \
        --config path3_handshape_benchmark/configs/keypoint_mlp.yaml --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bangla_handshape.class_alignment import discover_source, SourceSpec
from bangla_handshape.handshape_dataset import (
    enumerate_source, split_user_disjoint, split_random, cap_per_class,
)
from bangla_handshape.keypoint_baseline import build_keypoint_mlp
from bangla_handshape.train_utils import multihead_loss, evaluate
from path3_handshape_benchmark.train_baseline import (
    _provided_test_dir, _append_csv_row, _init_seed, RESULT_COLUMNS,
)

CACHE_DIR = "work_dir/_kp_cache"
MIN_TRAIN = 20  # skip a source with fewer detected train keypoints


def _kkey(path):
    return os.path.normpath(os.path.abspath(path))


def _load_cache(name):
    f = os.path.join(CACHE_DIR, f"{name}.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    paths = [str(p) for p in d["paths"]]
    kp, det = d["kp"], d["detected"]
    return {_kkey(p): (kp[i], int(det[i])) for i, p in enumerate(paths)}


def _collect(items, cache):
    """items: (path,label,user) -> (X[list], y[list]) for DETECTED entries only."""
    X, y = [], []
    for path, label, _u in items:
        hit = cache.get(_kkey(path))
        if hit is None or hit[1] == 0:
            continue
        X.append(hit[0]); y.append(label)
    return X, y


def _split_for_source(spec, items, sp, seed):
    """Mirror train_baseline's split selection, returning (train_items, eval_items)."""
    val_users = set(sp.get("val_users", []))
    test_users = set(sp.get("test_users", []))
    force_random = bool(sp.get("force_random", False))
    held_out = _provided_test_dir(spec.root)
    if (not force_random) and spec.name in ("bdsl47_digits", "bdsl47_letters"):
        tr, va, _te = split_user_disjoint(items, val_users, test_users)
    elif held_out is not None:
        eval_spec = SourceSpec(name=spec.name, root=held_out,
                               num_classes=spec.num_classes,
                               class_to_idx=spec.class_to_idx)
        tr, va = items, enumerate_source(eval_spec)
    else:
        tr, va, _te = split_random(items, seed=int(sp.get("seed", 0)),
                                   val_frac=float(sp.get("random_val_frac", 0.10)),
                                   test_frac=float(sp.get("random_test_frac", 0.10)))
    tr = cap_per_class(tr, int(sp.get("max_train_per_class", 0)), seed=int(sp.get("seed", 0)))
    return tr, va


def _train_one_seed(cfg, seed, results_csv="results_final.csv"):
    _init_seed(seed)
    base_exp = cfg.get("Experiment_name", "bhc_keypoint_mlp")
    work_dir = cfg.get("work_dir", f"./work_dir/{base_exp}")
    os.makedirs(work_dir, exist_ok=True)
    sp = cfg.get("split", {})

    kept_names, tr_X, tr_s, tr_y, va_X, va_s, va_y, ncls = [], [], [], [], [], [], [], []
    for name, root in cfg["sources"].items():
        if not os.path.isdir(root):
            print(f"[WARN] missing source {name} at {root}; skip"); continue
        cache = _load_cache(name)
        if cache is None:
            print(f"[WARN] no keypoint cache for {name}; run extract_keypoints.py; skip")
            continue
        spec = discover_source(name, root)
        items = enumerate_source(spec)
        tr_items, va_items = _split_for_source(spec, items, sp, seed)
        Xt, yt = _collect(tr_items, cache)
        Xv, yv = _collect(va_items, cache)
        det_tr = len(Xt)
        if det_tr < MIN_TRAIN or len(Xv) == 0:
            print(f"[skip] {name}: detected train={det_tr}, eval={len(Xv)} "
                  f"(too few keypoints — likely drawn-overlay/no-hand source)")
            continue
        s = len(kept_names)
        kept_names.append(name); ncls.append(spec.num_classes)
        tr_X += Xt; tr_s += [s] * len(Xt); tr_y += yt
        va_X += Xv; va_s += [s] * len(Xv); va_y += yv
        print(f"[keypoint] {name}: train={len(Xt)} eval={len(Xv)} "
              f"classes={spec.num_classes}")

    if not kept_names:
        raise RuntimeError("no source had usable keypoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def _ds(X, s, y):
        return TensorDataset(torch.tensor(np.asarray(X, dtype=np.float32)),
                             torch.tensor(np.asarray(s, dtype=np.int64)),
                             torch.tensor(np.asarray(y, dtype=np.int64)))
    bs = int(cfg.get("batch_size", 256))
    dl_tr = DataLoader(_ds(tr_X, tr_s, tr_y), batch_size=bs, shuffle=True, drop_last=True)
    dl_va = DataLoader(_ds(va_X, va_s, va_y), batch_size=bs, shuffle=False)

    mcfg = cfg.get("mlp", {})
    model = build_keypoint_mlp(ncls, in_dim=63,
                               hidden=int(mcfg.get("hidden", 256)),
                               depth=int(mcfg.get("depth", 3)),
                               dropout=float(mcfg.get("dropout", 0.2))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("base_lr", 1e-3)),
                            weight_decay=float(cfg.get("weight_decay", 1e-2)))
    num_epoch = int(cfg.get("num_epoch", 50))
    best = {i: 0.0 for i in range(len(kept_names))}
    for epoch in range(num_epoch):
        model.train()
        for x, s, y in dl_tr:
            x, s, y = x.to(device), s.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = multihead_loss(model(x, s), y)
            loss.backward(); opt.step()
        accs = evaluate(model, dl_va, device)
        for i, a in accs.items():
            best[i] = max(best[i], a)
        if (epoch + 1) % 10 == 0 or epoch == num_epoch - 1:
            print(f"[seed {seed}] epoch {epoch+1}/{num_epoch} "
                  + " ".join(f"{kept_names[i]}={best[i]*100:.1f}" for i in best))

    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for i, name in enumerate(kept_names):
        _append_csv_row(results_csv, {
            "Timestamp": ts, "Experiment": f"{base_exp}_{name}_seed{seed}",
            "Epoch": num_epoch, "Top1_Acc": f"{best[i]:.6f}", "Top5_Acc": "",
            "Top5_Policy": "best_over_epochs", "WorkDir": work_dir,
        })
    print(f"=== DONE keypoint_mlp seed{seed} ===")


def main():
    ap = argparse.ArgumentParser()
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
