"""A6 robustness — leave-one-user-out (LOUO) for the keypoint MLP on BdSL47.

Rotates the held-out signer across all 10 BdSL47 users; for each fold, trains on
the other 9 and evaluates on the held-out one. Reports per-source Top-1 mean±std
ACROSS SIGNERS — the honest signer-noise variance the design doc mandates
(kills the "you picked an easy test signer" critique behind the single-fold
100%/85% numbers).

Uses FINAL-epoch accuracy (not best-over-epochs) so there is no epoch-peeking on
the held-out fold. Cheap: cached keypoints + tiny MLP, ~10 folds in a few minutes.

Usage (bdsl_graph):
    python -m path3_handshape_benchmark.louo_keypoint --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bangla_handshape.class_alignment import discover_source
from bangla_handshape.handshape_dataset import enumerate_source
from bangla_handshape.keypoint_baseline import build_keypoint_mlp
from bangla_handshape.train_utils import multihead_loss, evaluate
from path3_handshape_benchmark.train_keypoint import _load_cache, _kkey
from path3_handshape_benchmark.train_baseline import _append_csv_row, _init_seed

SOURCES = {
    "bdsl47_digits":  "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}
USERS = list(range(1, 11))


def _arrays_for(items, cache, src_idx):
    X, s, y = [], [], []
    for path, label, _u in items:
        hit = cache.get(_kkey(path))
        if hit is None or hit[1] == 0:
            continue
        X.append(hit[0]); s.append(src_idx); y.append(label)
    return X, s, y


def _run_fold(specs, caches, ncls, held_user, seed, num_epoch, device):
    """Train on users != held_user, eval on held_user. Returns {src_i: final_acc}."""
    _init_seed(seed)
    trX, trS, trY, vaX, vaS, vaY = [], [], [], [], [], []
    for si, (name, spec) in enumerate(specs):
        items = enumerate_source(spec)
        tr_items = [it for it in items if it[2] != held_user]
        va_items = [it for it in items if it[2] == held_user]
        a = _arrays_for(tr_items, caches[name], si)
        b = _arrays_for(va_items, caches[name], si)
        trX += a[0]; trS += a[1]; trY += a[2]
        vaX += b[0]; vaS += b[1]; vaY += b[2]

    def _ds(X, s, y):
        return TensorDataset(torch.tensor(np.asarray(X, np.float32)),
                             torch.tensor(np.asarray(s, np.int64)),
                             torch.tensor(np.asarray(y, np.int64)))
    dl_tr = DataLoader(_ds(trX, trS, trY), batch_size=256, shuffle=True, drop_last=True)
    dl_va = DataLoader(_ds(vaX, vaS, vaY), batch_size=256, shuffle=False)

    model = build_keypoint_mlp(ncls, in_dim=63, hidden=256, depth=3, dropout=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    accs = {}
    for _ in range(num_epoch):
        model.train()
        for x, s, y in dl_tr:
            x, s, y = x.to(device), s.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            multihead_loss(model(x, s), y).backward(); opt.step()
        accs = evaluate(model, dl_va, device)   # final-epoch (overwrites each epoch)
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--num-epoch", type=int, default=50)
    ap.add_argument("--results-csv", default="results_final.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs, caches, ncls = [], {}, []
    for name, root in SOURCES.items():
        spec = discover_source(name, root)
        cache = _load_cache(name)
        if cache is None:
            sys.exit(f"no keypoint cache for {name}; run extract_keypoints.py first")
        specs.append((name, spec)); caches[name] = cache; ncls.append(spec.num_classes)

    # per_source[name] = list of per-(user,seed) accuracies
    per_source = {name: [] for name, _ in specs}
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for seed in args.seeds:
        for u in USERS:
            accs = _run_fold(specs, caches, ncls, u, seed, args.num_epoch, device)
            for si, (name, _spec) in enumerate(specs):
                a = accs.get(si, 0.0)
                per_source[name].append(a)
                _append_csv_row(args.results_csv, {
                    "Timestamp": ts,
                    "Experiment": f"bhc_keypoint_mlp_louo_{name}_testuser{u:02d}_seed{seed}",
                    "Epoch": args.num_epoch, "Top1_Acc": f"{a:.6f}", "Top5_Acc": "",
                    "Top5_Policy": "final_epoch", "WorkDir": "./work_dir/bhc_keypoint_mlp_louo",
                })
            print(f"[seed {seed}] held user {u:2d}: "
                  + " ".join(f"{specs[si][0]}={accs.get(si,0)*100:5.1f}" for si in range(len(specs))),
                  flush=True)

    print("\n=== LOUO summary (Top-1 mean +/- std across signers) ===")
    for name in per_source:
        v = np.array(per_source[name]) * 100
        print(f"  {name:16s} {v.mean():5.2f} +/- {v.std():4.2f}  "
              f"(min {v.min():.1f}, max {v.max():.1f}, folds={len(v)})")


if __name__ == "__main__":
    main()
