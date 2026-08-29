"""Tier-2 method — pose⊕RGB fusion for BdSL47 (does a signer-agnostic hand-pose
stream close the appearance model's signer-independent gap?).

On the IDENTICAL split, trains three heads over the same items:
  * appearance : head on FROZEN DINOv2 features (the RGB stream)
  * pose       : head on the 63-d MediaPipe hand-keypoint vector (from
                 data_prep/extract_keypoints.py — signer-agnostic geometry)
  * fusion     : head on the concatenation [feat ⊕ kp]   (the proposed method)

Run the SI config AND the SD config (force_random) and compare the gap per
modality; the headline is  gap_fusion < gap_appearance  (pose recovers SI acc).

Frozen DINOv2 features are extracted ONCE per (source, split) and cached, so the
backbone is never re-forwarded across seeds. Keypoints come from the .npz cache
work_dir/_kp_cache/<source>.npz (aligned by absolute image path). One row per
(modality, source, seed) -> --results-csv, Experiment = <base>_<modality>_<source>_seed<N>
(so tools/summarize_seeds.py aggregates it like any other config).

Usage:
    python -m benchmark.baselines.train_fusion \
        --config benchmark/configs/fusion_bdsl47_si.yaml --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Weights are pre-warmed in the HF cache; compute nodes have no outbound HTTPS.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from banglahandshape.class_alignment import discover_source, SourceSpec
from banglahandshape.handshape_dataset import (
    enumerate_source, split_user_disjoint, split_random, cap_per_class,
)
from benchmark.baselines.train_baseline import (
    _build_transforms, _provided_test_dir, _append_csv_row, _init_seed,
)

KP_CACHE = "work_dir/_kp_cache"
FEAT_CACHE = "work_dir/_fusion_cache"


def _norm(p):
    return os.path.normpath(os.path.abspath(str(p)))


def _kp_lookup(source):
    """abs-path -> (kp[63], detected) from the keypoint cache, or None if absent."""
    f = os.path.join(KP_CACHE, f"{source}.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    paths = [_norm(p) for p in d["paths"]]
    kp = d["kp"].astype(np.float32)
    det = d["detected"].astype(np.uint8)
    return {p: (kp[i], int(det[i])) for i, p in enumerate(paths)}


class _PathDS(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), i


@torch.no_grad()
def _extract_feats(paths, timm_name, image_size, device, batch=64, nw=4):
    import timm
    backbone = timm.create_model(timm_name, pretrained=True, num_classes=0,
                                 dynamic_img_size=True).to(device).eval()
    D = int(getattr(backbone, "num_features", 384))
    ds = _PathDS(paths, _build_transforms(image_size))
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=nw,
                        pin_memory=torch.cuda.is_available())
    feats = np.zeros((len(paths), D), dtype=np.float32)
    for x, idx in loader:
        f = backbone(x.to(device, non_blocking=True)).float().cpu().numpy()
        feats[idx.numpy()] = f
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return feats


def _cached_feats(tag, paths, timm_name, image_size, device, nw):
    """Extract-once frozen features, validated against the exact path list (the
    split is deterministic, so a matching cache is bit-identical). Re-extract on
    any path-list mismatch."""
    os.makedirs(FEAT_CACHE, exist_ok=True)
    short = timm_name.split(".")[0].replace("/", "_")
    out = os.path.join(FEAT_CACHE, f"{tag}_{short}.npz")
    key = np.array([_norm(p) for p in paths])
    if os.path.exists(out):
        d = np.load(out, allow_pickle=True)
        if len(d["paths"]) == len(key) and bool(np.all(d["paths"] == key)):
            return d["feats"].astype(np.float32)
    feats = _extract_feats(paths, timm_name, image_size, device, nw=nw)
    np.savez_compressed(out, feats=feats, paths=key)
    return feats


class _Head(nn.Module):
    """MLP head: in_dim -> [hidden]*depth -> num_classes (depth 0 = linear)."""

    def __init__(self, in_dim, num_classes, hidden=256, depth=2, dropout=0.2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(max(0, depth)):
            layers += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _train_head(Xtr, ytr, Xva, yva, num_classes, device, *, seed,
                epochs=50, lr=1e-3, wd=1e-2, batch=256, hidden=256,
                depth=2, dropout=0.2):
    """Best-over-epochs val Top-1 for a head trained on precomputed vectors."""
    _init_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32).to(device)
    head = _Head(Xtr.shape[1], num_classes, hidden, depth, dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr_t)
    best = 0.0
    for _ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            # BatchNorm needs >1 sample; skip a size-1 tail batch.
            if len(idx) < 2:
                continue
            xb = Xtr_t[idx].to(device)
            yb = ytr_t[idx].to(device)
            opt.zero_grad()
            lossf(head(xb), yb).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(Xva_t).argmax(1).cpu().numpy()
        acc = float((pred == yva).mean()) if len(yva) else 0.0
        best = max(best, acc)
    return best


def _split_source(spec, sp, seed):
    """Mirror train_baseline._train_one_seed exactly, so fusion numbers are
    directly comparable to the appearance baseline (T1/T3). Returns (train_items,
    eval_items) where eval is the VAL partition (what train_baseline reports)."""
    items = enumerate_source(spec)
    force_random = bool(sp.get("force_random", False))
    val_users = set(sp.get("val_users", []))
    test_users = set(sp.get("test_users", []))
    held = _provided_test_dir(spec.root)
    if (not force_random) and spec.name in ("bdsl47_digits", "bdsl47_letters"):
        tr, va, _te = split_user_disjoint(items, val_users, test_users)
    elif held is not None:
        tr = items
        va = enumerate_source(SourceSpec(spec.name, held, spec.num_classes,
                                         spec.class_to_idx))
    else:
        tr, va, _te = split_random(items, seed=int(sp.get("seed", 0)),
                                   val_frac=float(sp.get("random_val_frac", 0.10)),
                                   test_frac=float(sp.get("random_test_frac", 0.10)))
    tr = cap_per_class(tr, int(sp.get("max_train_per_class", 0)),
                       seed=int(sp.get("seed", 0)))
    return tr, va


def _assemble(items, kp, tag, timm_name, image_size, device, nw):
    """-> (feats[N,D], kp[N,63], labels[N], detected[N])."""
    paths = [it[0] for it in items]
    labels = np.array([it[1] for it in items], dtype=np.int64)
    feats = _cached_feats(tag, paths, timm_name, image_size, device, nw)
    KP = np.zeros((len(paths), 63), dtype=np.float32)
    det = np.zeros(len(paths), dtype=np.uint8)
    if kp is not None:
        for i, p in enumerate(paths):
            hit = kp.get(_norm(p))
            if hit is not None:
                KP[i], det[i] = hit[0], hit[1]
    return feats, KP, labels, det


def _run_one_seed(cfg, seed, results_csv):
    base = cfg.get("Experiment_name", "bhc_fusion")
    sp = cfg.get("split", {})
    enc = cfg.get("encoder", {})
    timm_name = enc.get("timm_name", "vit_base_patch14_dinov2.lvd142m")
    image_size = int(enc.get("image_size", cfg.get("image_size", 224)))
    head = cfg.get("head", {})
    hp = dict(epochs=int(cfg.get("num_epoch", 50)),
              lr=float(cfg.get("base_lr", 1e-3)),
              wd=float(cfg.get("weight_decay", 1e-2)),
              batch=int(cfg.get("batch_size", 256)),
              hidden=int(head.get("hidden", 256)),
              depth=int(head.get("depth", 2)),
              dropout=float(head.get("dropout", 0.2)))
    nw = int(cfg.get("num_workers", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_tag = "sd" if bool(sp.get("force_random", False)) else "si"
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for name, root in cfg["sources"].items():
        if not os.path.isdir(root):
            print(f"[WARN] missing source {name} at {root}; skip", flush=True)
            continue
        spec = discover_source(name, root)
        tr_items, va_items = _split_source(spec, sp, seed)
        kp = _kp_lookup(name)
        if kp is None:
            print(f"[WARN] no keypoint cache for {name} "
                  f"(run extract_keypoints.py); appearance-only for this source",
                  flush=True)
        # feature cache key includes the split so SI/SD don't collide
        Xtr, Ktr, ytr, dtr = _assemble(
            tr_items, kp, f"{name}_{split_tag}_train", timm_name, image_size, device, nw)
        Xva, Kva, yva, dva = _assemble(
            va_items, kp, f"{name}_{split_tag}_val", timm_name, image_size, device, nw)
        det_rate = 100.0 * (dtr.sum() + dva.sum()) / max(1, len(dtr) + len(dva))
        print(f"[{name}/{split_tag} seed{seed}] train={len(ytr)} val={len(yva)} "
              f"C={spec.num_classes} kp_det={det_rate:.1f}%", flush=True)

        modalities = {"appearance": (Xtr, Xva)}
        if kp is not None:
            modalities["pose"] = (Ktr, Kva)
            modalities["fusion"] = (np.concatenate([Xtr, Ktr], axis=1),
                                    np.concatenate([Xva, Kva], axis=1))
        for mod, (A, B) in modalities.items():
            acc = _train_head(A, ytr, B, yva, spec.num_classes, device,
                              seed=seed, **hp)
            print(f"    {mod:11s} val Top-1 = {acc*100:.2f}%", flush=True)
            _append_csv_row(results_csv, {
                "Timestamp": timestamp,
                "Experiment": f"{base}_{mod}_{name}_seed{seed}",
                "Epoch": hp["epochs"],
                "Top1_Acc": f"{acc:.6f}",
                "Top5_Acc": "",
                "Top5_Policy": "best_over_epochs",
                "WorkDir": cfg.get("work_dir", f"./work_dir/{base}"),
            })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--results-csv", default="results_final.csv")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg.get("work_dir", "./work_dir/bhc_fusion"), exist_ok=True)
    for seed in args.seeds:
        _run_one_seed(cfg, seed, args.results_csv)
    print("=== FUSION DONE ===", flush=True)


if __name__ == "__main__":
    main()
