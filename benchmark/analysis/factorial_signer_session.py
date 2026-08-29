"""A: factorial signer / session-proxy design (reviewer major-revision, PLAN §A).

The manuscript's Eq. 1 wrote the SD->SI gap as a *sum* of separately-estimated
signer, session, and data terms. The reviewer's point (§1/§2): that decomposition
is **not identified** -- the paired estimator confounds the three, so the 28.2 pp
is an upper bound, not a controlled signer effect. This module runs the controlled
factorial that *is* identifiable from BdSL47, and hands the per-cell accuracies to
``tools/mixed_effects.py`` for the coefficient / variance-component table.

Design (fixed test block per held-out signer ``t``; **training-set size matched**
across all conditions -- when we add ``n`` frames of ``t`` we remove ``n``
class-matched frames of the other signers):

    cond  signer-seen  added-frames-of-t
    A     no           --                     (SI baseline)
    B     yes          FAR block  (temporally separated  = pseudo *other session*)
    D     yes          NEAR block (adjacent to test      = same-burst / near-dup)

  beta1  = A(B) - A(A)  -> controlled signer-exposure effect (clean-ish)
  beta2  = A(D) - A(B)  -> extra lift from adjacent/near-dup frames (session/burst)
  sigma^2_signer        -> held-out-signer difficulty (random intercept)

HONEST LIMITATIONS (stated in the paper, per the plan):
  * BdSL47 is ~single-session per signer, so "FAR block" is a *temporal-separation
    proxy* for an independent session, not a true one. A clean beta1 needs a
    repeated-session corpus (that is §B / RSBdSL38, future collection).
  * The 4th cell (signer-unseen + near-dup, "C") needs *cross-signer* near-duplicate
    detection -- that is §C's job -- so it is omitted here and the design is nested
    (dup is measured only within signer-seen). No full 2x2 interaction is claimed.

The classifier is a head on FROZEN backbone features (``--timm-name``, optionally a
LoRA ``--weights`` checkpoint): the factorial then reads as "does adding signer t's
frames to the head's training data recover accuracy on t?", cheap enough to run the
whole LOUO x condition x seed grid locally. Report it as a frozen-feature probe.

Example:
  python -m benchmark.analysis.factorial_signer_session \\
      --source bdsl47_letters --timm-name vit_base_patch14_dinov2.lvd142m \\
      --seeds 0 1 2 --output results/A_factorial_letters.csv
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from banglahandshape.class_alignment import discover_source
from banglahandshape.handshape_dataset import HandshapeDataset, enumerate_source
from banglahandshape.dinov2_lora import build_dinov2_lora
from benchmark.baselines.train_baseline import _build_transforms

DEFAULT_ROOTS = {
    "bdsl47_digits": "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}
_SAMPLE_RE = re.compile(r"\((\d+)\)")   # "Sign 0 - Sample (123).jpg" -> 123 (temporal order)


# --------------------------------------------------------------------------- #
def _temporal_key(path):
    """Integer temporal order from the '(k)' sample index; fall back to the name."""
    m = _SAMPLE_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else 0


@torch.no_grad()
def _extract_features(model, entries, spec, transform, batch_size, device):
    """Frozen features in `entries` order (shuffle=False) so they pair with labels."""
    ds = HandshapeDataset([(spec, entries)], transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=torch.cuda.is_available())
    feats = []
    model.eval()
    for x, _src, _y in loader:
        feats.append(model.features(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(feats)


def _fit_head(Xtr, ytr, Xte, yte, num_classes, device, epochs=120, seed=0):
    """Train a linear handshape head on frozen features; return top-1 on the test block."""
    torch.manual_seed(seed)
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte = torch.tensor(Xte, dtype=torch.float32, device=device)
    net = nn.Linear(Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n, bs = len(Xtr), 256
    for _ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(net(Xtr[idx]), ytr[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte).argmax(1).cpu().numpy()
    return float((pred == np.asarray(yte)).mean())


# --------------------------------------------------------------------------- #
def _blocks_for_signer(idx_by_class, paths, test_frac, add_frac):
    """Per class, order signer t's frames by temporal key and carve
    FAR (earliest add_frac) / NEAR (add_frac just before test) / TEST (last test_frac).
    Returns three flat index arrays (into the global feature matrix)."""
    far, near, test = [], [], []
    for _c, idxs in idx_by_class.items():
        order = sorted(idxs, key=lambda i: _temporal_key(paths[i]))
        n = len(order)
        n_test = max(1, int(round(n * test_frac)))
        n_add = max(1, int(round(n * add_frac)))
        test_blk = order[n - n_test:]
        near_blk = order[max(0, n - n_test - n_add):n - n_test]
        far_blk = order[:min(n_add, max(0, n - n_test - n_add))]
        test.extend(test_blk); near.extend(near_blk); far.extend(far_blk)
    return np.array(far, int), np.array(near, int), np.array(test, int)


def _size_matched_base(base_idx, classes, add_idx, rng):
    """Remove |add_idx|, class-matched, from the base pool so every condition trains
    on the same number of frames with the same class histogram."""
    if len(add_idx) == 0:
        return base_idx
    drop = []
    add_by_c = {}
    for i in add_idx:
        add_by_c[classes[i]] = add_by_c.get(classes[i], 0) + 1
    base_by_c = {}
    for i in base_idx:
        base_by_c.setdefault(classes[i], []).append(i)
    for c, k in add_by_c.items():
        pool = base_by_c.get(c, [])
        if not pool:
            continue
        pool = list(pool); rng.shuffle(pool)
        drop.extend(pool[:min(k, len(pool))])
    drop = set(drop)
    return np.array([i for i in base_idx if i not in drop], int)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = args.source or ["bdsl47_digits", "bdsl47_letters"]
    transform = _build_transforms(args.image_size)
    rows = []

    for name in sources:
        root = DEFAULT_ROOTS[name]
        if not os.path.isdir(root):
            print(f"[skip] {name}: {root} not found", flush=True)
            continue
        spec = discover_source(name, root)
        entries = enumerate_source(spec)                      # (path, class, user)
        paths = [e[0] for e in entries]
        classes = np.array([e[1] for e in entries])
        signers = np.array([e[2] for e in entries])
        K = spec.num_classes

        # one frozen forward pass over the whole source
        model = build_dinov2_lora([K], timm_name=args.timm_name,
                                  lora_rank=args.lora_rank, lora_targets=args.lora_targets,
                                  pretrained=True).to(device)
        if args.weights:
            sd = torch.load(args.weights, map_location="cpu")
            model.backbone.load_state_dict(sd, strict=False)
            print(f"[weights] loaded {args.weights} into backbone", flush=True)
        X = _extract_features(model, entries, spec, transform, args.batch_size, device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{name}] features {X.shape}; signers={sorted(set(signers.tolist()))}", flush=True)

        for t in sorted(set(signers.tolist())):
            t_mask = signers == t
            t_idx = np.where(t_mask)[0]
            idx_by_class = {}
            for i in t_idx:
                idx_by_class.setdefault(classes[i], []).append(i)
            far, near, test = _blocks_for_signer(idx_by_class, paths,
                                                 args.test_frac, args.add_frac)
            base_all = np.where(~t_mask)[0]                    # every other signer's frames
            yte = classes[test]
            if len(test) == 0 or len(base_all) == 0:
                continue
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                # A: signer unseen. Size-match to the max add so all conds equal size.
                n_add = max(len(far), len(near))
                baseA = _size_matched_base(base_all, classes,
                                           rng.choice(base_all, size=min(n_add, len(base_all)),
                                                      replace=False), rng)
                conds = {
                    "A": baseA,
                    "B": np.concatenate([_size_matched_base(base_all, classes, far, rng), far]),
                    "D": np.concatenate([_size_matched_base(base_all, classes, near, rng), near]),
                }
                for cond, tr_idx in conds.items():
                    acc = _fit_head(X[tr_idx], classes[tr_idx], X[test], yte, K,
                                    device, epochs=args.epochs, seed=seed)
                    rows.append(dict(source=name, cond=cond, signer=int(t), seed=int(seed),
                                     n_train=int(len(tr_idx)), n_test=int(len(test)),
                                     top1=round(acc * 100, 3)))
                    print(f"  [{name}] t={t:>2} seed={seed} cond={cond} "
                          f"n_tr={len(tr_idx)} top1={acc*100:.2f}", flush=True)

    if args.output and rows:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        import csv
        with open(args.output, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[written] {args.output}  ({len(rows)} rows)", flush=True)
        print(f"  next: python tools/mixed_effects.py --csv {args.output}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", choices=list(DEFAULT_ROOTS))
    ap.add_argument("--timm-name", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lora-targets", nargs="*", default=[])
    ap.add_argument("--weights", default=None, help="optional backbone state_dict (.pt)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--add-frac", type=float, default=0.30)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
