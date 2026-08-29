"""E: signer-decodability probe battery (reviewer major-revision, EXPERIMENT_PLAN §E).

Measures how recoverable *signer identity* is from a representation's frozen features
on BdSL47 -- the quantity the manuscript labelled ``I(Phi)``. The reviewer's point 8/12:
that name overclaims (a linear probe is not mutual information) and a nonlinear probe
may recover far more. So this reports THREE decodability numbers per representation,
all as balanced accuracy (mean per-signer recall; chance = 1/#signers):

  * linear             -- single Linear layer  (the paper's original probe)
  * mlp                -- 2-layer MLP           (reviewer's mandatory nonlinear probe)
  * class-conditional  -- per handshape class, fit a signer probe, average
                          ("is identity recoverable AFTER fixing the handshape?")

If linear << mlp, the "identity removed" story is wrong -- which is the whole point of
running it. The representation is a frozen timm backbone (``--timm-name``), optionally
with a LoRA checkpoint loaded (``--lora-rank/--lora-targets/--weights``) so adapted /
pose-distilled / adversarial encoders can be probed too.

Needs a GPU + BdSL47 on disk. Example:
  python -m path3_handshape_benchmark.signer_probe \\
      --source bdsl47_letters --timm-name vit_base_patch14_dinov2.lvd142m \\
      --val-users 4 --test-users 5 --output results/E_decodability_dinov2b.md
  # adapted encoder:
  ... --lora-rank 8 --lora-targets attn.qkv attn.proj mlp.fc1 mlp.fc2 --weights <ckpt.pt>
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bangla_handshape.class_alignment import discover_source
from bangla_handshape.handshape_dataset import HandshapeDataset, enumerate_source
from bangla_handshape.dinov2_lora import build_dinov2_lora
from path3_handshape_benchmark.train_baseline import _build_transforms

DEFAULT_ROOTS = {
    "bdsl47_digits": "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}


# --------------------------------------------------------------------------- #
def balanced_accuracy(y_true, y_pred):
    """Mean per-class recall (chance = 1/#classes)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    recalls = []
    for c in np.unique(y_true):
        m = y_true == c
        if m.sum():
            recalls.append((y_pred[m] == c).mean())
    return float(np.mean(recalls)) if recalls else 0.0


def _stratified_idx(groups, frac_test=0.30, seed=0):
    """Per-group (signer) train/test index split, so every signer is in both."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * frac_test)))
        te.extend(idx[:k]); tr.extend(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(te))


@torch.no_grad()
def _extract_features(model, entries, spec, transform, batch_size, device):
    """Features in `entries` order (shuffle=False), so they pair with entries' user ids."""
    ds = HandshapeDataset([(spec, entries)], transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=torch.cuda.is_available())
    feats = []
    model.eval()
    for x, _src, _y in loader:
        feats.append(model.features(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(feats)


def _fit_probe(X, y, kind, device, epochs=200, seed=0):
    """Train a signer classifier; return balanced accuracy on a held-out split."""
    y = np.asarray(y)
    uniq = np.unique(y)
    if len(uniq) < 2:
        return float("nan")
    remap = {u: i for i, u in enumerate(uniq)}
    y = np.array([remap[v] for v in y])
    tr, te = _stratified_idx(y, seed=seed)
    torch.manual_seed(seed)
    Xtr = torch.tensor(X[tr], dtype=torch.float32, device=device)
    ytr = torch.tensor(y[tr], dtype=torch.long, device=device)
    Xte = torch.tensor(X[te], dtype=torch.float32, device=device)
    D, K = X.shape[1], len(uniq)
    if kind == "linear":
        net = nn.Linear(D, K)
    elif kind == "mlp":
        net = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, K))
    else:
        raise ValueError(kind)
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n, bs = len(tr), 256
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
    return balanced_accuracy(y[te], pred)


def class_conditional_decodability(X, signers, classes, device, seed=0):
    """Per handshape class, fit a linear signer probe; average balanced accuracy.
    Answers: is signer identity recoverable once the handshape is held fixed?"""
    accs = []
    for c in np.unique(classes):
        m = classes == c
        if len(np.unique(signers[m])) < 2 or m.sum() < 20:
            continue
        accs.append(_fit_probe(X[m], signers[m], "linear", device, epochs=150, seed=seed))
    return float(np.nanmean(accs)) if accs else float("nan")


# --------------------------------------------------------------------------- #
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sources = args.source or ["bdsl47_digits", "bdsl47_letters"]
    val_users = set(args.val_users); test_users = set(args.test_users)
    transform = _build_transforms(args.image_size)

    rows = []
    for name in sources:
        root = DEFAULT_ROOTS[name]
        if not os.path.isdir(root):
            print(f"[skip] {name}: {root} not found", flush=True)
            continue
        spec = discover_source(name, root)
        entries = enumerate_source(spec)                      # (path, class, user)
        all_users = sorted({u for _, _, u in entries})
        train_users = [u for u in all_users if u not in val_users and u not in test_users]
        tr_entries = [e for e in entries if e[2] in train_users]
        if not tr_entries:
            print(f"[skip] {name}: no training-signer images", flush=True)
            continue

        # one frozen forward pass over the training-signer images
        model = build_dinov2_lora([spec.num_classes], timm_name=args.timm_name,
                                  lora_rank=args.lora_rank, lora_targets=args.lora_targets,
                                  pretrained=True).to(device)
        if args.weights:
            sd = torch.load(args.weights, map_location="cpu")
            model.backbone.load_state_dict(sd, strict=False)
            print(f"[weights] loaded {args.weights} into backbone", flush=True)
        X = _extract_features(model, tr_entries, spec, transform, args.batch_size, device)
        signers = np.array([e[2] for e in tr_entries])
        classes = np.array([e[1] for e in tr_entries])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        chance = 100.0 / len(train_users)
        lin = _fit_probe(X, signers, "linear", device, seed=args.seed) * 100
        mlp = _fit_probe(X, signers, "mlp", device, seed=args.seed) * 100
        cc = class_conditional_decodability(X, signers, classes, device, seed=args.seed) * 100
        print(f"[{name}] signers={len(train_users)} chance={chance:.1f}%  "
              f"linear={lin:.1f}  mlp={mlp:.1f}  class-cond={cc:.1f}", flush=True)
        rows.append((name, len(train_users), chance, lin, mlp, cc))

    if args.output and rows:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            fh.write(f"# Signer decodability (balanced acc %) -- {args.timm_name}"
                     f"{' + LoRA' if args.lora_rank else ''}\n\n")
            fh.write("| Source | #signers | chance | linear | MLP (nonlinear) | class-cond |\n")
            fh.write("|---|---|---|---|---|---|\n")
            for n, ns, ch, li, ml, cc in rows:
                fh.write(f"| {n} | {ns} | {ch:.1f} | {li:.1f} | {ml:.1f} | {cc:.1f} |\n")
        print(f"[written] {args.output}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", nargs="+", choices=list(DEFAULT_ROOTS))
    ap.add_argument("--timm-name", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lora-targets", nargs="*", default=[])
    ap.add_argument("--weights", default=None, help="optional backbone state_dict (.pt)")
    ap.add_argument("--val-users", type=int, nargs="*", default=[4])
    ap.add_argument("--test-users", type=int, nargs="*", default=[5])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
