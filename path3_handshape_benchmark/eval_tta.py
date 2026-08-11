"""Tier-2 method — test-time adaptation (Tent) on an unseen BdSL47 signer.

Deployment-realistic story: we have a signer-independent (SI) BdSL47 encoder that
has NEVER seen the test signer. At inference we get a batch of that new signer's
images with NO labels. Can we recover accuracy by adapting to them online?

Tent (Wang et al., ICLR 2021) does exactly this: put the network in train() mode
but freeze everything except the affine (weight/bias) params of every LayerNorm,
then minimize the mean prediction entropy over the unlabeled test batch. Low
entropy = confident predictions; the only thing that moves is the LN affine, so
the feature statistics re-normalize to the new signer's distribution while the
learned representation and the classification head stay fixed.

Protocol (mirrors plot_confusion.py's encoder-load + refit-head pattern, and
eval_cross_dataset.py's disjoint-label handling):

  1. Build the LoRA DINOv2 model exactly as the trained SI config; load the
     BACKBONE-only checkpoint with strict=False (heads are never checkpointed).
  2. For each BdSL47 source: user-disjoint split (val=user 4, test=user 5). Fit a
     FRESH, differentiable torch Linear head on the FROZEN train features so
     entropy gradients can flow through features() to the LN affine params. The
     head is FIXED during adaptation.
  3. Measure SI Top-1 on the held-out TEST signer (user 5) BEFORE adaptation.
  4. Tent: adapt LN affine params by entropy minimization on the SAME unlabeled
     test images (transductive TTA is the standard Tent setting), then re-measure.

We report BOTH the val-user (4) and test-user (5) numbers, but the headline is
the TEST-user before -> after delta. One row per (source, seed) for BEFORE and
AFTER is appended to --results-csv using train_baseline's RESULT_COLUMNS.

Usage (bdsl_graph):
    python -m path3_handshape_benchmark.eval_tta \
        --si-dir work_dir/bhc_bdsl47_si --seed 0 --epoch 0 \
        --sources bdsl47_digits bdsl47_letters \
        --tta-steps 10 --tta-lr 1e-3 --batch-size 64
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import re
import sys

# Keep timm/HF strictly offline (the cluster login node has no internet); must be
# set before timm/transformers get imported anywhere downstream.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bangla_handshape.class_alignment import discover_source
from bangla_handshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, split_user_disjoint,
)
from bangla_handshape.dinov2_lora import build_dinov2_lora
from path3_handshape_benchmark.train_baseline import (
    _build_transforms, _append_csv_row, _init_seed,
)
from path3_handshape_benchmark.train_probe_cached import _extract

# Source-of-truth roots for the two user-aware BdSL47 sources.
SOURCE_ROOTS = {
    "bdsl47_digits":  "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}


def _latest_epoch(si_dir, seed):
    """Auto-detect the highest encoder_seed{seed}_epoch{E}.pt in si_dir, or None."""
    pat = os.path.join(si_dir, f"encoder_seed{seed}_epoch*.pt")
    best_e, best_p = -1, None
    for p in glob.glob(pat):
        m = re.search(rf"encoder_seed{seed}_epoch(\d+)\.pt$", os.path.basename(p))
        if m and int(m.group(1)) > best_e:
            best_e, best_p = int(m.group(1)), p
    return (best_e, best_p) if best_p is not None else (None, None)


def _loader(spec, entries, transform, bs, nw, shuffle=False):
    ds = HandshapeDataset([(spec, entries)], transform=transform)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      pin_memory=torch.cuda.is_available())


def _fit_torch_head(Xtr, ytr, num_classes, device, epochs=300, lr=1e-2):
    """Fit a differentiable nn.Linear head on FROZEN train features.

    A torch head (not sklearn) is required so entropy gradients from the Tent
    objective flow back through features() to the LayerNorm affine params. The
    head is returned in eval mode and is FIXED (grad off) during adaptation.
    """
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    head = nn.Linear(Xtr_t.shape[1], int(num_classes)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n = Xtr_t.shape[0]
    bs = 256
    head.train()
    for _ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(head(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def _collect_ln_params(backbone):
    """Return the affine (weight, bias) params of every LayerNorm in the backbone.

    These are the ONLY params Tent adapts. Everything else — base ViT weights,
    LoRA A/B, the classification head — stays frozen.
    """
    params = []
    for m in backbone.modules():
        if isinstance(m, nn.LayerNorm):
            # build_dinov2_lora's freeze_non_lora() turned grad OFF for everything
            # except the LoRA A/B matrices — including these LN affines. Tent must
            # be able to update them, so re-enable grad here (else the optimizer
            # step is a silent no-op and before == after exactly).
            if m.weight is not None:
                m.weight.requires_grad_(True)
                params.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad_(True)
                params.append(m.bias)
    return params


@torch.no_grad()
def _eval_head(model, head, loader, device):
    """Top-1 of head(features(x)) on a loader. model/head in eval mode."""
    model.eval()
    head.eval()
    correct, total = 0, 0
    for x, _src, y in loader:
        feats = model.features(x.to(device, non_blocking=True))
        preds = head(feats).argmax(1).cpu()
        correct += int((preds == y).sum())
        total += int(y.numel())
    return (correct / total) if total else 0.0


def _adapt_loss(logits, method, div_weight):
    """Return (loss, mean_entropy, batch_diversity).

    'tent' minimizes per-sample entropy only. 'im' (Information Maximization,
    SHOT-style) also MAXIMIZES the batch-marginal entropy (diversity), which is
    the collapse guard: plain Tent on an already-overconfident model drives every
    prediction onto one confident class (entropy -> 0 but accuracy drops); the
    diversity term penalizes that degenerate all-same-class solution.
    """
    logp = torch.log_softmax(logits, dim=1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=1).mean()            # per-sample confidence (minimize)
    p_bar = p.mean(dim=0)
    div = -(p_bar * torch.log(p_bar + 1e-8)).sum()  # batch-marginal diversity (maximize)
    if method == "tent":
        return ent, float(ent), float(div)
    return ent - div_weight * div, float(ent), float(div)


def _tent_adapt(model, head, loader, ln_params, steps, lr, device,
                method="im", div_weight=1.0):
    """Adapt only the LayerNorm affine params on the UNLABELED loader. Episodic —
    the caller snapshots/restores LN state around this so a re-run is clean.

    LayerNorm has no batch-statistic behaviour, so eval() mode is correct here and
    it disables the LoRA dropout noise; grads still flow to the (now-trainable) LN
    affine params. The head is fixed. Prints the entropy AND diversity trajectory
    so a no-op (entropy flat) or a collapse (diversity crashing) is visible.
    """
    model.eval()
    head.eval()
    opt = torch.optim.Adam(ln_params, lr=lr)
    e0 = d0 = e1 = d1 = None
    for _step in range(steps):
        for x, _src, _y in loader:
            opt.zero_grad()
            loss, e, d = _adapt_loss(head(model.features(x.to(device, non_blocking=True))),
                                     method, div_weight)
            loss.backward()
            opt.step()
            e1, d1 = e, d
            if e0 is None:
                e0, d0 = e, d
    print(f"    [{method}] {len(ln_params)} LN params; entropy {e0:.4f}->{e1:.4f} "
          f"diversity {d0:.4f}->{d1:.4f} ({steps} steps, lr={lr}, div_w={div_weight})")


def _run_source(name, seed, epoch, ckpt, tta_steps, tta_lr, bs, nw,
                image_size, results_csv, work_dir, method="im", div_weight=1.0):
    root = SOURCE_ROOTS[name]
    if not os.path.isdir(root):
        print(f"[skip] {name}: source root not found: {root!r}")
        return
    spec = discover_source(name, root)
    items = enumerate_source(spec)
    # user-disjoint SI split — mirror bdsl47_si.yaml (val=user 4, test=user 5).
    tr, va, te = split_user_disjoint(items, val_users={4}, test_users={5})
    print(f"[{name}] classes={spec.num_classes}  train={len(tr)} "
          f"val(user4)={len(va)} test(user5)={len(te)}")
    if not tr or not te:
        print(f"[skip] {name}: empty train or test-user split")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the LoRA DINOv2 model exactly as the SI config, then load the
    # backbone-only checkpoint (heads are never checkpointed). strict=False.
    model = build_dinov2_lora(
        num_classes_per_source=[spec.num_classes],
        timm_name="vit_small_patch14_dinov2.lvd142m",
        lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
        lora_targets=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        pretrained=True, full_finetune=False,
    )
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    print(f"[{name}] loaded {os.path.basename(ckpt)} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    model = model.to(device)

    transform = _build_transforms(image_size)
    tr_loader = _loader(spec, tr, transform, bs, nw)
    va_loader = _loader(spec, va, transform, bs, nw) if va else None
    # shuffle the TTA/eval loader so each Tent step sees mixed batches.
    te_loader_adapt = _loader(spec, te, transform, bs, nw, shuffle=True)
    te_loader_eval = _loader(spec, te, transform, bs, nw)

    # Fit a FIXED, differentiable head on FROZEN train features.
    Xtr, ytr = _extract(model, tr_loader, device)
    head = _fit_torch_head(Xtr, ytr, spec.num_classes, device)

    # BEFORE adaptation.
    before_test = _eval_head(model, head, te_loader_eval, device)
    before_val = _eval_head(model, head, va_loader, device) if va_loader else None
    if before_val is not None:
        print(f"[{name}] BEFORE  val(user4) Top-1 = {before_val*100:.2f}%")
    print(f"[{name}] BEFORE  test(user5) Top-1 = {before_test*100:.2f}%")

    # Snapshot LN affine state so adaptation is episodic (clean re-run).
    ln_params = _collect_ln_params(model.backbone)
    ln_snapshot = [p.detach().clone() for p in ln_params]

    # Tent: adapt LN affine on the UNLABELED test-user images.
    _tent_adapt(model, head, te_loader_adapt, ln_params, tta_steps, tta_lr, device,
                method=method, div_weight=div_weight)

    # AFTER adaptation (report on the SAME test-user set — transductive TTA).
    after_test = _eval_head(model, head, te_loader_eval, device)
    after_val = _eval_head(model, head, va_loader, device) if va_loader else None
    if after_val is not None:
        print(f"[{name}] AFTER   val(user4) Top-1 = {after_val*100:.2f}%")
    print(f"[{name}] AFTER   test(user5) Top-1 = {after_test*100:.2f}%")

    delta = (after_test - before_test) * 100.0
    print(f"[{name}] TEST-user delta (after - before) = {delta:+.2f} pp "
          f"({before_test*100:.2f}% -> {after_test*100:.2f}%)")

    # Restore LN affine so a subsequent source/re-run is unaffected.
    with torch.no_grad():
        for p, s in zip(ln_params, ln_snapshot):
            p.copy_(s)

    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for tag, acc in (("before", before_test), ("after", after_test)):
        _append_csv_row(results_csv, {
            "Timestamp": ts,
            "Experiment": f"bhc_tta_{tag}_{name}_seed{seed}",
            "Epoch": epoch,
            "Top1_Acc": f"{acc:.6f}",
            "Top5_Acc": "",
            "Top5_Policy": "tta",
            "WorkDir": work_dir,
        })


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--si-dir", default="work_dir/bhc_bdsl47_si",
                    help="dir with encoder_seed<N>_epoch<E>.pt backbone checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=0,
                    help="checkpoint epoch to load; 0 = auto-detect latest")
    ap.add_argument("--sources", nargs="+",
                    default=["bdsl47_digits", "bdsl47_letters"],
                    choices=list(SOURCE_ROOTS.keys()))
    ap.add_argument("--tta-steps", type=int, default=3,
                    help="number of adaptation passes over the test images")
    ap.add_argument("--tta-lr", type=float, default=1e-3,
                    help="Adam LR for the LayerNorm affine params")
    ap.add_argument("--tta-method", choices=["im", "tent"], default="im",
                    help="im = entropy-min + diversity (collapse-guarded, default); "
                         "tent = entropy-min only")
    ap.add_argument("--div-weight", type=float, default=1.0,
                    help="weight on the batch-diversity term (im only); higher = "
                         "stronger anti-collapse")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--results-csv", default="results_final.csv")
    args = ap.parse_args()

    _init_seed(args.seed)

    # Resolve the checkpoint (auto-detect latest epoch when --epoch 0).
    if args.epoch and args.epoch > 0:
        ckpt = os.path.join(args.si_dir,
                            f"encoder_seed{args.seed}_epoch{args.epoch}.pt")
        epoch = args.epoch
        if not os.path.isfile(ckpt):
            print(f"[no-op] checkpoint not found: {ckpt}\n"
                  f"        Train the SI encoder first "
                  f"(train_baseline --config .../bdsl47_si.yaml). Exiting cleanly.")
            sys.exit(0)
    else:
        epoch, ckpt = _latest_epoch(args.si_dir, args.seed)
        if ckpt is None:
            print(f"[no-op] no encoder_seed{args.seed}_epoch*.pt in {args.si_dir!r}\n"
                  f"        Train the SI encoder first "
                  f"(train_baseline --config .../bdsl47_si.yaml). Exiting cleanly.")
            sys.exit(0)
        print(f"[auto] latest checkpoint: {os.path.basename(ckpt)} (epoch {epoch})")

    work_dir = os.path.abspath(args.si_dir)
    print(f"TTA[{args.tta_method}] — seed={args.seed} epoch={epoch} "
          f"steps={args.tta_steps} lr={args.tta_lr} div_w={args.div_weight}\n"
          f"  adapting: LayerNorm affine only (head + backbone otherwise frozen); "
          f"transductive on the TEST-user set.")

    for name in args.sources:
        _run_source(name, args.seed, epoch, ckpt, args.tta_steps, args.tta_lr,
                    args.batch_size, args.num_workers, args.image_size,
                    args.results_csv, work_dir,
                    method=args.tta_method, div_weight=args.div_weight)


if __name__ == "__main__":
    main()
