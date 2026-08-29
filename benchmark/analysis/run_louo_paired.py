"""E2 (per-signer paired SD/SI) + E11 (per-fold TTA) --- one held-out signer per run.

Replaces the confounded random-SD minus user-disjoint-SI gap with a test-matched
paired estimator. For held-out signer s and a source, FIX s's test images T_s
(30%, a fixed seed-independent split so both models share T_s) and compare:
  M_unseen : LoRA trained on the OTHER signers only            -> A_unseen(T_s)
  M_seen   : LoRA trained on other signers + s's TRAIN 70%     -> A_seen(T_s)
Delta_s = A_seen - A_unseen  is the honest signer-overlap effect. Val (early stop)
is a fixed 10% of the OTHER signers, identical for both models. We also run
information-max TTA on M_unseen over T_s (E11) and save the M_unseen backbone.

Run one task per signer (a 10-task array). Rows -> --results-csv:
  bhc_paired_{unseen,seen,unseen_tta}_<source>_user{ss}_seed<N>
"""
from __future__ import annotations
import argparse, datetime as dt, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from banglahandshape.class_alignment import discover_source
from banglahandshape.handshape_dataset import HandshapeDataset, enumerate_source
from banglahandshape.dinov2_lora import build_dinov2_lora
from banglahandshape.train_utils import train_one_epoch, evaluate
from benchmark.baselines.train_baseline import (
    _build_transforms, _append_csv_row, _init_seed,
)
from benchmark.analysis.eval_tta import _collect_ln_params, _adapt_loss

SOURCE_ROOTS = {
    "bdsl47_digits":  "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}
LORA = dict(lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
            lora_targets=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])


def _loader(spec, items, tf, bs, nw, shuffle):
    ds = HandshapeDataset([(spec, items)], transform=tf)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      drop_last=shuffle, pin_memory=torch.cuda.is_available())


def _train_model(spec, tr, va, te, epochs, lr, wd, bs, nw, device, seed):
    """Train a single-source DINOv2-S+LoRA; select epoch by val, report T_s acc
    (0--1) at the best-val epoch."""
    _init_seed(seed)
    tf = _build_transforms(224)
    ltr = _loader(spec, tr, tf, bs, nw, True)
    lva = _loader(spec, va, tf, bs, nw, False) if va else None
    lte = _loader(spec, te, tf, bs, nw, False)
    model = build_dinov2_lora(num_classes_per_source=[spec.num_classes],
                              timm_name="vit_small_patch14_dinov2.lvd142m",
                              pretrained=True, full_finetune=False, **LORA).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val, best_te = -1.0, 0.0
    for ep in range(epochs):
        train_one_epoch(model, ltr, opt, device, log_every=300, grad_clip=1.0, scaler=scaler)
        v = evaluate(model, lva, device).get(0, 0.0) if lva else 0.0
        t = evaluate(model, lte, device).get(0, 0.0)
        if v >= best_val:
            best_val, best_te = v, t
    return model, best_te


def _tta(model, spec, te, bs, nw, device, steps=3, lr=1e-3, divw=1.0):
    """Information-max TTA on the trained model's LN affines over the unlabeled
    T_s; return adapted T_s accuracy (0--1). Uses the model's own trained head."""
    tf = _build_transforms(224)
    ln = _collect_ln_params(model.backbone)      # re-enables grad on LN affines
    model.eval()
    opt = torch.optim.Adam(ln, lr=lr)
    lad = _loader(spec, te, tf, bs, nw, True)
    for _ in range(steps):
        for x, src, _y in lad:
            opt.zero_grad()
            out = model(x.to(device, non_blocking=True), src.to(device))
            if not out:
                continue
            loss, _, _ = _adapt_loss(out[0][2], "im", divw)
            loss.backward(); opt.step()
    return evaluate(model, _loader(spec, te, tf, bs, nw, False), device).get(0, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-user", type=int, required=True)
    ap.add_argument("--source", default="bdsl47_letters", choices=list(SOURCE_ROOTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--base-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--results-csv", default="results/bhc_paired.csv")
    ap.add_argument("--work-dir", default="work_dir/bhc_louo_paired")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.work_dir, exist_ok=True)
    spec = discover_source(args.source, SOURCE_ROOTS[args.source])
    items = enumerate_source(spec)
    s = args.test_user
    s_items = [it for it in items if it[2] == s]
    others = [it for it in items if it[2] not in (s, -1)]
    if len(s_items) < 5 or len(others) < 50:
        print(f"[skip] user {s}: too few images (s={len(s_items)}, others={len(others)})")
        return
    # FIXED T_s split (seed-independent -> M_seen/M_unseen share the test images)
    r = np.random.RandomState(1234)
    si = r.permutation(len(s_items)); c = max(1, int(0.30 * len(s_items)))
    s_test = [s_items[i] for i in si[:c]]; s_train = [s_items[i] for i in si[c:]]
    oi = r.permutation(len(others)); ov = max(1, int(0.10 * len(others)))
    others_val = [others[i] for i in oi[:ov]]; others_train = [others[i] for i in oi[ov:]]
    print(f"[user {s}/{args.source}] |T_s|={len(s_test)} |s_train|={len(s_train)} "
          f"|others_train|={len(others_train)} |val|={len(others_val)}", flush=True)

    hp = dict(epochs=args.epochs, lr=args.base_lr, wd=args.weight_decay,
              bs=args.batch_size, nw=args.num_workers, device=device, seed=args.seed)

    Mu, a_unseen = _train_model(spec, others_train, others_val, s_test, **hp)
    a_unseen_tta = _tta(Mu, spec, s_test, args.batch_size, args.num_workers, device)
    torch.save(Mu.backbone.state_dict(),
               os.path.join(args.work_dir, f"unseen_{args.source}_user{s:02d}_seed{args.seed}.pt"))
    del Mu
    if device.type == "cuda":
        torch.cuda.empty_cache()
    Ms, a_seen = _train_model(spec, others_train + s_train, others_val, s_test, **hp)
    del Ms
    print(f"[user {s}] A_unseen={a_unseen*100:.1f}  A_seen={a_seen*100:.1f}  "
          f"Delta={100*(a_seen-a_unseen):+.1f}  A_unseen+TTA={a_unseen_tta*100:.1f}", flush=True)

    ts = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for tag, acc in (("unseen", a_unseen), ("seen", a_seen), ("unseen_tta", a_unseen_tta)):
        _append_csv_row(args.results_csv, {
            "Timestamp": ts,
            "Experiment": f"bhc_paired_{tag}_{args.source}_user{s:02d}_seed{args.seed}",
            "Epoch": args.epochs, "Top1_Acc": f"{acc:.6f}", "Top5_Acc": "",
            "Top5_Policy": "paired_louo", "WorkDir": args.work_dir,
        })
    print("wrote rows to", args.results_csv)


if __name__ == "__main__":
    main()
