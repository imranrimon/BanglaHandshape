"""E14 (revision) --- SIGNER/session-aware k-shot.

The paper's k-shot samples k FRAMES per class, which with burst-captured data can
overstate low-shot performance (near-duplicate frames). Here each class's training
examples come from k distinct SIGNERS (all their frames of that class), so 'k-shot'
means k signers, not k correlated frames. Evaluated on the SI val signer (user 4);
one training per (k, draw) invocation -> a small array with multiple random draws.

Rows -> --results-csv: bhc_kshotS_k{k}_draw{d}_<source>_seed<N>
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
from banglahandshape.handshape_dataset import HandshapeDataset, enumerate_source, split_user_disjoint
from banglahandshape.dinov2_lora import build_dinov2_lora
from banglahandshape.train_utils import train_one_epoch, evaluate
from benchmark.baselines.train_baseline import _build_transforms, _append_csv_row, _init_seed

SR = {"bdsl47_digits": "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
      "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters"}
LORA = dict(lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
            lora_targets=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])


def _signer_kshot(train_items, k, rng):
    """For each class, keep all frames of k randomly-chosen train signers."""
    by_cls = {}
    for path, label, u in train_items:
        by_cls.setdefault(label, {}).setdefault(u, []).append((path, label, u))
    out = []
    for label, per_signer in by_cls.items():
        signers = list(per_signer)
        rng.shuffle(signers)
        for u in signers[:k]:
            out.extend(per_signer[u])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="bdsl47_letters", choices=list(SR))
    ap.add_argument("--k", type=int, required=True)          # k signers per class
    ap.add_argument("--draw", type=int, default=0)           # random draw id
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--results-csv", default="results/bhc_kshot_signer.csv")
    args = ap.parse_args()
    _init_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = discover_source(args.source, SR[args.source]); items = enumerate_source(spec)
    tr, va, _te = split_user_disjoint(items, {4}, {5})
    rng = np.random.RandomState(1000 + args.draw)
    sub = _signer_kshot(tr, args.k, rng)
    print(f"[{args.source} k={args.k} draw={args.draw}] train={len(sub)} of {len(tr)}, val={len(va)}", flush=True)

    tf = _build_transforms(224)
    def _loader(it, sh):
        return DataLoader(HandshapeDataset([(spec, it)], transform=tf),
                          batch_size=args.batch_size, shuffle=sh, num_workers=args.num_workers,
                          drop_last=sh, pin_memory=torch.cuda.is_available())
    model = build_dinov2_lora(num_classes_per_source=[spec.num_classes],
                              timm_name="vit_small_patch14_dinov2.lvd142m",
                              pretrained=True, full_finetune=False, **LORA).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))
    ltr, lva = _loader(sub, True), _loader(va, False)
    best = 0.0
    for ep in range(args.epochs):
        train_one_epoch(model, ltr, opt, dev, log_every=300, grad_clip=1.0, scaler=scaler)
        best = max(best, evaluate(model, lva, dev).get(0, 0.0))
    print(f"[k={args.k} draw={args.draw}] SI val Top-1 = {best*100:.2f}%", flush=True)
    _append_csv_row(args.results_csv, {
        "Timestamp": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "Experiment": f"bhc_kshotS_k{args.k}_draw{args.draw}_{args.source}_seed{args.seed}",
        "Epoch": args.epochs, "Top1_Acc": f"{best:.6f}", "Top5_Acc": "",
        "Top5_Policy": "signer_kshot", "WorkDir": "work_dir/bhc_kshot_signer"})


if __name__ == "__main__":
    main()
