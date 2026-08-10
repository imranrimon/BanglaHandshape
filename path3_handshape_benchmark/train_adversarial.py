"""Path 3 — signer-ADVERSARIAL LoRA-DINOv2 training on BdSL47 (Tier-2 method).

Adds a signer-classification branch behind a Gradient Reversal Layer on top of
the shared backbone (see `bangla_handshape/adversarial.py`), so the backbone is
driven to produce signer-INVARIANT handshape features. Trains with

    total = class_CE + signer_CE_through_GRL

on the user-disjoint (signer-independent) split — the SAME split
`train_baseline._train_one_seed` uses for `bdsl47_si.yaml`, so the reported
SI Top-1 is directly comparable to the plain baseline. The headline claim is
that adversarial training RAISES SI accuracy / shrinks the SD-SI gap.

Signer space: the signer-id map is built from the TRAIN users only, unioned
across the two BdSL47 sources (the same people appear in both digits & letters);
val items get signer_label = -1 and are ignored by the signer loss (val is a
class-accuracy-only measurement, matching the baseline).

Usage:
    python -m path3_handshape_benchmark.train_adversarial ^
        --config path3_handshape_benchmark/configs/bdsl47_si_adv.yaml ^
        --seeds 0 1 2 --results-csv results_final.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys

# Force offline HF/timm resolution BEFORE timm gets imported anywhere.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bangla_handshape.class_alignment import discover_source, write_inventory
from bangla_handshape.handshape_dataset import enumerate_source, split_user_disjoint
from bangla_handshape.adversarial import build_adversarial
from path3_handshape_benchmark.train_baseline import (
    RESULT_COLUMNS, _append_csv_row, _build_transforms, _init_seed,
)

# The two BdSL47 sources are the only ones with real user metadata.
BDSL47_SOURCES = ("bdsl47_digits", "bdsl47_letters")


class AdvDataset(Dataset):
    """Flat dataset over (path, src_idx, class_label, signer_label) tuples.

    Mirrors HandshapeDataset's decode path (PIL -> transform) but also carries a
    per-item signer_label (>=0 for train users mapped to a contiguous signer id,
    -1 for val items which are excluded from the signer loss)."""

    def __init__(self, items, transform):
        self.items = list(items)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, src_idx, class_label, signer_label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, src_idx, class_label, signer_label


def _lambda_schedule(adv_lambda, epoch, num_epoch):
    """DANN schedule: warm lambda up from 0 -> adv_lambda over training so the
    (initially useless) signer head doesn't destabilise the backbone early on.
    p = epoch / num_epoch in [0, 1)."""
    if num_epoch <= 1:
        return float(adv_lambda)
    p = epoch / float(num_epoch)
    return float(adv_lambda) * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def _train_one_seed(cfg, seed, results_csv="results_final.csv"):
    _init_seed(seed)
    base_exp = cfg.get("Experiment_name", "bhc_bdsl47_si_adv")
    work_dir = cfg.get("work_dir", f"./work_dir/{base_exp}")
    os.makedirs(work_dir, exist_ok=True)

    # --- discover the two BdSL47 sources (order of the dict = src_idx order) --
    sources = []
    for name, root in cfg["sources"].items():
        if name not in BDSL47_SOURCES:
            print(f"[WARN] {name} is not a BdSL47 source (no user IDs); skipping")
            continue
        if not os.path.isdir(root):
            print(f"[WARN] missing source: {name} at {root}; skipping")
            continue
        sources.append(discover_source(name, root))
    if not sources:
        raise RuntimeError("no BdSL47 sources on disk (adversarial training "
                           "needs user IDs)")
    write_inventory(sources, os.path.join(work_dir, "source_inventory.json"))

    sp = cfg.get("split", {})
    val_users = set(sp.get("val_users", []))
    test_users = set(sp.get("test_users", []))

    # --- per-source user-disjoint split (SAME as train_baseline / bdsl47_si) --
    per_source_train = []   # list of (spec, train_entries)
    per_source_val = []     # list of (spec, val_entries)
    train_user_ids = set()  # union of TRAIN users across sources
    for src_idx, spec in enumerate(sources):
        items = enumerate_source(spec)
        tr, va, _te = split_user_disjoint(items, val_users, test_users)
        per_source_train.append((spec, tr))
        per_source_val.append((spec, va))
        for _p, _lab, uid in tr:
            train_user_ids.add(uid)

    # --- signer-id map: TRAIN users only, unioned across sources -------------
    # Same physical person appears in digits & letters, so a single shared map
    # is correct. Contiguous [0..S-1] in sorted user-id order (deterministic).
    signer_id_of = {uid: i for i, uid in enumerate(sorted(train_user_ids))}
    num_signers = len(signer_id_of)
    print(f"[seed {seed}] signer space: {num_signers} train signers "
          f"{sorted(train_user_ids)} -> ids 0..{num_signers-1}")

    # --- build flat item lists -----------------------------------------------
    train_items, val_items = [], []
    for src_idx, (spec, tr) in enumerate(per_source_train):
        for path, class_label, uid in tr:
            train_items.append((path, src_idx, class_label, signer_id_of[uid]))
    for src_idx, (spec, va) in enumerate(per_source_val):
        for path, class_label, _uid in va:
            val_items.append((path, src_idx, class_label, -1))  # signer ignored

    transform = _build_transforms(int(cfg.get("image_size", 224)))
    ds_train = AdvDataset(train_items, transform)
    ds_val = AdvDataset(val_items, transform)

    nw = int(cfg.get("num_workers", 0))
    extra = dict(persistent_workers=True, prefetch_factor=2) if nw > 0 else {}
    loader_train = DataLoader(
        ds_train, batch_size=int(cfg.get("batch_size", 64)),
        shuffle=True, num_workers=nw, drop_last=True,
        pin_memory=torch.cuda.is_available(), **extra,
    )
    loader_val = DataLoader(
        ds_val, batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False, num_workers=nw,
        pin_memory=torch.cuda.is_available(), **extra,
    )

    # --- model ----------------------------------------------------------------
    enc_cfg = cfg["encoder"]
    num_classes_per_source = [s.num_classes for s in sources]
    adv_lambda = float(cfg.get("adv_lambda", 1.0))
    model = build_adversarial(
        num_classes_per_source=num_classes_per_source,
        num_signers=num_signers,
        lambd=adv_lambda,
        timm_name=enc_cfg.get("timm_name", "vit_small_patch14_dinov2.lvd142m"),
        lora_rank=int(enc_cfg.get("lora_rank", 8)),
        lora_alpha=float(enc_cfg.get("lora_alpha", 16.0)),
        lora_dropout=float(enc_cfg.get("lora_dropout", 0.05)),
        lora_targets=enc_cfg.get("lora_targets") or
        ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        pretrained=bool(enc_cfg.get("pretrained", True)),
        full_finetune=bool(enc_cfg.get("full_finetune", False)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[seed {seed}] LoRA replacements={model.num_lora_replacements}; "
          f"train items={len(ds_train)}; val items={len(ds_val)}; "
          f"adv_lambda={adv_lambda}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("base_lr", 5e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )
    num_epoch = int(cfg.get("num_epoch", 5))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    use_schedule = bool(cfg.get("adv_lambda_schedule", True))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_per_source = {i: 0.0 for i in range(len(sources))}
    source_names = [s.name for s in sources]

    for epoch in range(num_epoch):
        # DANN warm-up (falls back to constant if adv_lambda_schedule: False).
        model.lambd = _lambda_schedule(adv_lambda, epoch, num_epoch) \
            if use_schedule else adv_lambda
        print(f"[seed {seed}] epoch {epoch+1}/{num_epoch}  lambda={model.lambd:.4f}")
        model.train()
        cl_losses, sg_losses = [], []
        for x, src_idx, class_label, signer_label in loader_train:
            x = x.to(device, non_blocking=True)
            src_idx = src_idx.to(device, non_blocking=True)
            class_label = class_label.to(device, non_blocking=True)
            signer_label = signer_label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            use_amp = device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp):
                class_out, signer_logits = model(x, src_idx)
                # class loss: mean CE over all present samples (sum/n_total),
                # matching train_utils.multihead_loss semantics.
                losses, n_total = [], 0
                for _i, mask, logits in class_out:
                    tgt = class_label[mask]
                    losses.append(F.cross_entropy(logits, tgt, reduction="sum"))
                    n_total += int(mask.sum().item())
                class_loss = (torch.stack(losses).sum() / max(1, n_total)
                              if losses else
                              torch.zeros((), device=device, requires_grad=True))
                # signer loss over valid (train) samples only; GRL already
                # negates the backbone gradient, so ADD it to the total.
                valid = signer_label >= 0
                if valid.any():
                    signer_loss = F.cross_entropy(signer_logits[valid],
                                                  signer_label[valid])
                else:
                    signer_loss = torch.zeros((), device=device,
                                              requires_grad=True)
                total = class_loss + signer_loss
            scaler.scale(total).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            cl_losses.append(float(class_loss.item()))
            sg_losses.append(float(signer_loss.item()))
        print(f"  train class_loss {sum(cl_losses)/max(1,len(cl_losses)):.4f}  "
              f"signer_loss {sum(sg_losses)/max(1,len(sg_losses)):.4f}")

        # --- eval: class Top-1 per source on val (user 4) --------------------
        model.eval()
        n_correct = {i: 0 for i in range(len(sources))}
        n_total = {i: 0 for i in range(len(sources))}
        with torch.no_grad():
            for x, src_idx, class_label, _signer in loader_val:
                x = x.to(device, non_blocking=True)
                src_idx = src_idx.to(device, non_blocking=True)
                class_label = class_label.to(device, non_blocking=True)
                class_out, _ = model(x, src_idx)
                for i, mask, logits in class_out:
                    tgt = class_label[mask]
                    preds = logits.argmax(dim=-1)
                    n_correct[i] += int((preds == tgt).sum().item())
                    n_total[i] += int(mask.sum().item())
        for i in range(len(sources)):
            acc = n_correct[i] / max(1, n_total[i])
            print(f"  val {source_names[i]}: class Top-1 = {acc*100:.2f}%")
            best_per_source[i] = max(best_per_source[i], acc)

        # --- diagnostic: signer-head accuracy on TRAIN (should DROP over time
        #     if the backbone is becoming signer-invariant) --------------------
        sg_correct, sg_total = 0, 0
        with torch.no_grad():
            for x, src_idx, _class_label, signer_label in loader_train:
                x = x.to(device, non_blocking=True)
                signer_label = signer_label.to(device, non_blocking=True)
                valid = signer_label >= 0
                if not valid.any():
                    continue
                feats = model.features(x)
                sg_logits = model.signer_head(feats)
                preds = sg_logits.argmax(dim=-1)
                sg_correct += int((preds[valid] == signer_label[valid]).sum().item())
                sg_total += int(valid.sum().item())
        sg_acc = sg_correct / max(1, sg_total)
        print(f"  [diag] signer-head train acc = {sg_acc*100:.2f}%  "
              f"(chance {100.0/max(1,num_signers):.2f}%; should DROP if invariance works)")

        ckpt = os.path.join(work_dir, f"encoder_seed{seed}_epoch{epoch+1}.pt")
        torch.save(model.backbone.state_dict(), ckpt)

    # --- write one row per (source, seed) ------------------------------------
    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for i in range(len(sources)):
        exp = f"{base_exp}_{source_names[i]}_seed{seed}"
        _append_csv_row(results_csv, {
            "Timestamp": timestamp,
            "Experiment": exp,
            "Epoch": num_epoch,
            "Top1_Acc": f"{best_per_source[i]:.6f}",
            "Top5_Acc": "",
            "Top5_Policy": "best_over_epochs",
            "WorkDir": work_dir,
        })
        print(f"[seed {seed}] {exp}: best SI Top-1 = {best_per_source[i]*100:.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
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
