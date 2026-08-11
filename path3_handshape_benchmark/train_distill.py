"""Phase-2 METHOD — distill the signer-agnostic pose stream into the RGB
appearance student, so the student is signer-invariant BUT uses IMAGES ONLY at
test time.

Prior result (train_fusion.py / A6): the MediaPipe hand-keypoint MLP has a
near-zero SD->SI gap (hand geometry is signer-agnostic), while the DINOv2+LoRA
appearance model has a large one (it latches onto signer identity). Here we make
the appearance model inherit pose's invariance via knowledge distillation:

  * Teacher : a keypoint-MLP trained on the SAME SI TRAIN split (63-d MediaPipe
              vectors from work_dir/_kp_cache/<source>.npz, aligned by ABSOLUTE
              image path). Trained on train signers ONLY, then frozen.
  * Student : DINOv2-S + LoRA (rank 8, alpha 16, dropout 0.05), multi-head over
              the BdSL47 sources — IDENTICAL encoder/split to bdsl47_si.yaml so
              SI Top-1 is directly comparable to the plain LoRA baseline.

  Loss per batch =
        CE(student_logits, label)
      + lambda_kd * KL( softmax(student_logits / Ts) || softmax(teacher_logits / Ts) )
                    * Ts^2                                  # temperature-scaled KD
  where the KD term is applied ONLY to samples with a DETECTED hand (undetected
  samples still contribute to CE, just not to KD). Ts, lambda_kd from config.

At TEST/eval the student is IMAGE-ONLY — the teacher and keypoints are not used.
We report SI val Top-1 per source (best-over-epochs, val user 4 exactly like
train_baseline). One row per (source, seed) -> --results-csv, RESULT_COLUMNS,
Experiment = <Experiment_name>_<source>_seed<N>, Top5_Policy = best_over_epochs.

Usage:
    python -m path3_handshape_benchmark.train_distill \
        --config path3_handshape_benchmark/configs/bdsl47_si_distill.yaml \
        --seeds 0 1 2 --results-csv results/bhc_distill.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Weights are pre-warmed in the HF cache; compute nodes have no outbound HTTPS.
# Must be set BEFORE timm (-> huggingface_hub) import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from bangla_handshape.class_alignment import discover_source, SourceSpec
from bangla_handshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, split_user_disjoint, split_random,
    cap_per_class, load_img_cache,
)
from bangla_handshape.dinov2_lora import build_dinov2_lora
from bangla_handshape.keypoint_baseline import build_keypoint_mlp
from bangla_handshape.train_utils import evaluate
from path3_handshape_benchmark.train_baseline import (
    RESULT_COLUMNS, _init_seed, _build_transforms, _build_cached_transforms,
    _append_csv_row, _provided_test_dir,
)
# Reuse the fusion keypoint-cache reader (abs-path -> (kp[63], detected)).
from path3_handshape_benchmark.train_fusion import _kp_lookup, _norm

KP_CACHE = "work_dir/_kp_cache"


def _split_source(spec, sp):
    """Mirror train_baseline._train_one_seed's split selection EXACTLY so the SI
    Top-1 is directly comparable to the plain LoRA baseline. Returns
    (train_items, eval_items) where eval is the VAL partition (val user 4)."""
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


def _kp_matrix(items, kp):
    """-> (KP[N,63] float32, detected[N] uint8) aligned to `items` by abs path."""
    KP = np.zeros((len(items), 63), dtype=np.float32)
    det = np.zeros(len(items), dtype=np.uint8)
    if kp is not None:
        for i, it in enumerate(items):
            hit = kp.get(_norm(it[0]))
            if hit is not None:
                KP[i], det[i] = hit[0], hit[1]
    return KP, det


def _train_teacher(KP, labels, det, num_classes, device, *, seed,
                   epochs=50, hidden=256, lr=1e-3, wd=1e-2, batch=256,
                   dropout=0.2, depth=3):
    """Train a single-source keypoint-MLP teacher on the SI TRAIN split, on
    samples WITH a detected hand only (undetected 63-d vectors are all-zero and
    carry no pose signal). Returns the frozen model in eval mode.

    Single-source, so the multi-head builder is given [num_classes] and always
    routed with src_idx == 0.
    """
    _init_seed(seed)
    keep = det.astype(bool)
    Xtr = torch.tensor(KP[keep], dtype=torch.float32)
    ytr = torch.tensor(labels[keep], dtype=torch.long)
    teacher = build_keypoint_mlp([int(num_classes)], in_dim=63, hidden=hidden,
                                 depth=depth, dropout=dropout).to(device)
    if len(Xtr) < 2:
        # Not enough detected samples to train a meaningful teacher; return an
        # untrained (frozen) one — the KD mask will zero out its contribution
        # since there are ~no detected samples anyway.
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher
    opt = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=wd)
    n = len(Xtr)
    for _ep in range(epochs):
        teacher.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            xb = Xtr[idx].to(device)
            yb = ytr[idx].to(device)
            src = torch.zeros(len(idx), dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            out = teacher(xb, src)
            # single source -> exactly one (src_idx, mask, logits) tuple
            loss = F.cross_entropy(out[0][2], yb[out[0][1]])
            loss.backward()
            opt.step()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def _teacher_logits(teacher, KP, det, num_classes, device, batch=256):
    """Soft (pre-temperature) logits for EVERY train item. Undetected rows keep
    zero logits but are masked out of the KD term downstream, so their value is
    irrelevant."""
    logits = np.zeros((len(KP), int(num_classes)), dtype=np.float32)
    keep = np.nonzero(det.astype(bool))[0]
    if len(keep) == 0:
        return logits
    X = torch.tensor(KP[keep], dtype=torch.float32)
    teacher.eval()
    for i in range(0, len(keep), batch):
        idx = keep[i:i + batch]
        xb = torch.tensor(KP[idx], dtype=torch.float32).to(device)
        src = torch.zeros(len(idx), dtype=torch.long, device=device)
        out = teacher(xb, src)
        logits[idx] = out[0][2].float().cpu().numpy()
    return logits


def _kd_loss(student_logits, teacher_logits, kd_mask, temperature):
    """Temperature-scaled distillation term, averaged over the DETECTED samples.

        KL( softmax(student/Ts) || softmax(teacher/Ts) ) * Ts^2

    Implemented with F.kl_div(log q_student, p_teacher) so gradients flow through
    the student only (teacher is detached). Zero if no sample has a detected
    hand in this batch. `kd_mask`: bool (N,) marking detected-hand samples."""
    if kd_mask.sum() == 0:
        return student_logits.new_zeros(())
    s = student_logits[kd_mask]
    t = teacher_logits[kd_mask].detach()
    Ts = float(temperature)
    log_p_student = F.log_softmax(s / Ts, dim=-1)
    p_teacher = F.softmax(t / Ts, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (Ts * Ts)


class _DistillDataset(HandshapeDataset):
    """HandshapeDataset that also returns the item's TEACHER LOGITS and a
    detected-hand flag, aligned by construction order (the flattened `items`
    order is identical to the (spec, entries) order we build the aux arrays in).

    Returns (image, src_idx, label, teacher_logits[C_source], detected)."""

    def __init__(self, sources_with_entries, teacher_logits_per_source,
                 detected_per_source, **kw):
        super().__init__(sources_with_entries, **kw)
        # Flatten aux arrays in the SAME order the base class flattens `items`.
        tl, dt_ = [], []
        for src_idx, (_spec, entries) in enumerate(sources_with_entries):
            tlog = teacher_logits_per_source[src_idx]
            dflag = detected_per_source[src_idx]
            for j in range(len(entries)):
                tl.append(tlog[j])
                dt_.append(int(dflag[j]))
        self._teacher_logits = tl
        self._detected = dt_

    def __getitem__(self, idx):
        img, src_idx, label = super().__getitem__(idx)
        tlog = torch.as_tensor(self._teacher_logits[idx], dtype=torch.float32)
        return img, src_idx, label, tlog, int(self._detected[idx])


def _collate(batch):
    """Custom collate: teacher-logit vectors differ in width across sources, so
    they can't be stacked into one tensor. Keep them as a per-sample list; the
    train loop indexes them by the batch's per-source mask (single label space
    within a mask, so widths match there)."""
    imgs = torch.stack([b[0] for b in batch])
    src = torch.tensor([b[1] for b in batch], dtype=torch.long)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    tlogs = [b[3] for b in batch]           # list of (C_source,) tensors
    det = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return imgs, src, labels, tlogs, det


def _train_one_seed(cfg, seed, results_csv="results_final.csv"):
    _init_seed(seed)
    base_exp = cfg.get("Experiment_name", "bhc_distill")
    work_dir = cfg.get("work_dir", f"./work_dir/{base_exp}")
    os.makedirs(work_dir, exist_ok=True)

    sources = []
    for name, root in cfg["sources"].items():
        if not os.path.isdir(root):
            print(f"[WARN] missing source: {name} at {root}; skipping")
            continue
        sources.append(discover_source(name, root))
    if not sources:
        raise RuntimeError("no sources on disk")

    sp = cfg.get("split", {})
    kd = cfg.get("kd", {})
    lambda_kd = float(kd.get("lambda", 1.0))
    temperature = float(kd.get("temperature", 2.0))
    teacher_epochs = int(kd.get("teacher_epochs", 50))
    teacher_hidden = int(kd.get("teacher_hidden", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Per-source split + teacher soft-logit computation -------------------
    train_pairs, val_pairs = [], []
    teacher_logits_per_source, detected_per_source = [], []
    for spec in sources:
        tr_items, va_items = _split_source(spec, sp)
        train_pairs.append((spec, tr_items))
        val_pairs.append((spec, va_items))

        kp = _kp_lookup(spec.name)
        if kp is None:
            print(f"[WARN] no keypoint cache for {spec.name} "
                  f"(run extract_keypoints.py); NO KD signal for this source "
                  f"(CE-only)", flush=True)
        KPtr, dtr = _kp_matrix(tr_items, kp)
        ytr = np.array([it[1] for it in tr_items], dtype=np.int64)
        det_rate = 100.0 * dtr.sum() / max(1, len(dtr))
        print(f"[{spec.name} seed{seed}] train={len(tr_items)} val={len(va_items)} "
              f"C={spec.num_classes} kp_det={det_rate:.1f}%", flush=True)

        # Teacher trained on TRAIN signers only, then frozen.
        teacher = _train_teacher(KPtr, ytr, dtr, spec.num_classes, device,
                                 seed=seed, epochs=teacher_epochs,
                                 hidden=teacher_hidden)
        tlog = _teacher_logits(teacher, KPtr, dtr, spec.num_classes, device)
        teacher_logits_per_source.append(tlog)
        detected_per_source.append(dtr)
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Datasets / loaders (student sees IMAGES; KD aux carried alongside) --
    transform = _build_transforms(int(cfg.get("image_size", 224)))
    cached_tf = _build_cached_transforms()
    img_cache_index, img_cache_dir = load_img_cache()
    _ck = dict(img_cache_index=img_cache_index, img_cache_dir=img_cache_dir,
               cached_transform=cached_tf)
    ds_train = _DistillDataset(train_pairs, teacher_logits_per_source,
                               detected_per_source, transform=transform, **_ck)
    # Eval uses IMAGES ONLY -> the plain HandshapeDataset + standard evaluate().
    ds_val = HandshapeDataset(val_pairs, transform=transform, **_ck)

    nw = int(cfg.get("num_workers", 0))
    extra = dict(persistent_workers=True, prefetch_factor=2) if nw > 0 else {}
    loader_train = DataLoader(
        ds_train, batch_size=int(cfg.get("batch_size", 64)), shuffle=True,
        num_workers=nw, drop_last=True, pin_memory=torch.cuda.is_available(),
        collate_fn=_collate, **extra,
    )
    loader_val = DataLoader(
        ds_val, batch_size=int(cfg.get("batch_size", 64)), shuffle=False,
        num_workers=nw, pin_memory=torch.cuda.is_available(), **extra,
    )

    # ---- Student: DINOv2-S + LoRA, multi-head — identical to bdsl47_si -------
    enc_cfg = cfg["encoder"]
    model = build_dinov2_lora(
        num_classes_per_source=ds_train.num_classes_per_source(),
        timm_name=enc_cfg.get("timm_name", "vit_small_patch14_dinov2.lvd142m"),
        lora_rank=int(enc_cfg.get("lora_rank", 8)),
        lora_alpha=float(enc_cfg.get("lora_alpha", 16.0)),
        lora_dropout=float(enc_cfg.get("lora_dropout", 0.05)),
        lora_targets=enc_cfg.get("lora_targets") or [],
        pretrained=bool(enc_cfg.get("pretrained", True)),
        full_finetune=bool(enc_cfg.get("full_finetune", False)),
    ).to(device)
    print(f"[seed {seed}] LoRA replacements={model.num_lora_replacements}; "
          f"train items={len(ds_train)}; val items={len(ds_val)}; "
          f"lambda_kd={lambda_kd} Ts={temperature}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("base_lr", 5e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )
    num_epoch = int(cfg.get("num_epoch", 50))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    log_every = int(cfg.get("log_every", 50))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_per_source = {i: 0.0 for i in range(len(sources))}

    # Full-state resume (mirror train_baseline).
    resume_path = os.path.join(work_dir, f"resume_seed{seed}.pt")
    start_epoch = 0
    if os.path.exists(resume_path):
        try:
            ck = torch.load(resume_path, map_location=device)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = int(ck["epoch"])
            best_per_source = ck["best_per_source"]
            print(f"[seed {seed}] resumed from epoch {start_epoch}/{num_epoch}")
        except Exception as e:
            print(f"[seed {seed}] resume failed ({e}); starting fresh")

    for epoch in range(start_epoch, num_epoch):
        print(f"[seed {seed}] epoch {epoch+1}/{num_epoch}")
        _train_kd_epoch(model, loader_train, optimizer, device, scaler,
                        lambda_kd=lambda_kd, temperature=temperature,
                        grad_clip=grad_clip, log_every=log_every)
        accs = evaluate(model, loader_val, device)   # IMAGES ONLY
        for src_i, acc in accs.items():
            name = ds_train.source_names()[src_i]
            print(f"  val {name}: Top-1 = {acc*100:.2f}%")
            best_per_source[src_i] = max(best_per_source[src_i], acc)
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "best_per_source": best_per_source},
                   resume_path)
        if (epoch + 1) % int(cfg.get("save_interval", 1)) == 0 or epoch == num_epoch - 1:
            ckpt = os.path.join(work_dir, f"encoder_seed{seed}_epoch{epoch+1}.pt")
            torch.save(model.backbone.state_dict(), ckpt)
            print(f"  saved {ckpt}")

    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for src_i, acc in best_per_source.items():
        name = ds_train.source_names()[src_i]
        _append_csv_row(results_csv, {
            "Timestamp": timestamp,
            "Experiment": f"{base_exp}_{name}_seed{seed}",
            "Epoch": num_epoch,
            "Top1_Acc": f"{acc:.6f}",
            "Top5_Acc": "",
            "Top5_Policy": "best_over_epochs",
            "WorkDir": work_dir,
        })


def _train_kd_epoch(model, loader, optimizer, device, scaler, *,
                    lambda_kd, temperature, grad_clip=1.0, log_every=50):
    """One epoch of CE + KD. Preserves the true-source-index contract: forward
    returns (true_src_idx, mask_in_batch, logits); we index the batch's
    teacher-logit list and detected-flag by the SAME mask, so per-source KD is
    correctly attributed even when a batch is source-ordered."""
    import time
    model.train()
    use_amp = scaler is not None and device.type == "cuda"
    losses = []
    t0 = time.time()
    for step, (x, src_idx, labels, tlogs, det) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        src_idx = src_idx.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        det = det.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        def _compute():
            outputs = model(x, src_idx)   # [(true_src_i, mask, logits), ...]
            ce_terms, kd_terms, n_ce = [], [], 0
            for src_i, mask, logits in outputs:
                if not mask.any():
                    continue
                tgt = labels[mask]
                ce_terms.append(F.cross_entropy(logits, tgt, reduction="sum"))
                n_ce += int(mask.sum().item())
                # Teacher logits for THIS source's samples, in mask order. tlogs
                # is a per-sample list keyed by batch position; select by mask.
                pos = torch.nonzero(mask, as_tuple=False).flatten().tolist()
                t_sel = torch.stack([tlogs[p] for p in pos]).to(device)
                kd_mask = det[mask].bool()
                kd_terms.append(_kd_loss(logits, t_sel, kd_mask, temperature))
            if n_ce == 0:
                return x.new_zeros((), requires_grad=True)
            ce = torch.stack(ce_terms).sum() / n_ce
            kd = torch.stack(kd_terms).mean() if kd_terms else ce.new_zeros(())
            return ce + lambda_kd * kd

        if use_amp:
            with torch.cuda.amp.autocast():
                loss = _compute()
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = _compute()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(float(loss.item()))
        if (step + 1) % log_every == 0:
            print(f"  step {step+1}/{len(loader)}  "
                  f"loss {sum(losses[-log_every:])/min(log_every, len(losses)):.4f}")
    print(f"  epoch done. mean loss {sum(losses)/max(1,len(losses)):.4f}  "
          f"duration {time.time()-t0:.1f}s")
    return float(sum(losses) / max(1, len(losses)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--results-csv", default="results_final.csv")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg.get("work_dir", "./work_dir/bhc_bdsl47_si_distill"),
                exist_ok=True)
    for seed in args.seeds:
        _train_one_seed(cfg, seed, results_csv=args.results_csv)
    print("=== DISTILL DONE ===", flush=True)


if __name__ == "__main__":
    main()
