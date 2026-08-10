"""Generic training-loop helpers shared by Path 1 and Path 3."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def multihead_loss(per_source_outputs, labels):
    """per_source_outputs: list of (source_idx, mask, logits); labels: (N,) int.

    Returns scalar mean cross-entropy weighted by per-source contribution.
    """
    losses = []
    n_total = 0
    for _src_i, mask, logits in per_source_outputs:
        if mask.any():
            tgt = labels[mask]
            losses.append(F.cross_entropy(logits, tgt, reduction="sum"))
            n_total += int(mask.sum().item())
    if n_total == 0:
        return torch.zeros((), device=labels.device, requires_grad=True)
    total = torch.stack(losses).sum() / n_total
    return total


def multihead_topk(per_source_outputs, labels, k=1):
    """Per-source Top-k accuracies (returns dict source_idx -> accuracy)."""
    out = {}
    for src_idx, mask, logits in per_source_outputs:
        if not mask.any():
            continue
        tgt = labels[mask]
        topk = logits.topk(min(k, logits.size(-1)), dim=-1).indices
        correct = (topk == tgt.unsqueeze(-1)).any(dim=-1).float().mean().item()
        out[src_idx] = correct
    return out


def train_one_epoch(model, loader, optimizer, device, log_every=50,
                    grad_clip: float = 1.0, scaler=None):
    model.train()
    use_amp = scaler is not None and device.type == "cuda"
    losses = []
    t0 = time.time()
    for step, (x, src_idx, labels) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        src_idx = src_idx.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:  # mixed precision — uses tensor cores, ~2-3x faster
            with torch.cuda.amp.autocast():
                outputs = model(x, src_idx)
                loss = multihead_loss(outputs, labels)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x, src_idx)
            loss = multihead_loss(outputs, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(float(loss.item()))
        if (step + 1) % log_every == 0:
            print(f"  step {step+1}/{len(loader)}  loss {sum(losses[-log_every:])/min(log_every, len(losses)):.4f}")
    print(f"  epoch done. mean loss {sum(losses)/max(1,len(losses)):.4f}  duration {time.time()-t0:.1f}s")
    return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n_correct = {}
    n_total = {}
    for x, src_idx, labels in loader:
        x = x.to(device, non_blocking=True)
        src_idx = src_idx.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(x, src_idx)
        for src_i, mask, logits in outputs:
            tgt = labels[mask]
            preds = logits.argmax(dim=-1)
            n_correct[src_i] = n_correct.get(src_i, 0) + int((preds == tgt).sum().item())
            n_total[src_i] = n_total.get(src_i, 0) + int(mask.sum().item())
    return {i: (n_correct.get(i, 0) / max(1, n_total.get(i, 0))) for i in n_total}
