"""Signer-adversarial branch for the LoRA-DINOv2 multi-head classifier.

Adds a Gradient Reversal Layer (GRL) + a signer-classification head on top of
the shared backbone features. The class heads are trained normally; the signer
head is trained to *predict* the signer, but because its gradient is reversed
before it reaches the backbone, the backbone is pushed to produce features from
which the signer *cannot* be predicted. The result is a signer-INVARIANT
handshape representation — the Tier-2 method for closing the signer-independent
(SI) accuracy gap.

DANN-style domain adaptation (Ganin & Lempitsky, 2015), with the signer id
playing the role of the "domain". Used only on BdSL47 (the sole sources with
real user metadata) via `benchmark/baselines/train_adversarial.py`.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from banglahandshape.dinov2_lora import build_dinov2_lora, MultiHeadLoRADinov2


class GradReverse(torch.autograd.Function):
    """Identity in the forward pass; negates (and scales by lambd) the gradient
    in the backward pass. This is the Gradient Reversal Layer: the head above it
    minimises signer loss while the backbone below it is driven to MAXIMISE it,
    yielding signer-invariant features."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Negated (and lambd-scaled) gradient to x; None for the lambd arg.
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    """Functional wrapper around the Gradient Reversal Layer."""
    return GradReverse.apply(x, lambd)


class AdversarialMultiHead(nn.Module):
    """Wrap a `MultiHeadLoRADinov2` with a GRL + signer-classification head.

    Preserves the base multi-head contract for the class outputs: `forward`
    returns `(class_out, signer_logits)` where `class_out` is the SAME list of
    `(true_source_idx, mask, logits)` tuples the base model emits — the first
    element is the true head index `i`, NOT an enumerate position — so the
    shared per-source attribution / CSV plumbing works unchanged. `signer_logits`
    is `(N, num_signers)` for the whole batch (all samples, all sources).
    """

    def __init__(self, base: MultiHeadLoRADinov2, num_signers: int,
                 lambd: float = 1.0):
        super().__init__()
        self.base = base
        self.num_signers = int(num_signers)
        self.lambd = float(lambd)
        self.signer_head = nn.Linear(base.feat_dim, int(num_signers))
        # The signer head trains from scratch and must receive gradients.
        self.signer_head.weight.requires_grad_(True)
        self.signer_head.bias.requires_grad_(True)
        nn.init.trunc_normal_(self.signer_head.weight, std=0.02)
        nn.init.zeros_(self.signer_head.bias)

    # --- expose the base interface used by the shared plumbing --------------
    @property
    def backbone(self):
        return self.base.backbone

    @property
    def feat_dim(self):
        return self.base.feat_dim

    @property
    def num_lora_replacements(self):
        return self.base.num_lora_replacements

    def features(self, x):
        return self.base.features(x)

    # -----------------------------------------------------------------------
    def forward(self, x, src_idx: torch.Tensor):
        """Compute features ONCE, route them through both the per-source class
        heads and the GRL->signer head.

        Returns (class_out, signer_logits):
          * class_out: list of (i, mask, base.heads[i](feats[mask])) for every
            source i present in the batch (true head index, base contract).
          * signer_logits: (N, num_signers) from the signer head fed reversed
            features.
        """
        feats = self.base.features(x)
        class_out = []
        for i, head in enumerate(self.base.heads):
            mask = (src_idx == i)
            if mask.any():
                class_out.append((i, mask, head(feats[mask])))
        signer_logits = self.signer_head(grad_reverse(feats, self.lambd))
        return class_out, signer_logits


def build_adversarial(num_classes_per_source: List[int], num_signers: int,
                      lambd: float = 1.0, **lora_kwargs) -> AdversarialMultiHead:
    """Build a `MultiHeadLoRADinov2` (via `build_dinov2_lora`) and wrap it with a
    GRL + signer head. `lora_kwargs` are forwarded verbatim to
    `build_dinov2_lora` (timm_name, lora_rank, lora_alpha, lora_dropout,
    lora_targets, pretrained, full_finetune)."""
    base = build_dinov2_lora(num_classes_per_source=num_classes_per_source,
                             **lora_kwargs)
    model = AdversarialMultiHead(base, num_signers=num_signers, lambd=lambd)
    # Guarantee the signer head is trainable regardless of any freezing done
    # inside build_dinov2_lora (freeze_non_lora only touches the backbone).
    for p in model.signer_head.parameters():
        p.requires_grad_(True)
    return model
