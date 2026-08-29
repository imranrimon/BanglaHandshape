"""LoRA fine-tuning utilities for a frozen DINOv2 backbone.

Wraps each `nn.Linear` in the targeted submodules with a low-rank adapter
without modifying the base weights. Uses timm's DINOv2 weights via
`timm.create_model('vit_small_patch14_dinov2.lvd142m', ...)` (also used by
the Option-B feature extractor).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear and add a low-rank trainable delta.

        y = base(x) + (alpha/rank) * B(A(x))
    where A and B are small linear maps with rank-r intermediate dim.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear must wrap an nn.Linear")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(self.rank, 1)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.drop(self.lora_A(x)))


def apply_lora_to_linears(module: nn.Module, target_substrings: List[str],
                          rank: int = 8, alpha: float = 16.0,
                          dropout: float = 0.0) -> int:
    """Recursively replace `nn.Linear` whose qualified name contains any of
    `target_substrings` with `LoRALinear`. Returns the count of replacements.
    """
    n_replaced = 0
    for parent_name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            qual = f"{parent_name}.{child_name}".lstrip(".")
            if not any(sub in qual for sub in target_substrings):
                continue
            new_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, new_layer)
            n_replaced += 1
    return n_replaced


def freeze_non_lora(module: nn.Module) -> int:
    """Set requires_grad=False for everything that isn't in a LoRALinear's A/B
    matrices. Returns the count of trainable params after freezing.
    """
    for p in module.parameters():
        p.requires_grad_(False)
    for sub in module.modules():
        if isinstance(sub, LoRALinear):
            sub.lora_A.weight.requires_grad_(True)
            sub.lora_B.weight.requires_grad_(True)
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_dinov2_lora(num_classes_per_source: List[int],
                      timm_name: str = "vit_small_patch14_dinov2.lvd142m",
                      lora_rank: int = 8,
                      lora_alpha: float = 16.0,
                      lora_dropout: float = 0.0,
                      lora_targets: Optional[List[str]] = None,
                      pretrained: bool = True,
                      full_finetune: bool = False):
    """Build a multi-head classifier over a timm ViT backbone.

    Three adaptation regimes, selected by args (not by name):
      * linear probe   -> lora_targets=[] (nothing wrapped; backbone frozen).
      * LoRA           -> lora_targets non-empty with lora_rank>0.
      * full fine-tune -> full_finetune=True (whole backbone trainable, no LoRA).

    Despite the historical name, `timm_name` may be ANY timm ViT (DINOv2,
    ImageNet-supervised, SigLIP, MAE, ...); LoRA targets match on submodule-name
    substrings (attn.qkv/proj + mlp.fc1/fc2 are shared across timm ViT blocks).
    """
    import timm

    # dynamic_img_size=True lets ViTs — especially DINOv2, which is natively
    # 518px — accept our 224px inputs by interpolating the positional embeddings.
    backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0,
                                 dynamic_img_size=True)
    feat_dim = int(getattr(backbone, "num_features", 384))

    if full_finetune:
        # Upper-bound baseline: no LoRA; the entire backbone trains end-to-end.
        for p in backbone.parameters():
            p.requires_grad_(True)
        n = 0
    else:
        if lora_targets is None:
            # Sensible default: tune the attention QKV, attention out-proj, and the MLP.
            lora_targets = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
        n = apply_lora_to_linears(backbone, lora_targets,
                                  rank=lora_rank, alpha=lora_alpha,
                                  dropout=lora_dropout)
        freeze_non_lora(backbone)

    return MultiHeadLoRADinov2(backbone, feat_dim,
                               num_classes_per_source=num_classes_per_source,
                               num_lora_replacements=n)


class MultiHeadLoRADinov2(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int,
                 num_classes_per_source: List[int], num_lora_replacements: int):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = int(feat_dim)
        self.heads = nn.ModuleList(
            [nn.Linear(feat_dim, n) for n in num_classes_per_source]
        )
        # Heads are trained from scratch.
        for h in self.heads:
            h.weight.requires_grad_(True)
            h.bias.requires_grad_(True)
            nn.init.trunc_normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)
        self.num_lora_replacements = int(num_lora_replacements)

    def features(self, x):
        """(N, 3, H, W) -> (N, feat_dim)"""
        return self.backbone(x)

    def forward(self, x, src_idx: torch.Tensor):
        """Return a list of (source_idx, mask_in_batch, logits) per source
        contributing to the batch. Emitting the true source_idx (not the
        enumerate position) is required for correct per-source attribution:
        forward skips sources absent from the batch, so the list position does
        NOT equal the head index. This avoids running every head on every sample.
        """
        feats = self.features(x)
        out = []
        for i, head in enumerate(self.heads):
            mask = (src_idx == i)
            if mask.any():
                out.append((i, mask, head(feats[mask])))
        return out
