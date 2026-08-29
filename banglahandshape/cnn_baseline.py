"""Multi-head CNN baselines (ResNet) for the sister-paper Tier-1 comparison.

These are the "old-school" reference points the published Bangla-handshape
papers actually used (from-scratch CNNs and ImageNet-pretrained transfer). The
module mirrors `dinov2_lora.MultiHeadLoRADinov2` exactly so it is a drop-in for
`train_baseline` and `eval_cross_dataset`:

  * `forward(x, src_idx)` -> list of `(source_idx, mask, logits)` — same
    contract (true source index, only sources present in the batch), so the
    per-source attribution in `train_utils.evaluate` stays correct.
  * `features(x)` -> `(N, feat_dim)` pooled features, for the S2 transfer matrix.
  * `.backbone` is the feature extractor whose `state_dict()` is checkpointed.
  * `.num_lora_replacements = 0` (so the shared training log line still prints).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MultiHeadCNN(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int,
                 num_classes_per_source: List[int]):
        super().__init__()
        self.backbone = backbone            # (N,3,H,W) -> (N, feat_dim)
        self.feat_dim = int(feat_dim)
        self.heads = nn.ModuleList(
            [nn.Linear(feat_dim, n) for n in num_classes_per_source]
        )
        for h in self.heads:
            nn.init.trunc_normal_(h.weight, std=0.02)
            nn.init.zeros_(h.bias)
        self.num_lora_replacements = 0

    def features(self, x):
        return self.backbone(x)

    def forward(self, x, src_idx: torch.Tensor):
        feats = self.features(x)
        out = []
        for i, head in enumerate(self.heads):
            mask = (src_idx == i)
            if mask.any():
                out.append((i, mask, head(feats[mask])))
        return out


def build_cnn(num_classes_per_source: List[int],
              arch: str = "resnet18",
              pretrained: bool = False,
              freeze_backbone: bool = False):
    """Build a multi-head ResNet.

    arch: 'resnet18' | 'resnet34' | 'resnet50'.
    pretrained: load ImageNet weights (transfer-learning baseline) vs random
        init (from-scratch baseline).
    freeze_backbone: if True, only the heads train (a CNN linear-probe control).
    """
    from torchvision import models

    ctors = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    if arch not in ctors:
        raise ValueError(f"unsupported CNN arch: {arch} (have {list(ctors)})")

    # torchvision >=0.13 weights API; 'DEFAULT' picks the best ImageNet weights.
    try:
        net = ctors[arch](weights=("DEFAULT" if pretrained else None))
    except TypeError:  # very old torchvision without the `weights` kwarg
        net = ctors[arch](pretrained=pretrained)

    feat_dim = int(net.fc.in_features)
    net.fc = nn.Identity()                  # backbone now returns pooled features
    if freeze_backbone:
        for p in net.parameters():
            p.requires_grad_(False)
    return MultiHeadCNN(net, feat_dim, num_classes_per_source)
