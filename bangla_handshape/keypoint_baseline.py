"""A6 modality control — multi-head MLP over MediaPipe hand keypoints.

Pose counterpart to the appearance encoders (DINOv2/CNN). Input is a 63-d
normalized hand-keypoint vector (21 landmarks x xyz) from `extract_keypoints.py`;
output follows the same contract as `MultiHeadLoRADinov2` / `MultiHeadCNN`:

    forward(x, src_idx) -> list of (source_idx, mask, logits)   # present sources only
    features(x)         -> trunk embedding

so `train_utils.{multihead_loss,evaluate}` and the CSV plumbing work unchanged.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class MultiHeadKeypointMLP(nn.Module):
    def __init__(self, num_classes_per_source: List[int], in_dim: int = 63,
                 hidden: int = 256, depth: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(1, depth)):
            layers += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden, nc)
                                    for nc in num_classes_per_source])
        self.feature_dim = hidden
        # For parity with the other builders' logging in train_baseline.
        self.num_lora_replacements = 0

    def features(self, x):
        return self.trunk(x)

    def forward(self, x, src_idx):
        feats = self.trunk(x)
        out = []
        for i, head in enumerate(self.heads):
            mask = (src_idx == i)
            if mask.any():
                out.append((i, mask, head(feats[mask])))
        return out


def build_keypoint_mlp(num_classes_per_source, in_dim=63, hidden=256, depth=3,
                       dropout=0.2):
    return MultiHeadKeypointMLP(num_classes_per_source, in_dim=in_dim,
                                hidden=hidden, depth=depth, dropout=dropout)
