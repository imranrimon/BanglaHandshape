"""Smoke tests for the bangla_handshape shared library + Path 1 / Path 3 plumbing.

Verifies on synthetic data:
  * `discover_source` reads class folders correctly
  * `HandshapeDataset` returns the expected `(image, src_idx, label)` tuples
  * `LoRALinear` forwards and trains its A/B matrices only
  * `apply_lora_to_linears` swaps the right submodules
  * `freeze_non_lora` leaves only A/B matrices (and the head if added) trainable
  * `MultiHeadLoRADinov2` forward returns one (mask, logits) per source present in batch

DINOv2 timm download is NOT required: we substitute a tiny stand-in nn.Module
that has the same forward contract (returns (N, feat_dim) given (N, 3, H, W)).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
import torch.nn as nn

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# class_alignment + handshape_dataset
# ---------------------------------------------------------------------------

def _make_fake_flat_source(tmp_path, n_classes=3, n_per_class=4):
    root = tmp_path / "fake_flat"
    root.mkdir()
    from PIL import Image
    for c in range(n_classes):
        d = root / str(c)
        d.mkdir()
        for i in range(n_per_class):
            Image.new("RGB", (16, 16), color=(c * 30, i * 30, 100)).save(d / f"img_{i}.png")
    return root


def _make_fake_user_source(tmp_path, n_users=2, n_classes=2, n_per=3):
    root = tmp_path / "fake_user"
    root.mkdir()
    from PIL import Image
    for u in range(1, n_users + 1):
        ud = root / f"User {u:02d} (Female, 25)"
        ud.mkdir()
        for c in range(n_classes):
            d = ud / f"Sign {c}"
            d.mkdir()
            for i in range(n_per):
                Image.new("RGB", (16, 16), color=(u * 50, c * 80, i * 30)).save(d / f"u{u}_c{c}_{i}.png")
    return root


def test_discover_flat_source(tmp_path):
    from bangla_handshape.class_alignment import discover_source
    root = _make_fake_flat_source(tmp_path)
    spec = discover_source("flat_test", str(root))
    assert spec.num_classes == 3
    assert sorted(spec.class_to_idx.keys()) == ["0", "1", "2"]


def test_handshape_dataset_returns_tensor_tuples(tmp_path):
    from bangla_handshape.class_alignment import discover_source
    from bangla_handshape.handshape_dataset import HandshapeDataset, enumerate_source
    root = _make_fake_flat_source(tmp_path, n_classes=2, n_per_class=3)
    spec = discover_source("t", str(root))
    items = enumerate_source(spec)
    ds = HandshapeDataset([(spec, items)], transform=None, image_size=16)
    assert len(ds) == 6
    img, src, label = ds[0]
    assert isinstance(img, torch.Tensor) and img.shape == (3, 16, 16)
    assert src == 0 and 0 <= label < 2


def test_handshape_dataset_user_disjoint_split(tmp_path):
    from bangla_handshape.class_alignment import discover_source
    from bangla_handshape.handshape_dataset import (
        enumerate_source, split_user_disjoint,
    )
    root = _make_fake_user_source(tmp_path, n_users=4, n_classes=2, n_per=2)
    spec = discover_source("bdsl47_digits", str(root))   # name triggers user-aware enum
    items = enumerate_source(spec)
    train, val, test = split_user_disjoint(items, val_users={2}, test_users={3})
    train_users = {ent[2] for ent in train}
    assert 2 not in train_users and 3 not in train_users
    assert all(ent[2] == 2 for ent in val)
    assert all(ent[2] == 3 for ent in test)


# ---------------------------------------------------------------------------
# LoRA + multi-head model
# ---------------------------------------------------------------------------

class _StubBackbone(nn.Module):
    """Mimics a DINOv2 backbone for testing: (N,3,H,W) -> (N, feat_dim)."""
    def __init__(self, feat_dim=32):
        super().__init__()
        self.num_features = feat_dim
        self.conv = nn.Conv2d(3, feat_dim, kernel_size=4, stride=2)
        self.attn = nn.ModuleDict({
            "qkv":  nn.Linear(feat_dim, 3 * feat_dim),
            "proj": nn.Linear(feat_dim, feat_dim),
        })
        self.mlp = nn.ModuleDict({
            "fc1": nn.Linear(feat_dim, 2 * feat_dim),
            "fc2": nn.Linear(2 * feat_dim, feat_dim),
        })

    def forward(self, x):
        z = self.conv(x).mean(dim=(2, 3))
        z = z + self.attn["proj"](self.attn["qkv"](z)[:, :z.size(-1)])
        z = z + self.mlp["fc2"](torch.relu(self.mlp["fc1"](z)))
        return z


def test_lora_linear_only_AB_train():
    from bangla_handshape.dinov2_lora import LoRALinear
    base = nn.Linear(8, 16)
    wrap = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(2, 8)
    y = wrap(x)
    assert y.shape == (2, 16)
    # base frozen
    assert not base.weight.requires_grad
    # A/B trainable
    assert wrap.lora_A.weight.requires_grad
    assert wrap.lora_B.weight.requires_grad
    loss = y.pow(2).mean()
    loss.backward()
    assert wrap.lora_A.weight.grad is not None
    assert wrap.lora_B.weight.grad is not None
    assert base.weight.grad is None


def test_apply_lora_replaces_targeted_submodules():
    from bangla_handshape.dinov2_lora import (
        apply_lora_to_linears, freeze_non_lora, LoRALinear,
    )
    bb = _StubBackbone(feat_dim=32)
    n = apply_lora_to_linears(bb, ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
    assert n == 4   # all four targeted Linears wrapped
    # ensure replacements are LoRALinear
    assert isinstance(bb.attn["qkv"], LoRALinear)
    assert isinstance(bb.mlp["fc1"], LoRALinear)
    n_train = freeze_non_lora(bb)
    assert n_train > 0


def test_multihead_lora_dinov2_forward_routes_per_source():
    from bangla_handshape.dinov2_lora import (
        apply_lora_to_linears, freeze_non_lora, MultiHeadLoRADinov2,
    )
    bb = _StubBackbone(feat_dim=32)
    apply_lora_to_linears(bb, ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"], rank=4)
    freeze_non_lora(bb)
    model = MultiHeadLoRADinov2(bb, feat_dim=32,
                                num_classes_per_source=[3, 5, 4],
                                num_lora_replacements=4)
    x = torch.randn(6, 3, 32, 32)
    src = torch.tensor([0, 0, 1, 1, 2, 2])
    outputs = model(x, src)
    assert len(outputs) == 3   # all three sources represented
    for src_i, (mask, logits) in enumerate(outputs):
        assert mask.sum().item() == 2
        assert logits.shape == (2, [3, 5, 4][src_i])


def test_train_utils_loss_and_topk_match_shapes():
    from bangla_handshape.train_utils import multihead_loss, multihead_topk
    # Construct 3-sample batch: 2 from src 0, 1 from src 1
    logits0 = torch.randn(2, 3, requires_grad=True)
    logits1 = torch.randn(1, 5, requires_grad=True)
    mask0 = torch.tensor([True, True, False])
    mask1 = torch.tensor([False, False, True])
    labels = torch.tensor([0, 2, 4])
    outs = [(mask0, logits0), (mask1, logits1)]
    loss = multihead_loss(outs, labels)
    assert torch.isfinite(loss)
    loss.backward()
    accs = multihead_topk(outs, labels, k=1)
    assert set(accs.keys()) == {0, 1}
