"""PyTorch Dataset for the unified Bangla handshape corpus across 4 sources.

Each item: (image_tensor, source_idx, label_within_source). Use a sampler that
balances source contributions if you need per-source equal exposure during
training; the bare dataset returns items in deterministic listing order.

User-disjoint splits are built from BdSL47's `User XX (...)/Sign N/` layout
(two sources have explicit user metadata in folder names). Other sources lack
user metadata; for them we fall back to a deterministic random 80/10/10 split
seeded by `--seed`.
"""

from __future__ import annotations

import os
import random
import re
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from bangla_handshape.class_alignment import SourceSpec


_USER_RE = re.compile(r"^User\s*(\d+)", re.I)
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_image(name):
    return os.path.splitext(name)[1].lower() in _IMG_EXTS


def _enumerate_with_user(root, class_to_idx):
    """For BdSL47 layout: <root>/User XX (...)/Sign N/<image>."""
    out = []
    for user in sorted(os.listdir(root)):
        user_dir = os.path.join(root, user)
        if not os.path.isdir(user_dir):
            continue
        m = _USER_RE.match(user)
        user_id = int(m.group(1)) if m else -1
        for sign in os.listdir(user_dir):
            sign_dir = os.path.join(user_dir, sign)
            if not os.path.isdir(sign_dir) or sign not in class_to_idx:
                continue
            label = class_to_idx[sign]
            for fn in os.listdir(sign_dir):
                if _is_image(fn):
                    out.append((os.path.join(sign_dir, fn), label, user_id))
    return out


def _enumerate_flat(root, class_to_idx):
    """For BdSL-MNIST / BSLD_45 / BDSL49 layout: <root>/<class>/<image>."""
    out = []
    for cls in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir) or cls not in class_to_idx:
            continue
        label = class_to_idx[cls]
        for fn in os.listdir(cls_dir):
            if _is_image(fn):
                out.append((os.path.join(cls_dir, fn), label, -1))
    return out


def enumerate_source(spec: SourceSpec):
    """Return list of (image_path, label_within_source, user_id_or_minus1)."""
    if spec.name in ("bdsl47_digits", "bdsl47_letters"):
        return _enumerate_with_user(spec.root, spec.class_to_idx)
    return _enumerate_flat(spec.root, spec.class_to_idx)


def split_user_disjoint(entries, val_users, test_users):
    """For sources with user_id, partition by user."""
    train, val, test = [], [], []
    for ent in entries:
        u = ent[2]
        if u in test_users:
            test.append(ent)
        elif u in val_users:
            val.append(ent)
        else:
            train.append(ent)
    return train, val, test


def split_random(entries, seed=0, val_frac=0.10, test_frac=0.10):
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    return (
        shuffled[n_test + n_val:],   # train
        shuffled[n_test:n_test + n_val],
        shuffled[:n_test],
    )


def cap_per_class(entries, max_per_class, seed=0):
    """Few-shot (A4): keep at most `max_per_class` items per label, chosen
    deterministically from `seed`. `entries`: (path, label, user_id) list.
    A non-positive / falsy cap is a no-op (returns entries unchanged)."""
    if not max_per_class or int(max_per_class) <= 0:
        return entries
    rng = random.Random(seed)
    by_label = {}
    for e in entries:
        by_label.setdefault(e[1], []).append(e)
    out = []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        out.extend(items[: int(max_per_class)])
    return out


class HandshapeDataset(Dataset):
    """Multi-source aggregated handshape image dataset.

    Each entry is (PIL or transformed tensor, source_idx, label). The label is
    *within* its source's label space; downstream multi-head models route by
    source_idx. Pass image_size and a torchvision-style transform via the
    `transform` argument.
    """

    def __init__(
        self,
        sources_with_entries: List[Tuple[SourceSpec, List]],
        transform=None,
        image_size: int = 224,
    ):
        self.sources = [s for s, _e in sources_with_entries]
        self.image_size = image_size
        self.transform = transform
        # Flatten: (path, source_idx, label_within_source)
        self.items = []
        for src_idx, (_spec, entries) in enumerate(sources_with_entries):
            for path, label, _user_id in entries:
                self.items.append((path, src_idx, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, src_idx, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            # default: just resize + tensor in [0, 1]
            img = img.resize((self.image_size, self.image_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1)
        return img, src_idx, label

    def num_classes_per_source(self):
        return [s.num_classes for s in self.sources]

    def source_names(self):
        return [s.name for s in self.sources]
