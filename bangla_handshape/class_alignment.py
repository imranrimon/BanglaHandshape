"""Class taxonomy bookkeeping across the 4 still-image Bangla SL datasets.

These datasets have *partially* overlapping label sets:

  * BdSL-MNIST            -> 37 classes (Bangla alphabet, integer-named folders)
  * BdSL47 Sign Digits    -> 10 classes (digits 0-9)
  * BdSL47 Sign Letters   -> 37 classes (Bangla letters)
  * BSLD_45               -> 45 classes
  * BDSL 49 Recognition   -> 49 classes (authors' train/test split)

Folder names in every source are integer-stringified ("0", "1", ...). Without
the original papers' label dictionaries we can't be 100% sure two folders named
"0" across two sources mean the same Bangla character, so we keep each source's
labels disjoint by default and let downstream code unify via an explicit
mapping table only when one is verified.

This module just enumerates each source's class folders and builds either:
  * a per-source label space (multi-task, one head per source — used by Path 1
    and Path 3 baseline reporting), or
  * a unified label space (only when --alignment-json is supplied).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SourceSpec:
    """One image-dataset source on disk."""
    name: str
    root: str               # absolute path to the directory of class folders
    num_classes: int        # cached
    class_to_idx: Dict[str, int]


# Default discovery layout for the 5 sources in this repo.
DEFAULT_SOURCES = {
    "bdsl_mnist":          "data/BdSL-MNIST",
    "bdsl47_digits":       "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters":      "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
    "bsld_45":             "data/BSLD_45/Train",
    "bdsl49_recognition":  "data/bdsl49_extracted/Recognition_1/Recognition_1/train",
}


def _list_class_folders(root):
    if not os.path.isdir(root):
        return []
    return sorted([d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))])


def _list_class_folders_recursive_one_level(root):
    """For sources like BdSL47 where structure is `User XX/Sign N/...` we
    enumerate distinct `Sign N` folders from across all users.
    """
    if not os.path.isdir(root):
        return []
    sub_dirs = []
    for user in sorted(os.listdir(root)):
        ud = os.path.join(root, user)
        if not os.path.isdir(ud):
            continue
        for sign in os.listdir(ud):
            if os.path.isdir(os.path.join(ud, sign)):
                sub_dirs.append(sign)
    return sorted(set(sub_dirs))


def discover_source(name, root):
    """Build a SourceSpec for one source by inspecting disk."""
    if name in ("bdsl47_digits", "bdsl47_letters"):
        classes = _list_class_folders_recursive_one_level(root)
    else:
        classes = _list_class_folders(root)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return SourceSpec(name=name, root=root, num_classes=len(classes),
                      class_to_idx=class_to_idx)


def discover_default(repo_root="."):
    """Return SourceSpec list for the 5 default sources, skipping missing ones."""
    out = []
    for name, rel in DEFAULT_SOURCES.items():
        root = os.path.join(repo_root, rel)
        if os.path.isdir(root):
            out.append(discover_source(name, root))
    return out


def write_inventory(sources, out_path):
    payload = {
        "sources": [
            {
                "name": s.name,
                "root": s.root,
                "num_classes": s.num_classes,
                "classes": list(s.class_to_idx.keys()),
            }
            for s in sources
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_inventory(in_path):
    with open(in_path) as f:
        payload = json.load(f)
    return [
        SourceSpec(
            name=s["name"], root=s["root"],
            num_classes=s["num_classes"],
            class_to_idx={c: i for i, c in enumerate(s["classes"])},
        ) for s in payload["sources"]
    ]
