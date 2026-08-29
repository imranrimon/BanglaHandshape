"""Tier-3 unified handshape taxonomy — map each source's disjoint *local* classes
into ONE canonical handshape label space, enabling a unified benchmark and
leave-one-dataset-out (LODO) cross-corpus classification.

The five sources keep disjoint label spaces by default (see class_alignment.py):
folders are integer-named and we can't assume "0" in one dataset is the same
Bangla handshape as "0" in another. This module is the *opt-in* unification the
repo anticipates — activated only by a VERIFIED alignment JSON.

Alignment JSON schema (propose_alignment.py drafts it; a signer/linguist verifies):

    {
      "version": "0.1",
      "verified": false,                     # flip to true ONLY after expert review
      "canonical": {                         # contiguous ids "0".."K-1"
        "0": {"name": "A-hand", "hamnosys": "", "notes": ""},
        ...
      },
      "map": {                               # source -> {local_class_folder: canonical_id}
        "bdsl47_digits": {"Sign 0": 0, "Sign 1": 3, ...},
        "bdsl_mnist":    {"0": 12, "1": 7, ...},
        ...
      }
    }

`map` keys are the on-disk class-folder names (== SourceSpec.class_to_idx keys).
A local class absent from `map` is UNMAPPED: excluded from the unified / LODO
label space and reported in the coverage table. An unverified alignment is for
exploration only — never report unified numbers off an unverified map.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

from banglahandshape.class_alignment import SourceSpec


def load_alignment(path: str) -> dict:
    with open(path) as f:
        al = json.load(f)
    if "canonical" not in al or "map" not in al:
        raise ValueError(f"{path}: alignment JSON needs 'canonical' and 'map' keys")
    # canonical ids must be contiguous 0..K-1
    ids = sorted(int(k) for k in al["canonical"])
    if ids and ids != list(range(len(ids))):
        raise ValueError(f"{path}: canonical ids must be contiguous 0..K-1, got {ids[:5]}...")
    return al


def num_canonical(alignment: dict) -> int:
    return len(alignment["canonical"])


def canonical_name(alignment: dict, cid: int) -> str:
    return alignment["canonical"].get(str(cid), {}).get("name", str(cid))


def source_local_to_canonical(spec: SourceSpec, alignment: dict) -> Dict[int, int]:
    """{local_label_idx -> canonical_id} for one source (mapped classes only)."""
    m = alignment["map"].get(spec.name, {})
    out = {}
    for folder, local_idx in spec.class_to_idx.items():
        if folder in m:
            out[int(local_idx)] = int(m[folder])
    return out


def remap_entries(entries: List[Tuple], local2canon: Dict[int, int]) -> List[Tuple]:
    """(path, local_label, user_id) -> (path, canonical_label, user_id),
    dropping items whose local class is unmapped."""
    out = []
    for path, label, user in entries:
        c = local2canon.get(int(label))
        if c is not None:
            out.append((path, int(c), user))
    return out


def coverage(sources: List[SourceSpec], alignment: dict) -> dict:
    """Per source: mapped/total classes; per canonical id: how many sources have it."""
    K = num_canonical(alignment)
    per_source = {}
    canon_sources = {c: set() for c in range(K)}
    for spec in sources:
        l2c = source_local_to_canonical(spec, alignment)
        per_source[spec.name] = {"mapped": len(l2c), "total": spec.num_classes}
        for cid in l2c.values():
            if cid in canon_sources:
                canon_sources[cid].add(spec.name)
    return {
        "num_canonical": K,
        "verified": bool(alignment.get("verified", False)),
        "per_source": per_source,
        "canonical_source_count": {c: len(s) for c, s in canon_sources.items()},
        "shared_by_2plus": sum(1 for s in canon_sources.values() if len(s) >= 2),
    }
