"""E16 step 3: compile a linguist-verified workbook into the canonical alignment JSON.

Deterministic reducer. Input is the verification workbook (built by
tools/build_taxonomy_workbook.py, then edited by a native-signer / SL-linguist and
saved as e.g. alignment/handshape_alignment.verified.csv). Output is the
contiguous-id alignment JSON that banglahandshape/handshape_taxonomy.py loads and
benchmark/analysis/run_lodo.py consumes.

Decision column semantics (see docs/E16_taxonomy_verification/README.md):
    confirm            keep this class in its current canonical_id group
    split              this class does NOT belong -> give it a fresh handshape
    merge_into:<id>    this class's WHOLE group folds into canonical group <id>
    reassign:<id>      move just THIS class into existing group <id>
    drop | (blank)     exclude this class from the unified space (undecided == drop)

<id> targets refer to the ORIGINAL canonical_id shown in the workbook; merges are
resolved transitively. Output canonical ids are renumbered contiguous 0..K-1.

The output is marked "verified": true ONLY with --mark-verified (never silently),
because run_lodo refuses to report unified numbers off an unverified map.

    python -m tools.compile_verified_alignment \
        --workbook alignment/handshape_alignment.verified.csv \
        --out alignment/handshape_alignment.json --mark-verified
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import OrderedDict, defaultdict

_MERGE = re.compile(r"^merge_into:\s*(\d+)\s*$", re.I)
_REASSIGN = re.compile(r"^reassign:\s*(\d+)\s*$", re.I)


def _resolve(target, merge_map, _seen=None):
    """Follow merge_into chains to a fixed point (cycle-guarded)."""
    _seen = _seen or set()
    while target in merge_map and target not in _seen:
        _seen.add(target)
        target = merge_map[target]
    return target


def compile_alignment(rows, version="1.0", mark_verified=False):
    # --- Phase A: group-level merges (merge_into folds a whole original group) ---
    merge_map = {}                      # original_gid -> target original_gid
    for r in rows:
        gid = (r.get("canonical_id") or "").strip()
        m = _MERGE.match((r.get("decision") or "").strip())
        if m and gid.isdigit():
            tgt = m.group(1)
            if gid in merge_map and merge_map[gid] != tgt:
                print(f"[warn] group {gid} has conflicting merge_into "
                      f"({merge_map[gid]} vs {tgt}); keeping first")
            else:
                merge_map[gid] = tgt

    # --- Phase B: per-row provisional key + collect member metadata ---
    key_rows = defaultdict(list)        # provisional_key -> [rows]
    row_key = []                        # (row, key_or_None)
    split_n = 0
    for r in rows:
        dec = (r.get("decision") or "").strip()
        gid = (r.get("canonical_id") or "").strip()
        key = None
        if dec.lower() == "confirm":
            key = ("g", _resolve(gid, merge_map))
        elif _MERGE.match(dec):
            key = ("g", _resolve(gid, merge_map))          # folded into target group
        elif _REASSIGN.match(dec):
            key = ("g", _resolve(_REASSIGN.match(dec).group(1), merge_map))
        elif dec.lower() == "split":
            key = ("s", r.get("source", ""), r.get("class_folder", ""))
        # drop / blank / unknown -> unmapped
        row_key.append((r, key))
        if key is not None:
            key_rows[key].append(r)

    # --- Phase C: assign contiguous canonical ids (stable: original-group order,
    #     then splits in encounter order) ---
    def _order(k):
        if k[0] == "g":
            return (0, int(k[1]) if str(k[1]).isdigit() else 1 << 30)
        return (1, 0)
    ordered_keys = sorted(key_rows.keys(), key=_order)
    # keep encounter order among equal-rank splits
    seen, final_keys = set(), []
    for r, k in row_key:
        if k is not None and k not in seen:
            seen.add(k); final_keys.append(k)
    final_keys.sort(key=_order)                            # groups by id, splits after
    key_to_cid = {k: i for i, k in enumerate(final_keys)}

    # --- Phase D: canonical names + map ---
    canonical = OrderedDict()
    for k in final_keys:
        cid = key_to_cid[k]
        members = key_rows[k]
        def first(col):
            for r in members:
                v = (r.get(col) or "").strip()
                if v:
                    return v
            return ""
        name = first("canonical_name") or first("proposed_bengali") or first("auto_name") or str(cid)
        notes = "; ".join(sorted({(r.get("notes") or "").strip()
                                  for r in members if (r.get("notes") or "").strip()}))
        canonical[str(cid)] = {"name": name, "hamnosys": first("hamnosys"), "notes": notes}

    mapping = defaultdict(dict)
    for r, k in row_key:
        if k is None:
            continue
        src = (r.get("source") or "").strip()
        folder = (r.get("class_folder") or "").strip()
        if src and folder:
            mapping[src][folder] = key_to_cid[k]

    alignment = {
        "version": version,
        "verified": bool(mark_verified),
        "canonical": canonical,
        "map": {s: dict(m) for s, m in mapping.items()},
    }
    return alignment


def _summary(al):
    K = len(al["canonical"])
    canon_src = defaultdict(set)
    for src, m in al["map"].items():
        for _folder, cid in m.items():
            canon_src[cid].add(src)
    shared = sum(1 for s in canon_src.values() if len(s) >= 2)
    mapped = sum(len(m) for m in al["map"].values())
    return K, mapped, shared


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workbook", default="alignment/handshape_alignment.verified.csv")
    ap.add_argument("--out", default="alignment/handshape_alignment.json")
    ap.add_argument("--version", default="1.0")
    ap.add_argument("--mark-verified", action="store_true",
                    help="set verified:true (only pass this after real expert review)")
    args = ap.parse_args()

    with open(args.workbook, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.workbook}: no rows")

    al = compile_alignment(rows, version=args.version, mark_verified=args.mark_verified)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(al, f, indent=2, ensure_ascii=False)

    K, mapped, shared = _summary(al)
    n_dec = sum(1 for r in rows if (r.get("decision") or "").strip())
    print(f"[compile] {len(rows)} workbook rows ({n_dec} decided) -> "
          f"{mapped} classes mapped into {K} canonical handshapes "
          f"({shared} span >=2 sources).  verified={al['verified']}")
    print(f"[compile] wrote {args.out}")
    if not args.mark_verified:
        print("[compile] NOTE: verified=false (pass --mark-verified only after a "
              "qualified annotator has reviewed). run_lodo will warn until then.")
    if shared == 0:
        print("[compile] WARNING: 0 cross-source canonical groups — LODO would be "
              "trivial. Re-check merge_into/reassign decisions or the proposer threshold.")


if __name__ == "__main__":
    main()
