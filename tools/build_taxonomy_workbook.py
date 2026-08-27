"""E16 — assemble a linguist-facing verification workbook for the unified taxonomy.

The compute step (`propose_alignment.py`) already drafted a candidate canonical
grouping (`alignment/handshape_alignment.proposed.json`) with per-class cross-source
similarity evidence (`alignment/handshape_alignment_review.csv`). Those are
machine artifacts (`cluster_N` names). This turns them into ONE workbook a native
signer / SL-linguist edits, seeded with the E15 Bengali labels so BdSL47 rows already
show their grapheme.

Output `alignment/verification_workbook.csv`, one row per member class, sorted so each
candidate canonical group is contiguous, with:
  - group context: canonical_id, group_size, group_sources (how many distinct
    sources the group already spans — groups with >=2 are the ones that make LODO
    non-trivial, so prioritise confirming those);
  - the member: source, class_folder, n_images, proposed_bengali;
  - the evidence: top / second cross-source match + cosine, needs_review;
  - blank DECISION columns for the expert: decision, canonical_name, hamnosys, notes.

Decision codes (see docs/E16_taxonomy_verification/README.md):
  confirm | split | merge_into:<id> | reassign:<id> | drop | (blank = undecided)

Run (no GPU, inputs already on disk):
    python -m tools.build_taxonomy_workbook
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict


def _load_bengali(path):
    """{(source, class_folder): proposed_bengali} from the E15 draft, if present."""
    out = {}
    if path and os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                b = r.get("confirmed_bengali") or r.get("proposed_bengali") or ""
                out[(r["source"], r["class_folder"])] = b
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review-csv", default="alignment/handshape_alignment_review.csv")
    ap.add_argument("--proposed-json", default="alignment/handshape_alignment.proposed.json")
    ap.add_argument("--bengali-csv",
                    default="docs/E15_handshape_annotation/sign_to_bengali_DRAFT.csv")
    ap.add_argument("--out", default="alignment/verification_workbook.csv")
    args = ap.parse_args()

    with open(args.review_csv, newline="", encoding="utf-8") as f:
        members = list(csv.DictReader(f))
    canon = {}
    if os.path.exists(args.proposed_json):
        with open(args.proposed_json, encoding="utf-8") as f:
            canon = json.load(f).get("canonical", {})
    beng = _load_bengali(args.bengali_csv)

    # group stats
    size = defaultdict(int)
    srcs = defaultdict(set)
    for m in members:
        cid = m["canonical_id"]
        size[cid] += 1
        srcs[cid].add(m["source"])

    # sort: cross-source groups first (most useful for LODO), then by group size desc
    def keyf(m):
        cid = m["canonical_id"]
        return (-len(srcs[cid]), -size[cid], int(cid) if cid.isdigit() else 0,
                m["source"], m["class_folder"])
    members.sort(key=keyf)

    cols = ["canonical_id", "group_size", "group_sources", "auto_name",
            "decision", "canonical_name", "hamnosys",           # <- expert fills these
            "source", "class_folder", "proposed_bengali", "n_images",
            "top_cross_source_match", "top_cross_source_cosine",
            "second_cross_source_match", "second_cosine", "needs_review", "notes"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in members:
            cid = m["canonical_id"]
            w.writerow({
                "canonical_id": cid,
                "group_size": size[cid],
                "group_sources": len(srcs[cid]),
                "auto_name": canon.get(cid, {}).get("name", ""),
                "decision": "", "canonical_name": "", "hamnosys": "",
                "source": m["source"], "class_folder": m["class_folder"],
                "proposed_bengali": beng.get((m["source"], m["class_folder"]), ""),
                "n_images": m.get("n_images", ""),
                "top_cross_source_match": m.get("top_cross_source_match", ""),
                "top_cross_source_cosine": m.get("top_cross_source_cosine", ""),
                "second_cross_source_match": m.get("second_cross_source_match", ""),
                "second_cosine": m.get("second_cosine", ""),
                "needs_review": m.get("needs_review", ""),
                "notes": "",
            })

    n_groups = len(size)
    n_multi = sum(1 for c in srcs if len(srcs[c]) >= 2)
    print(f"[E16] {len(members)} member classes over {n_groups} candidate groups "
          f"({n_multi} span >=2 sources).")
    print(f"[E16] wrote {args.out}")
    if n_multi <= 1:
        print("[E16] WARNING: <=1 cross-source group. The 0.92-threshold proposal "
              "under-merges — re-run propose_alignment.py at --sim-threshold 0.90 "
              "with ALL 5 sources on disk before verification (see README).")


if __name__ == "__main__":
    main()
