"""Tier-3, step 1 ("propose") — compute-assisted DRAFT of a cross-corpus
handshape alignment.

The five still-image sources keep DISJOINT local label spaces (see
bangla_handshape/class_alignment.py): integer-named folders can't be assumed to
mean the same Bangla handshape across datasets. Tier-3 unifies them via a
VERIFIED alignment JSON (bangla_handshape/handshape_taxonomy.py). Verifying
by hand is the bottleneck; this script drafts the map so a signer/linguist only
has to CHECK it, not build it from scratch.

Pipeline:
  1. Enumerate every source (discover_default), sample up to --per-class images
     per local class.
  2. Extract FROZEN DINOv2 features (same pattern as train_fusion.py); the
     prototype for a (source, class) is the L2-normalized mean feature.
  3. Pool all prototypes across ALL sources and cluster them by cosine
     similarity (average-linkage agglomerative). Each cluster = one canonical
     handshape id (contiguous 0..K-1).
  4. Emit:
       * a proposed alignment JSON in the handshape_taxonomy.py schema, with
         "verified": false  (NEVER report unified numbers off an unverified map);
       * a human-review CSV carrying the cross-source evidence (nearest
         prototype from a DIFFERENT source) so the reviewer can sanity-check /
         edit each row.

The JSON round-trips through handshape_taxonomy.load_alignment: canonical ids
are contiguous strings "0".."K-1", and `map` keys are the on-disk class-folder
names (== SourceSpec.class_to_idx keys).

Usage:
    python -m path3_handshape_benchmark.propose_alignment \
        --out alignment/handshape_alignment.proposed.json \
        --review-csv alignment/handshape_alignment_review.csv \
        [--per-class 60] [--sim-threshold 0.55] \
        [--limit-sources bdsl47_digits bdsl_mnist]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Weights are pre-warmed in the HF cache; compute nodes have no outbound HTTPS.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from bangla_handshape.class_alignment import discover_default
from bangla_handshape.handshape_dataset import enumerate_source
from path3_handshape_benchmark.train_baseline import _build_transforms
from path3_handshape_benchmark.train_fusion import _PathDS


# --------------------------------------------------------------------------- #
# Feature extraction (mirror train_fusion.py's frozen-feature pattern).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _extract_feats(paths, timm_name, image_size, device, batch=64, nw=4):
    """(N, D) frozen DINOv2 features for `paths`, in input order."""
    import timm
    backbone = timm.create_model(timm_name, pretrained=True, num_classes=0,
                                 dynamic_img_size=True).to(device).eval()
    D = int(getattr(backbone, "num_features", 384))
    ds = _PathDS(paths, _build_transforms(image_size))
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=nw,
                        pin_memory=torch.cuda.is_available())
    feats = np.zeros((len(paths), D), dtype=np.float32)
    for x, idx in loader:
        f = backbone(x.to(device, non_blocking=True)).float().cpu().numpy()
        feats[idx.numpy()] = f
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return feats


def _l2norm(m, axis=-1, eps=1e-12):
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=axis, keepdims=True)
    return m / np.maximum(n, eps)


# --------------------------------------------------------------------------- #
# Prototype building.
# --------------------------------------------------------------------------- #
def _build_prototypes(sources, per_class, timm_name, image_size, device,
                      batch, nw, rng):
    """Return parallel lists:
        source_names[i], folder_names[i], local_idxs[i], n_images[i]
    and a (K, D) matrix of L2-normalized prototype vectors.

    For each source, groups image paths by local_label_idx, samples up to
    `per_class` per class, extracts features in ONE batched pass per source,
    then averages per class and L2-normalizes."""
    src_names, folders, local_idxs, n_imgs, protos = [], [], [], [], []
    for spec in sources:
        # invert class_to_idx: local_idx -> folder_name
        idx_to_folder = {i: f for f, i in spec.class_to_idx.items()}
        entries = enumerate_source(spec)  # (path, local_label_idx, user_id)
        by_class = {}
        for path, label, _user in entries:
            by_class.setdefault(int(label), []).append(path)
        if not by_class:
            print(f"[WARN] {spec.name}: no images enumerated; skip", flush=True)
            continue

        # deterministic per-class sampling, flattened for one batched forward
        flat_paths, class_slices = [], []
        for local_idx in sorted(by_class):
            paths = list(by_class[local_idx])
            rng.shuffle(paths)
            paths = paths[: max(1, int(per_class))]
            start = len(flat_paths)
            flat_paths.extend(paths)
            class_slices.append((local_idx, start, len(flat_paths)))

        print(f"[{spec.name}] classes={len(class_slices)} "
              f"sampled_images={len(flat_paths)}", flush=True)
        feats = _extract_feats(flat_paths, timm_name, image_size, device,
                               batch=batch, nw=nw)

        for local_idx, lo, hi in class_slices:
            if hi <= lo:
                continue
            proto = feats[lo:hi].mean(axis=0)
            protos.append(proto)
            src_names.append(spec.name)
            folders.append(idx_to_folder.get(local_idx, str(local_idx)))
            local_idxs.append(int(local_idx))
            n_imgs.append(int(hi - lo))

    if not protos:
        return src_names, folders, local_idxs, n_imgs, np.zeros((0, 0),
                                                                 dtype=np.float32)
    P = _l2norm(np.stack(protos, axis=0), axis=1)  # (K, D)
    return src_names, folders, local_idxs, n_imgs, P


# --------------------------------------------------------------------------- #
# Clustering.
# --------------------------------------------------------------------------- #
def _cluster_sklearn(P, sim_threshold):
    """AgglomerativeClustering with cosine average-linkage.
    distance_threshold = 1 - sim_threshold. Returns int labels or None if
    sklearn is unavailable."""
    try:
        from sklearn.cluster import AgglomerativeClustering
    except Exception:
        return None
    if P.shape[0] == 1:
        return np.zeros(1, dtype=np.int64)
    dist = float(max(0.0, 1.0 - sim_threshold))
    # sklearn>=1.2 uses `metric=`; older uses `affinity=`.
    try:
        model = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=dist)
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None, affinity="cosine", linkage="average",
            distance_threshold=dist)
    return model.fit_predict(P).astype(np.int64)


def _cluster_greedy(P, sim_threshold):
    """Greedy average-linkage agglomerative merging on cosine similarity.
    Start every prototype in its own group; repeatedly merge the two groups
    whose mean-prototype cosine is highest, while that max >= sim_threshold.
    Average-linkage here == cosine of the (L2-normalized) group MEAN vectors."""
    n = P.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    members = [[i] for i in range(n)]           # prototype indices per group
    means = _l2norm(P.copy(), axis=1)            # group mean (normalized)
    alive = list(range(n))
    while len(alive) > 1:
        M = means[alive]                         # (g, D)
        sim = M @ M.T                            # cosine (rows normalized)
        np.fill_diagonal(sim, -np.inf)
        flat = int(np.argmax(sim))
        a, b = divmod(flat, sim.shape[1])
        best = sim[a, b]
        if best < sim_threshold:
            break
        ga, gb = alive[a], alive[b]              # merge gb into ga
        members[ga].extend(members[gb])
        merged = P[members[ga]].mean(axis=0)
        means[ga] = _l2norm(merged[None, :], axis=1)[0]
        alive.remove(gb)
    # relabel alive groups to contiguous 0..K-1 (stable by group order)
    labels = np.empty(n, dtype=np.int64)
    for new_id, g in enumerate(sorted(alive)):
        for idx in members[g]:
            labels[idx] = new_id
    return labels


def _contiguous(labels):
    """Relabel arbitrary int labels to contiguous 0..K-1, ordered by first
    appearance (keeps output deterministic and matches the taxonomy schema)."""
    remap, out = {}, np.empty(len(labels), dtype=np.int64)
    for i, l in enumerate(labels):
        l = int(l)
        if l not in remap:
            remap[l] = len(remap)
        out[i] = remap[l]
    return out


# --------------------------------------------------------------------------- #
# Cross-source evidence for the review CSV.
# --------------------------------------------------------------------------- #
def _cross_source_matches(P, src_names):
    """For each prototype, the top-2 most-similar prototypes from a DIFFERENT
    source. Returns (top_idx, top_cos, second_idx, second_cos) arrays; index -1
    means no cross-source prototype exists."""
    n = P.shape[0]
    top_i = np.full(n, -1, dtype=np.int64)
    sec_i = np.full(n, -1, dtype=np.int64)
    top_c = np.zeros(n, dtype=np.float32)
    sec_c = np.zeros(n, dtype=np.float32)
    if n == 0:
        return top_i, top_c, sec_i, sec_c
    sim = P @ P.T
    same = np.array([[src_names[i] == src_names[j] for j in range(n)]
                     for i in range(n)], dtype=bool)
    masked = sim.copy()
    masked[same] = -np.inf
    for i in range(n):
        row = masked[i]
        order = np.argsort(row)[::-1]
        order = [j for j in order if np.isfinite(row[j])]
        if order:
            top_i[i] = order[0]
            top_c[i] = float(row[order[0]])
        if len(order) > 1:
            sec_i[i] = order[1]
            sec_c[i] = float(row[order[1]])
    return top_i, top_c, sec_i, sec_c


# --------------------------------------------------------------------------- #
# Emit.
# --------------------------------------------------------------------------- #
def _write_json(out_path, labels, src_names, folders, n_imgs):
    K = int(labels.max()) + 1 if len(labels) else 0
    # canonical group stats
    group_srcs = {c: set() for c in range(K)}
    group_n = {c: 0 for c in range(K)}
    for c, s in zip(labels, src_names):
        group_srcs[int(c)].add(s)
        group_n[int(c)] += 1
    canonical = {}
    for c in range(K):
        canonical[str(c)] = {
            "name": f"cluster_{c}",
            "hamnosys": "",
            "notes": f"auto: {group_n[c]} classes from {len(group_srcs[c])} sources",
        }
    mp = {}
    for c, s, folder in zip(labels, src_names, folders):
        mp.setdefault(s, {})[folder] = int(c)
    payload = {
        "version": "0.1",
        "verified": False,
        "canonical": canonical,
        "map": mp,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return K, group_srcs, group_n


def _write_review_csv(csv_path, labels, src_names, folders, local_idxs, n_imgs,
                      top_i, top_c, sec_i, sec_c, sim_threshold):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    flag_cut = sim_threshold + 0.05

    def _fmt(idx):
        if idx is None or int(idx) < 0:
            return ""
        return f"{src_names[int(idx)]}:{folders[int(idx)]}"

    rows = []
    for i in range(len(labels)):
        rows.append({
            "canonical_id": int(labels[i]),
            "source": src_names[i],
            "class_folder": folders[i],
            "n_images": int(n_imgs[i]),
            "top_cross_source_match": _fmt(top_i[i]),
            "top_cross_source_cosine": f"{float(top_c[i]):.4f}"
                                       if int(top_i[i]) >= 0 else "",
            "second_cross_source_match": _fmt(sec_i[i]),
            "second_cosine": f"{float(sec_c[i]):.4f}"
                             if int(sec_i[i]) >= 0 else "",
            "needs_review": 1 if (int(top_i[i]) < 0
                                  or float(top_c[i]) < flag_cut) else 0,
        })
    rows.sort(key=lambda r: (r["canonical_id"], r["source"]))
    fields = ["canonical_id", "source", "class_folder", "n_images",
              "top_cross_source_match", "top_cross_source_cosine",
              "second_cross_source_match", "second_cosine", "needs_review"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return sum(r["needs_review"] for r in rows)


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",
                    default="alignment/handshape_alignment.proposed.json")
    ap.add_argument("--review-csv",
                    default="alignment/handshape_alignment_review.csv")
    ap.add_argument("--timm-name", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--sim-threshold", type=float, default=0.55)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit-sources", nargs="+", default=None,
                    help="optional subset of source names for a quick dry run")
    args = ap.parse_args()

    rng = random.Random(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    sources = discover_default(".")
    if args.limit_sources:
        keep = set(args.limit_sources)
        sources = [s for s in sources if s.name in keep]
    if not sources:
        raise SystemExit(
            "No sources discovered. Re-fetch data (docs/SISTER_PAPER_DATA_REFETCH.md) "
            "or pass valid --limit-sources names.")
    print(f"[sources] {[s.name for s in sources]}", flush=True)

    src_names, folders, local_idxs, n_imgs, P = _build_prototypes(
        sources, args.per_class, args.timm_name, args.image_size, device,
        args.batch, args.num_workers, rng)
    if P.shape[0] == 0:
        raise SystemExit("No prototypes built (no images found).")
    print(f"[prototypes] {P.shape[0]} across {len(set(src_names))} sources, "
          f"D={P.shape[1]}", flush=True)

    labels = _cluster_sklearn(P, args.sim_threshold)
    if labels is None:
        print("[cluster] sklearn unavailable; greedy fallback", flush=True)
        labels = _cluster_greedy(P, args.sim_threshold)
    else:
        print("[cluster] sklearn AgglomerativeClustering (cosine/average)",
              flush=True)
    labels = _contiguous(labels)

    top_i, top_c, sec_i, sec_c = _cross_source_matches(P, src_names)

    K, group_srcs, group_n = _write_json(args.out, labels, src_names, folders,
                                          n_imgs)
    n_flagged = _write_review_csv(args.review_csv, labels, src_names, folders,
                                  local_idxs, n_imgs, top_i, top_c, sec_i,
                                  sec_c, args.sim_threshold)

    shared = sum(1 for c in range(K) if len(group_srcs[c]) >= 2)
    mean_size = (sum(group_n.values()) / K) if K else 0.0
    print("=== PROPOSED ALIGNMENT (DRAFT — verified=false) ===", flush=True)
    print(f"  canonical groups K           : {K}", flush=True)
    print(f"  shared by >=2 sources        : {shared}", flush=True)
    print(f"  mean group size (classes)    : {mean_size:.2f}", flush=True)
    print(f"  classes flagged needs_review : {n_flagged} / {len(labels)}",
          flush=True)
    print(f"  JSON  -> {args.out}", flush=True)
    print(f"  CSV   -> {args.review_csv}", flush=True)
    print("  NEXT: a signer/linguist edits the review CSV, then flips "
          "verified=true.", flush=True)


if __name__ == "__main__":
    main()
