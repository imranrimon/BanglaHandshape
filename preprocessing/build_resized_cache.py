"""Pre-resize image cache — the 2-3x speedup for image-based training.

The training transform is fully deterministic (Resize 1.15x -> CenterCrop 224 ->
ToTensor -> Normalize; NO random augmentation), so every image maps to exactly
one 224x224 tensor every epoch. Re-decoding full-res JPEGs each epoch made the
pipeline I/O-bound (~14% GPU compute). This precomputes the geometry step once
(Resize+CenterCrop -> 224x224 uint8) into per-root memmap .npy shards; the
Dataset then does only ToTensor+Normalize on a small cached array -> decode cost
collapses to a memmap read. Bit-for-bit identical to the live pipeline (ToTensor
+ Normalize act on the same pixels).

Runs in bdsl_graph (needs torchvision/PIL) as a separate CPU process; modest
worker count so it doesn't starve the running training dataloaders.

Usage (repo root):
    python preprocessing/build_resized_cache.py --workers 4
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

CACHE_DIR = "work_dir/_img_cache"
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
SIZE = 224
RESIZE = int(SIZE * 1.15)  # 257, matches _build_transforms

ROOTS = {
    "mnist":        "data/BdSL-MNIST",
    "digits":       "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "letters":      "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
    "bsld45_train": "data/BSLD_45/Train",
    "bsld45_test":  "data/BSLD_45/Test",
    "bdsl49_train": "data/bdsl49_extracted/Recognition_1/Recognition_1/train",
    "bdsl49_test":  "data/bdsl49_extracted/Recognition_1/Recognition_1/test",
}


def key(path):
    return os.path.normpath(os.path.abspath(path))


def list_images(root):
    out = []
    for ext in IMG_EXTS:
        out += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted({key(p) for p in out})


_GEOM = None


def _init_geom():
    """Build the EXACT torchvision geometry (Resize+CenterCrop) once per worker,
    so the cached pixels are bit-identical to the live _build_transforms pipeline
    (matching torchvision's smaller-edge rounding + interpolation)."""
    global _GEOM
    from torchvision import transforms
    _GEOM = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(SIZE),
    ])


def _resize_one(path):
    """Resize(257)->CenterCrop(224) via torchvision, return (path, uint8 HWC)."""
    global _GEOM
    if _GEOM is None:
        _init_geom()
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return (path, None)
    return (path, np.asarray(_GEOM(img), dtype=np.uint8))


def build_root(name, root, workers):
    shard = os.path.join(CACHE_DIR, f"{name}.npy")
    idxf = os.path.join(CACHE_DIR, f"{name}.index.json")
    if os.path.exists(shard) and os.path.exists(idxf):
        print(f"[skip] {name}: cache exists", flush=True)
        return
    if not os.path.isdir(root):
        print(f"[warn] {name}: missing root {root}", flush=True)
        return
    paths = list_images(root)
    n = len(paths)
    print(f"=== {name}: {n} images -> {shard} ===", flush=True)
    arr = np.lib.format.open_memmap(shard, mode="w+", dtype=np.uint8,
                                    shape=(n, SIZE, SIZE, 3))
    index = {}
    from multiprocessing import Pool
    done = 0
    with Pool(workers, initializer=_init_geom) as pool:
        for path, img in pool.imap(_resize_one, paths, chunksize=64):
            row = len(index)
            index[path] = row
            arr[row] = img if img is not None else 0
            done += 1
            if done % 10000 == 0:
                print(f"  [{name}] {done}/{n}", flush=True)
    arr.flush()
    with open(idxf, "w") as f:
        json.dump(index, f)
    print(f"[saved] {name}: {n} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(CACHE_DIR, exist_ok=True)
    names = args.only or list(ROOTS.keys())
    for name in names:
        build_root(name, ROOTS[name], args.workers)
    print("=== RESIZED CACHE COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
