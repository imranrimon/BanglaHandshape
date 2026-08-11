"""Hand-crop ablation — cache the PIXEL bounding box of the detected hand.

Runs in the ISOLATED `mp_kp` conda env (mediapipe 1.0.0 Tasks API), NOT
`bdsl_graph` — mediapipe is kept out of the training env so it can't perturb the
running campaign. Mirrors `extract_keypoints.py` (same model bundle, same
HandLandmarker options, same SOURCES / list_images / key helpers, same
--only/--force cache-skip logic, same cv2 BGR->RGB read).

DIFFERENCE from extract_keypoints.py: the keypoint cache stores only normalized
wrist-centred coords, so the RAW pixel bbox is not recoverable from it. Here,
instead of normalizing the landmarks we compute the pixel-space bounding box of
the 21 landmarks, expand it by a margin (default 0.25 of the bbox's max side),
and clamp it to the image bounds. Undetected images get detected=0 and
bbox=[0,0,w,h] (the full image — the trainer treats that as "no crop"). The
per-source detection rate is itself a benchmark statistic.

Output is a per-source `.npz` at `work_dir/_bbox_cache/<source>.npz` with:
    paths    (n,)  object  absolute normalized image paths (same order as extract_keypoints.py)
    bbox     (n,4) int32   [x0,y0,x1,y1] in PIXELS, margin-expanded + clamped
    detected (n,)  uint8   1 if a hand was found

Usage (repo root):
    /c/Users/rimon/anaconda3/envs/mp_kp/python.exe preprocessing/extract_hand_bbox.py \
        --only bdsl47_digits bdsl47_letters
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python import vision

MODEL = ".kp_models/hand_landmarker.task"
CACHE_DIR = "work_dir/_bbox_cache"
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")

SOURCES = {
    "bdsl_mnist":     r"data/BdSL-MNIST",
    "bdsl47_digits":  r"data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": r"data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
    "bsld_45":        r"data/BSLD_45/Train",
    "bdsl49_recognition": r"data/bdsl49_extracted/Recognition_1/Recognition_1/train",
}

# default = only the SD/SI comparison sources (BdSL47); --only can name any source
DEFAULT_SOURCES = ["bdsl47_digits", "bdsl47_letters"]


def key(path):
    return os.path.normpath(os.path.abspath(path))


def list_images(root):
    out = []
    for ext in IMG_EXTS:
        out += glob.glob(os.path.join(root, "**", ext), recursive=True)
    # de-dup (case-insensitive globs can double-count on Windows) + stable order
    return sorted({key(p) for p in out})


def make_landmarker():
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(opts)


def landmark_bbox(landmarks, w, h, margin):
    """Pixel bbox over the 21 landmarks, margin-expanded and clamped to [0,w]/[0,h]."""
    xs = np.array([p.x for p in landmarks], dtype=np.float32) * w  # normalized -> pixels
    ys = np.array([p.y for p in landmarks], dtype=np.float32) * h
    x0, y0 = float(xs.min()), float(ys.min())
    x1, y1 = float(xs.max()), float(ys.max())
    pad = margin * max(x1 - x0, y1 - y0)     # margin fraction of the bbox's max side
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    x0 = int(round(min(max(x0, 0.0), w)))    # clamp to image bounds
    y0 = int(round(min(max(y0, 0.0), h)))
    x1 = int(round(min(max(x1, 0.0), w)))
    y1 = int(round(min(max(y1, 0.0), h)))
    return [x0, y0, x1, y1]


def extract_source(name, root, landmarker, margin, log_every=2000):
    imgs = list_images(root)
    n = len(imgs)
    bbox = np.zeros((n, 4), dtype=np.int32)
    detected = np.zeros((n,), dtype=np.uint8)
    t0 = time.time()
    for i, path in enumerate(imgs):
        im = cv2.imread(path)
        if im is None:
            continue
        h, w = im.shape[:2]
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_img)
        if res.hand_landmarks:
            bbox[i] = landmark_bbox(res.hand_landmarks[0], w, h, margin)
            detected[i] = 1
        else:
            bbox[i] = [0, 0, w, h]           # undetected -> full image ("no crop")
        if (i + 1) % log_every == 0:
            rate = 100.0 * detected[: i + 1].sum() / (i + 1)
            print(f"  [{name}] {i+1}/{n}  det={rate:4.1f}%  "
                  f"{(time.time()-t0)/(i+1)*1000:.0f} ms/img", flush=True)
    rate = 100.0 * detected.sum() / max(1, n)
    print(f"[{name}] DONE n={n} detected={int(detected.sum())} ({rate:.1f}%) "
          f"in {time.time()-t0:.0f}s", flush=True)
    # Unicode (not object) dtype so the .npz is pickle-free and loads in the
    # numpy-1.x training env (an object array pickles numpy 2.x's numpy._core).
    return np.asarray(imgs, dtype="U"), bbox, detected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of source names; default = bdsl47_digits bdsl47_letters")
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="bbox margin as a fraction of the bbox's max side")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(MODEL):
        sys.exit(f"missing model bundle: {MODEL}")
    names = args.only or DEFAULT_SOURCES
    landmarker = make_landmarker()
    for name in names:
        root = SOURCES[name]
        out = os.path.join(CACHE_DIR, f"{name}.npz")
        if os.path.exists(out) and not args.force:
            print(f"[skip] {name}: cache exists ({out})", flush=True)
            continue
        if not os.path.isdir(root):
            print(f"[warn] {name}: missing root {root}", flush=True)
            continue
        print(f"=== extracting {name} bbox from {root} (margin={args.margin}) ===", flush=True)
        paths, bbox, detected = extract_source(name, root, landmarker, args.margin)
        np.savez_compressed(out, paths=paths, bbox=bbox, detected=detected)
        print(f"[saved] {out}", flush=True)
    print("=== HAND BBOX EXTRACTION COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
