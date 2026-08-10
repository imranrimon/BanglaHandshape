"""A6 modality control — extract MediaPipe hand keypoints from handshape images.

Runs in the ISOLATED `mp_kp` conda env (mediapipe 1.0.0 Tasks API), NOT
`bdsl_graph` — mediapipe is kept out of the training env so it can't perturb the
running campaign. Output is a per-source `.npz` keypoint cache keyed by absolute
image path; the MLP trainer (`train_keypoint.py`, bdsl_graph) reads the cache and
never touches mediapipe.

Each detected hand -> 21 landmarks (x,y,z) -> a 63-d vector, made
translation/scale invariant (wrist-centred, unit max-radius) and mirrored to a
canonical right hand via MediaPipe handedness. Undetected images get a zero
vector + detected=0 (the trainer drops them). The per-source detection rate is
itself a benchmark statistic (e.g. BSLD_45 = 0% because the dataset ships images
with the hand skeleton already drawn on — see project note).

Usage (repo root):
    /c/Users/rimon/anaconda3/envs/mp_kp/python.exe preprocessing/extract_keypoints.py \
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
CACHE_DIR = "work_dir/_kp_cache"
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")

SOURCES = {
    "bdsl_mnist":     r"data/BdSL-MNIST",
    "bdsl47_digits":  r"data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": r"data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
    "bsld_45":        r"data/BSLD_45/Train",
    "bdsl49_recognition": r"data/bdsl49_extracted/Recognition_1/Recognition_1/train",
}


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


def normalize(landmarks, handed_label):
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)  # (21,3)
    if handed_label == "Left":              # mirror to a canonical right hand
        pts[:, 0] = -pts[:, 0]
    pts = pts - pts[0:1]                     # wrist-centred (translation-invariant)
    scale = float(np.linalg.norm(pts, axis=1).max())
    if scale > 1e-6:
        pts = pts / scale                    # unit max-radius (scale-invariant)
    return pts.reshape(-1).astype(np.float32)  # (63,)


def extract_source(name, root, landmarker, log_every=2000):
    imgs = list_images(root)
    n = len(imgs)
    kp = np.zeros((n, 63), dtype=np.float32)
    detected = np.zeros((n,), dtype=np.uint8)
    t0 = time.time()
    for i, path in enumerate(imgs):
        im = cv2.imread(path)
        if im is None:
            continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_img)
        if res.hand_landmarks:
            handed = "Right"
            if res.handedness and res.handedness[0]:
                handed = res.handedness[0][0].category_name
            kp[i] = normalize(res.hand_landmarks[0], handed)
            detected[i] = 1
        if (i + 1) % log_every == 0:
            rate = 100.0 * detected[: i + 1].sum() / (i + 1)
            print(f"  [{name}] {i+1}/{n}  det={rate:4.1f}%  "
                  f"{(time.time()-t0)/(i+1)*1000:.0f} ms/img", flush=True)
    rate = 100.0 * detected.sum() / max(1, n)
    print(f"[{name}] DONE n={n} detected={int(detected.sum())} ({rate:.1f}%) "
          f"in {time.time()-t0:.0f}s", flush=True)
    return np.array(imgs, dtype=object), kp, detected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of source names; default = all")
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(MODEL):
        sys.exit(f"missing model bundle: {MODEL}")
    names = args.only or list(SOURCES.keys())
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
        print(f"=== extracting {name} from {root} ===", flush=True)
        paths, kp, detected = extract_source(name, root, landmarker)
        np.savez_compressed(out, paths=paths, kp=kp, detected=detected)
        print(f"[saved] {out}", flush=True)
    print("=== KEYPOINT EXTRACTION COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
