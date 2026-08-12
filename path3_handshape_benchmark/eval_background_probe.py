"""E8 (revision) --- Background-only leakage probe + hand-only reference.

The clean version of the hand-crop ablation. Using the MediaPipe hand box
(work_dir/_bbox_cache), we build BACKGROUND-ONLY images (hand box greyed out) and
HAND-ONLY images (everything outside the box greyed out), extract frozen DINOv2
features, and fit a handshape classifier under BOTH the SD (random) and SI
(user-disjoint) splits.

If a BACKGROUND-ONLY classifier predicts handshape far above chance under the
random (SD) split --- and much more than under the user-disjoint (SI) split --- that
is direct evidence of background/session leakage, exactly the Threats-to-Validity
concern. Hand-only is the complementary reference. Writes results/T_background_probe.md.

Usage:
  python -m path3_handshape_benchmark.eval_background_probe --timm-name vit_base_patch14_dinov2.lvd142m
"""
from __future__ import annotations
import argparse, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from bangla_handshape.class_alignment import discover_source
from bangla_handshape.handshape_dataset import (
    enumerate_source, split_user_disjoint, split_random,
)
from path3_handshape_benchmark.train_baseline import _build_transforms

SOURCE_ROOTS = {
    "bdsl47_digits":  "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}
BBOX = "work_dir/_bbox_cache"


def _norm(p):
    return os.path.normpath(os.path.abspath(str(p)))


class _MaskDS(Dataset):
    """mode: 'full' (no mask) | 'bg' (grey out the hand box) | 'hand' (grey out
    everything OUTSIDE the box). Undetected images -> returned as 'full'."""
    def __init__(self, items, bbox_lut, transform, mode):
        self.items = items; self.lut = bbox_lut; self.tf = transform; self.mode = mode

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _u = self.items[i]
        img = Image.open(path).convert("RGB")
        box = self.lut.get(_norm(path))
        if box is not None and self.mode != "full":
            x0, y0, x1, y1 = [int(v) for v in box]
            if self.mode == "bg":
                ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
            else:  # hand: grey everything, then paste the hand region back
                hand = img.crop((x0, y0, x1, y1))
                grey = Image.new("RGB", img.size, (128, 128, 128))
                grey.paste(hand, (x0, y0)); img = grey
        return self.tf(img), label


@torch.no_grad()
def _feats(model, items, lut, mode, device, bs=64, nw=4):
    ds = _MaskDS(items, lut, _build_transforms(224), mode)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=torch.cuda.is_available())
    X, y = [], []
    for xb, yb in loader:
        X.append(model(xb.to(device, non_blocking=True)).float().cpu().numpy())
        y.append(yb.numpy())
    return np.concatenate(X), np.concatenate(y)


def _acc(Xtr, ytr, Xev, yev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    return 100.0 * float((clf.predict(sc.transform(Xev)) == yev).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--sources", nargs="+", default=["bdsl47_digits", "bdsl47_letters"])
    ap.add_argument("--out", default="results/T_background_probe.md")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    import timm
    model = timm.create_model(args.timm_name, pretrained=True, num_classes=0,
                              dynamic_img_size=True).to(device).eval()

    rows = []
    for src in args.sources:
        root = SOURCE_ROOTS[src]
        if not os.path.isdir(root):
            print(f"[skip] {src}: missing {root}"); continue
        bf = os.path.join(BBOX, f"{src}.npz")
        if not os.path.exists(bf):
            print(f"[skip] {src}: no bbox cache {bf}"); continue
        d = np.load(bf, allow_pickle=False)
        lut = {_norm(p): d["bbox"][i] for i, p in enumerate(d["paths"]) if int(d["detected"][i]) == 1}
        spec = discover_source(src, root)
        items = enumerate_source(spec)
        chance = 100.0 / spec.num_classes
        print(f"[{src}] {len(items)} imgs, {spec.num_classes} classes, "
              f"{len(lut)} boxes, chance={chance:.1f}%")

        splits = {
            "SD (random)":        split_random(items, seed=0, val_frac=0.10, test_frac=0.10)[:2],
            "SI (user-disjoint)": split_user_disjoint(items, {4}, {5})[:2],
        }
        for mode in ("full", "bg", "hand"):
            Xall, yall = _feats(model, items, lut, mode, device)
            pos = {id(it): k for k, it in enumerate(items)}
            for sp, (tr, ev) in splits.items():
                ti = np.array([pos[id(it)] for it in tr]); ei = np.array([pos[id(it)] for it in ev])
                acc = _acc(Xall[ti], yall[ti], Xall[ei], yall[ei])
                rows.append(dict(source=src, mode=mode, split=sp, chance=chance, acc=acc))
                print(f"    mode={mode:5s} {sp:20s} acc={acc:.1f}% (chance {chance:.1f})")

    L = ["# T_background_probe --- handshape recoverable from BACKGROUND ONLY?\n",
         "Frozen-DINOv2 handshape classifier trained on background-only / hand-only / "
         "full images, under SD (random) and SI (user-disjoint) splits. A high "
         "background-only SD accuracy (>> chance, >> SI) is direct leakage evidence.\n",
         "| Source | Region | Split | chance | acc |", "|---|---|---|---:|---:|"]
    for r in rows:
        L.append(f"| {r['source']} | {r['mode']} | {r['split']} | {r['chance']:.1f} | {r['acc']:.1f} |")
    open(args.out, "w").write("\n".join(L) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
