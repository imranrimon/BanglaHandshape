"""E7 (revision) --- Quantitative Grad-CAM attribution (SD vs SI).

Replaces the qualitative Grad-CAM claim with a number: the fraction of Grad-CAM
mass that falls INSIDE the MediaPipe hand box (R_hand), averaged over test images,
for the SD and SI encoders. If the SD model has systematically LOWER R_hand (more
attention outside the hand) than SI, that supports "SD attends to background". The
hand box (original pixels, work_dir/_bbox_cache) is mapped into the 224 input via
the Resize(258)+CenterCrop(224) geometry.

Writes results/T_attribution.md.

Usage:
  python -m path3_handshape_benchmark.eval_attribution --n 200
"""
from __future__ import annotations
import argparse, glob, os, re, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from bangla_handshape.class_alignment import discover_source
from bangla_handshape.handshape_dataset import HandshapeDataset, enumerate_source, split_user_disjoint
from bangla_handshape.dinov2_lora import build_dinov2_lora
from path3_handshape_benchmark.train_baseline import _build_transforms
from path3_handshape_benchmark.train_probe_cached import _extract

SOURCE_ROOTS = {"bdsl47_digits": "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
                "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters"}
LORA = dict(lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
            lora_targets=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
BBOX = "work_dir/_bbox_cache"


def _norm(p):
    return os.path.normpath(os.path.abspath(str(p)))


def _latest(dirp, seed):
    best = (-1, None)
    for p in glob.glob(os.path.join(dirp, f"encoder_seed{seed}_epoch*.pt")):
        m = re.search(r"epoch(\d+)\.pt$", p)
        if m and int(m.group(1)) > best[0]:
            best = (int(m.group(1)), p)
    return best[1]


def _box_to_224(box, W, H):
    """Map an original-pixel box to the 224 input under Resize(258)+CenterCrop(224)."""
    s = 258.0 / min(W, H)
    nw, nh = W * s, H * s
    ox, oy = (nw - 224) / 2.0, (nh - 224) / 2.0
    x0 = np.clip(box[0] * s - ox, 0, 224); y0 = np.clip(box[1] * s - oy, 0, 224)
    x1 = np.clip(box[2] * s - ox, 0, 224); y1 = np.clip(box[3] * s - oy, 0, 224)
    return x0, y0, x1, y1


def _fit_head(model, spec, tr, device):
    from sklearn.linear_model import LogisticRegression
    ds = HandshapeDataset([(spec, tr)], transform=_build_transforms(224))
    X, y = _extract(model, DataLoader(ds, batch_size=64, num_workers=4), device)
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    W = np.zeros((spec.num_classes, X.shape[1]), np.float32); b = np.zeros(spec.num_classes, np.float32)
    for j, c in enumerate(clf.classes_):
        W[int(c)] = clf.coef_[j] if clf.coef_.shape[0] > 1 else clf.coef_[0] * (1 if j == 1 else -1)
        b[int(c)] = clf.intercept_[j] if len(clf.intercept_) > 1 else clf.intercept_[0] * (1 if j == 1 else -1)
    head = torch.nn.Linear(X.shape[1], spec.num_classes).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(W)); head.bias.copy_(torch.tensor(b))
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def _cam(model, head, x, device):
    """Grad-CAM for the predicted class from the last block's patch tokens."""
    feats = {}
    blk = model.backbone.blocks[-1]
    h1 = blk.register_forward_hook(lambda m, i, o: feats.__setitem__("o", o))
    grads = {}
    h2 = blk.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    x = x.to(device).requires_grad_(False)
    logits = head(model.features(x))
    cls = logits.argmax(1)
    model.zero_grad(set_to_none=True)
    logits[torch.arange(len(x)), cls].sum().backward()
    o, g = feats["o"], grads["g"]                       # (N,T,C)
    npf = getattr(model.backbone, "num_prefix_tokens", 1)
    o, g = o[:, npf:, :], g[:, npf:, :]
    w = g.mean(dim=1, keepdim=True)                     # (N,1,C)
    cam = F.relu((w * o).sum(-1))                        # (N,P)
    P = cam.shape[1]; s = int(round(P ** 0.5))
    cam = cam.reshape(len(x), 1, s, s)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
    cam = cam / (cam.flatten(1).sum(1).clamp_min(1e-8)[:, None, None])
    h1.remove(); h2.remove()
    return cam.detach().cpu().numpy()                   # (N,224,224), sums to 1


def _r_hand(dirp, seed, spec, items, lut, device, n):
    ck = _latest(dirp, seed)
    if not ck:
        return None
    model = build_dinov2_lora(num_classes_per_source=[spec.num_classes],
                              timm_name="vit_small_patch14_dinov2.lvd142m",
                              pretrained=True, full_finetune=False, **LORA).to(device).eval()
    model.backbone.load_state_dict(torch.load(ck, map_location="cpu"), strict=False)
    tr, va, _ = split_user_disjoint(items, {4}, {5})
    head = _fit_head(model, spec, tr, device)
    # sample eval images WITH a detected box
    ev = [it for it in va if _norm(it[0]) in lut][:n]
    tf = _build_transforms(224); rr = []
    for it in ev:
        img = Image.open(it[0]).convert("RGB"); W, H = img.size
        x = tf(img).unsqueeze(0)
        cam = _cam(model, head, x, device)[0]
        x0, y0, x1, y1 = _box_to_224(lut[_norm(it[0])], W, H)
        inside = cam[int(y0):int(y1), int(x0):int(x1)].sum()
        rr.append(float(inside))                         # cam sums to 1 -> inside = R_hand
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(np.mean(rr)), float(np.std(rr)), len(rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--si-dir", default="work_dir/bhc_bdsl47_si")
    ap.add_argument("--sd-dir", default="work_dir/bhc_bdsl47_sd")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", nargs="+", default=["bdsl47_letters"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="results/T_attribution.md")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    rows = []
    for src in args.sources:
        bf = os.path.join(BBOX, f"{src}.npz")
        if not os.path.isdir(SOURCE_ROOTS[src]) or not os.path.exists(bf):
            print(f"[skip] {src}"); continue
        d = np.load(bf, allow_pickle=False)
        lut = {_norm(p): d["bbox"][i] for i, p in enumerate(d["paths"]) if int(d["detected"][i]) == 1}
        spec = discover_source(src, SOURCE_ROOTS[src]); items = enumerate_source(spec)
        for tag, dirp in (("SI", args.si_dir), ("SD", args.sd_dir)):
            res = _r_hand(dirp, args.seed, spec, items, lut, device, args.n)
            if res is None:
                print(f"[skip] {tag} {src}: no checkpoint"); continue
            mean, std, k = res
            rows.append(dict(source=src, model=tag, r_hand=mean, std=std, n=k))
            print(f"[{src}/{tag}] R_hand = {mean*100:.1f}% +/- {std*100:.1f} (n={k})")

    L = ["# T_attribution --- Grad-CAM mass inside the hand box (R_hand)\n",
         "Fraction of Grad-CAM mass (maps normalized to sum 1) falling inside the "
         "MediaPipe hand box, averaged over eval images. Lower R_hand = more "
         "attention on background/context. Compare SD vs SI.\n",
         "| Source | Model | R_hand (%) | std | n |", "|---|---|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['source']} | {r['model']} | {r['r_hand']*100:.1f} | {r['std']*100:.1f} | {r['n']} |")
    open(args.out, "w").write("\n".join(L) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
