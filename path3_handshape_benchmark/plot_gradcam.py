"""SF2 — Grad-CAM: signer-DEPENDENT (SD) vs signer-INDEPENDENT (SI) on BdSL47.

The "explain the shortcut" figure. Renders a side-by-side Grad-CAM comparison of
the SD and SI BdSL47 encoders on the SAME test images, so the reader can *see*
what the identity shortcut buys:

  * the SD model (trained with the same signer in train+test) is free to attend
    to background / skin / signer-identity cues, and
  * the SI model (user-disjoint) has no such shortcut, so it must attend to the
    hand shape.

Output grid: rows = sample test images (from the SI test users), columns =
  [ original image | SD Grad-CAM overlay | SI Grad-CAM overlay ].

Why a re-fit head (same rationale as plot_confusion.py): train_baseline.py
checkpoints ONLY the backbone (`model.backbone.state_dict()`), not the per-source
classification heads. So for each encoder we load the LoRA-adapted backbone and
fit a fresh linear head on that encoder's TRAIN features (logistic-regression,
then wrapped into an nn.Linear so gradients flow for Grad-CAM). The class the CAM
targets is that head's predicted class for the image — i.e. "where did THIS
encoder look to make ITS prediction". The attention structure is driven entirely
by the encoder, so this faithfully contrasts the two representations.

Grad-CAM on a timm ViT: hook the last transformer block's token output (forward)
and its gradient w.r.t. the target-class logit (backward). Prefix tokens (CLS +
any DINOv2 register tokens) are stripped via `backbone.num_prefix_tokens`; the
remaining patch tokens reshape to a gh×gw grid, CAM = ReLU(sum_c mean_grad_c *
token_c), upsampled to the image size and overlaid (jet, alpha 0.5).

Usage (bdsl_graph):
    HF_HUB_OFFLINE=1 python -m path3_handshape_benchmark.plot_gradcam \
        --source bdsl47_letters --seed 0 --n 6 \
        --output results/SF2_gradcam_sd_vs_si.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# Weights are cached; compute/login nodes may be offline. Set this BEFORE any
# timm / huggingface_hub import so create_model never attempts a network fetch.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bangla_handshape.class_alignment import discover_source, SourceSpec
from bangla_handshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, split_user_disjoint,
)
from bangla_handshape.dinov2_lora import build_dinov2_lora
from path3_handshape_benchmark.train_baseline import _build_transforms
from path3_handshape_benchmark.train_probe_cached import _extract

# Source root path source-of-truth (mirrors bdsl47_si.yaml `sources:`).
DEFAULT_ROOTS = {
    "bdsl47_digits":  "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}

# ImageNet normalization used by _build_transforms; needed to de-normalize the
# tensor back to a displayable RGB image for the overlay.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _latest_epoch(enc_dir, seed):
    """Largest E such that encoder_seed<seed>_epoch<E>.pt exists in enc_dir, or
    None if the dir has no matching checkpoint."""
    if not os.path.isdir(enc_dir):
        return None
    pat = re.compile(rf"encoder_seed{seed}_epoch(\d+)\.pt$")
    epochs = []
    for p in glob.glob(os.path.join(enc_dir, f"encoder_seed{seed}_epoch*.pt")):
        m = pat.search(os.path.basename(p))
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


def _load_encoder(enc_dir, seed, epoch, num_classes, args, device):
    """Build a LoRA DINOv2 (matching bdsl47_si/sd.yaml) and load the backbone-only
    checkpoint. Returns (model, ckpt_path)."""
    model = build_dinov2_lora(
        num_classes_per_source=[num_classes],
        timm_name=args.timm_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=list(args.lora_targets),
        pretrained=True,
        full_finetune=False,
    )
    ckpt = os.path.join(enc_dir, f"encoder_seed{seed}_epoch{epoch}.pt")
    state = torch.load(ckpt, map_location=device)
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    print(f"  loaded {os.path.basename(ckpt)} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    model = model.to(device)
    model.eval()
    return model, ckpt


def _fit_linear_head(model, spec, train_entries, transform, args, device):
    """Fit a fresh classifier on frozen encoder features (sklearn logistic
    regression, exactly like plot_confusion), then wrap its weights into a torch
    nn.Linear so class logits are differentiable for Grad-CAM."""
    from sklearn.linear_model import LogisticRegression

    ds = HandshapeDataset([(spec, train_entries)], transform=transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())
    Xtr, ytr = _extract(model, loader, device)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(Xtr, ytr)

    feat_dim = Xtr.shape[1]
    # LogisticRegression drops classes with no samples; map its coef rows back to
    # the full [0, num_classes) label space so head index == source label index.
    head = nn.Linear(feat_dim, spec.num_classes)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-1e4)  # unseen classes never win argmax
        coef = clf.coef_
        intercept = clf.intercept_
        classes = clf.classes_
        if coef.shape[0] == 1:  # binary LR: one row = class classes_[1]
            head.weight[int(classes[1])] = torch.from_numpy(coef[0]).float()
            head.bias[int(classes[1])] = float(intercept[0])
            head.weight[int(classes[0])] = torch.from_numpy(-coef[0]).float()
            head.bias[int(classes[0])] = float(-intercept[0])
        else:
            for row, cls in enumerate(classes):
                head.weight[int(cls)] = torch.from_numpy(coef[row]).float()
                head.bias[int(cls)] = float(intercept[row])
    head = head.to(device)
    head.eval()
    return head


class _ViTGradCAM:
    """Grad-CAM over the last transformer block of a timm ViT backbone.

    Hooks the last block's token output (forward) and gradient (backward), strips
    prefix tokens, reshapes patch tokens to a square grid, and returns a 0..1 CAM.
    """

    def __init__(self, backbone):
        self.backbone = backbone
        self.num_prefix = int(getattr(backbone, "num_prefix_tokens", 1))
        blocks = getattr(backbone, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise RuntimeError("backbone has no `.blocks`; not a timm ViT?")
        self.target = blocks[-1]
        self._acts = None
        self._grads = None
        self._fh = self.target.register_forward_hook(self._save_act)
        # full_backward_hook captures grad w.r.t. the block's OUTPUT tensor.
        self._bh = self.target.register_full_backward_hook(self._save_grad)

    def _save_act(self, _module, _inp, out):
        self._acts = out  # (N, T, C)

    def _save_grad(self, _module, _grad_in, grad_out):
        self._grads = grad_out[0]  # (N, T, C)

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def cam(self, x, target_logit_fn):
        """x: (1, 3, H, W). target_logit_fn(feat)->scalar logit to backprop.
        Returns a (gh, gw) float CAM normalized to 0..1."""
        self.backbone.zero_grad(set_to_none=True)
        feat = self.backbone(x)                 # triggers forward hook
        logit = target_logit_fn(feat)
        logit.backward()

        acts = self._acts[0]                    # (T, C)
        grads = self._grads[0]                  # (T, C)
        acts = acts[self.num_prefix:]           # drop CLS (+ register tokens)
        grads = grads[self.num_prefix:]
        n_patch = acts.shape[0]
        gh = int(round(n_patch ** 0.5))
        assert gh * gh == n_patch, (
            f"patch tokens ({n_patch}) not a perfect square; cannot reshape to a grid")
        weights = grads.mean(dim=0)             # (C,) GAP over patch tokens
        cam = torch.relu((acts * weights).sum(dim=1))   # (n_patch,)
        cam = cam.reshape(gh, gh)
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 0:
            cam = cam / denom
        return cam.detach().float().cpu().numpy()


def _denorm_image(tensor):
    """(3, H, W) normalized tensor -> (H, W, 3) uint8 RGB for display."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _overlay(rgb_uint8, cam, cmap, alpha=0.5):
    """Overlay a 0..1 CAM (upsampled to image size) on an RGB uint8 image."""
    h, w = rgb_uint8.shape[:2]
    cam_t = torch.from_numpy(cam)[None, None].float()
    cam_up = F.interpolate(cam_t, size=(h, w), mode="bilinear",
                           align_corners=False)[0, 0].numpy()
    heat = cmap(cam_up)[..., :3]  # RGBA -> RGB in 0..1
    base = rgb_uint8.astype(np.float32) / 255.0
    blended = (1.0 - alpha) * base + alpha * heat
    return np.clip(blended, 0.0, 1.0)


def _predicted_class_logit_fn(head, device):
    """Returns a closure feat->logit for the head's argmax (predicted) class."""
    def fn(feat):
        logits = head(feat)                 # (1, num_classes)
        cls = int(logits.argmax(dim=1).item())
        return logits[0, cls]
    return fn


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--si-dir", default="work_dir/bhc_bdsl47_si",
                    help="dir with the SI encoder_seed<N>_epoch<E>.pt checkpoints")
    ap.add_argument("--sd-dir", default="work_dir/bhc_bdsl47_sd",
                    help="dir with the SD encoder_seed<N>_epoch<E>.pt checkpoints")
    ap.add_argument("--source", default="bdsl47_letters",
                    choices=["bdsl47_letters", "bdsl47_digits"],
                    help="which BdSL47 source to sample test images from")
    ap.add_argument("--root", default=None, help="override the source root dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=0,
                    help="checkpoint epoch; 0 = auto-detect the latest per dir")
    ap.add_argument("--n", type=int, default=6, help="number of sample test images")
    ap.add_argument("--val-users", nargs="+", type=int, default=[4])
    ap.add_argument("--test-users", nargs="+", type=int, default=[5])
    ap.add_argument("--timm-name", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", nargs="+",
                    default=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.5, help="overlay opacity")
    ap.add_argument("--output", default="results/SF2_gradcam_sd_vs_si.png")
    args = ap.parse_args()

    # --- locate checkpoints (auto-detect latest epoch if --epoch 0) -----------
    sd_epoch = args.epoch or _latest_epoch(args.sd_dir, args.seed)
    si_epoch = args.epoch or _latest_epoch(args.si_dir, args.seed)
    if sd_epoch is None or si_epoch is None:
        missing = []
        if sd_epoch is None:
            missing.append(f"SD ({args.sd_dir}) — run bdsl47_sd.yaml first")
        if si_epoch is None:
            missing.append(f"SI ({args.si_dir}) — run bdsl47_si.yaml first")
        print("no checkpoints found for seed "
              f"{args.seed}: " + "; ".join(missing))
        return 0
    print(f"SD encoder @ epoch {sd_epoch}, SI encoder @ epoch {si_epoch} "
          f"(seed {args.seed})")

    # --- source + SI user-disjoint split (matches bdsl47_si.yaml) -------------
    root = args.root or DEFAULT_ROOTS.get(args.source)
    if not root or not os.path.isdir(root):
        print(f"source root not found: {root!r} — pass --root or re-fetch data")
        return 0
    spec = discover_source(args.source, root)
    print(f"source {spec.name}: {spec.num_classes} classes  root={root}")

    items = enumerate_source(spec)
    train_entries, _val, test_entries = split_user_disjoint(
        items, set(args.val_users), set(args.test_users))
    print(f"  train={len(train_entries)}  test(SI users {args.test_users})="
          f"{len(test_entries)}")
    if not train_entries or not test_entries:
        print("empty train or test split — check --val-users/--test-users and data")
        return 0

    # Deterministically pick N sample test images (spread across the test set).
    n = min(args.n, len(test_entries))
    idxs = np.linspace(0, len(test_entries) - 1, num=n, dtype=int)
    sample_entries = [test_entries[i] for i in idxs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = _build_transforms(args.image_size)

    # --- build both encoders + re-fit heads -----------------------------------
    print("[SD] loading encoder + fitting head")
    sd_model, _ = _load_encoder(args.sd_dir, args.seed, sd_epoch,
                                spec.num_classes, args, device)
    sd_head = _fit_linear_head(sd_model, spec, train_entries, transform, args, device)

    print("[SI] loading encoder + fitting head")
    si_model, _ = _load_encoder(args.si_dir, args.seed, si_epoch,
                                spec.num_classes, args, device)
    si_head = _fit_linear_head(si_model, spec, train_entries, transform, args, device)

    # --- compute Grad-CAMs per sample -----------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm as mpl_cm

    jet = mpl_cm.get_cmap("jet")

    sd_cam = _ViTGradCAM(sd_model.backbone)
    si_cam = _ViTGradCAM(si_model.backbone)

    rows = []  # (orig_uint8, sd_overlay, si_overlay, label)
    for path, label, _user in sample_entries:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        orig = _denorm_image(x[0])

        sd_map = sd_cam.cam(x, _predicted_class_logit_fn(sd_head, device))
        si_map = si_cam.cam(x, _predicted_class_logit_fn(si_head, device))
        rows.append((
            orig,
            _overlay(orig, sd_map, jet, args.alpha),
            _overlay(orig, si_map, jet, args.alpha),
            label,
        ))

    sd_cam.remove()
    si_cam.remove()

    # --- render the grid ------------------------------------------------------
    out = args.output
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    col_titles = ["original", f"SD Grad-CAM\n(epoch {sd_epoch})",
                  f"SI Grad-CAM\n(epoch {si_epoch})"]
    fig, axes = plt.subplots(len(rows), 3,
                             figsize=(3 * 2.6, len(rows) * 2.6),
                             squeeze=False)
    for r, (orig, sd_ov, si_ov, label) in enumerate(rows):
        panels = [orig, sd_ov, si_ov]
        for c, panel in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(panel)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=17)
        axes[r][0].set_ylabel(f"class {label}", fontsize=15)
    fig.suptitle(
        f"Grad-CAM: SD vs. SI attention on {spec.name} test images",
        fontsize=19)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.splitext(out)[0] + "." + ext, dpi=600, bbox_inches="tight")
    print(f"wrote {os.path.splitext(out)[0]}.{{pdf,png}} @600dpi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
