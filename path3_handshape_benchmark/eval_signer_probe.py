"""E3 (revision) --- Signer-decodability probe.

Directly tests the mechanism claim behind pose-guided distillation: does a
representation encode *signer identity*? For each representation we freeze it,
extract features over BdSL47 train-signer images, fit a signer classifier
(logistic regression) on a random image split, and report BALANCED signer accuracy
on held-out images of the SAME signers. Chance = 100/num_signers. LOWER = more
signer-invariant.

Representations probed (each if its checkpoint exists):
  * pose            : 63-d MediaPipe keypoint vector (work_dir/_kp_cache)
  * dinov2b_frozen  : frozen DINOv2-B features (no adaptation)
  * lora_plain      : work_dir/bhc_bdsl47_si         backbone
  * lora_adv        : work_dir/bhc_bdsl47_si_adv      backbone
  * lora_distill    : work_dir/bhc_bdsl47_si_distill  backbone  (ours)

Pair this with SI handshape accuracy (Table in the paper) to make the invariance
scatter (E6). Writes results/T_signer_probe.md.

Usage:
  python -m path3_handshape_benchmark.eval_signer_probe --seed 0
"""
from __future__ import annotations
import argparse, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch.utils.data import DataLoader
from bangla_handshape.class_alignment import discover_source
from bangla_handshape.handshape_dataset import HandshapeDataset, enumerate_source
from bangla_handshape.dinov2_lora import build_dinov2_lora
from path3_handshape_benchmark.train_baseline import _build_transforms
from path3_handshape_benchmark.train_probe_cached import _extract

SOURCE_ROOTS = {
    "bdsl47_digits":  "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}
LORA = dict(lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
            lora_targets=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
KP_CACHE = "work_dir/_kp_cache"


def _latest(dirp, seed):
    import glob, re
    best_e, best_p = -1, None
    for p in glob.glob(os.path.join(dirp, f"encoder_seed{seed}_epoch*.pt")):
        m = re.search(rf"epoch(\d+)\.pt$", p)
        if m and int(m.group(1)) > best_e:
            best_e, best_p = int(m.group(1)), p
    return best_p


def _norm(p):
    return os.path.normpath(os.path.abspath(str(p)))


def _load_backbone(timm_name, ckpt, device):
    m = build_dinov2_lora(num_classes_per_source=[2], timm_name=timm_name,
                          pretrained=True, full_finetune=False, **LORA)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        m.backbone.load_state_dict(state, strict=False)
    return m.to(device).eval()


@torch.no_grad()
def _feats_for_paths(model, spec, items, device, bs=64, nw=4):
    ds = HandshapeDataset([(spec, items)], transform=_build_transforms(224))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=torch.cuda.is_available())
    X, _ = _extract(model, loader, device)
    return X


def _signer_balanced_acc(X, signer, rng):
    """Fit LR signer classifier on a random 70/30 image split; balanced acc on test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    n = len(signer); idx = rng.permutation(n); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(X[tr]), signer[tr])
    pred = clf.predict(sc.transform(X[te]))
    return 100.0 * balanced_accuracy_score(signer[te], pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", nargs="+", default=["bdsl47_digits", "bdsl47_letters"])
    ap.add_argument("--out", default="results/T_signer_probe.md")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    rows = []  # (source, rep, chance, balanced_signer_acc, n_signers)
    for src in args.sources:
        root = SOURCE_ROOTS[src]
        if not os.path.isdir(root):
            print(f"[skip] {src}: missing {root}"); continue
        spec = discover_source(src, root)
        items = enumerate_source(spec)
        # train signers only (exclude the SI val/test users 4,5)
        items = [it for it in items if it[2] not in (4, 5) and it[2] != -1]
        signer = np.array([it[2] for it in items], dtype=np.int64)
        nsign = len(set(signer.tolist()))
        chance = 100.0 / max(1, nsign)
        print(f"[{src}] {len(items)} train-signer images, {nsign} signers, chance={chance:.1f}%")

        # pose representation (keypoint vectors), aligned by path
        kpf = os.path.join(KP_CACHE, f"{src}.npz")
        if os.path.exists(kpf):
            d = np.load(kpf, allow_pickle=False)
            lut = {_norm(p): i for i, p in enumerate(d["paths"])}
            kp = d["kp"].astype(np.float32)
            keep = [(i, lut[_norm(it[0])]) for i, it in enumerate(items) if _norm(it[0]) in lut]
            if keep:
                ii = np.array([k[0] for k in keep]); jj = np.array([k[1] for k in keep])
                acc = _signer_balanced_acc(kp[jj], signer[ii], rng)
                rows.append((src, "pose (keypoints)", chance, acc, nsign))
                print(f"    pose: signer balanced acc = {acc:.1f}%")

        # appearance encoders
        encoders = [
            ("dinov2b_frozen", "vit_base_patch14_dinov2.lvd142m", None),
            ("lora_plain",   "vit_small_patch14_dinov2.lvd142m", _latest("work_dir/bhc_bdsl47_si", args.seed)),
            ("lora_adv",     "vit_small_patch14_dinov2.lvd142m", _latest("work_dir/bhc_bdsl47_si_adv", args.seed)),
            ("lora_distill", "vit_small_patch14_dinov2.lvd142m", _latest("work_dir/bhc_bdsl47_si_distill", args.seed)),
        ]
        for name, timm_name, ckpt in encoders:
            if name != "dinov2b_frozen" and not ckpt:
                print(f"    [skip] {name}: no checkpoint"); continue
            model = _load_backbone(timm_name, ckpt, device)
            X = _feats_for_paths(model, spec, items, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            acc = _signer_balanced_acc(X, signer, rng)
            rows.append((src, name, chance, acc, nsign))
            print(f"    {name}: signer balanced acc = {acc:.1f}%")

    # write markdown
    L = ["# T_signer_probe --- signer decodability (BdSL47 train signers)\n",
         "Balanced signer-classification accuracy of a logistic-regression probe on "
         "FROZEN features (70/30 image split within the train signers). Chance = "
         "100/num_signers. LOWER = more signer-invariant. Pair with SI handshape "
         "accuracy to test whether pose-distillation reduces signer information.\n",
         "| Source | Representation | #signers | chance | signer bal-acc |",
         "|---|---|---:|---:|---:|"]
    for src, rep, ch, acc, ns in rows:
        L.append(f"| {src} | {rep} | {ns} | {ch:.1f} | {acc:.1f} |")
    open(args.out, "w").write("\n".join(L) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
