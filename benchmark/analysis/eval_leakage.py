"""E1 (revision) --- Near-duplicate / session-leakage audit for BdSL47.

Tests whether the ~99% SD result is partly explained by near-identical frames from
the same capture burst appearing in both train and eval under a random split. For
each eval image we find its maximum cosine similarity to ANY train image (in DINOv2
feature space) and report the near-duplicate rate (fraction with max-sim above a
threshold), separately for the SD (random) and SI (user-disjoint) splits.

If SD eval images have many near-exact train neighbours while SI ones do not, the SD
inflation is (partly) session/near-duplicate leakage, not signer recognition ---
exactly the confound flagged in the paper's Threats to Validity.

Writes results/T_leakage.md (+ SF7_leakage.png histogram).

Usage:
  python -m benchmark.analysis.eval_leakage --timm-name vit_base_patch14_dinov2.lvd142m
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
from banglahandshape.class_alignment import discover_source
from banglahandshape.handshape_dataset import (
    HandshapeDataset, enumerate_source, split_user_disjoint, split_random,
)
from benchmark.baselines.train_baseline import _build_transforms

SOURCE_ROOTS = {
    "bdsl47_digits":  "data/BdSL47/Bangla Sign Language Dataset - Sign Digits",
    "bdsl47_letters": "data/BdSL47/Bangla Sign Language Dataset - Sign Letters",
}


@torch.no_grad()
def _feats(model, spec, items, device, bs=64, nw=4):
    ds = HandshapeDataset([(spec, items)], transform=_build_transforms(224))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=torch.cuda.is_available())
    out = []
    for x, _s, _y in loader:
        f = model(x.to(device, non_blocking=True))
        f = torch.nn.functional.normalize(f.float(), dim=1)
        out.append(f.cpu())
    return torch.cat(out) if out else torch.zeros(0)


def _max_sim_to_train(train_f, eval_f, device, chunk=512):
    """For each eval row, max cosine sim to any train row (features are L2-normed)."""
    tf = train_f.to(device)
    out = np.zeros(len(eval_f), dtype=np.float32)
    for i in range(0, len(eval_f), chunk):
        ef = eval_f[i:i + chunk].to(device)
        sims = ef @ tf.T                      # cosine (normed)
        out[i:i + chunk] = sims.max(dim=1).values.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--sources", nargs="+", default=["bdsl47_digits", "bdsl47_letters"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.95, 0.98, 0.99])
    ap.add_argument("--out", default="results/T_leakage.md")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    import timm
    model = timm.create_model(args.timm_name, pretrained=True, num_classes=0,
                              dynamic_img_size=True).to(device).eval()

    rows, hist = [], {}
    for src in args.sources:
        root = SOURCE_ROOTS[src]
        if not os.path.isdir(root):
            print(f"[skip] {src}: missing {root}"); continue
        spec = discover_source(src, root)
        items = enumerate_source(spec)
        feats_all = _feats(model, spec, items, device)          # (N,D) normed, order = items
        idx = {id(it): i for i, it in enumerate(items)}

        for split_name, (tr, ev) in {
            "SD (random)":      (lambda: split_random(items, seed=0, val_frac=0.10, test_frac=0.10)[:2])(),
            "SI (user-disjoint)": (lambda: split_user_disjoint(items, {4}, {5})[:2])(),
        }.items():
            tr_i = np.array([idx[id(it)] for it in tr]); ev_i = np.array([idx[id(it)] for it in ev])
            if len(tr_i) == 0 or len(ev_i) == 0:
                continue
            ms = _max_sim_to_train(feats_all[tr_i], feats_all[ev_i], device)
            hist[(src, split_name)] = ms
            rec = dict(source=src, split=split_name, n_eval=len(ev_i),
                       mean=float(ms.mean()), median=float(np.median(ms)))
            for t in args.thresholds:
                rec[f"ge{t}"] = 100.0 * float((ms >= t).mean())
            rows.append(rec)
            print(f"[{src}/{split_name}] n_eval={len(ev_i)} mean_maxsim={ms.mean():.3f} "
                  + " ".join(f">= {t}:{rec[f'ge{t}']:.1f}%" for t in args.thresholds))

    L = ["# T_leakage --- near-duplicate audit (max cosine sim of each eval image to any train image)\n",
         "DINOv2 feature space. `>=t` = %% of eval images with a train neighbour of "
         "cosine similarity at least `t` (near-duplicate rate). A large SD/SI gap "
         "indicates session/near-duplicate leakage under the random split.\n",
         "| Source | Split | n_eval | mean | median | "
         + " | ".join(f">={t}" for t in args.thresholds) + " |",
         "|---|---|---:|---:|---:|" + "|".join(["---:"] * len(args.thresholds)) + "|"]
    for r in rows:
        L.append(f"| {r['source']} | {r['split']} | {r['n_eval']} | {r['mean']:.3f} | "
                 f"{r['median']:.3f} | " + " | ".join(f"{r[f'ge{t}']:.1f}" for t in args.thresholds) + " |")
    open(args.out, "w").write("\n".join(L) + "\n")
    print("wrote", args.out)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        srcs = sorted({k[0] for k in hist})
        fig, axes = plt.subplots(1, max(1, len(srcs)), figsize=(5 * max(1, len(srcs)), 3.2), squeeze=False)
        for ax, src in zip(axes[0], srcs):
            for (s, sp), ms in hist.items():
                if s == src:
                    ax.hist(ms, bins=40, alpha=0.5, label=sp, density=True)
            ax.set_title(f"{src}: max train-sim per eval image"); ax.set_xlabel("cosine similarity")
            ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig("results/SF7_leakage.png", dpi=150)
        print("wrote results/SF7_leakage.png")
    except Exception as e:
        print("[warn] figure skipped:", e)


if __name__ == "__main__":
    main()
