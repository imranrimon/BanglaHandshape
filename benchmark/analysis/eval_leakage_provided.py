"""E13/E17 fallback --- near-duplicate leakage audit on a source's OWN provided
train/test split (no signer IDs required).

For datasets that ship an author-provided train/test split but no participant labels
(RSBdSL38, BDSL49-Recognition), we cannot build a signer-disjoint split. But we CAN
ask the leakage question directly: for each *test* image, what is its maximum cosine
similarity (DINOv2 feature space) to ANY *train* image, and what fraction exceed a
near-duplicate threshold? A high near-duplicate rate across the authors' own
train/test boundary means their published accuracy is (partly) explained by
near-identical frames leaking across the split --- the same confound E1 found on
BdSL47's random split (82-88% at >=0.98 for SD vs 0% for the signer-disjoint SI
split). This needs no author data, so it stands even if the identity-grouping
requests are declined.

Writes results/T_leakage_provided_<source>.md.

Usage:
  python -m benchmark.analysis.eval_leakage_provided --sources rsbdsl38
  python -m benchmark.analysis.eval_leakage_provided --sources bdsl49_recognition
"""
from __future__ import annotations
import argparse, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from banglahandshape.class_alignment import discover_source, SourceSpec
from banglahandshape.handshape_dataset import enumerate_source
from benchmark.analysis.eval_leakage import _feats, _max_sim_to_train

# (train_root, test_root) for sources that ship a provided split but no user IDs.
ROOTS = {
    "rsbdsl38": ("data/RSBdSL38/train", "data/RSBdSL38/test"),
    "bdsl49_recognition": (
        "data/bdsl49_extracted/Recognition_1/Recognition_1/train",
        "data/bdsl49_extracted/Recognition_1/Recognition_1/test"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timm-name", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--sources", nargs="+", default=list(ROOTS))
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.95, 0.98, 0.99])
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    import timm
    model = timm.create_model(args.timm_name, pretrained=True, num_classes=0,
                              dynamic_img_size=True).to(device).eval()

    for src in args.sources:
        train_root, test_root = ROOTS[src]
        if not (os.path.isdir(train_root) and os.path.isdir(test_root)):
            print(f"[skip] {src}: missing {train_root} or {test_root}"); continue
        spec = discover_source(src, train_root)
        tr_items = enumerate_source(spec)
        eval_spec = SourceSpec(name=src, root=test_root,
                               num_classes=spec.num_classes, class_to_idx=spec.class_to_idx)
        te_items = enumerate_source(eval_spec)
        print(f"[{src}] train={len(tr_items)} test={len(te_items)} classes={spec.num_classes}", flush=True)

        tr_f = _feats(model, spec, tr_items, device)
        te_f = _feats(model, eval_spec, te_items, device)
        ms = _max_sim_to_train(tr_f, te_f, device)

        rec = dict(mean=float(ms.mean()), median=float(np.median(ms)))
        for t in args.thresholds:
            rec[f"ge{t}"] = 100.0 * float((ms >= t).mean())
        print(f"[{src}] mean_maxsim={ms.mean():.3f} "
              + " ".join(f">= {t}:{rec[f'ge{t}']:.1f}%" for t in args.thresholds), flush=True)

        L = [f"# T_leakage_provided --- {src}: near-duplicate audit on the AUTHORS' OWN train/test split\n",
             "For each test image, max cosine similarity (DINOv2-B feature space) to any train image. "
             "`>=t` = %% of test images with a train neighbour at cosine similarity >= t (near-duplicate rate). "
             "No signer IDs used --- this audits the provided split directly. A high rate means the "
             "published accuracy is partly near-duplicate leakage across the authors' own boundary.\n",
             f"- source: **{src}**   train={len(tr_items)}   test={len(te_items)}   classes={spec.num_classes}",
             f"- mean max-sim = {rec['mean']:.3f}   median = {rec['median']:.3f}\n",
             "| Threshold | test images with a near-dup train neighbour (%) |",
             "|---|---:|"]
        for t in args.thresholds:
            L.append(f"| >= {t} | {rec[f'ge{t}']:.1f} |")
        out = os.path.join(args.out_dir, f"T_leakage_provided_{src}.md")
        open(out, "w").write("\n".join(L) + "\n")
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
