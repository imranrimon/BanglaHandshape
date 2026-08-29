"""SF5 — per-held-out-signer LOUO Top-1: appearance (DINOv2-B) vs pose (keypoint MLP),
BdSL47 Digits + Letters. Reads the fixed-data LOUO CSVs, de-dups to the latest run per
(source, signer, seed), averages over seeds. Stacked column-width panels (10 signers
read cleanly), shared style, PDF+PNG."""
import csv, re, os
import numpy as np
from tools.figstyle import apply, despine, ygrid, PAL, COL
apply()
import matplotlib.pyplot as plt

SRC = {"appearance": "results/bhc_louo_appearance.csv", "pose": "results/bhc_louo_keypoint.csv"}


def load(path):
    latest = {}   # (source,user,seed) -> (timestamp, acc)
    for r in csv.DictReader(open(path)):
        m = re.search(r"louo_bdsl47_(digits|letters)_testuser(\d+)_seed(\d+)", r["Experiment"])
        if not m:
            continue
        k = (m.group(1), int(m.group(2)), int(m.group(3)))
        ts = r["Timestamp"]
        if k not in latest or ts > latest[k][0]:
            latest[k] = (ts, float(r["Top1_Acc"]) * 100)
    agg = {}
    for (src, u, s), (ts, a) in latest.items():
        agg.setdefault((src, u), []).append(a)
    return {k: float(np.mean(v)) for k, v in agg.items()}


data = {name: load(p) for name, p in SRC.items()}
users = list(range(1, 11))
fig, axes = plt.subplots(2, 1, figsize=(COL, 3.7), sharex=True)
for ax, src in zip(axes, ("digits", "letters")):
    app = [data["appearance"].get((src, u), np.nan) for u in users]
    pos = [data["pose"].get((src, u), np.nan) for u in users]
    x = np.arange(len(users)); w = 0.4
    ax.bar(x - w / 2, app, w, label="Appearance (DINOv2-B)", color=PAL["appearance"],
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, pos, w, label="Pose (keypoint MLP)", color=PAL["pose"],
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([f"{u:02d}" for u in users])
    ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(f"{src.capitalize()}\nTop-1 (%)")
    despine(ax); ygrid(ax)
    print(f"  {src}: appearance mean={np.nanmean(app):.1f} pose mean={np.nanmean(pos):.1f}")
axes[-1].set_xlabel("Held-out test signer")
axes[0].legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.02),
               handletextpad=0.4, columnspacing=1.3)
fig.tight_layout()
os.makedirs("paper/figs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"paper/figs/SF5_per_signer.{ext}", bbox_inches="tight")
print("wrote paper/figs/SF5_per_signer.{pdf,png}")
