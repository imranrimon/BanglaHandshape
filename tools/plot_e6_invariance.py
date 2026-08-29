"""E6 (reviewer): invariance scatter — signer decodability vs SI accuracy.

x = signer decodability (balanced acc of a signer classifier on frozen features,
tab:probe; chance 12.5%, lower = more invariant); y = SI handshape Top-1
(tab:mitigate). If invariance drove accuracy the points would trend up-left; they
do not --- the signer-adversary scrubs identity without helping, while pose-distill
(ours) and plain LoRA keep identity fully decodable yet score highest.

Numbers transcribed from paper/main.tex tab:probe + tab:mitigate (no new compute);
update ROWS if those tables move in the fold-in pass. Emits paper/figs/E6_invariance.*
"""
import os
from tools.figstyle import apply, despine, ygrid, PAL, DOUBLE
apply()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# label -> (decodability D/L, SI acc D/L, color, marker, size)
ROWS = [
    ("Pose (keypoints)",            (54.1, 44.7),  (99.9, 85.5), PAL["pose"],       "o", 46),
    ("DINOv2-B (frozen)",           (100.0, 99.4), (81.3, 68.2), PAL["neutral"],    "s", 40),
    ("LoRA (plain)",                (99.9, 99.2),  (94.4, 85.6), PAL["lora"],       "D", 40),
    ("LoRA + pose-distill (ours)",  (99.8, 98.7),  (95.2, 88.2), PAL["ours"],       "*", 150),
    ("LoRA + adversary",            (42.8, 48.3),  (93.9, 85.7), PAL["adversary"],  "^", 48),
]
CHANCE = 12.5

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.9))
for ax, si, name in zip(axes, (0, 1), ("Digits", "Letters")):
    for label, dec, acc, col, mk, sz in ROWS:
        ax.scatter(dec[si], acc[si], s=sz, marker=mk, c=col,
                   edgecolors="black", linewidths=0.6, zorder=3)
    ax.set_xlim(0, 108)
    ax.axvline(CHANCE, ls=(0, (2, 2)), c="0.6", lw=0.8, zorder=1)
    ax.annotate("chance", xy=(CHANCE, ax.get_ylim()[1]), xytext=(2, -2),
                textcoords="offset points", rotation=90, fontsize=6.5,
                color="0.5", va="top", ha="left")
    ax.set_title(name, pad=4)
    ax.set_xlabel("Signer decodability (%)")
    ax.margins(y=0.16)
    despine(ax)
    ygrid(ax)
axes[0].set_ylabel("Signer-independent Top-1 (%)")

handles = [Line2D([0], [0], marker=mk, color="w", markerfacecolor=col,
                  markeredgecolor="black", markersize=11 if mk == "*" else 7,
                  linewidth=0, label=label)
           for label, _, _, col, mk, _ in ROWS]
fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02),
           handletextpad=0.3, columnspacing=1.0)
fig.text(0.5, 0.11, "lower  →  more signer-invariant", ha="center",
         fontsize=7, style="italic", color="0.4")

fig.tight_layout(rect=(0, 0.14, 1, 1))
os.makedirs("paper/figs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"paper/figs/E6_invariance.{ext}", bbox_inches="tight")
print("wrote paper/figs/E6_invariance.{pdf,png}")
