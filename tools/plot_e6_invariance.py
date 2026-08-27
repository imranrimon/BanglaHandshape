"""E6 (reviewer): invariance scatter — signer decodability vs SI accuracy.

x = signer decodability (balanced acc of a signer classifier on frozen features,
tab:probe; chance 12.5%, lower = more invariant); y = SI handshape Top-1
(tab:mitigate). If invariance drove accuracy the points would trend; they do not
--- the signer-adversary scrubs identity without helping, while pose-distill (ours)
and plain LoRA keep identity fully decodable yet score highest.

Numbers transcribed from paper/main.tex tab:probe + tab:mitigate (no new compute).
Design: clean shared legend (no overlapping inline labels), large fonts, no
in-figure title (the message lives in the LaTeX caption). Emits paper/figs/E6_invariance.*
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "legend.fontsize": 12, "axes.linewidth": 0.8,
})

# label -> (decodability D/L, SI acc D/L, color, marker, size)
ROWS = [
    ("Pose (keypoints)",           (54.1, 44.7), (99.9, 84.7), "#2a9d8f", "o", 130),
    ("DINOv2-B (frozen)",          (100.0, 99.3), (81.4, 67.7), "#8d99ae", "s", 120),
    ("LoRA (plain)",               (99.7, 99.3), (94.4, 85.6), "#457b9d", "D", 120),
    ("LoRA + pose-distill (ours)", (100.0, 99.1), (95.2, 88.2), "#e63946", "*", 320),
    ("LoRA + adversary",           (51.0, 37.8), (93.9, 85.7), "#f4a261", "^", 130),
]
CHANCE = 12.5

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=False)
for ax, si, name in zip(axes, (0, 1), ("Digits", "Letters")):
    for label, dec, acc, col, mk, sz in ROWS:
        ax.scatter(dec[si], acc[si], s=sz, marker=mk, c=col,
                   edgecolors="black", linewidths=0.9, zorder=3)
    ax.set_xlim(5, 108)
    ax.axvline(CHANCE, ls=":", c="0.5", lw=1.2, zorder=1)
    ymin, ymax = ax.get_ylim()
    ax.text(CHANCE + 2.0, ymax - 0.05 * (ymax - ymin), "chance", rotation=90,
            fontsize=10, color="0.45", va="top")
    ax.set_title(name, fontsize=15, pad=8)
    ax.set_xlabel("Signer decodability (%)")
    ax.margins(y=0.14)
    ax.grid(alpha=0.3, zorder=0)
axes[0].set_ylabel("Signer-independent Top-1 (%)")

# one shared legend below (no overlapping inline labels)
handles = [Line2D([0], [0], marker=mk, color="w", markerfacecolor=col,
                  markeredgecolor="black", markersize=15 if mk == "*" else 11,
                  linewidth=0, label=label)
           for label, _, _, col, mk, _ in ROWS]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02), columnspacing=1.6, handletextpad=0.4)
# shared x-axis descriptor (won't clip — placed in reserved bottom margin)
fig.text(0.5, 0.135, r"lower $\rightarrow$ more signer-invariant", ha="center",
         fontsize=11, style="italic", color="0.35")

fig.tight_layout(rect=(0, 0.20, 1, 1))
os.makedirs("paper/figs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"paper/figs/E6_invariance.{ext}", dpi=600, bbox_inches="tight")
print("wrote paper/figs/E6_invariance.{pdf,png} @600dpi")
