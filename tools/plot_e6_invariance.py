"""E6 (reviewer §9/§35): invariance scatter — signer decodability vs SI accuracy.

Reviewer asked for a scatter of x = signer decodability (balanced acc of a signer
classifier on frozen features, tab:probe) vs y = signer-independent handshape
accuracy (tab:mitigate), across representations, to show the two axes are
DECOUPLED: removing signer identity (adversary, low x) does not raise y, and our
pose-distill gain arrives without removing it (high x).

Numbers are transcribed from paper/main.tex tab:probe + tab:mitigate (no new
compute). Chance signer acc = 12.5% (8 signers). Emits paper/figs/E6_invariance.*
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, decodability D/L, SI acc D/L, marker note)
ROWS = [
    ("Pose (keypoints)",       (54.1, 44.7), (99.9, 84.7)),
    ("DINOv2-B (frozen)",      (100.0, 99.3), (81.4, 67.7)),
    ("LoRA (plain)",           (99.7, 99.3), (94.4, 85.6)),
    ("LoRA + pose-distill (ours)", (100.0, 99.1), (95.2, 88.2)),
    ("LoRA + adversary",       (51.0, 37.8), (93.9, 85.7)),
]
CHANCE = 12.5
COL = {"Pose (keypoints)": "#2a9d8f", "DINOv2-B (frozen)": "#8d99ae",
       "LoRA (plain)": "#457b9d", "LoRA + pose-distill (ours)": "#e63946",
       "LoRA + adversary": "#f4a261"}

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), sharey=False)
for ax, si, name in zip(axes, (0, 1), ("Digits", "Letters")):
    for label, dec, acc in ROWS:
        x, y = dec[si], acc[si]
        star = "*" if "ours" in label else "o"
        ax.scatter(x, y, s=150 if star == "*" else 90, marker=star,
                   c=COL[label], edgecolors="black", linewidths=0.7,
                   zorder=3, label=label)
        dx = 1.5 if x < 90 else -2.0
        ha = "left" if x < 90 else "right"
        ax.annotate(label.replace(" + ", "\n+ "), (x, y), fontsize=7.3,
                    xytext=(x + dx, y + (0.9 if si == 0 else 0.9)), ha=ha,
                    va="bottom", zorder=4)
    ax.axvline(CHANCE, ls=":", c="gray", lw=1)
    ax.text(CHANCE + 1, ax.get_ylim()[0] if False else 0, "", fontsize=7)
    ax.set_title(f"{name}", fontsize=11)
    ax.set_xlabel("Signer decodability (balanced acc, %)  —  lower = more invariant")
    ax.set_xlim(20, 108)
    ax.grid(alpha=0.25, zorder=0)
axes[0].set_ylabel("Signer-independent accuracy (%)")
# a single de-duplicated legend
h, l = axes[0].get_legend_handles_labels()
seen = dict(zip(l, h))
fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=5,
           fontsize=7.2, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Invariance is decoupled from accuracy: removing signer identity "
             "(adversary) does not raise SI accuracy;\nour gain (pose-distill) "
             "arrives with signer identity still fully decodable",
             fontsize=9.5, y=1.02)
fig.tight_layout(rect=(0, 0.06, 1, 0.99))
os.makedirs("paper/figs", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"paper/figs/E6_invariance.{ext}", dpi=200, bbox_inches="tight")
print("wrote paper/figs/E6_invariance.{png,pdf}")
