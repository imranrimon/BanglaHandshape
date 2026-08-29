"""Shared publication figure style for the BanglaHandshape paper.

One import so every figure looks like it came from the same (careful) hand:
colorblind-safe Okabe-Ito palette, despined axes, hairline value-axis grid only,
column/double-column sizing for a two-column IEEE/CVPR layout, vector PDF at the
body font size. No in-figure titles (the LaTeX caption carries the message) and no
chartjunk. Call `apply()` at import; use `PAL`, `COL`, `DOUBLE`, `despine`,
`barlabels`.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Two-column IEEE/CVPR: text column ~3.35in, full width ~7.0in.
COL = 3.35
DOUBLE = 7.0

# Okabe-Ito colorblind-safe palette, mapped to the paper's recurring roles.
PAL = {
    "pose":       "#009E73",  # geometry / keypoints (bluish green)
    "appearance": "#D55E00",  # RGB frozen / appearance (vermilion)
    "lora":       "#0072B2",  # LoRA-adapted RGB (blue)
    "ours":       "#CC79A7",  # our pose-distilled model (reddish purple)
    "adversary":  "#E69F00",  # signer-adversary (orange)
    "neutral":    "#7F7F7F",  # frozen baseline / reference (grey)
    "accent":     "#0072B2",
    "bar":        "#0072B2",
    "bar2":       "#D55E00",
    "gap":        "#B5406A",  # single-hue gap bars
}


def apply():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.grid": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "grid.color": "#B0B0B0",
        "legend.frameon": False,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "lines.linewidth": 1.2,
        "figure.constrained_layout.use": False,
        "pdf.fonttype": 42,   # editable/embeddable TrueType, not Type-3 (venue-safe)
        "ps.fonttype": 42,
    })


def despine(ax, left=True, bottom=True):
    """Keep only the left+bottom spines (or fewer)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(top=False, right=False)


def ygrid(ax):
    """Hairline horizontal grid behind the data on the value axis only."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major")


def barlabels(ax, bars, fmt="{:.0f}", dy=1.0, fontsize=6.5, color="#222222"):
    """Compact value labels above bars (skips NaN)."""
    for b in bars:
        h = b.get_height()
        if h != h:  # NaN
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize, color=color)
