"""Shared plotting style for Chapter 9 figures.

Every figure script imports this module so that fonts, colors, and output
settings are consistent across the chapter.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BLUE = "#3b75af"
ORANGE = "#e1812c"
GREEN = "#3a923a"
RED = "#c03d3e"
PURPLE = "#9372b2"
GRAY = "#7f7f7f"
BROWN = "#8c613c"
CLASS_COLORS = [BLUE, ORANGE]
PALETTE = [BLUE, ORANGE, GREEN, RED, PURPLE, BROWN, GRAY]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 2.0,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
})


def save(fig, name):
    """Save a figure as a 150-dpi PNG in the figures/ directory."""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def smooth(x, k):
    """Trailing moving average of a 1-D array with window k (shorter at the start)."""
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i + 1 - k)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out

