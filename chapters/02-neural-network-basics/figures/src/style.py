"""Shared plotting style for Chapter 2 figures.

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


def scatter_classes(ax, X, y, s=28, edge="white"):
    """Scatter a two-class dataset with the chapter's class colors."""
    for c in (0, 1):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=s, c=CLASS_COLORS[c], edgecolors=edge,
                   linewidths=0.6, label=f"class {c}", zorder=3)


def decision_background(ax, predict, xlim, ylim, n=300, levels=(0.5,)):
    """Shade the plane by predicted probability of class 1 and draw the 0.5 contour.

    `predict` maps an (N, 2) array to an (N,) array of probabilities.
    """
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    gx, gy = np.meshgrid(xs, ys)
    p = predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bo", ["#c9dcef", "#ffffff", "#f7d9bd"])
    ax.contourf(gx, gy, p, levels=np.linspace(0, 1, 21), cmap=cmap, zorder=0)
    ax.contour(gx, gy, p, levels=list(levels), colors="k", linewidths=1.5, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def make_moons(n=200, noise=0.15, seed=0):
    """Two interleaving half-circles (our own implementation, no sklearn needed)."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    t0 = rng.uniform(0, np.pi, n0)
    t1 = rng.uniform(0, np.pi, n1)
    X0 = np.c_[np.cos(t0), np.sin(t0)]
    X1 = np.c_[1 - np.cos(t1), 0.5 - np.sin(t1)]
    X = np.r_[X0, X1] + rng.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    perm = rng.permutation(n)
    return X[perm], y[perm]
