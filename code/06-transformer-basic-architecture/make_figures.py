"""Generate the figures for Chapter 6 (Transformer: Basic Architecture).

Usage:  python make_figures.py
        (writes PNG files to chapters/06-transformer-basic-architecture/figures/)

Requires numpy and matplotlib. Everything is computed from formulas or seeded
random numbers; nothing is trained.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "chapters" / "06-transformer-basic-architecture" / "figures"
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- positional encodings
def sinusoidal(n_pos, d):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(0, d, 2)[None, :]
    angle = pos / (10000 ** (i / d))
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


def fig_positional_encodings():
    pe = sinusoidal(100, 128)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xlabel("encoding dimension")
    ax.set_ylabel("position $p$")
    ax.set_title("Sinusoidal positional encodings ($d = 128$, positions 0-99)")
    fig.colorbar(im, ax=ax)
    save(fig, "positional-encoding-heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    for dim in [0, 2, 8, 20, 60]:
        ax.plot(pe[:, dim], label=f"dim {dim}")
    ax.set_xlabel("position $p$")
    ax.set_ylabel("value")
    ax.set_title("Individual dimensions: low dimensions oscillate fast, high dimensions slowly")
    ax.legend(ncol=5, fontsize=8, loc="lower left")
    save(fig, "positional-encoding-curves.png")

    sim = pe @ pe.T
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(sim, cmap="viridis")
    ax.set_xlabel("position")
    ax.set_ylabel("position")
    ax.set_title("Dot products between encodings")
    fig.colorbar(im, ax=ax)
    save(fig, "positional-encoding-similarity.png")


# ---------------------------------------------------------------- masks
def fig_masks():
    n = 6
    lengths_src, pad_from = 6, 4          # last two source tokens are padding
    causal = np.tril(np.ones((n, n)))
    pad = np.ones((n, n))
    pad[:, pad_from:] = 0
    combined = causal * pad
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    titles = ["Padding mask\n(keys 5-6 are padding)", "Causal mask\n(decoder self-attention)",
              "Padding + causal"]
    for ax, m, t in zip(axes, [pad, causal, combined], titles):
        ax.imshow(m, cmap="Greens", vmin=0, vmax=1.3)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, "0" if m[i, j] else "−∞", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(n), [str(k + 1) for k in range(n)])
        ax.set_yticks(range(n), [str(k + 1) for k in range(n)])
        ax.set_xlabel("key position $s$")
        ax.set_ylabel("query position $t$")
        ax.set_title(t)
    save(fig, "attention-masks.png")


# ---------------------------------------------------------------- softmax saturation
def fig_softmax_scaling():
    rng = np.random.default_rng(0)
    dks = [4, 16, 64, 256, 1024]
    n_keys, trials = 32, 200
    ent_scaled, ent_unscaled, max_scaled, max_unscaled = [], [], [], []

    def softmax(z):
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    for dk in dks:
        q = rng.standard_normal((trials, 1, dk))
        k = rng.standard_normal((trials, n_keys, dk))
        s = (q * k).sum(-1)
        for scale, ent, mx in [(math.sqrt(dk), ent_scaled, max_scaled), (1.0, ent_unscaled, max_unscaled)]:
            p = softmax(s / scale)
            ent.append(float((-(p * np.log(p + 1e-30)).sum(-1)).mean()))
            mx.append(float(p.max(-1).mean()))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(dks, ent_scaled, "o-", label=r"scaled by $1/\sqrt{d_k}$")
    axes[0].plot(dks, ent_unscaled, "s--", label="unscaled")
    axes[0].axhline(math.log(n_keys), color="gray", lw=0.8, ls=":", label="uniform")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("$d_k$")
    axes[0].set_ylabel("entropy of weights (nats)")
    axes[0].legend(fontsize=8)
    axes[1].plot(dks, max_scaled, "o-", label=r"scaled by $1/\sqrt{d_k}$")
    axes[1].plot(dks, max_unscaled, "s--", label="unscaled")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("$d_k$")
    axes[1].set_ylabel("largest weight")
    axes[1].legend(fontsize=8)
    fig.suptitle("Attention over 32 random keys: without scaling, the softmax saturates as $d_k$ grows")
    save(fig, "softmax-scaling.png")
    return dict(zip(dks, zip(ent_scaled, ent_unscaled, max_scaled, max_unscaled)))


if __name__ == "__main__":
    fig_positional_encodings()
    fig_masks()
    print(fig_softmax_scaling())
