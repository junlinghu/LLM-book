"""Generate the figures for Chapter 14 (Advanced Transformer Topics).

Usage:  python make_figures.py        (writes PNG files next to this script)

Requires numpy and matplotlib. Both figures are computed from formulas for the
base Transformer configuration of Vaswani et al. (2017); nothing is trained.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- parameter breakdown and cost
def base_param_breakdown(d=512, d_ff=2048, N=6, V=37000):
    enc_attn = N * 4 * d * d
    enc_ffn = N * 2 * d * d_ff
    dec_self = N * 4 * d * d
    dec_cross = N * 4 * d * d
    dec_ffn = N * 2 * d * d_ff
    emb = V * d
    return {"encoder self-attention": enc_attn, "encoder FFN": enc_ffn,
            "decoder self-attention": dec_self, "decoder cross-attention": dec_cross,
            "decoder FFN": dec_ffn, "shared embedding": emb}


def fig_param_breakdown():
    b = base_param_breakdown()
    total = sum(b.values())
    fig, ax = plt.subplots(figsize=(8, 2.4))
    left = 0
    colors = plt.cm.tab10.colors
    for (name, v), c in zip(b.items(), colors):
        ax.barh([0], [v / 1e6], left=left / 1e6, color=c, label=f"{name}: {v / 1e6:.1f}M")
        left += v
    ax.set_yticks([])
    ax.set_xlabel("parameters (millions)")
    ax.set_title(f"Base Transformer weight matrices by component (formula total: {total / 1e6:.1f}M)")
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=8)
    save(fig, "param-breakdown.png")
    return b, total


def fig_attention_cost():
    d, d_ff = 512, 2048
    n = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    # per encoder layer, per token: weight FLOPs vs attention-score FLOPs
    weights = 2 * (4 * d * d + 2 * d * d_ff) * np.ones_like(n, dtype=float)
    attn = 4 * n * d * 1.0  # QK^T and AV: 2*n*d each
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(n, weights / 1e6, "o-", label="weight multiplications ($24d^2$)")
    ax.plot(n, attn / 1e6, "s-", label="attention scores and mixing ($4nd$)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length $n$")
    ax.set_ylabel("forward MFLOPs per token per layer")
    ax.set_title("Encoder layer of the base model: attention overtakes the weights at $n = 6d$")
    ax.legend(fontsize=8)
    save(fig, "attention-cost.png")


if __name__ == "__main__":
    print(fig_param_breakdown())
    fig_attention_cost()
