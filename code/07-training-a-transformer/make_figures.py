"""Generate the figures for Chapter 7 (Training a Transformer).

Usage:  python make_figures.py
        (writes PNG files to chapters/07-training-a-transformer/figures/)

Requires numpy, matplotlib, and torch. Everything is seeded; the only figure
that involves training is the cross-attention heatmap, which trains a tiny
encoder-decoder Transformer on sequence reversal for a few minutes on a CPU.
The parameter-breakdown and attention-cost figures are computed from formulas
for the base Transformer configuration of Vaswani et al. (2017).
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parents[2] / "chapters" / "07-training-a-transformer" / "figures"
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


# ---------------------------------------------------------------- label smoothing
def fig_label_smoothing():
    V, eps, y = 10, 0.1, 3
    onehot = np.eye(V)[y]
    smooth = (1 - eps) * onehot + eps / V
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    axes[0].bar(range(V), onehot, color="tab:blue")
    axes[0].set_title("One-hot target")
    axes[1].bar(range(V), smooth, color="tab:orange")
    axes[1].set_title(r"Smoothed target, $\epsilon_{ls} = 0.1$, $V = 10$")
    for ax in axes:
        ax.set_xticks(range(V))
        ax.set_xlabel("token id")
    axes[0].set_ylabel("target probability")
    save(fig, "label-smoothing.png")


# ---------------------------------------------------------------- learning-rate schedule
def fig_lr_schedule():
    d, warm = 512, 4000
    steps = np.arange(1, 100_001)
    lr = d ** -0.5 * np.minimum(steps ** -0.5, steps * warm ** -1.5)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(steps, lr)
    ax.axvline(warm, color="gray", ls=":", lw=0.8)
    ax.text(warm * 1.1, lr.max() * 0.95, "end of warmup (step 4,000)", fontsize=8)
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title("The original Transformer schedule ($d = 512$, 4,000 warmup steps)")
    save(fig, "lr-schedule.png")
    return float(lr.max())


# ---------------------------------------------------------------- trained toy model: cross-attention
PAD, BOS, EOS = 0, 1, 2
N_DIGITS = 10                      # tokens 3..12 are the digits 0..9
VOCAB = 3 + N_DIGITS


class ToyTransformer(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, max_len=32):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.register_buffer("pe", torch.tensor(sinusoidal(max_len, d), dtype=torch.float32))
        self.tf = nn.Transformer(d, heads, layers, layers, dim_feedforward=4 * d,
                                 dropout=0.0, batch_first=True)
        self.out = nn.Linear(d, VOCAB)
        self.d = d

    def embed(self, x):
        return self.emb(x) * math.sqrt(self.d) + self.pe[: x.size(1)]

    def forward(self, src, tgt_in):
        causal = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1))
        h = self.tf(self.embed(src), self.embed(tgt_in), tgt_mask=causal,
                    src_key_padding_mask=src == PAD, tgt_key_padding_mask=tgt_in == PAD,
                    memory_key_padding_mask=src == PAD)
        return self.out(h)


def reversal_batch(batch, length, gen):
    digits = torch.randint(3, VOCAB, (batch, length), generator=gen)
    src = digits
    tgt = torch.flip(digits, dims=[1])
    tgt_in = torch.cat([torch.full((batch, 1), BOS), tgt], dim=1)
    tgt_out = torch.cat([tgt, torch.full((batch, 1), EOS)], dim=1)
    return src, tgt_in, tgt_out


def train_toy(steps=1500, length=8, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = ToyTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for step in range(steps):
        src, tgt_in, tgt_out = reversal_batch(64, length, gen)
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, VOCAB), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  toy step {step} loss {loss.item():.3f}")
    return model, gen


def cross_attention_weights(model, src, tgt_in):
    """Average cross-attention weights of the last decoder layer (over heads)."""
    captured = {}
    layer = model.tf.decoder.layers[-1]
    orig = layer.multihead_attn.forward

    def hook(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        out, w = orig(*args, **kwargs)
        captured["w"] = w
        return out, w

    layer.multihead_attn.forward = hook
    try:
        model.eval()
        with torch.no_grad():
            model(src, tgt_in)
    finally:
        layer.multihead_attn.forward = orig
    return captured["w"][0].numpy()


def fig_cross_attention():
    model, gen = train_toy()
    src, tgt_in, tgt_out = reversal_batch(256, 8, gen)
    model.eval()
    with torch.no_grad():
        acc = (model(src, tgt_in).argmax(-1) == tgt_out).float().mean().item()
    print(f"  toy teacher-forced token accuracy on fresh data: {acc:.3f}")
    w = cross_attention_weights(model, src[:1], tgt_in[:1])
    src_labels = [str(t - 3) for t in src[0].tolist()]
    tgt_labels = ["<bos>"] + [str(t - 3) for t in tgt_in[0, 1:].tolist()]
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.imshow(w, cmap="Blues")
    ax.set_xticks(range(len(src_labels)), src_labels)
    ax.set_yticks(range(len(tgt_labels)), tgt_labels)
    ax.set_xlabel("source token (encoder)")
    ax.set_ylabel("decoder input token")
    ax.set_title("Cross-attention of a toy model\ntrained to reverse 8 digits")
    save(fig, "cross-attention-reversal.png")
    return acc


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
    fig_label_smoothing()
    print("peak lr", fig_lr_schedule())
    print("toy accuracy", fig_cross_attention())
    print(fig_param_breakdown())
    fig_attention_cost()
