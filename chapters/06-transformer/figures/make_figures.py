"""Generate the figures for Chapter 6 (Transformer).

Usage:  python make_figures.py        (writes PNG files next to this script)

Requires numpy, matplotlib, and torch. Everything is seeded; the only figure
that involves training is the cross-attention heatmap, which trains a tiny
encoder-decoder Transformer on sequence reversal for a few minutes on a CPU.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

OUT = Path(__file__).resolve().parent
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


if __name__ == "__main__":
    fig_positional_encodings()
    fig_masks()
    print(fig_softmax_scaling())
    fig_label_smoothing()
    print("peak lr", fig_lr_schedule())
    print(fig_param_breakdown())
    fig_attention_cost()
    print("toy accuracy", fig_cross_attention())
