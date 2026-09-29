"""Generate the figures for Chapter 8 (Generative Pretraining).

Usage:  python make_figures.py [name ...]
        (writes PNG files to chapters/08-generative-pretraining-gpt/figures/;
        names: lr, isoflop, temperature, nucleus, icl, training; default: all)

Requires numpy, matplotlib, torch, tiktoken, transformers, and datasets. Everything
is seeded. Two figures involve real computation on a CPU: the training curves rerun
the mini GPT training of Section 8.4 on tiny Shakespeare (a few minutes), and the
in-context learning figure evaluates GPT-2 small and medium on 100 SST-2 reviews
(several minutes, plus downloading the models and dataset on first use).
"""
import math
import os
import random
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "chapters" / "08-generative-pretraining-gpt" / "figures"
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- learning-rate schedule
def fig_lr_schedule():
    """GPT-3 175B: linear warmup over 375M tokens, cosine decay to 10% over 260B tokens,
    then constant until 300B tokens (Brown et al. 2020, Appendix B). The batch-size ramp is ignored."""
    peak, warm, decay_end, total = 0.6e-4, 375e6, 260e9, 300e9
    tokens = np.linspace(0, total, 3001)
    lr = np.where(tokens < warm, peak * tokens / warm,
                  peak * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * np.clip((tokens - warm) / (decay_end - warm), 0, 1)))))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(tokens / 1e9, lr * 1e4)
    ax.axvline(decay_end / 1e9, color="gray", ls=":", lw=0.8)
    ax.text(decay_end / 1e9 - 3, peak * 1e4 * 0.8, "end of cosine decay\n(260B tokens)", fontsize=8, ha="right")
    ax.set_xlabel("training tokens (billions)")
    ax.set_ylabel(r"learning rate ($\times 10^{-4}$)")
    ax.set_title("GPT-3 175B learning-rate schedule: warmup, cosine decay to 10%, then constant")
    save(fig, "lr-schedule.png")


# ---------------------------------------------------------------- Chinchilla parametric fit
def fig_isoflop():
    """Loss predicted by the parametric fit of Hoffmann et al. (2022), L = E + A/P^a + B/D^b,
    along curves of constant compute C = 6PD."""
    E, A, B, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28
    fig, ax = plt.subplots(figsize=(7, 3.6))
    P = np.logspace(7, 12.5, 400)
    for C in [1e19, 1e21, 1e23, 1e25]:
        D = C / (6 * P)
        L = E + A / P**alpha + B / D**beta
        line, = ax.plot(P, L, label=f"C = {C:.0e} FLOPs")
        G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))          # closed-form optimum, as in Code 8.4.3
        Po = G * (C / 6) ** (beta / (alpha + beta))
        Do = C / (6 * Po)
        Lo = E + A / Po**alpha + B / Do**beta
        ax.plot(Po, Lo, "o", color=line.get_color())
        ax.annotate(f"{Do / Po:.0f} tokens/param", (Po, Lo), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color=line.get_color())
    ax.set_xscale("log")
    ax.set_ylim(1.8, 4.5)
    ax.set_xlabel("parameters P")
    ax.set_ylabel("predicted loss (nats per token)")
    ax.set_title("Chinchilla parametric fit: each compute budget has an optimal model size")
    ax.legend(fontsize=8)
    save(fig, "isoflop-curves.png")


# ---------------------------------------------------------------- mini GPT (same as Section 8.2's gpt.py)
class CausalSelfAttention(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.h, self.dropout = h, dropout

    def forward(self, x):
        B, n, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.view(B, n, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, n, d))


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc, self.proj = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(d), CausalSelfAttention(d, h, dropout)
        self.ln2, self.mlp = nn.LayerNorm(d), MLP(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self, V, n_max, N, d, h, dropout):
        super().__init__()
        self.n_max = n_max
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(n_max, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d, h, dropout) for _ in range(N))
        self.ln_f = nn.LayerNorm(d)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * N))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets):
        x = self.drop(self.tok(idx) + self.pos(torch.arange(idx.size(1))))
        for block in self.blocks:
            x = block(x)
        logits = self.ln_f(x) @ self.tok.weight.T
        return logits, F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def shakespeare_tokens():
    """Tiny Shakespeare in GPT-2 tokens, renumbered to the tokens that occur (as in Code 8.4.1)."""
    import tiktoken
    path = HERE / "input.txt"
    downloaded = not path.exists()
    if downloaded:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor(enc.encode(path.read_text(encoding="utf-8")))
    if downloaded:
        os.remove(path)
    vocab = ids.unique()
    return torch.searchsorted(vocab, ids), len(vocab)


def fig_training_curves():
    """The training run of Code 8.4.1, evaluated every 50 steps instead of every 200."""
    data, V = shakespeare_tokens()
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    n, B = 128, 16

    def get_batch(src, gen):
        i = torch.randint(len(src) - n - 1, (B,), generator=gen)
        blk = torch.stack([src[j: j + n + 1] for j in i.tolist()])
        return blk[:, :-1], blk[:, 1:]

    torch.manual_seed(0)
    model = GPT(V, n, N=4, d=128, h=4, dropout=0.1)
    decay = [p for name, p in model.named_parameters() if p.dim() == 2 and "pos" not in name]
    no_decay = [p for name, p in model.named_parameters() if p.dim() < 2 or "pos" in name]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}], lr=1e-3, betas=(0.9, 0.95))
    steps, warmup, min_ratio = 800, 50, 0.1

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        progress = (s - warmup) / (steps - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    @torch.no_grad()
    def evaluate(src, iters=10):
        model.eval()
        gen = torch.Generator().manual_seed(123)
        losses = []
        for _ in range(iters):
            x, y = get_batch(src, gen)
            losses.append(model(x, y)[1].item())
        model.train()
        return sum(losses) / len(losses)

    gen = torch.Generator().manual_seed(0)
    record = [(0, evaluate(train_data), evaluate(val_data))]
    P = sum(p.numel() for p in model.parameters())
    for step in range(steps):
        x, y = get_batch(train_data, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 50 == 0:
            record.append((step + 1, evaluate(train_data), evaluate(val_data)))
            print(f"  step {step + 1}: train {record[-1][1]:.3f}, val {record[-1][2]:.3f}")
    steps_done, tr, va = map(np.array, zip(*record))
    tokens = steps_done * B * n
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(tokens / 1e6, tr, label="training loss")
    ax.plot(tokens / 1e6, va, label="validation loss")
    ax.axhline(math.log(V), color="gray", ls=":", lw=0.8)
    ax.text(tokens[-1] / 1e6, math.log(V) - 0.35, rf"uniform over the {V:,} tokens: $\ln V$ = {math.log(V):.2f}",
            ha="right", fontsize=8)
    ax.axhline(5.859, color="tab:red", ls="--", lw=0.8)
    ax.text(tokens[-1] / 1e6, 5.859 + 0.15, "bigram baseline (validation) = 5.86", ha="right", fontsize=8,
            color="tab:red")
    ax.set_xlabel("training tokens seen (millions)")
    ax.set_ylabel("loss (nats per token)")
    ax.set_title(f"Mini GPT ({P / 1e6:.2f}M parameters) on tiny Shakespeare")
    ax.legend(fontsize=8, loc="upper center")
    save(fig, "mini-gpt-training.png")
    i = int(np.argmin(va))
    return {"best val": (int(steps_done[i]), float(va[i])), "final": (float(tr[-1]), float(va[-1]))}


# ---------------------------------------------------------------- sampling with GPT-2
PROMPT = "The city council met on Tuesday night to discuss the plan for the new park."


def fig_temperature():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    top = logits.topk(12).indices
    labels = [repr(tok.decode([t]))[1:-1].replace("\n", "\\n") for t in top.tolist()]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, tau in zip(axes, [0.5, 1.0, 1.5]):
        p = (logits / tau).softmax(-1)[top].numpy()
        ax.bar(range(len(top)), p)
        ax.set_xticks(range(len(top)), labels, rotation=70, fontsize=7)
        ax.set_title(rf"$\tau$ = {tau}: top-12 mass {p.sum():.2f}")
    axes[0].set_ylabel("probability")
    fig.suptitle("GPT-2's next-token distribution after the prompt, at three temperatures", y=1.04)
    save(fig, "temperature.png")
    return {tau: [(labels[i], round(float(q), 3)) for i, q in enumerate((logits / tau).softmax(-1)[top][:3])]
            for tau in [0.5, 1.0, 1.5]}


def fig_nucleus_size(steps=100, p=0.9):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    g = torch.Generator().manual_seed(0)
    sizes = []
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1]
        for _ in range(steps):
            probs, order = logits.softmax(-1).sort(descending=True)
            keep = int(((probs.cumsum(-1) - probs) < p).sum())
            sizes.append(keep)
            q = probs[:keep] / probs[:keep].sum()
            t = int(order[torch.multinomial(q, 1, generator=g)])
            out = model(torch.tensor([[t]]), past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[0, -1]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(range(1, steps + 1), sizes, ".-", lw=0.8)
    ax.set_xlabel("generation step")
    ax.set_ylabel("tokens in the nucleus (log scale)")
    ax.set_title(f"Size of the top-p set (p = {p}) while GPT-2 samples 100 tokens")
    save(fig, "nucleus-size.png")
    return {"min": min(sizes), "median": int(np.median(sizes)), "max": max(sizes),
            "steps with 1 token": sum(s == 1 for s in sizes)}


# ---------------------------------------------------------------- in-context learning (Section 8.6)
def fig_in_context_learning():
    from datasets import load_dataset
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    sst2 = load_dataset("stanfordnlp/sst2")
    train = [ex for ex in sst2["train"] if 8 <= len(ex["sentence"].split()) <= 25]
    test = list(sst2["validation"])[:100]
    words = {0: " negative", 1: " positive"}

    def prompt(demos, sentence):
        shots = "".join(f"Review: {d['sentence'].strip()}\nSentiment:{words[d['label']]}\n\n" for d in demos)
        return shots + f"Review: {sentence.strip()}\nSentiment:"

    @torch.no_grad()
    def accuracy(model, tok, k, seed):
        rng = random.Random(seed)
        label_ids = [tok.encode(words[0])[0], tok.encode(words[1])[0]]
        correct = 0
        for ex in test:
            demos = rng.sample(train, k)
            ids = tok(prompt(demos, ex["sentence"]), return_tensors="pt").input_ids
            logits = model(ids).logits[0, -1, label_ids]
            correct += int(logits.argmax()) == ex["label"]
        return correct / len(test)

    ks = [0, 1, 4, 8]
    results = {}
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, label in [("gpt2", "GPT-2 small (124M)"), ("gpt2-medium", "GPT-2 medium (355M)")]:
        tok = GPT2TokenizerFast.from_pretrained(name)
        model = GPT2LMHeadModel.from_pretrained(name).eval()
        accs = [np.mean([accuracy(model, tok, k, s) for s in ([0] if k == 0 else [0, 1, 2])]) for k in ks]
        results[name] = [round(float(a), 3) for a in accs]
        print(" ", name, results[name])
        ax.plot(ks, accs, "o-", label=label)
    majority = max(np.mean([ex["label"] for ex in test]), 1 - np.mean([ex["label"] for ex in test]))
    ax.axhline(majority, color="gray", ls=":", lw=0.8)
    ax.text(8, majority + 0.01, "majority class", ha="right", fontsize=8)
    ax.set_xticks(ks)
    ax.set_xlabel("examples in the prompt (k)")
    ax.set_ylabel("accuracy on 100 SST-2 reviews")
    ax.set_title("In-context learning with GPT-2")
    ax.legend(fontsize=8)
    save(fig, "in-context-learning.png")
    return results


FIGURES = {"lr": fig_lr_schedule, "isoflop": fig_isoflop, "temperature": fig_temperature,
           "nucleus": fig_nucleus_size, "icl": fig_in_context_learning, "training": fig_training_curves}

if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or FIGURES:        # e.g. "python make_figures.py lr isoflop"; default: all
        result = FIGURES[name]()
        if result is not None:
            print(name, result)
