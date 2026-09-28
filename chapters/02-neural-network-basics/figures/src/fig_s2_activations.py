"""Figures for Section 2.2: activation functions."""
import math

import numpy as np
import matplotlib.pyplot as plt

from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, BROWN, save

erf = np.vectorize(math.erf)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def phi(x):  # standard normal pdf
    return np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def Phi(x):  # standard normal cdf
    return 0.5 * (1 + erf(x / math.sqrt(2)))


ACTS = {
    "step": (lambda x: (x > 0).astype(float), lambda x: np.zeros_like(x)),
    "sigmoid": (sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))),
    "tanh": (np.tanh, lambda x: 1 - np.tanh(x) ** 2),
    "ReLU": (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)),
    "GELU": (lambda x: x * Phi(x), lambda x: Phi(x) + x * phi(x)),
    "SiLU / Swish": (lambda x: x * sigmoid(x),
                     lambda x: sigmoid(x) * (1 + x * (1 - sigmoid(x)))),
}
COLORS = [GRAY, BLUE, ORANGE, GREEN, PURPLE, RED]

# ---------------------------------------------------------------- Figure 2.4
x = np.linspace(-6, 6, 1201)
fig, axes = plt.subplots(2, 6, figsize=(17, 5.6), sharex=True)
for j, ((name, (f, df)), c) in enumerate(zip(ACTS.items(), COLORS)):
    y, dy = f(x), df(x)
    top, bot = axes[0, j], axes[1, j]
    if name == "step":
        # draw the jump as two pieces so that no vertical line is implied
        top.plot(x[x < 0], y[x < 0], color=c)
        top.plot(x[x > 0], y[x > 0], color=c)
        bot.plot(x, dy, color=c)
        bot.annotate("undefined at 0", (0, 0), xytext=(-5.5, 0.55), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="k"))
    else:
        top.plot(x, y, color=c)
        bot.plot(x, dy, color=c)
    # shade saturated regions: where the derivative is below 5% of its max
    sat = dy < 0.05 * max(dy.max(), 1e-9)
    if name != "step":
        for ax in (top, bot):
            ax.fill_between(x, 0, 1, where=sat, transform=ax.get_xaxis_transform(),
                            color=RED, alpha=0.08, lw=0)
    top.set_title(name)
    top.axhline(0, color="k", lw=0.6)
    bot.axhline(0, color="k", lw=0.6)
    bot.set_xlabel("$z$")
    bot.set_ylim(-0.25, 1.2)
axes[0, 0].set_ylabel("$g(z)$")
axes[1, 0].set_ylabel("$g'(z)$")
for j in range(6):
    axes[0, j].set_ylim(-1.3, 1.6 if j < 3 else 6.2)
fig.text(0.5, -0.02, "Shaded: regions where the derivative is below 5% of its maximum "
         "(saturated, or 'dead' for ReLU's negative side)", ha="center", fontsize=10, color=RED)
fig.tight_layout()
save(fig, "fig2-07-activation-gallery.png")

# ---------------------------------------------------------------- Figure 2.3
rng = np.random.default_rng(0)
xs = np.linspace(-3, 3, 400)[:, None]
fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=False)
titles = ["no activation (linear layers only)", "ReLU between layers", "tanh between layers"]
acts = [lambda z: z, lambda z: np.maximum(0, z), np.tanh]
for ax, title, g in zip(axes, titles, acts):
    for k, col in zip(range(4), [BLUE, ORANGE, GREEN, PURPLE]):
        r = np.random.default_rng(100 + k)
        W1, b1 = r.normal(0, 1.2, (1, 8)), r.normal(0, 1.0, 8)
        W2, b2 = r.normal(0, 0.6, (8, 8)), r.normal(0, 0.5, 8)
        W3, b3 = r.normal(0, 0.6, (8, 1)), r.normal(0, 0.5, 1)
        h = g(xs @ W1 + b1)
        h = g(h @ W2 + b2)
        out = h @ W3 + b3
        ax.plot(xs[:, 0], out[:, 0], color=col, lw=2)
    ax.set_title(title)
    ax.set_xlabel("input $x$")
axes[0].set_ylabel("network output")
fig.suptitle("Four randomly initialized 1-8-8-1 networks", y=1.02)
save(fig, "fig2-06-why-nonlinearity.png")

# ---------------------------------------------------------------- Figure 2.5
x = np.linspace(-4, 3, 701)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, c in [("ReLU", GREEN), ("GELU", PURPLE), ("SiLU / Swish", RED)]:
    f, df = ACTS[name]
    axes[0].plot(x, f(x), color=c, label=name)
    axes[1].plot(x, df(x), color=c, label=name)
gx = np.linspace(-4, 3, 7001)
g = ACTS["GELU"][0](gx)
s = ACTS["SiLU / Swish"][0](gx)
print(f"GELU min {g.min():.4f} at {gx[g.argmin()]:.3f}; SiLU min {s.min():.4f} at {gx[s.argmin()]:.3f}")
axes[0].annotate(f"GELU min ≈ {g.min():.2f}\nat z ≈ {gx[g.argmin()]:.2f}", (gx[g.argmin()], g.min()),
                 xytext=(-3.9, 1.0), arrowprops=dict(arrowstyle="->", color=PURPLE), fontsize=9, color=PURPLE)
axes[0].annotate(f"SiLU min ≈ {s.min():.2f}\nat z ≈ {gx[s.argmin()]:.2f}", (gx[s.argmin()], s.min()),
                 xytext=(-3.9, -0.9), arrowprops=dict(arrowstyle="->", color=RED), fontsize=9, color=RED)
axes[0].set_title("Smooth ReLU-like activations")
axes[1].set_title("Their derivatives")
for ax in axes:
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("$z$")
    ax.legend(loc="upper left")
axes[0].set_ylim(-1.2, 3.1)
axes[1].set_ylim(-0.3, 1.3)
save(fig, "fig2-08-relu-gelu-silu.png")
