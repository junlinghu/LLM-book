"""Figures for Section 2.7: vectorization."""
import time

import numpy as np
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, save

# ---------------------------------------------------------------- Figure 2.24: scalar vs vectorized timing
rng = np.random.default_rng(0)
sizes = [50, 100, 200, 400, 800]
t_scalar, t_vec = [], []
for n in sizes:
    X = rng.normal(size=(n, 32))
    W = rng.normal(size=(32, 64))
    # vectorized
    t0 = time.perf_counter()
    for _ in range(20):
        _ = X @ W
    t_vec.append((time.perf_counter() - t0) / 20)
    # scalar (Python loops) — fewer reps for large n
    reps = 5 if n <= 200 else 2
    t0 = time.perf_counter()
    for _ in range(reps):
        out = np.zeros((n, 64))
        for i in range(n):
            for j in range(64):
                s = 0.0
                for k in range(32):
                    s += X[i, k] * W[k, j]
                out[i, j] = s
    t_scalar.append((time.perf_counter() - t0) / reps)
    print(n, "scalar", t_scalar[-1], "vec", t_vec[-1], "speedup", t_scalar[-1] / t_vec[-1])

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(sizes, t_scalar, "o-", color=RED, label="naive Python triple loop")
ax.plot(sizes, t_vec, "s-", color=GREEN, label="NumPy matrix multiply (X @ W)")
ax.set_yscale("log")
ax.set_xlabel("batch size $n$  (matrix is $n \\times 32$ times $32 \\times 64$)")
ax.set_ylabel("seconds per multiply (log scale)")
ax.set_title("Why we vectorize: same math, orders-of-magnitude less time")
ax.legend()
save(fig, "fig2-28-scalar-vs-vectorized.png")

# ---------------------------------------------------------------- Figure 2.25: shape diagram for a minibatch forward pass
fig, ax = plt.subplots(figsize=(13, 3.4))
ax.set_xlim(0, 17)
ax.set_ylim(0.2, 3.6)
ax.axis("off")
ax.set_title("Shapes through a vectorized forward pass (batch of 32, 2 features, 8 hidden units, 2 classes)")


def tensor(x, y, w, h, label, sub, color):
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor=color, alpha=0.25, ec="k", lw=1.5))
    ax.text(x + w / 2, y + h / 2 + 0.18, label, ha="center", va="center", fontsize=12)
    ax.text(x + w / 2, y + h / 2 - 0.25, sub, ha="center", va="center", fontsize=9.5)


def op(x, text):
    ax.text(x, 2.0, text, fontsize=13, ha="center", va="center")


tensor(0.2, 1.2, 1.4, 1.6, "$X$", r"$32\times 2$", BLUE)
op(2.0, r"$\times$")
tensor(2.4, 1.3, 1.4, 1.4, r"$W^{(1)}$", r"$2\times 8$", ORANGE)
op(4.3, r"$+$")
tensor(4.8, 1.6, 1.3, 0.8, r"$b^{(1)}$", r"$8$", ORANGE)
op(6.9, "→ ReLU →")
tensor(7.7, 1.2, 1.6, 1.6, r"$H$", r"$32\times 8$", GREEN)
op(9.7, r"$\times$")
tensor(10.1, 1.3, 1.4, 1.4, r"$W^{(2)}$", r"$8\times 2$", ORANGE)
op(12.0, r"$+$")
tensor(12.5, 1.6, 1.3, 0.8, r"$b^{(2)}$", r"$2$", ORANGE)
op(14.3, r"$=$")
tensor(14.8, 1.2, 1.6, 1.6, r"$Z$", r"$32\times 2$", PURPLE)
ax.text(8.5, 0.55, "each bias vector is broadcast (copied) across all 32 rows of the batch",
        ha="center", fontsize=10, color=GRAY)
save(fig, "fig2-29-shapes.png")

# ---------------------------------------------------------------- Figure 2.26: transpose pattern
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.set_xlim(0, 12)
ax.set_ylim(0.0, 3.1)
ax.axis("off")
ax.set_title(r"Backprop through $Z = H W$: the transpose pattern")
ax.text(6, 2.6,
        r"$Z=HW$  with  $H\in\mathbb{R}^{n\times d},\; W\in\mathbb{R}^{d\times k},\; Z\in\mathbb{R}^{n\times k}$",
        ha="center", fontsize=12)
ax.text(3, 1.5,
        r"$\dfrac{\partial L}{\partial W} = H^{\top} \dfrac{\partial L}{\partial Z}$"
        "\n"
        r"shape: $(d\times n)\,(n\times k) = (d\times k)$",
        ha="center", fontsize=12,
        bbox=dict(fc="#e8f0ff", ec=BLUE, boxstyle="round,pad=0.4"))
ax.text(9, 1.5,
        r"$\dfrac{\partial L}{\partial H} = \dfrac{\partial L}{\partial Z}\, W^{\top}$"
        "\n"
        r"shape: $(n\times k)\,(k\times d) = (n\times d)$",
        ha="center", fontsize=12,
        bbox=dict(fc="#fff0e8", ec=ORANGE, boxstyle="round,pad=0.4"))
ax.text(6, 0.35, "Every gradient has the same shape as its parameter — use this as a debugging check.",
        ha="center", fontsize=10, color=GRAY)
save(fig, "fig2-30-transpose-pattern.png")
