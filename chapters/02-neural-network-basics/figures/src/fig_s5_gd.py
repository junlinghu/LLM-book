"""Figures for Section 2.5: gradient descent."""
import numpy as np
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, BROWN, save

# A simple 2-D quadratic bowl: L(w) = 0.5 (a w1^2 + b w2^2)
a, b = 4.0, 1.0


def loss(w):
    return 0.5 * (a * w[0] ** 2 + b * w[1] ** 2)


def grad(w):
    return np.array([a * w[0], b * w[1]])


def run_gd(w0, lr, steps=40):
    path = [np.array(w0, float)]
    w = path[0].copy()
    for _ in range(steps):
        w = w - lr * grad(w)
        path.append(w.copy())
    return np.array(path)


# ---------------------------------------------------------------- Figure 2.15: learning rates on 1-D
xs = np.linspace(-3, 3, 400)
f = 0.5 * a * xs ** 2  # 1-D slice along w1


def run_1d(x0, lr, steps=12):
    xs_ = [x0]
    x = x0
    for _ in range(steps):
        x = x - lr * (a * x)
        xs_.append(x)
    return np.array(xs_)


fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
configs = [(0.05, "too small ($\\eta=0.05$)", BLUE),
           (0.35, "good ($\\eta=0.35$)", GREEN),
           (0.55, "too large ($\\eta=0.55$)", RED)]
for ax, (lr, title, col) in zip(axes, configs):
    path = run_1d(2.5, lr, steps=12 if lr < 0.5 else 5)
    span = max(3.0, np.abs(path).max() * 1.1)
    xx = np.linspace(-span, span, 400)
    ax.plot(xx, 0.5 * a * xx ** 2, color=GRAY, lw=2)
    ax.plot(path, 0.5 * a * path ** 2, "o-", color=col, ms=6)
    for i in range(len(path) - 1):
        ax.annotate("", xy=(path[i + 1], 0.5 * a * path[i + 1] ** 2),
                    xytext=(path[i], 0.5 * a * path[i] ** 2),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
    ax.set_title(title)
    ax.set_xlabel("$w$")
for ax in axes:
    ax.set_ylabel("$L(w)$")
fig.suptitle(r"Gradient descent on $L(w)=\frac{1}{2}\cdot 4\,w^2$", y=1.05)
save(fig, "fig2-18-lr-1d.png")

# ---------------------------------------------------------------- Figure 2.16: 2-D paths
w1 = np.linspace(-3, 3, 200)
w2 = np.linspace(-3, 3, 200)
W1, W2 = np.meshgrid(w1, w2)
Z = 0.5 * (a * W1 ** 2 + b * W2 ** 2)

fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
configs = [(0.05, "too small ($\\eta=0.05$)", BLUE, 60),
           (0.35, "good ($\\eta=0.35$)", GREEN, 25),
           (0.48, "oscillating ($\\eta=0.48$)", ORANGE, 25),
           (0.53, "diverging ($\\eta=0.53$)", RED, 25)]
start = np.array([2.5, 2.5])
for ax, (lr, title, col, steps) in zip(axes, configs):
    cs = ax.contour(W1, W2, Z, levels=12, colors=GRAY, linewidths=0.8)
    ax.clabel(cs, fmt=lambda v: f"{v:.0f}", fontsize=8, inline=True)
    path = run_gd(start, lr, steps=steps)
    # keep the path inside the view for the diverging case
    visible = np.all(np.abs(path) < 3.2, axis=1)
    # include one point just outside so the arrow of divergence is visible
    if not visible.all():
        first_out = int(np.argmax(~visible))
        path = path[: max(first_out, 1) + 1]
        path[-1] = np.clip(path[-1], -3.1, 3.1)
    ax.plot(path[:, 0], path[:, 1], "o-", color=col, ms=4, lw=1.5)
    ax.scatter(*start, s=80, c="k", zorder=5, marker="s")
    ax.scatter(0, 0, s=140, c="gold", edgecolors="k", zorder=5, marker="*")
    ax.set_title(title)
    ax.set_xlabel("$w_1$")
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
axes[0].set_ylabel("$w_2$")
fig.suptitle(r"Paths on $L(w)=\frac{1}{2}(4w_1^2 + w_2^2)$; ★ = optimum", y=1.02)
save(fig, "fig2-19-lr-2d.png")
print("final positions", [run_gd(start, lr, 25)[-1] for lr, *_ in configs])

# ---------------------------------------------------------------- Figure 2.17: SGD vs batch
# Fit 1-D linear regression y = 2x + noise with batch vs SGD vs minibatch
rng = np.random.default_rng(0)
N = 200
X = rng.uniform(-1, 1, N)
y = 2.0 * X + rng.normal(0, 0.4, N)


def batch_path(lr=0.05, steps=80):
    w = -1.5
    hist = [w]
    for _ in range(steps):
        g = np.mean((w * X - y) * X) * 2
        w -= lr * g
        hist.append(w)
    return np.array(hist)


def sgd_path(lr=0.05, steps=80, seed=0):
    r = np.random.default_rng(seed)
    w = -1.5
    hist = [w]
    for t in range(steps):
        i = r.integers(0, N)
        g = 2 * (w * X[i] - y[i]) * X[i]
        w -= lr * g
        hist.append(w)
    return np.array(hist)


def mini_path(lr=0.05, steps=80, bs=16, seed=0):
    r = np.random.default_rng(seed)
    w = -1.5
    hist = [w]
    for _ in range(steps):
        idx = r.choice(N, bs, replace=False)
        g = 2 * np.mean((w * X[idx] - y[idx]) * X[idx])
        w -= lr * g
        hist.append(w)
    return np.array(hist)


STEPS = 150
paths = {
    "batch GD (all 200 examples)": (batch_path(lr=0.1, steps=STEPS), BLUE),
    "SGD (1 example)": (sgd_path(lr=0.1, steps=STEPS), RED),
    "minibatch (16 examples)": (mini_path(lr=0.1, steps=STEPS), PURPLE),
}
w_star = np.sum(X * y) / np.sum(X * X)   # least-squares optimum for this data
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for name, (path, col) in paths.items():
    axes[0].plot(path, color=col, lw=1.6, label=name)
    full = [np.mean((w * X - y) ** 2) for w in path]
    axes[1].plot(full, color=col, lw=1.6, label=name)
axes[0].axhline(w_star, color=GREEN, ls="--", lw=1, label=f"optimum $w^*={w_star:.2f}$")
axes[0].set_xlabel("update step")
axes[0].set_ylabel("weight $w$")
axes[0].set_title("Parameter trajectory")
axes[0].legend(fontsize=9)
axes[1].set_yscale("log")
axes[1].set_xlabel("update step")
axes[1].set_ylabel("full-dataset MSE (log scale)")
axes[1].set_title("Loss on the whole training set")
axes[1].legend(fontsize=9)
fig.suptitle(r"Fitting $y \approx wx$ with the same learning rate ($\eta=0.1$) and three batch sizes", y=1.02)
print("w* =", w_star, "final w:", {k: round(v[0][-1], 3) for k, v in paths.items()})
save(fig, "fig2-20-sgd-vs-batch.png")

# ---------------------------------------------------------------- Figure 2.18: epochs vs iterations schematic is a bar/timeline
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.set_xlim(0, 12)
ax.set_ylim(0.5, 2.6)
ax.axis("off")
# draw one epoch as 4 minibatches
for e, y0, lab in [(0, 2.0, "epoch 1"), (1, 1.0, "epoch 2")]:
    ax.add_patch(plt.Rectangle((0.5, y0 - 0.35), 10.5, 0.7, fill=False, ec=GRAY, lw=1.5, ls="--"))
    ax.text(0.2, y0, lab, ha="right", va="center", fontsize=11)
    for i in range(4):
        x0 = 0.8 + i * 2.6
        ax.add_patch(plt.Rectangle((x0, y0 - 0.25), 2.2, 0.5, facecolor=[BLUE, ORANGE, GREEN, PURPLE][i],
                                   alpha=0.35, ec="k", lw=1))
        ax.text(x0 + 1.1, y0, f"minibatch {i + 1}\n(iteration {e * 4 + i + 1})",
                ha="center", va="center", fontsize=9)
ax.set_title("One epoch = one pass over the training set  ·  one iteration = one minibatch update", pad=12)
save(fig, "fig2-21-epoch-iteration.png")
