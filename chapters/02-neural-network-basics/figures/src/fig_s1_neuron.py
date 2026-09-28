"""Figures for Section 2.1: the artificial neuron and the perceptron."""
import numpy as np
import matplotlib.pyplot as plt

from style import (BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, save, scatter_classes)


def perceptron_train(X, y, lr=1.0, epochs=20, seed=0):
    """Perceptron rule with labels in {-1, +1}. Returns history of (w, b, i)."""
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    b = 0.0
    history = [(w.copy(), b, None)]
    errors_per_epoch = []
    for _ in range(epochs):
        errors = 0
        for i in rng.permutation(len(X)):
            if y[i] * (X[i] @ w + b) <= 0:  # misclassified (or on the boundary)
                w += lr * y[i] * X[i]
                b += lr * y[i]
                errors += 1
                history.append((w.copy(), b, i))
        errors_per_epoch.append(errors)
        if errors == 0:
            break
    return history, errors_per_epoch


def draw_line(ax, w, b, xlim, **kw):
    xs = np.linspace(*xlim, 2)
    if abs(w[1]) > 1e-12:
        ax.plot(xs, -(w[0] * xs + b) / w[1], **kw)
    elif abs(w[0]) > 1e-12:
        ax.axvline(-b / w[0], **kw)


# ---------------------------------------------------------------- Figure 2.2
rng = np.random.default_rng(3)
# Points on both sides of the (unknown to the learner) line x1 - 0.5*x2 = 1.0,
# with a margin around it so that the data are linearly separable.
X = rng.uniform(-2.2, 3.2, (200, 2))
margin = X[:, 0] - 0.5 * X[:, 1] - 1.0
keep = np.abs(margin) > 0.25
X = X[keep][:40]
y = np.where(margin[keep][:40] > 0, 1.0, -1.0)
history, errs = perceptron_train(X, y, lr=0.5, epochs=50, seed=3)
print("separable: updates", len(history) - 1, "errors/epoch", errs)

steps = [1, 4, 9, len(history) - 1]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
xlim, ylim = (-2.5, 3.5), (-2.5, 3.5)
y01 = (y > 0).astype(int)
for ax, k in zip(axes, steps):
    w, b, i = history[k]
    scatter_classes(ax, X, y01)
    draw_line(ax, w, b, xlim, color="k", lw=2)
    if i is not None:
        ax.scatter(*X[i], s=160, facecolors="none", edgecolors=RED, lw=2, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    title = f"after update {k}" if k < len(history) - 1 else f"final (update {k})"
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
axes[0].set_ylabel("$x_2$")
axes[0].legend(loc="upper left", framealpha=0.9)
save(fig, "fig2-02-perceptron-learning.png")

# ---------------------------------------------------------------- Figure 2.3
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
y_xor = np.array([-1, 1, 1, -1], float)
hist_x, errs_x = perceptron_train(X_xor, y_xor, lr=1.0, epochs=30, seed=0)
print("xor errors/epoch", errs_x)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
scatter_classes(ax, X_xor, (y_xor > 0).astype(int), s=180, edge="k")
w, b, _ = hist_x[-1]
seen = []
for (wk, bk, _) in hist_x[::-1]:
    key = (tuple(wk), bk)
    if np.any(wk != 0) and key not in seen:
        seen.append(key)
for j, (wk, bk) in enumerate(seen[:6]):
    draw_line(ax, np.array(wk), bk, (-0.6, 1.7), color=GRAY, lw=1.3, alpha=0.8, ls="--",
              label="boundaries visited\nduring training" if j == 0 else None)
for (p, lab) in zip(X_xor, ["(0,0) → 0", "(0,1) → 1", "(1,0) → 1", "(1,1) → 0"]):
    ax.annotate(lab, p, xytext=(8, 8), textcoords="offset points", fontsize=10)
ax.set_xlim(-0.5, 1.7)
ax.set_ylim(-1.15, 1.6)
ax.set_aspect("equal")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("XOR: no single line separates the classes")
ax.legend(loc="lower right", fontsize=9)

ax = axes[1]
ax.plot(range(1, len(errs)+1), errs, "o-", color=GREEN, label="separable data (40 points)")
ax.plot(range(1, len(errs_x)+1), errs_x, "s-", color=RED, label="XOR (4 points)")
ax.set_xlabel("epoch")
ax.set_ylabel("mistakes during epoch")
ax.set_title("Perceptron mistakes per epoch")
ax.set_ylim(bottom=-0.2)
ax.legend()
save(fig, "fig2-03-perceptron-xor.png")

# ---------------------------------------------------------------- Figure 2.4
w = np.array([2.0, 1.0])
b = -2.0
fig, ax = plt.subplots(figsize=(5.6, 5.2))
g = np.linspace(-1, 3, 200)
gx, gy = np.meshgrid(g, g)
z = w[0] * gx + w[1] * gy + b
cs = ax.contourf(gx, gy, z, levels=np.arange(-7, 8, 1), cmap="RdBu_r", alpha=0.35)
ax.contour(gx, gy, z, levels=[0], colors="k", linewidths=2)
cl = ax.contour(gx, gy, z, levels=[-4, -2, 2, 4], colors=GRAY, linewidths=0.8, linestyles="--")
ax.clabel(cl, fmt=lambda v: f"z={v:.0f}", fontsize=9)
p0 = np.array([0.6, 0.8])  # a point on the boundary: 2*0.6+0.8-2 = 0
ax.annotate("", xy=p0 + w / np.linalg.norm(w) * 1.2, xytext=p0,
            arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2.5))
ax.text(*(p0 + w / np.linalg.norm(w) * 1.25 + [0.05, 0.05]), r"$\mathbf{w}=(2,1)$", color=PURPLE, fontsize=12)
ax.text(2.0, 2.5, r"$z > 0$: predict 1", fontsize=11, ha="center", bbox=dict(fc="white", ec="none", alpha=0.85))
ax.text(-0.35, 0.5, r"$z < 0$: predict 0", fontsize=11, ha="center", bbox=dict(fc="white", ec="none", alpha=0.85))
ax.text(1.9, -0.75, r"$2x_1 + x_2 - 2 = 0$", fontsize=11, rotation=-58)
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_aspect("equal")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title(r"Decision boundary of $z = \mathbf{w}^\top\mathbf{x} + b$")
save(fig, "fig2-04-hyperplane.png")

# ---------------------------------------------------------------- Figure 2.5
rng = np.random.default_rng(7)
xr = rng.uniform(0, 10, 30)
yr = 1.5 * xr + 2 + rng.normal(0, 2.0, 30)
A = np.c_[xr, np.ones_like(xr)]
w_lin, b_lin = np.linalg.lstsq(A, yr, rcond=None)[0]

xc = np.r_[rng.normal(3, 1.3, 30), rng.normal(7, 1.3, 30)]
yc = np.r_[np.zeros(30), np.ones(30)]
wl, bl = 0.0, 0.0
for _ in range(20000):  # plain gradient descent on binary cross-entropy
    p = 1 / (1 + np.exp(-(wl * xc + bl)))
    wl -= 0.05 * np.mean((p - yc) * xc)
    bl -= 0.05 * np.mean(p - yc)
print(f"linear fit w={w_lin:.2f} b={b_lin:.2f}; logistic w={wl:.2f} b={bl:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.scatter(xr, yr, color=BLUE, s=25, zorder=3)
xs = np.linspace(0, 10, 100)
ax.plot(xs, w_lin * xs + b_lin, color="k", label=f"$\\hat y = {w_lin:.2f}x + {b_lin:.2f}$")
ax.set_title("Neuron with no activation = linear regression")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend()
ax = axes[1]
ax.scatter(xc, yc + rng.uniform(-0.03, 0.03, 60), c=[ORANGE if t else BLUE for t in yc], s=25, zorder=3)
ax.plot(xs, 1 / (1 + np.exp(-(wl * xs + bl))), color="k",
        label=f"$\\hat p = \\sigma({wl:.2f}x {bl:+.2f})$")
ax.axhline(0.5, color=GRAY, lw=1, ls="--")
ax.axvline(-bl / wl, color=GRAY, lw=1, ls=":")
ax.set_title("Neuron with sigmoid = logistic regression")
ax.set_xlabel("$x$")
ax.set_ylabel("$P(y=1 \\mid x)$")
ax.legend(loc="upper left")
save(fig, "fig2-05-linear-logistic.png")
