"""Figures for Section 2.1: the artificial neuron and the perceptron."""
import numpy as np
import matplotlib.pyplot as plt

from style import GREEN, RED, GRAY, save, scatter_classes


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


# Separable data: used only for the mistakes-per-epoch comparison in Figure 2.2
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

# ---------------------------------------------------------------- Figure 2.2
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
