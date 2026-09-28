"""Figures for Section 2.3: the multi-layer perceptron."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, BROWN, save, scatter_classes, decision_background, make_moons
from mlp_common import mlp_train, predict_proba, mlp_forward


# ---------------------------------------------------------------- Figure 2.6: network diagram
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.set_xlim(-0.5, 8.5)
ax.set_ylim(-0.8, 5.5)
ax.axis("off")
ax.set_title("A multi-layer perceptron with one hidden layer (2-4-1)")

layers = {
    "input":  [(1.0, 3.5), (1.0, 1.5)],
    "hidden": [(4.0, 4.5), (4.0, 3.2), (4.0, 1.8), (4.0, 0.5)],
    "output": [(7.0, 2.5)],
}
labels = {
    "input":  [r"$x_1$", r"$x_2$"],
    "hidden": [r"$h_1$", r"$h_2$", r"$h_3$", r"$h_4$"],
    "output": [r"$\hat y$"],
}
colors = {"input": BLUE, "hidden": ORANGE, "output": GREEN}

# weights as lines
for i, p in enumerate(layers["input"]):
    for j, q in enumerate(layers["hidden"]):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=GRAY, lw=0.9, alpha=0.7, zorder=0)
for j, p in enumerate(layers["hidden"]):
    for q in layers["output"]:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=GRAY, lw=0.9, alpha=0.7, zorder=0)

# bias nodes
ax.add_patch(Circle((1.0, 4.7), 0.28, facecolor="white", edgecolor=PURPLE, lw=1.5, zorder=2))
ax.text(1.0, 4.7, "+1", ha="center", va="center", fontsize=10, color=PURPLE)
for q in layers["hidden"]:
    ax.plot([1.0, q[0]], [4.7, q[1]], color=PURPLE, lw=0.8, ls=":", alpha=0.7, zorder=0)
ax.add_patch(Circle((4.0, 5.2), 0.28, facecolor="white", edgecolor=PURPLE, lw=1.5, zorder=2))
ax.text(4.0, 5.2, "+1", ha="center", va="center", fontsize=10, color=PURPLE)
ax.plot([4.0, 7.0], [5.2, 2.5], color=PURPLE, lw=0.8, ls=":", alpha=0.7, zorder=0)

for name in ("input", "hidden", "output"):
    for (xy, lab) in zip(layers[name], labels[name]):
        ax.add_patch(Circle(xy, 0.38, facecolor="white", edgecolor=colors[name], lw=2.2, zorder=2))
        ax.text(*xy, lab, ha="center", va="center", fontsize=12, zorder=3)

ax.text(1.0, 0.0, "input layer\n(2 units)", ha="center", fontsize=11, color=BLUE)
ax.text(4.0, -0.5, "hidden layer\n(4 units + nonlinearity)", ha="center", fontsize=11, color=ORANGE)
ax.text(7.0, 0.7, "output layer\n(1 unit)", ha="center", fontsize=11, color=GREEN)
ax.annotate(r"$W^{(1)}\in\mathbb{R}^{2\times 4}$", xy=(2.5, 4.0), fontsize=11, ha="center",
            bbox=dict(fc="white", ec=GRAY, alpha=0.9))
ax.annotate(r"$W^{(2)}\in\mathbb{R}^{4\times 1}$", xy=(5.5, 4.2), fontsize=11, ha="center",
            bbox=dict(fc="white", ec=GRAY, alpha=0.9))
save(fig, "fig2-09-mlp-architecture.png")

# ---------------------------------------------------------------- Figure 2.8: XOR solved
X_xor = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y_xor = np.array([0., 1., 1., 0.])
params = mlp_train(X_xor, y_xor, n_hidden=2, act="tanh", lr=1.5, epochs=3000, seed=30, l2=0)
print("XOR final loss", params["losses"][-1], "preds", predict_proba(params, X_xor).round(3))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
scatter_classes(ax, X_xor, y_xor.astype(int), s=180, edge="k")
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect("equal")
ax.set_title("XOR inputs")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(loc="lower right")

ax = axes[1]
decision_background(ax, lambda Z: predict_proba(params, Z), (-0.5, 1.5), (-0.5, 1.5))
scatter_classes(ax, X_xor, y_xor.astype(int), s=180, edge="k")
ax.set_aspect("equal")
ax.set_title("Decision boundary of trained 2-2-1 MLP")
ax.set_xlabel("$x_1$")

# hidden unit features
ax = axes[2]
_, h, *_ = mlp_forward(X_xor, params["W1"], params["b1"], params["W2"], params["b2"], "tanh")
for c in (0, 1):
    m = y_xor == c
    ax.scatter(h[m, 0], h[m, 1], s=180, c=[BLUE, ORANGE][c], edgecolors="k",
               linewidths=0.8, label=f"class {c}", zorder=3)
done = set()
for i, (a, b) in enumerate(h):
    if i in done:
        continue
    group = [j for j in range(4) if np.linalg.norm(h[j] - h[i]) < 0.15]
    done.update(group)
    lab = " & ".join(f"({int(X_xor[j,0])},{int(X_xor[j,1])})" for j in group)
    ax.annotate(lab, (a, b), xytext=(-85 if a > 0.5 else 8, -22 if a > 0.5 else 10), textcoords="offset points", fontsize=9)
print("hidden activations for XOR inputs:\n", h.round(3))
ax.set_title("Hidden-unit features")
ax.set_xlabel("$h_1 = \\tanh(z_1)$")
ax.set_ylabel("$h_2 = \\tanh(z_2)$")
ax.legend(loc="best")
ax.set_aspect("equal")
# show that the image of the four points is linearly separable
# draw a separating line in hidden space by logistic fit in NumPy
w = params["W2"].ravel()
b = float(params["b2"][0])
# the output decision is sigmoid(w·h + b) = 0.5  ⇒  w·h + b = 0
xs = np.linspace(h[:, 0].min() - 0.3, h[:, 0].max() + 0.3, 2)
if abs(w[1]) > 1e-8:
    ax.plot(xs, -(w[0] * xs + b) / w[1], "k--", lw=1.5, label="output boundary")
ax.legend(loc="lower right", fontsize=9)
save(fig, "fig2-11-xor-mlp.png")

# ---------------------------------------------------------------- Figure 2.9: two moons
X, y = make_moons(n=300, noise=0.18, seed=1)
params_m = mlp_train(X, y.astype(float), n_hidden=16, act="relu", lr=0.8, epochs=600, seed=0)
print("moons final loss", params_m["losses"][-1], "acc",
      ((predict_proba(params_m, X) > 0.5) == y).mean())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
scatter_classes(ax, X, y, s=18)
ax.set_title("Two moons (not linearly separable)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_aspect("equal")
ax.legend(loc="upper right")
ax = axes[1]
decision_background(ax, lambda Z: predict_proba(params_m, Z),
                     (X[:, 0].min() - 0.3, X[:, 0].max() + 0.3),
                     (X[:, 1].min() - 0.3, X[:, 1].max() + 0.3))
scatter_classes(ax, X, y, s=18)
ax.set_title("16-unit ReLU MLP decision boundary")
ax.set_xlabel("$x_1$")
ax.set_aspect("equal")
save(fig, "fig2-12-two-moons.png")

# ---------------------------------------------------------------- Figure 2.10: universal approx
rng = np.random.default_rng(0)
def target(x):
    return np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x)

xs = np.linspace(0, 1, 200)[:, None]
ys = target(xs[:, 0])
X_train = rng.uniform(0, 1, (80, 1))
y_train = target(X_train[:, 0]) + rng.normal(0, 0.05, 80)

def train_regressor(n_h, epochs=6000, lr=0.01, seed=0):
    """Fit a 1-n_h-1 ReLU network with PyTorch and Adam (Adam is covered in Chapter 3)."""
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(1, n_h), torch.nn.ReLU(), torch.nn.Linear(n_h, 1))
    with torch.no_grad():  # place the ReLU kinks inside [0, 1]
        net[0].weight.copy_(torch.randn(n_h, 1) * 4)
        net[0].bias.copy_(-net[0].weight[:, 0] * torch.rand(n_h))
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)[:, None]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(Xt) - yt) ** 2).mean()
        loss.backward()
        opt.step()
    print(f"universal approx: {n_h} hidden units, final train MSE {loss.item():.4f}")

    def pred(X):
        with torch.no_grad():
            return net(torch.tensor(X, dtype=torch.float32)).numpy().ravel()
    return pred

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for ax, nh in zip(axes, [1, 3, 10, 50]):
    pred = train_regressor(nh, seed=nh)
    ax.plot(xs[:, 0], ys, color=GRAY, lw=2, label="true $f(x)$")
    ax.scatter(X_train[:, 0], y_train, s=12, color=BLUE, alpha=0.5, zorder=3, label="noisy samples")
    ax.plot(xs[:, 0], pred(xs), color=RED, lw=2, label=f"MLP ({nh} hidden)")
    ax.set_title(f"{nh} hidden unit{'s' if nh > 1 else ''}")
    ax.set_xlabel("$x$")
axes[0].set_ylabel("$y$")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Universal approximation in 1-D: more hidden units → richer fits", y=1.05)
save(fig, "fig2-13-universal-approx.png")
