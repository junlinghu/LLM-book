"""Train the Scalar-engine MLP on XOR and two moons (Figure for Section 2.6)."""
import time

import numpy as np
import matplotlib.pyplot as plt

from style import BLUE, ORANGE, GREEN, RED, GRAY, save, scatter_classes, decision_background, make_moons
from scalar_mlp import train, logit


def prob_grid(params, Z):
    return np.array([1 / (1 + np.exp(-logit(params, list(z)).value)) for z in Z])


X_xor = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y_xor = [0, 1, 1, 0]
t0 = time.perf_counter()
p_xor, h_xor = train(X_xor, Y_xor, n_hidden=4, lr=1.0, steps=300, seed=3)
print(f"XOR: final loss {h_xor[-1]:.4f} in {time.perf_counter() - t0:.1f}s")

Xm, ym = make_moons(n=100, noise=0.12, seed=4)
t0 = time.perf_counter()
p_m, h_m = train(Xm.tolist(), ym.tolist(), n_hidden=8, lr=1.0, steps=300, seed=0)
tm = time.perf_counter() - t0
acc = np.mean([(logit(p_m, list(x)).value > 0) == bool(t) for x, t in zip(Xm, ym)])
print(f"moons: final loss {h_m[-1]:.4f}, train accuracy {acc:.2f}, {tm:.1f}s")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
ax = axes[0]
ax.plot(h_xor, color=RED, label="XOR (4 points, 4 hidden)")
ax.plot(h_m, color=BLUE, label="two moons (100 points, 8 hidden)")
ax.set_xlabel("gradient-descent step")
ax.set_ylabel("mean binary cross-entropy")
ax.set_title("Training with the Scalar engine")
ax.legend()
ax = axes[1]
decision_background(ax, lambda Z: prob_grid(p_xor, Z), (-0.5, 1.5), (-0.5, 1.5), n=80)
scatter_classes(ax, np.array(X_xor), np.array(Y_xor), s=160, edge="k")
ax.set_aspect("equal")
ax.set_title("XOR decision boundary")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax = axes[2]
decision_background(ax, lambda Z: prob_grid(p_m, Z), (-1.6, 2.6), (-1.1, 1.6), n=80)
scatter_classes(ax, Xm, ym, s=18)
ax.set_aspect("equal")
ax.set_title(f"Two moons (train acc. {acc:.0%})")
ax.set_xlabel("$x_1$")
save(fig, "fig2-27-scalar-train.png")
