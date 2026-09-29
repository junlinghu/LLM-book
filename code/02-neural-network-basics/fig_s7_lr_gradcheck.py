"""Vectorized MLP: learning-rate sweep and gradient checking (Section 2.7)."""
import copy
import time

import numpy as np
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, save, make_moons
import numpy_mlp as nm

X, y = make_moons(n=400, noise=0.2, seed=2)

# ---------------------------------------------------------------- learning-rate sweep
fig, ax = plt.subplots(figsize=(7.5, 4.3))
for lr, col in [(0.003, BLUE), (0.03, ORANGE), (0.3, GREEN), (3.0, PURPLE), (8.0, RED)]:
    P = nm.init_params(2, 16, 2, seed=0)
    with np.errstate(all="ignore"):
        P, hist = nm.train_sgd(P, X, y, lr=lr, epochs=60, batch_size=32, seed=0)
    hist = np.array(hist)
    print(f"lr={lr}: final loss {hist[-1]:.4f}")
    ax.plot(np.arange(1, 61), np.where(np.isfinite(hist), hist, np.nan), color=col, label=f"$\\eta={lr}$")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("training cross-entropy (log scale)")
ax.set_title("Minibatch SGD on two moons (16 ReLU units, batch 32)")
ax.legend()
save(fig, "fig2-31-lr-sweep.png")

# ---------------------------------------------------------------- timing scalar vs vectorized training step
from scalar_mlp import make_params, logit, bce_from_logit, all_params
from scalar_autograd import Scalar
Xs, ys = X[:64], y[:64]
params = make_params(2, 16, seed=0)
t0 = time.perf_counter()
for _ in range(3):
    loss = sum((bce_from_logit(logit(params, list(x)), int(t)) for x, t in zip(Xs, ys)), Scalar(0.0)) * (1 / 64)
    for p in all_params(params):
        p.grad = 0.0
    loss.backward()
t_scalar = (time.perf_counter() - t0) / 3
P = nm.init_params(2, 16, 2, seed=0)
t0 = time.perf_counter()
for _ in range(300):
    Z, cache = nm.forward(P, Xs)
    l, dZ = nm.softmax_cross_entropy(Z, ys)
    g = nm.backward(P, cache, dZ)
t_vec = (time.perf_counter() - t0) / 300
print(f"one forward+backward on 64 examples: scalar {t_scalar*1e3:.1f} ms, vectorized {t_vec*1e3:.3f} ms, "
      f"ratio {t_scalar / t_vec:.0f}x")

# ---------------------------------------------------------------- gradient check
def numeric_grads(P, X, y, eps=1e-5):
    num = {}
    for k in P:
        num[k] = np.zeros_like(P[k])
        for idx in np.ndindex(P[k].shape):
            old = P[k][idx]
            P[k][idx] = old + eps
            lp, _ = nm.softmax_cross_entropy(nm.forward(P, X)[0], y)
            P[k][idx] = old - eps
            lm, _ = nm.softmax_cross_entropy(nm.forward(P, X)[0], y)
            P[k][idx] = old
            num[k][idx] = (lp - lm) / (2 * eps)
    return num


def rel_error(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)


def buggy_backward(P, cache, dZ2):
    g = nm.backward(P, cache, dZ2)
    X_, Z1, H = cache
    dH = dZ2 @ P["W2"].T
    g["W1"] = X_.T @ dH           # BUG: forgot to multiply by the ReLU derivative
    return g


rng = np.random.default_rng(1)
Xc, yc = X[:20], y[:20]
P = nm.init_params(2, 5, 2, seed=3)
P["b1"] = rng.normal(0, 0.5, 5)   # nonzero biases so some ReLUs are off
num = numeric_grads(P, Xc, yc)
Z, cache = nm.forward(P, Xc)
_, dZ = nm.softmax_cross_entropy(Z, yc)
good = nm.backward(P, cache, dZ)
bad = buggy_backward(P, cache, dZ)
names = ["W1", "b1", "W2", "b2"]
err_good = [rel_error(good[k], num[k]) for k in names]
err_bad = [rel_error(bad[k], num[k]) for k in names]
for k, e1, e2 in zip(names, err_good, err_bad):
    print(f"{k}: correct {e1:.2e}   buggy {e2:.2e}")

fig, ax = plt.subplots(figsize=(7.5, 4))
xpos = np.arange(len(names))
ax.bar(xpos - 0.2, err_good, width=0.4, color=GREEN, label="correct backward")
ax.bar(xpos + 0.2, np.maximum(err_bad, 1e-16), width=0.4, color=RED, label="buggy backward (ReLU mask dropped)")
ax.axhline(1e-6, color=GRAY, ls="--", lw=1)
ax.text(3.45, 2.0e-6, "reference line at 1e-6", fontsize=9, color=GRAY, ha="right")
ax.set_yscale("log")
ax.set_xticks(xpos, [f"${n[0]}^{{({n[1]})}}$" for n in names])
ax.set_ylabel("relative error vs. finite differences")
ax.set_title("Gradient check: analytic vs. centered finite differences")
ax.set_ylim(1e-12, 30)
ax.legend(loc="upper right", fontsize=9)
save(fig, "fig2-32-gradcheck.png")
