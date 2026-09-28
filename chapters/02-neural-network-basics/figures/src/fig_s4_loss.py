"""Figures for Section 2.4: loss functions."""
import numpy as np
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, BROWN, save

# ---------------------------------------------------------------- Figure 2.12: MSE vs CE
p = np.linspace(1e-4, 1 - 1e-4, 500)
# target y=1
mse_1 = (1 - p) ** 2
ce_1 = -np.log(p)
# target y=0
mse_0 = (0 - p) ** 2
ce_0 = -np.log(1 - p)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(p, mse_1, color=BLUE, label="MSE, $y=1$")
ax.plot(p, ce_1, color=RED, label="cross-entropy, $y=1$")
ax.set_xlabel(r"predicted probability $\hat p$")
ax.set_ylabel("loss")
ax.set_title("When the true label is 1")
ax.legend()
ax.set_ylim(0, 4)
ax = axes[1]
ax.plot(p, mse_0, color=BLUE, label="MSE, $y=0$")
ax.plot(p, ce_0, color=RED, label="cross-entropy, $y=0$")
ax.set_xlabel(r"predicted probability $\hat p$")
ax.set_title("When the true label is 0")
ax.legend()
ax.set_ylim(0, 4)
save(fig, "fig2-15-mse-vs-ce.png")

# ---------------------------------------------------------------- Figure 2.11: softmax
logits_a = np.array([2.0, 0.5, -1.0])
logits_b = logits_a - logits_a.max()  # numerically stable (same probs)
logits_c = np.array([10.0, 8.5, 7.0])  # shifted by +8 → identical probs


def soft(z):
    e = np.exp(z - z.max())
    return e / e.sum()


fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
for ax, z, title in zip(
    axes,
    [logits_a, logits_c, np.array([0.0, 0.0, 0.0])],
    [r"$z=(2.0,\,0.5,\,-1.0)$", r"$z=(10,\,8.5,\,7)$  (+8 shift)", r"$z=(0,\,0,\,0)$  uniform"],
):
    p = soft(z)
    bars = ax.bar(["class 0", "class 1", "class 2"], p, color=[BLUE, ORANGE, GREEN], edgecolor="k")
    for rect, v in zip(bars, p):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_ylabel("softmax probability" if ax is axes[0] else "")
fig.suptitle("Softmax turns logits into a probability distribution", y=1.05)
print("probs a", soft(logits_a).round(4), "probs c", soft(logits_c).round(4))
save(fig, "fig2-14-softmax.png")

# ---------------------------------------------------------------- Figure 2.13: CE gradient intuition
# Plot ∂L/∂z = p-y for a binary logistic classifier as a function of logit z
z = np.linspace(-6, 6, 400)
y_true = 1.0
p = 1 / (1 + np.exp(-z))
grad_ce = p - y_true
# For MSE on sigmoid output: L=(σ(z)-y)^2 → dL/dz = 2(σ-y)σ(1-σ)
grad_mse = 2 * (p - y_true) * p * (1 - p)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(z, p, color=BLUE, label=r"$\hat p=\sigma(z)$")
ax.axhline(1, color=GRAY, ls="--", lw=1)
ax.set_xlabel("logit $z$")
ax.set_ylabel(r"predicted probability $\hat p$")
ax.set_title("Sigmoid output (true label $y=1$)")
ax.legend()
ax = axes[1]
ax.plot(z, grad_ce, color=RED, label=r"CE: $\partial L/\partial z = \hat p - y$")
ax.plot(z, grad_mse, color=BLUE, label=r"MSE: $\partial L/\partial z = 2(\hat p-y)\hat p(1-\hat p)$")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("logit $z$")
ax.set_ylabel(r"$\partial L / \partial z$")
ax.set_title("Gradient w.r.t. the logit")
ax.legend(fontsize=9)
# annotate saturation problem for MSE
ax.annotate("MSE gradient vanishes\nwhen confidently wrong", xy=(-4, grad_mse[np.argmin(np.abs(z + 4))]),
            xytext=(-5.9, -0.6), fontsize=9, arrowprops=dict(arrowstyle="->", color=BLUE), color=BLUE)
ax.annotate("CE keeps pushing", xy=(-4, grad_ce[np.argmin(np.abs(z + 4))]),
            xytext=(-5.5, -0.85), fontsize=9, arrowprops=dict(arrowstyle="->", color=RED), color=RED)
save(fig, "fig2-16-ce-vs-mse-grad.png")

# ---------------------------------------------------------------- Figure 2.14: next-token
# Tiny illustration: vocabulary of 5 tokens, one position
vocab = ["the", "cat", "sat", "on", "mat"]
true_idx = 1  # "cat"
logits = np.array([1.2, 2.5, 0.3, -0.5, 0.8])
probs = soft(logits)
nll = -np.log(probs[true_idx])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
bars = ax.bar(vocab, logits, color=GRAY)
bars[true_idx].set_color(GREEN)
ax.set_ylabel("logit $z_v$")
ax.set_title("Raw next-token scores (logits)")
ax = axes[1]
bars = ax.bar(vocab, probs, color=GRAY)
bars[true_idx].set_color(GREEN)
ax.axhline(probs[true_idx], color=GREEN, ls="--", lw=1)
ax.set_ylim(0, 0.72)
ax.set_ylabel("softmax probability")
ax.set_title(f"True token 'cat':  $-\\log p$ = {nll:.3f}  (this position's loss)")
for rect, v in zip(bars, probs):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
save(fig, "fig2-17-next-token-loss.png")
