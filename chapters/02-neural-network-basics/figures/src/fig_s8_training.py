"""Figures for Section 2.8: the training loop and generalization.

These experiments use PyTorch for speed. The overfitting experiments use the
Adam optimizer (introduced in Chapter 3) because plain gradient descent needs
far more steps to drive the training loss of a wide network toward zero.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, BROWN, save, make_moons

# ---------------------------------------------------------------- Figure 2.34: data split
fig, ax = plt.subplots(figsize=(11, 1.9))
ax.axis("off")
parts = [("training set (70%)\nfit the weights", 0.70, BLUE),
         ("validation (15%)\ntune, early-stop", 0.15, ORANGE),
         ("test (15%)\nreport once", 0.15, RED)]
x0 = 0
for lab, w, c in parts:
    ax.add_patch(plt.Rectangle((x0, 0.2), w, 0.6, facecolor=c, alpha=0.3, ec="k", lw=1.2))
    ax.text(x0 + w / 2, 0.5, lab, ha="center", va="center", fontsize=9.5)
    x0 += w
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(0, 1)
ax.set_title("Splitting a dataset before training")
save(fig, "fig2-34-data-split.png")

# ---------------------------------------------------------------- data for overfitting experiments
g = torch.Generator().manual_seed(0)


def f(x):
    return torch.sin(3 * x)


N_TRAIN, NOISE = 30, 0.3
xtr = torch.rand(N_TRAIN, 1, generator=g) * 2 - 1
ytr = f(xtr) + NOISE * torch.randn(N_TRAIN, 1, generator=g)
xva = torch.rand(300, 1, generator=g) * 2 - 1
yva = f(xva) + NOISE * torch.randn(300, 1, generator=g)


def make_model(width, seed=1):
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(1, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))
    with torch.no_grad():  # spread the tanh "kinks" over the input range
        m[0].weight.mul_(3.0)
        m[0].bias.uniform_(-3, 3)
    return m


def train(width, steps=20000, lr=0.003, log_every=20):
    m = make_model(width)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tr, va, its = [], [], []
    best = (float("inf"), None, None)
    for s in range(steps + 1):
        if s % log_every == 0:
            with torch.no_grad():
                ltr = ((m(xtr) - ytr) ** 2).mean().item()
                lva = ((m(xva) - yva) ** 2).mean().item()
            tr.append(ltr); va.append(lva); its.append(s)
            if lva < best[0]:
                best = (lva, s, {k: v.clone() for k, v in m.state_dict().items()})
        opt.zero_grad()
        loss = ((m(xtr) - ytr) ** 2).mean()
        loss.backward()
        opt.step()
    return m, np.array(its), np.array(tr), np.array(va), best


# ---------------------------------------------------------------- Figure 2.35: learning curves
m64, its, tr, va, best = train(64)
print(f"width 64: final train {tr[-1]:.4f} val {va[-1]:.3f}; best val {best[0]:.3f} at step {best[1]}")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]
ax.plot(its[1:], tr[1:], color=BLUE, label="training loss")
ax.plot(its[1:], va[1:], color=ORANGE, label="validation loss")
ax.axvline(best[1], color=GREEN, ls="--", lw=1.5, label=f"early stop (step {best[1]})")
ax.axhline(NOISE ** 2, color=GRAY, ls=":", lw=1.2, label=f"noise floor ({NOISE**2:.4f})")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("training step (log scale)")
ax.set_ylabel("MSE (log scale)")
ax.set_title("Learning curves: 64 hidden units, 30 training points")
ax.legend(fontsize=9)

ax = axes[1]
xs = torch.linspace(-1, 1, 400)[:, None]
m_best = make_model(64)
m_best.load_state_dict(best[2])
with torch.no_grad():
    ax.plot(xs[:, 0], f(xs)[:, 0], color=GRAY, lw=2, label="true function")
    ax.plot(xs[:, 0], m64(xs)[:, 0], color=RED, lw=2, label="final model (overfit)")
    ax.plot(xs[:, 0], m_best(xs)[:, 0], color=GREEN, lw=2, label="early-stopped model")
ax.scatter(xtr[:, 0], ytr[:, 0], color=BLUE, s=30, zorder=3, label="training points")
ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Overfitting vs. early stopping")
ax.legend(fontsize=9, loc="lower right")
save(fig, "fig2-35-learning-curves.png")

# ---------------------------------------------------------------- Figure 2.36: capacity
widths = [1, 3, 8, 64]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
summary = []
for ax, w in zip(axes, widths):
    m, its_w, tr_w, va_w, _ = train(w, steps=20000)
    summary.append((w, tr_w[-1], va_w[-1]))
    with torch.no_grad():
        ax.plot(xs[:, 0], f(xs)[:, 0], color=GRAY, lw=2)
        ax.plot(xs[:, 0], m(xs)[:, 0], color=RED, lw=2)
    ax.scatter(xtr[:, 0], ytr[:, 0], color=BLUE, s=22, zorder=3)
    ax.set_title(f"width {w}\ntrain {tr_w[-1]:.3f} · val {va_w[-1]:.3f}")
    ax.set_xlabel("$x$")
    ax.set_ylim(-2.2, 2.2)
axes[0].set_ylabel("$y$")
fig.suptitle("Model capacity: underfitting (left) to overfitting (right), 20k steps each", y=1.07)
save(fig, "fig2-36-capacity.png")
print("capacity summary", summary)

# ---------------------------------------------------------------- Figure 2.37: symmetry
X, y = make_moons(n=200, noise=0.15, seed=0)
Xt = torch.tensor(X, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.float32)


def symmetry_run(init):
    torch.manual_seed(0)
    lin1 = torch.nn.Linear(2, 4)
    lin2 = torch.nn.Linear(4, 1)
    with torch.no_grad():
        if init == "constant":
            lin1.weight.fill_(0.5); lin1.bias.fill_(0.0)
            lin2.weight.fill_(0.5); lin2.bias.fill_(0.0)
    opt = torch.optim.SGD(list(lin1.parameters()) + list(lin2.parameters()), lr=2.0)
    traj = []
    for _ in range(1000):
        traj.append(lin1.weight[:, 0].detach().clone().numpy())
        logits = lin2(torch.tanh(lin1(Xt))).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    return np.array(traj), loss.item()


fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=False)
for ax, init, title in zip(axes, ["constant", "random"],
                           ["All weights start at 0.5", "Random initialization"]):
    traj, final = symmetry_run(init)
    for k in range(4):
        ax.plot(traj[:, k], color=[BLUE, ORANGE, GREEN, PURPLE][k], lw=2,
                ls=["-", "--", "-.", ":"][k], label=f"hidden unit {k+1}")
    ax.set_title(f"{title}  (final loss {final:.3f})")
    ax.set_xlabel("SGD step")
    ax.set_ylabel("weight from $x_1$ into unit")
    ax.legend(fontsize=9)
    print(init, "final loss", final)
save(fig, "fig2-37-symmetry.png")
