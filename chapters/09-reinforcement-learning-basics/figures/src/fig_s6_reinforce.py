"""Figures and numbers for Section 9.6: REINFORCE, baselines, and step sizes.

Loads reinforce_results.npy written by run_reinforce_experiments.py
(run that script first; it takes several minutes on a CPU).
"""
import numpy as np
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, save, smooth

res = np.load("reinforce_results.npy", allow_pickle=True)
by = {}
for name, seed, rets, var in res:
    by.setdefault(name, []).append((seed, rets, var))

# ------------------------------------------------ numbers quoted in the text
for name in ("no baseline", "baseline"):
    R = np.array([r for _, r, _ in by[name]])
    first = [int(np.argmax(smooth(r, 20) >= 475)) if (smooth(r, 20) >= 475).any() else None
             for r in R]
    print(f"{name:12s} mean return eps 1-200: {R[:, :200].mean():6.1f}  eps 201-500: "
          f"{R[:, 200:500].mean():6.1f}  last 100: {R[:, -100:].mean():6.1f} "
          f"(per seed {np.round(R[:, -100:].mean(1)).astype(int).tolist()}); "
          f"first episode with 20-ep average >= 475: {first}")
V = {name: np.array([[v[1], v[2]] for _, _, var in by[name] for v in var]).reshape(
    len(by[name]), -1, 2) for name in ("no baseline", "baseline")}
eps_ck = [v[0] for v in by["baseline"][0][2]]
for name in ("no baseline", "baseline"):
    ratio = V[name][:, :, 0] / V[name][:, :, 1]
    print(f"variance ratio (plain / baseline) during '{name}' training, by checkpoint:",
          np.round(np.median(ratio, axis=0), 1).tolist(),
          " overall median", np.round(np.median(ratio), 1))

# ------------------------------------------------ Figure 9.14: learning curves
fig, ax = plt.subplots(figsize=(8.5, 4.2))
for name, col, label in [("no baseline", ORANGE, "REINFORCE"),
                         ("baseline", BLUE, "REINFORCE with a learned baseline")]:
    S = np.array([smooth(r, 20) for _, r, _ in by[name]])
    ax.plot(S.T, color=col, lw=0.6, alpha=0.35)
    ax.plot(S.mean(0), color=col, lw=2.4, label=f"{label} (mean of 5 seeds)")
ax.set_xlabel("episode")
ax.set_ylabel("episode return (20-episode average)")
ax.set_title("REINFORCE on CartPole-v1, with and without a baseline")
ax.set_ylim(0, 520)
ax.legend(loc="lower right")
save(fig, "fig9-14-reinforce-baseline.png")

# ------------------------------------------------ Figure 9.15: gradient variance
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
for k, col, lab in [(0, ORANGE, "without baseline: weights $G_t$"),
                    (1, BLUE, "with baseline: weights $G_t - V(s_t)$")]:
    m = np.exp(np.log(V["baseline"][:, :, k]).mean(0))      # geometric mean over seeds
    axes[0].plot(eps_ck, m, marker="o", color=col, label=lab)
axes[0].set_yscale("log")
axes[0].set_xlabel("training episode at which the policy was frozen")
axes[0].set_ylabel("total variance of one-episode\ngradient estimates")
axes[0].set_title("Variance of the two estimators")
axes[0].legend(fontsize=9)
for name, col, lab in [("baseline", BLUE, "policies trained with baseline"),
                       ("no baseline", ORANGE, "policies trained without baseline")]:
    ratio = V[name][:, :, 0] / V[name][:, :, 1]
    axes[1].plot(eps_ck, np.median(ratio, axis=0), marker="o", color=col, label=lab)
axes[1].axhline(1, color=GRAY, lw=1)
axes[1].set_xlabel("training episode at which the policy was frozen")
axes[1].set_ylabel("variance ratio (without / with)")
axes[1].set_title("How many times smaller with a baseline (median of 5 seeds)")
axes[1].legend(fontsize=9)
save(fig, "fig9-15-gradient-variance.png")

# ------------------------------------------------ Figure 9.17: step sizes
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, (name, title) in zip(axes, [("baseline", "learning rate $10^{-3}$"),
                                    ("baseline, lr=1e-2", "learning rate $10^{-2}$"),
                                    ("baseline, lr=3e-2", "learning rate $3 \\times 10^{-2}$")]):
    for (seed, r, _), col in zip(sorted(by[name], key=lambda x: x[0])[:3], [BLUE, GREEN, PURPLE]):
        ax.plot(smooth(r, 20), color=col, lw=1.4, label=f"seed {seed}")
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.set_ylim(0, 520)
axes[0].set_ylabel("return (20-episode average)")
axes[0].legend(loc="upper left")
fig.suptitle("REINFORCE with baseline: larger steps can collapse the policy", y=1.03)
save(fig, "fig9-17-step-size.png")
for name in ("baseline, lr=1e-2", "baseline, lr=3e-2"):
    R = [r for _, r, _ in by[name]]
    print(name, "peak 20-ep average per seed:", [round(smooth(r, 20).max()) for r in R],
          " last 100:", [round(r[-100:].mean(), 1) for r in R])
