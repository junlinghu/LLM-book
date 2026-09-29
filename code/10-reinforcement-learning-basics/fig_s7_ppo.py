"""Figures and numbers for Section 10.7: PPO.

Figure 10.18 is analytic. Figures 10.20 and 10.21 load ppo_results.npy (from
run_ppo_experiments.py) and reinforce_results.npy (from
run_reinforce_experiments.py); run those scripts first.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, save, smooth

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------ Figure 10.18: the clipped objective
eps = 0.2
rho = np.linspace(0.0, 2.0, 801)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=False)
for ax, A, title in [(axes[0], 1.0, "positive advantage ($\\hat{A} = +1$)"),
                     (axes[1], -1.0, "negative advantage ($\\hat{A} = -1$)")]:
    unclipped = rho * A
    clipped = np.clip(rho, 1 - eps, 1 + eps) * A
    L = np.minimum(unclipped, clipped)
    ax.plot(rho, unclipped, color=GRAY, ls="--", lw=1.5, label="unclipped $\\rho \\hat{A}$")
    ax.plot(rho, L, color=BLUE, lw=2.6, label="$L^{\\mathrm{CLIP}}$ = min(unclipped, clipped)")
    flat = (rho > 1 + eps) if A > 0 else (rho < 1 - eps)
    ax.fill_between(rho, -2.2, 2.2, where=flat, color=ORANGE, alpha=0.12,
                    label="gradient is zero here")
    for x in (1 - eps, 1 + eps):
        ax.axvline(x, color=GRAY, lw=0.8, ls=":")
    ax.axvline(1, color="k", lw=0.8)
    ax.plot([1], [A], "o", color="k", ms=5)
    ax.set_xlabel("probability ratio $\\rho = \\pi_\\theta / \\pi_{\\theta_{\\mathrm{old}}}$")
    ax.set_ylabel("objective (to maximize)")
    ax.set_title(title)
    ax.set_ylim(-2.1, 2.1)
    ax.legend(loc="upper left" if A > 0 else "lower left", fontsize=9)
    ax.set_xticks([0, 0.5, 1 - eps, 1, 1 + eps, 1.5, 2])
    ax.set_xticklabels(["0", "0.5", "$1-\\epsilon$", "1", "$1+\\epsilon$", "1.5", "2"])
save(fig, "fig9-18-clipped-objective.png")

# ------------------------------------------------ load training results
res = np.load(os.path.join(HERE, "ppo_results.npy"), allow_pickle=True)
logs = {}
for name, seed, log in res:
    logs.setdefault(name, []).append((seed, log))
MAIN = "PPO (clip 0.2, lambda 0.95)"


def first_solved(log, level=500.0):
    hit = np.flatnonzero(log[:, 1] >= level)
    return int(log[hit[0], 0]) if len(hit) else None


for name, runs in logs.items():
    L = np.array([log for _, log in runs])                    # seeds x iters x 4
    late = L[:, L[0, :, 0] > 50_000, 1]                       # second half of training
    drops = [int(((log[:, 1] < 200) & (np.maximum.accumulate(log[:, 1]) >= 500)).sum())
             for _, log in runs]
    print(f"{name:28s} first step with 10-ep mean 500: {[first_solved(l) for _, l in runs]}; "
          f"mean return over training {np.nanmean(L[:, :, 1]):.1f}; "
          f"mean return after 50k steps {np.nanmean(late):.1f}; final {L[:, -1, 1].round(1).tolist()}; "
          f"iterations below 200 after reaching 500: {drops}; "
          f"mean/max approx KL {np.nanmean(L[:, :, 2]):.4f}/{np.nanmax(L[:, :, 2]):.3f}; "
          f"mean clip fraction {np.nanmean(L[:, :, 3]):.3f}")

# REINFORCE with baseline, re-indexed by environment steps (reward is 1 per step)
rf = np.load(os.path.join(HERE, "reinforce_results.npy"), allow_pickle=True)
rf_runs = [rets for name, seed, rets, var in rf if name == "baseline"]
grid = np.arange(512, 100_001, 512)
rf_curves = []
for rets in rf_runs:
    steps = np.cumsum(rets)
    avg = np.array([rets[max(0, i - 9):i + 1].mean() for i in range(len(rets))])
    rf_curves.append(np.interp(grid, steps, avg, left=np.nan, right=np.nan))
rf_curves = np.array(rf_curves)
print("REINFORCE+baseline: mean 10-episode return at 50k/100k env steps:",
      np.round(np.nanmean(rf_curves[:, grid.searchsorted(50_000)]), 1),
      np.round(np.nanmean(rf_curves[:, -1]), 1),
      " steps used by 1000 episodes per seed:", [int(r.sum()) for r in rf_runs])

# ------------------------------------------------ Figure 10.20: PPO learning curve + diagnostics
fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
L = np.array([log for _, log in logs[MAIN]])
steps = L[0, :, 0]
for log in L:
    axes[0].plot(steps, log[:, 1], color=BLUE, lw=0.7, alpha=0.35)
axes[0].plot(steps, np.nanmean(L[:, :, 1], 0), color=BLUE, lw=2.4, label="PPO (mean of 5 seeds)")
axes[0].plot(grid, np.nanmean(rf_curves, 0), color=ORANGE, lw=2.0,
             label="REINFORCE + baseline (Section 10.6)")
axes[0].set_xlabel("environment steps")
axes[0].set_ylabel("return (last 10 episodes)")
axes[0].set_title("PPO on CartPole-v1")
axes[0].set_ylim(0, 520)
axes[0].legend(loc="lower right", fontsize=9)
for log in L:
    axes[1].plot(steps, smooth(log[:, 2], 5), color=PURPLE, lw=1.0, alpha=0.7)
axes[1].set_xlabel("environment steps")
axes[1].set_ylabel("approximate KL(old ‖ new)")
axes[1].set_title("How far each update moves the policy")
for log in L:
    axes[2].plot(steps, smooth(log[:, 3], 5), color=GREEN, lw=1.0, alpha=0.7)
axes[2].set_xlabel("environment steps")
axes[2].set_ylabel("clip fraction")
axes[2].set_title("Fraction of samples with clipped ratio")
save(fig, "fig9-20-ppo-cartpole.png")

# ------------------------------------------------ Figure 10.21: ablations
fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
for name, col in [(MAIN, BLUE), ("no clipping", RED)]:
    for _, log in logs[name]:
        axes[0].plot(log[:, 0], log[:, 1], color=col, lw=1.0, alpha=0.6)
        axes[1].plot(log[:, 0], smooth(log[:, 2], 5), color=col, lw=1.0, alpha=0.6)
    axes[0].plot([], [], color=col, label="clipping, $\\epsilon$ = 0.2" if name == MAIN else "no clipping")
axes[0].set_title("Clipping on and off (5 seeds each)")
axes[0].set_xlabel("environment steps")
axes[0].set_ylabel("return (last 10 episodes)")
axes[0].set_ylim(0, 520)
axes[0].legend(loc="lower right", fontsize=9)
axes[1].set_yscale("log")
axes[1].set_ylim(1e-4, 10)
axes[1].set_title("Approximate KL per update")
axes[1].set_xlabel("environment steps")
axes[1].set_ylabel("approximate KL (log scale)")
for name, col, lab in [("lambda = 0", RED, "$\\lambda$ = 0 (one-step TD)"),
                       (MAIN, BLUE, "$\\lambda$ = 0.95"),
                       ("lambda = 1", GREEN, "$\\lambda$ = 1 (Monte Carlo)")]:
    runs = logs[name][:3]
    M = np.array([log[:, 1] for _, log in runs])
    for m in M:
        axes[2].plot(steps, m, color=col, lw=0.7, alpha=0.3)
    axes[2].plot(steps, np.nanmean(M, 0), color=col, lw=2.2, label=lab)
axes[2].set_title("GAE $\\lambda$ (seeds 0-2)")
axes[2].set_xlabel("environment steps")
axes[2].set_ylabel("return (last 10 episodes)")
axes[2].set_ylim(0, 520)
axes[2].legend(loc="lower right", fontsize=9)
save(fig, "fig9-21-ppo-ablations.png")
for name in ("lambda = 0", MAIN, "lambda = 1"):
    M = np.array([log[:, 1] for _, log in logs[name][:3]])
    print(f"{name:28s} seeds 0-2: mean return over training {np.nanmean(M):.1f}, "
          f"first 500: {[first_solved(l) for _, l in logs[name][:3]]}")
