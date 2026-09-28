"""Figure and numbers for Section 9.5: DQN ablations on CartPole-v1.

Loads dqn_results.npy written by run_dqn_experiments.py (run that first;
it takes several minutes on a CPU).
"""
import numpy as np
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, GRAY, save

res = np.load("dqn_results.npy", allow_pickle=True)
grid = np.arange(0, 40_001, 500)
colors = {"full DQN": BLUE, "no target network": RED, "no replay": ORANGE}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
for name, col in colors.items():
    runs = [r for r in res if r[0] == name]
    curves = []
    for _, seed, ends, rets, qlog in runs:
        # return of the most recent 10 finished episodes, as a function of steps
        avg = np.array([rets[max(0, i - 9):i + 1].mean() for i in range(len(rets))])
        curves.append(np.interp(grid, ends, avg, left=np.nan))
        axes[1].plot(qlog[:, 0], qlog[:, 1], color=col, lw=0.8, alpha=0.6)
    curves = np.array(curves)
    for c in curves:
        axes[0].plot(grid, c, color=col, lw=0.7, alpha=0.35)
    axes[0].plot(grid, np.nanmean(curves, 0), color=col, lw=2.4, label=name)
    last = np.array([r[3][-20:].mean() for r in runs])
    best = np.array([max(r[3][max(0, i - 9):i + 1].mean() for i in range(len(r[3]))) for r in runs])
    qfinal = np.array([r[4][-1, 1] for r in runs])
    print(f"{name:18s} last-20-episode return per seed {np.round(last, 1).tolist()}, "
          f"best 10-episode average {np.round(best).tolist()}, final mean max-Q "
          f"{np.round(qfinal, 1).tolist()}")
    axes[1].plot([], [], color=col, label=name)
axes[0].set_xlabel("environment steps")
axes[0].set_ylabel("return (last 10 episodes)")
axes[0].set_title("DQN on CartPole-v1 (3 seeds each)")
axes[0].set_ylim(0, 520)
axes[0].legend(loc="upper left")
axes[1].axhline(100, color=GRAY, ls="--", lw=1.2)
axes[1].text(1000, 130, "upper bound on true values, $1/(1-\\gamma)$ = 100", fontsize=9,
             color=GRAY)
axes[1].set_yscale("log")
axes[1].set_xlabel("environment steps")
axes[1].set_ylabel("average predicted max-Q on a batch")
axes[1].set_title("Q-value estimates")
axes[1].legend(loc="upper left")
save(fig, "fig9-12-dqn-ablation.png")
