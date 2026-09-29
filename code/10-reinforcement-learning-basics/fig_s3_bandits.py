"""Figures and numbers for Section 10.3: bandits and exploration."""
import numpy as np
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, save, smooth
from bandit import run_testbed

# ------------------------------------------------ Figure 10.6: one testbed problem
rng = np.random.default_rng(0)
q_true = rng.normal(0, 1, 10)
samples = [rng.normal(q, 1, 2000) for q in q_true]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
parts = ax.violinplot(samples, positions=range(1, 11), showextrema=False, widths=0.8)
for b in parts["bodies"]:
    b.set_facecolor(BLUE)
    b.set_alpha(0.35)
ax.scatter(range(1, 11), q_true, color="k", zorder=3, s=25, label="true value $q_*(a)$")
best = q_true.argmax()
ax.scatter([best + 1], [q_true[best]], color=RED, zorder=4, s=60, label="best arm")
ax.axhline(0, color=GRAY, lw=1)
ax.set_xticks(range(1, 11))
ax.set_xlabel("arm $a$")
ax.set_ylabel("reward distribution")
ax.set_title("One 10-armed bandit problem: rewards are $q_*(a)$ plus unit-variance noise")
ax.set_ylim(-4.8, 6.0)
ax.legend(loc="upper right", ncol=2)
save(fig, "fig9-06-bandit-arms.png")
print("example problem true values:", np.round(q_true, 2), "best arm", best + 1)

# ------------------------------------------------ Figure 10.7: exploration strategies
agents = [
    ("greedy ($\\epsilon$ = 0)", dict(agent="eps", eps=0.0), GRAY),
    ("$\\epsilon$ = 0.01", dict(agent="eps", eps=0.01), RED),
    ("$\\epsilon$ = 0.1", dict(agent="eps", eps=0.1), BLUE),
    ("optimistic greedy ($Q_0$ = 5, $\\alpha$ = 0.1)", dict(agent="eps", eps=0.0, q0=5.0, alpha=0.1), PURPLE),
    ("UCB ($c$ = 2)", dict(agent="ucb", c=2.0), GREEN),
]
results = {}
for name, kw, col in agents:
    results[name] = run_testbed(**kw, n_problems=2000, steps=1000, seed=1)
    r, opt = results[name]
    print(f"{name:32s} mean reward steps 1-1000: {r.mean():.3f}  last 100: {r[-100:].mean():.3f}"
          f"  optimal last 100: {100 * opt[-100:].mean():.1f}%")
print("best achievable mean reward E[max q]:",
      np.round(np.random.default_rng(1).normal(0, 1, (2000, 10)).max(1).mean(), 3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
for name, kw, col in agents:
    r, opt = results[name]
    axes[0].plot(smooth(r, 10), color=col, lw=1.5, label=name)
    axes[1].plot(100 * smooth(opt, 10), color=col, lw=1.5, label=name)
axes[0].set_xlabel("step")
axes[0].set_ylabel("average reward")
axes[0].set_title("Average reward (2,000 problems)")
axes[1].set_xlabel("step")
axes[1].set_ylabel("% optimal action")
axes[1].set_title("How often the best arm is chosen")
axes[1].legend(loc="lower right", fontsize=9)
save(fig, "fig9-07-bandit-exploration.png")
