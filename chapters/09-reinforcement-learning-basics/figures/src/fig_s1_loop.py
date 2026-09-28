"""Figures and numbers for Section 9.1: the agent-environment loop."""
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from style import save
from gridworld import GridWorld, value_iteration, START
from grid_plot import draw_grid


# ------------------------------------------------ Code 9.1.1: the loop in Gymnasium
def run_episode(env, policy, seed):
    obs, info = env.reset(seed=seed)
    total, steps, done = 0.0, 0, False
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        done = terminated or truncated
    return total, steps


env = gym.make("CartPole-v1")
rng = np.random.default_rng(0)
random_policy = lambda obs: int(rng.integers(2))          # ignore the observation
lean_policy = lambda obs: int(obs[2] + obs[3] > 0)        # push toward the lean
for name, pol in [("random", random_policy), ("lean", lean_policy)]:
    returns = [run_episode(env, pol, seed)[0] for seed in range(100)]
    print(f"{name:7s} policy: mean return {np.mean(returns):6.1f} "
          f"(min {min(returns):.0f}, max {max(returns):.0f}) over 100 episodes")


# ------------------------------------------------ Code 9.1.2: visitation depends on the policy
def visit_counts(grid, policy_probs, episodes, rng, max_steps=200):
    counts = np.zeros(grid.n_states)
    for _ in range(episodes):
        s = grid.index[START]
        for _ in range(max_steps):
            counts[s] += 1
            a = rng.choice(grid.n_actions, p=policy_probs[s])
            s, r, done = grid.step(s, a)
            if done:
                counts[s] += 1
                break
    return counts / episodes


grid = GridWorld()
_, greedy = value_iteration(grid, gamma=0.9)
uniform = np.full((grid.n_states, grid.n_actions), 0.25)
eps = 0.2                                   # mostly greedy, sometimes random
mostly_greedy = np.full((grid.n_states, grid.n_actions), eps / 4)
mostly_greedy[np.arange(grid.n_states), greedy] += 1 - eps

rng = np.random.default_rng(1)
c_uniform = visit_counts(grid, uniform, 2000, rng)
c_greedy = visit_counts(grid, mostly_greedy, 2000, rng)
for name, c in [("uniform", c_uniform), ("mostly greedy", c_greedy)]:
    goal = c[grid.index[(0, 4)]]
    print(f"{name:13s}: {c.sum():5.1f} visits per episode, "
          f"reaches goal in {100 * goal:.0f}% of episodes")

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
vmax = max(c_uniform.max(), c_greedy.max())
for ax, c, title in [(axes[0], c_uniform, "uniform random policy"),
                     (axes[1], c_greedy, "mostly greedy policy ($\\epsilon$ = 0.2)")]:
    im = draw_grid(ax, grid.to_grid(c), cmap="Oranges", vmin=0, vmax=vmax,
                   fmt="{:.1f}", fontsize=9)
    ax.set_title(title)
fig.colorbar(im, ax=axes, shrink=0.8, label="average visits per episode")
save(fig, "fig9-02-visitation.png")
