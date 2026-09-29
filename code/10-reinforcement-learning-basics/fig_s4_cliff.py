"""Figure and numbers for Section 10.4: SARSA vs. Q-learning on CliffWalking-v1."""
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, GRAY, save, smooth


def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))     # random tie-break


def train(method, n_episodes=500, alpha=0.5, gamma=1.0, eps=0.1, seed=0):
    """Tabular SARSA or Q-learning; returns Q and the reward of every episode."""
    env = gym.make("CliffWalking-v1")
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    episode_rewards = []
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed * 10_000 + ep)
        a = eps_greedy(Q, s, eps, rng)
        total, done = 0.0, False
        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            a2 = eps_greedy(Q, s2, eps, rng)
            if method == "sarsa":        # on-policy: the action we will really take
                target = r + gamma * Q[s2, a2] * (not terminated)
            else:                        # Q-learning: the greedy action
                target = r + gamma * Q[s2].max() * (not terminated)
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
            total += r
        episode_rewards.append(total)
    return Q, np.array(episode_rewards)


def greedy_path(Q, max_steps=100):
    """Follow the greedy policy from the start; return visited states and return."""
    env = gym.make("CliffWalking-v1")
    s, _ = env.reset(seed=0)
    path, total = [s], 0.0
    for _ in range(max_steps):
        s, r, terminated, truncated, _ = env.step(int(Q[s].argmax()))
        path.append(s)
        total += r
        if terminated:
            break
    return path, total


if __name__ == "__main__":
    n_runs, n_ep = 50, 500
    curves, finals = {}, {}
    for method in ("sarsa", "qlearning"):
        runs = [train(method, n_ep, seed=i) for i in range(n_runs)]
        curves[method] = np.mean([r for _, r in runs], axis=0)
        finals[method] = runs[0][0]
        greedy_returns = [greedy_path(Q)[1] for Q, _ in runs]
        vals, counts = np.unique(greedy_returns, return_counts=True)
        print(f"{method:9s} online reward, last 100 episodes: {curves[method][-100:].mean():7.1f}"
              f" | greedy-policy return over {n_runs} runs: "
              + ", ".join(f"{v:.0f} (x{c})" for v, c in zip(vals, counts)))

    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
    ax_grid = fig.add_subplot(gs[0, :])
    ax_curve = fig.add_subplot(gs[1, :])
    # Grid with the learned greedy paths (seed-0 runs)
    ax_grid.set_xlim(-0.5, 11.5)
    ax_grid.set_ylim(3.5, -0.5)
    for c in range(1, 11):
        ax_grid.add_patch(plt.Rectangle((c - 0.5, 2.5), 1, 1, color="#555555"))
    ax_grid.text(5.5, 3, "the cliff  (reward −100, back to start)", color="white",
                 ha="center", va="center", fontsize=11)
    ax_grid.text(0, 3, "S", ha="center", va="center", fontsize=14, fontweight="bold")
    ax_grid.text(11, 3, "G", ha="center", va="center", fontsize=14, fontweight="bold")
    for method, col, off, label in [("sarsa", BLUE, -0.12, "SARSA (safe path)"),
                                    ("qlearning", RED, 0.12, "Q-learning (optimal path)")]:
        path, ret = greedy_path(finals[method])
        rc = np.array([divmod(s, 12) for s in path], float)
        ax_grid.plot(rc[:, 1] + off, rc[:, 0] + off, "-o", color=col, ms=4, lw=2.2,
                     label=f"{label}: return {ret:.0f}")
    ax_grid.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax_grid.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax_grid.grid(which="minor", color="k", lw=0.6)
    ax_grid.grid(which="major", visible=False)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_grid.tick_params(which="minor", length=0)
    for sp in ax_grid.spines.values():
        sp.set_visible(True)
    ax_grid.set_aspect("equal")
    ax_grid.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2)
    for method, col, label in [("sarsa", BLUE, "SARSA"), ("qlearning", RED, "Q-learning")]:
        ax_curve.plot(smooth(curves[method], 10), color=col, label=label)
    ax_curve.set_ylim(-110, 0)
    ax_curve.set_xlabel("episode")
    ax_curve.set_ylabel("reward per episode")
    ax_curve.set_title(f"Reward during training with $\\epsilon$ = 0.1 (mean of {n_runs} runs)")
    ax_curve.legend(loc="lower right")
    save(fig, "fig9-10-cliff-walking.png")
