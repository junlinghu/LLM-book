"""A vectorized 10-armed bandit testbed (Section 9.3).

Many independent bandit problems are simulated in parallel: row i of every
array belongs to problem i, so one step of the loop advances all of them.
"""
import numpy as np


def run_testbed(agent, n_problems=2000, n_arms=10, steps=1000, seed=0,
                eps=0.0, c=None, q0=0.0, alpha=None):
    """Run one agent on n_problems random bandits.

    agent is "eps" (epsilon-greedy, starting from estimates q0) or "ucb"
    (upper confidence bound with exploration constant c). Estimates are
    sample averages, or use a constant step size if alpha is given.
    Returns the average reward and the fraction of optimal actions per step.
    """
    rng = np.random.default_rng(seed)
    q_true = rng.normal(0.0, 1.0, (n_problems, n_arms))     # true arm values
    best = q_true.argmax(axis=1)
    Q = np.full((n_problems, n_arms), q0, dtype=float)      # estimates
    N = np.zeros((n_problems, n_arms))                      # pull counts
    rows = np.arange(n_problems)
    avg_reward, frac_optimal = np.zeros(steps), np.zeros(steps)
    for t in range(steps):
        if agent == "eps":
            greedy = argmax_random_ties(Q, rng)
            explore = rng.random(n_problems) < eps
            a = np.where(explore, rng.integers(n_arms, size=n_problems), greedy)
        else:  # UCB: try every arm once, then add an exploration bonus
            bonus = c * np.sqrt(np.log(t + 1) / np.maximum(N, 1e-12))
            a = argmax_random_ties(np.where(N == 0, np.inf, Q + bonus), rng)
        r = rng.normal(q_true[rows, a], 1.0)                 # noisy reward
        N[rows, a] += 1
        step = 1.0 / N[rows, a] if alpha is None else alpha
        Q[rows, a] += step * (r - Q[rows, a])                 # incremental update
        avg_reward[t] = r.mean()
        frac_optimal[t] = (a == best).mean()
    return avg_reward, frac_optimal


def argmax_random_ties(Q, rng):
    """Row-wise argmax that breaks ties uniformly at random."""
    noise = rng.random(Q.shape) * 1e-9
    return np.argmax(np.where(Q == Q.max(axis=1, keepdims=True), 1.0 + noise, 0.0), axis=1)
