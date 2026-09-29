"""Figure and numbers for Section 10.4: Monte Carlo vs. TD(0) prediction."""
import numpy as np
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, save
from gridworld import GridWorld, policy_evaluation, random_policy

env = GridWorld()
GAMMA = 0.9
V_true = policy_evaluation(env, random_policy(env), GAMMA)
nonterminal = np.flatnonzero(~env.terminal)


def generate_episode(env, rng):
    """One episode of the uniform random policy from a random non-terminal start.

    Returns the lists of visited states and the rewards that followed them.
    """
    s = rng.choice(nonterminal)
    states, rewards = [], []
    done = False
    while not done:
        a = rng.integers(env.n_actions)
        s2, r, done = env.step(s, a)
        states.append(s)
        rewards.append(r)
        s = s2
    return states, rewards


def mc_prediction(env, n_episodes, rng, alpha=None, gamma=GAMMA):
    """Every-visit Monte Carlo: move V(s) toward each observed return G_t.

    alpha=None uses the running mean of all returns seen from s.
    """
    V, N = np.zeros(env.n_states), np.zeros(env.n_states)
    history = []
    for _ in range(n_episodes):
        states, rewards = generate_episode(env, rng)
        G = 0.0
        for s, r in zip(reversed(states), reversed(rewards)):   # walk backward
            G = r + gamma * G                                   # return from s
            N[s] += 1
            step = 1.0 / N[s] if alpha is None else alpha
            V[s] += step * (G - V[s])
        history.append(V.copy())
    return np.array(history)


def td0_prediction(env, n_episodes, rng, alpha, gamma=GAMMA):
    """TD(0): after every step, move V(s) toward r + gamma * V(s')."""
    V = np.zeros(env.n_states)
    history = []
    for _ in range(n_episodes):
        s = rng.choice(nonterminal)
        done = False
        while not done:
            a = rng.integers(env.n_actions)
            s2, r, done = env.step(s, a)
            target = r + (0.0 if done else gamma * V[s2])
            V[s] += alpha * (target - V[s])
            s = s2
        history.append(V.copy())
    return np.array(history)


def rms(history):
    return np.sqrt(((history[:, nonterminal] - V_true[nonterminal]) ** 2).mean(axis=1))


if __name__ == "__main__":
    # A very long Monte Carlo run agrees with the Bellman solution.
    long_run = mc_prediction(env, 100_000, np.random.default_rng(123))[-1]
    print("max |long MC - Bellman| =", np.abs(long_run - V_true)[nonterminal].max().round(4))
    lengths = [len(generate_episode(env, np.random.default_rng(i))[0]) for i in range(2000)]
    print("mean episode length:", np.mean(lengths).round(1))

    n_runs, n_ep = 100, 300
    methods = [
        ("MC, running mean", lambda rng: mc_prediction(env, n_ep, rng), GRAY, "-"),
        ("MC, $\\alpha$ = 0.02", lambda rng: mc_prediction(env, n_ep, rng, alpha=0.02), RED, "-"),
        ("MC, $\\alpha$ = 0.05", lambda rng: mc_prediction(env, n_ep, rng, alpha=0.05), ORANGE, "-"),
        ("TD(0), $\\alpha$ = 0.02", lambda rng: td0_prediction(env, n_ep, rng, 0.02), BLUE, "--"),
        ("TD(0), $\\alpha$ = 0.05", lambda rng: td0_prediction(env, n_ep, rng, 0.05), GREEN, "--"),
        ("TD(0), $\\alpha$ = 0.15", lambda rng: td0_prediction(env, n_ep, rng, 0.15), PURPLE, "--"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name, fn, col, ls in methods:
        errs = np.array([rms(fn(np.random.default_rng(1000 + i))) for i in range(n_runs)])
        m = errs.mean(axis=0)
        ax.plot(np.arange(1, n_ep + 1), m, color=col, ls=ls, label=name)
        print(f"{name:24s} RMS error after 10/50/100/300 episodes: "
              f"{m[9]:.3f} {m[49]:.3f} {m[99]:.3f} {m[-1]:.3f}")
    ax.set_xlabel("episodes")
    ax.set_ylabel("RMS error of $V$ (averaged over 100 runs)")
    ax.set_title("Predicting $V^{\\pi}$ of the random policy: Monte Carlo vs. TD(0)")
    ax.set_ylim(0, 0.25)
    ax.legend()
    save(fig, "fig9-09-mc-vs-td.png")
