"""Figures and numbers for Section 10.2: MDPs, policies, and value functions."""
import numpy as np
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, GRAY, save
from gridworld import (GridWorld, policy_evaluation, value_iteration,
                       random_policy, START, GOAL)
from grid_plot import draw_grid

np.set_printoptions(precision=3, suppress=True)
env = GridWorld()
s0 = env.index[START]

# ---------------------------------------------------------- Figure 10.3: V^pi and V*
gamma = 0.9
V_rand = policy_evaluation(env, random_policy(env), gamma)
V_star, greedy = value_iteration(env, gamma)
print("V_random(start) =", round(V_rand[s0], 3), " V*(start) =", round(V_star[s0], 3))
print("V_random next to goal (0,3):", round(V_rand[env.index[(0, 3)]], 3),
      " (1,4):", round(V_rand[env.index[(1, 4)]], 3))
print("random-policy values:\n", env.to_grid(V_rand))

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
im = draw_grid(axes[0], env.to_grid(V_rand))
axes[0].set_title("$V^{\\pi}$ for the uniform random policy")
draw_grid(axes[1], env.to_grid(V_star), arrows=greedy, env=env)
axes[1].set_title("$V^{*}$ and a greedy optimal policy")
fig.colorbar(im, ax=axes, shrink=0.8, label="value")
fig.suptitle("Gridworld values with $\\gamma$ = 0.9", y=1.0)
save(fig, "fig9-04-gridworld-values.png")

# ---------------------------------------------------------- Figure 10.5: the discount factor
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
k = np.arange(0, 60)
for g, col in [(0.5, RED), (0.9, BLUE), (0.99, GREEN)]:
    axes[0].plot(k, g ** k, color=col, label=f"$\\gamma$ = {g}  (horizon ~ {1/(1-g):.0f})")
axes[0].set_xlabel("steps into the future, $k$")
axes[0].set_ylabel("weight $\\gamma^k$ on $r_{t+k+1}$")
axes[0].set_title("How much a future reward counts")
axes[0].legend()

gammas = np.linspace(0.3, 0.995, 60)
vr = [policy_evaluation(env, random_policy(env), g)[s0] for g in gammas]
vs = [value_iteration(env, g)[0][s0] for g in gammas]
axes[1].plot(gammas, vs, color=GREEN, label="optimal policy, $V^{*}$(S)")
axes[1].plot(gammas, vr, color=ORANGE, label="random policy, $V^{\\pi}$(S)")
axes[1].axhline(0, color=GRAY, lw=1)
axes[1].set_xlabel("discount factor $\\gamma$")
axes[1].set_ylabel("value of the start state S")
axes[1].set_title("Value of the start state vs. $\\gamma$")
axes[1].legend()
save(fig, "fig9-05-discount.png")
for g in (0.5, 0.9, 0.99):
    print(f"gamma={g}: V_random(S)={policy_evaluation(env, random_policy(env), g)[s0]:+.3f}"
          f"  V*(S)={value_iteration(env, g)[0][s0]:.3f}")


# ---------------------------------------------------------- sweeps needed (Code 10.2.2/10.2.3)
def count_sweeps(update, tol=1e-6):
    V, n = np.zeros(env.n_states), 0
    while True:
        V_new = update(V)
        n += 1
        if np.max(np.abs(V_new - V)) < tol:
            return n
        V = V_new


def pe_update(V, pi=random_policy(env)):
    Q = env.reward + gamma * V[env.next_state] * ~env.terminal[env.next_state]
    return np.where(env.terminal, 0.0, (pi * Q).sum(axis=1))


def vi_update(V):
    Q = env.reward + gamma * V[env.next_state] * ~env.terminal[env.next_state]
    return np.where(env.terminal, 0.0, Q.max(axis=1))


print("sweeps to tol 1e-6: policy evaluation", count_sweeps(pe_update),
      " value iteration", count_sweeps(vi_update))
