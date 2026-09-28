"""REINFORCE with and without a learned baseline on CartPole-v1 (Section 9.6)."""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


def mlp(inp, out, hidden=64):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, out))


class Policy(nn.Module):
    """A categorical policy: the network outputs one logit per action."""
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.logits = mlp(obs_dim, n_actions)

    def dist(self, s):
        return torch.distributions.Categorical(logits=self.logits(s))


def rollout(env, policy, seed=None):
    """Play one episode with the current policy; return states, actions, rewards."""
    s, _ = env.reset(seed=seed)
    states, actions, rewards = [], [], []
    done = False
    while not done:
        with torch.no_grad():
            a = int(policy.dist(torch.as_tensor(s, dtype=torch.float32)).sample())
        s2, r, terminated, truncated, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s, done = s2, terminated or truncated
    return (torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(actions), np.array(rewards, dtype=np.float32))


def rewards_to_go(rewards, gamma):
    """G_t = r_{t+1} + gamma r_{t+2} + ... for every step of one episode."""
    G, out = 0.0, np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        out[t] = G
    return torch.as_tensor(out)


def policy_loss(policy, states, actions, weights):
    """Surrogate whose gradient is the REINFORCE estimate: -sum_t log pi(a_t|s_t) w_t."""
    logp = policy.dist(states).log_prob(actions)
    return -(logp * weights).sum()


def train_reinforce(use_baseline, n_episodes=1000, gamma=0.99, lr=1e-3,
                    lr_value=1e-3, seed=0, variance_every=None, n_var=30):
    """Train REINFORCE; optionally measure gradient variance at checkpoints.

    Returns the episode returns and a list of (episode, var_no_baseline,
    var_with_baseline) tuples.
    """
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    policy = Policy(4, 2)
    value = mlp(4, 1)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_v = torch.optim.Adam(value.parameters(), lr=lr_value)
    returns, variances = [], []
    for ep in range(n_episodes):
        if variance_every and ep % variance_every == 0:
            variances.append((ep, *gradient_variance(env, policy, value, gamma, n_var,
                                                     seed=seed * 100_000 + ep)))
        states, actions, rewards = rollout(env, policy, seed=seed * 100_000 + ep)
        G = rewards_to_go(rewards, gamma)
        # The critic is trained in both settings so that the variance measurement
        # can use it; only use_baseline decides whether the policy update sees it.
        v = value(states).squeeze(1)
        v_loss = ((v - G) ** 2).mean()
        opt_v.zero_grad()
        v_loss.backward()
        opt_v.step()
        weights = G - v.detach() if use_baseline else G
        loss = policy_loss(policy, states, actions, weights)
        opt_pi.zero_grad()
        loss.backward()
        opt_pi.step()
        returns.append(rewards.sum())
    return np.array(returns), variances


def flat_grad(policy, states, actions, weights):
    policy.zero_grad()
    policy_loss(policy, states, actions, weights).backward()
    return torch.cat([p.grad.flatten() for p in policy.parameters()]).clone()


def gradient_variance(env, policy, value, gamma, n, seed):
    """Total variance (trace of the covariance) of single-episode gradient estimates,
    without and with the value baseline, for the current, frozen policy."""
    g_plain, g_base = [], []
    for i in range(n):
        states, actions, rewards = rollout(env, policy, seed=seed + 50_000 + i)
        G = rewards_to_go(rewards, gamma)
        with torch.no_grad():
            b = value(states).squeeze(1)
        g_plain.append(flat_grad(policy, states, actions, G))
        g_base.append(flat_grad(policy, states, actions, G - b))
    policy.zero_grad()
    tv = lambda g: torch.stack(g).var(dim=0).sum().item()
    return tv(g_plain), tv(g_base)
