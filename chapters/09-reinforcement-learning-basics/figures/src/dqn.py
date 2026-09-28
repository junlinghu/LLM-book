"""A small DQN for CartPole-v1, written from scratch in PyTorch (Section 9.5).

run_dqn(use_replay, use_target) trains one agent and returns the return of
every finished training episode together with the step at which it ended,
and the average predicted max-Q value on training batches every 1,000 steps.
"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


class QNetwork(nn.Module):
    """Maps a state to one Q-value per action."""
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, s):
        return self.net(s)


class ReplayBuffer:
    """A fixed-size circular buffer of transitions (s, a, r, s', terminated)."""
    def __init__(self, capacity, obs_dim, rng):
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.term = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.pos, self.rng = capacity, 0, 0, rng

    def add(self, s, a, r, s2, term):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.term[i] = s, a, r, s2, term
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(self.size, size=batch_size)
        return (torch.as_tensor(self.s[idx]), torch.as_tensor(self.a[idx]),
                torch.as_tensor(self.r[idx]), torch.as_tensor(self.s2[idx]),
                torch.as_tensor(self.term[idx]))

    def latest(self, batch_size):
        """The most recent transitions, in order (used when replay is off)."""
        idx = (self.pos - 1 - np.arange(batch_size)) % self.capacity
        return (torch.as_tensor(self.s[idx]), torch.as_tensor(self.a[idx]),
                torch.as_tensor(self.r[idx]), torch.as_tensor(self.s2[idx]),
                torch.as_tensor(self.term[idx]))


def run_dqn(use_replay=True, use_target=True, total_steps=40_000, seed=0,
            gamma=0.99, lr=5e-4, batch_size=64, buffer_size=50_000,
            target_every=500, learning_starts=1_000, eps_decay_steps=10_000):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    q = QNetwork(obs_dim, n_actions)
    q_target = QNetwork(obs_dim, n_actions)
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_size, obs_dim, rng)

    s, _ = env.reset(seed=seed)
    ep_return, returns, ends, qlog = 0.0, [], [], []
    for step in range(total_steps):
        # epsilon-greedy exploration, epsilon decaying linearly from 1 to 0.05
        eps = max(0.05, 1.0 - 0.95 * step / eps_decay_steps)
        if rng.random() < eps:
            a = int(rng.integers(n_actions))
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(s, dtype=torch.float32)).argmax())
        s2, r, terminated, truncated, _ = env.step(a)
        buf.add(s, a, r, s2, float(terminated))
        s, ep_return = s2, ep_return + r
        if terminated or truncated:
            returns.append(ep_return)
            ends.append(step)
            s, _ = env.reset()
            ep_return = 0.0

        if step >= learning_starts:
            bs, ba, br, bs2, bterm = (buf.sample(batch_size) if use_replay
                                      else buf.latest(batch_size))
            with torch.no_grad():                          # the TD target
                net = q_target if use_target else q
                target = br + gamma * (1 - bterm) * net(bs2).max(dim=1).values
            q_all = q(bs)
            pred = q_all.gather(1, ba[:, None]).squeeze(1)
            if step % 1000 == 0:
                qlog.append((step, q_all.max(dim=1).values.mean().item()))
            loss = nn.functional.smooth_l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            if use_target and step % target_every == 0:
                q_target.load_state_dict(q.state_dict())    # periodic copy
    return np.array(ends), np.array(returns), np.array(qlog)
