"""Proximal Policy Optimization from scratch for CartPole-v1 (Section 9.7)."""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


def layer(inp, out, std=np.sqrt(2)):
    """A linear layer with orthogonal weights and zero bias."""
    lin = nn.Linear(inp, out)
    nn.init.orthogonal_(lin.weight, std)
    nn.init.zeros_(lin.bias)
    return lin


class ActorCritic(nn.Module):
    """Separate policy (actor) and value (critic) networks."""
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(),
                                   layer(hidden, hidden), nn.Tanh(),
                                   layer(hidden, n_actions, std=0.01))
        self.critic = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(),
                                    layer(hidden, hidden), nn.Tanh(),
                                    layer(hidden, 1, std=1.0))

    def dist(self, s):
        return torch.distributions.Categorical(logits=self.actor(s))

    def value(self, s):
        return self.critic(s).squeeze(-1)


def collect_rollout(envs, obs, model, n_steps, gamma, stats):
    """Run every environment for n_steps with the current policy.

    Returns a dict of tensors with a leading (n_steps, n_envs) shape and the
    observations to continue from. Finished-episode returns go into stats.
    """
    n_envs = len(envs)
    buf = {k: [] for k in ("obs", "act", "logp", "rew", "term", "val")}
    for _ in range(n_steps):
        s = torch.as_tensor(np.array(obs), dtype=torch.float32)
        with torch.no_grad():
            d = model.dist(s)
            a = d.sample()
            logp, v = d.log_prob(a), model.value(s)
        rew, term = np.zeros(n_envs, np.float32), np.zeros(n_envs, np.float32)
        for i, env in enumerate(envs):
            s2, r, terminated, truncated, _ = env.step(int(a[i]))
            stats["ep_ret"][i] += r
            if truncated and not terminated:
                # A time limit is not a real ending: bootstrap from V(s_final).
                with torch.no_grad():
                    r += gamma * model.value(torch.as_tensor(s2, dtype=torch.float32)).item()
            if terminated or truncated:
                stats["finished"].append(stats["ep_ret"][i])
                stats["ep_ret"][i] = 0.0
                s2, _ = env.reset()
            rew[i], term[i] = r, float(terminated or truncated)
            obs[i] = s2
        for k, x in zip(buf, (s, a, logp, torch.as_tensor(rew), torch.as_tensor(term), v)):
            buf[k].append(x)
    return {k: torch.stack(x) for k, x in buf.items()}, obs


def compute_gae(rew, val, term, last_val, gamma, lam):
    """Generalized advantage estimation, computed backward in time.

    delta_t = r_t + gamma V(s_{t+1}) - V(s_t);  A_t = delta_t + gamma lam A_{t+1},
    with the recursion cut wherever an episode ended.
    """
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    next_adv, next_val = torch.zeros_like(last_val), last_val
    for t in reversed(range(T)):
        not_done = 1.0 - term[t]
        delta = rew[t] + gamma * next_val * not_done - val[t]
        next_adv = delta + gamma * lam * not_done * next_adv
        adv[t] = next_adv
        next_val = val[t]
    return adv, adv + val            # advantages and value targets (returns)


def ppo_update(model, opt, batch, clip_eps, epochs, n_minibatches,
               vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
    """Several epochs of minibatch updates on the clipped PPO loss."""
    N = batch["obs"].shape[0]
    mb_size = N // n_minibatches
    kls, clipfracs = [], []
    for _ in range(epochs):
        perm = torch.randperm(N)
        for start in range(0, N, mb_size):
            idx = perm[start:start + mb_size]
            d = model.dist(batch["obs"][idx])
            logp = d.log_prob(batch["act"][idx])
            log_ratio = logp - batch["logp"][idx]
            ratio = log_ratio.exp()
            adv = batch["adv"][idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)       # normalize per minibatch
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = ((model.value(batch["obs"][idx]) - batch["ret"][idx]) ** 2).mean()
            entropy = d.entropy().mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            with torch.no_grad():                              # diagnostics
                kls.append(((ratio - 1) - log_ratio).mean().item())
                clipfracs.append(((ratio - 1).abs() > clip_eps).float().mean().item())
    return np.mean(kls), np.mean(clipfracs)


def train_ppo(total_steps=100_000, n_envs=4, n_steps=128, gamma=0.99, lam=0.95,
              clip_eps=0.2, epochs=4, n_minibatches=4, lr=2.5e-4, seed=0,
              anneal_lr=True):
    torch.manual_seed(seed)
    envs = [gym.make("CartPole-v1") for _ in range(n_envs)]
    obs = [env.reset(seed=seed * 100 + i)[0] for i, env in enumerate(envs)]
    model = ActorCritic(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    stats = {"ep_ret": np.zeros(n_envs), "finished": []}
    n_iters = total_steps // (n_envs * n_steps)
    log = []
    for it in range(n_iters):
        if anneal_lr:                                   # linear decay to zero
            opt.param_groups[0]["lr"] = lr * (1 - it / n_iters)
        roll, obs = collect_rollout(envs, obs, model, n_steps, gamma, stats)
        with torch.no_grad():
            last_val = model.value(torch.as_tensor(np.array(obs), dtype=torch.float32))
        adv, ret = compute_gae(roll["rew"], roll["val"], roll["term"], last_val, gamma, lam)
        batch = {"obs": roll["obs"].reshape(-1, 4), "act": roll["act"].reshape(-1),
                 "logp": roll["logp"].reshape(-1), "adv": adv.reshape(-1),
                 "ret": ret.reshape(-1)}
        kl, clipfrac = ppo_update(model, opt, batch, clip_eps, epochs, n_minibatches)
        recent = stats["finished"][-10:]
        log.append(((it + 1) * n_envs * n_steps, np.mean(recent) if recent else np.nan,
                    kl, clipfrac))
    return np.array(log)
