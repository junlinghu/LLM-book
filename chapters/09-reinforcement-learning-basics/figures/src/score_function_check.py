"""Section 9.6: check the log-derivative trick on a three-armed bandit.

The policy is a softmax over logits theta. Its expected reward is
J(theta) = sum_a pi(a) r(a), whose exact gradient we can write down. The
score-function (REINFORCE) estimator averages grad log pi(a) * (r - b) over
sampled actions; we compare its mean and variance with and without a baseline.
"""
import numpy as np

theta = np.array([0.5, 0.0, -0.5])          # logits
r_mean = np.array([1.0, 2.0, 3.0])          # expected reward of each arm
pi = np.exp(theta) / np.exp(theta).sum()

# Exact gradient: dJ/dtheta_k = pi_k (r_k - J)
J = pi @ r_mean
exact = pi * (r_mean - J)

rng = np.random.default_rng(0)
n = 200_000
a = rng.choice(3, size=n, p=pi)
r = rng.normal(r_mean[a], 1.0)              # noisy rewards
score = np.eye(3)[a] - pi                   # grad of log softmax: onehot(a) - pi
for name, b in [("no baseline", 0.0), ("baseline b = J", J)]:
    g = score * (r - b)[:, None]            # one gradient estimate per sample
    print(f"{name:15s} mean {np.round(g.mean(0), 3)}  total variance {g.var(0).sum():.3f}")
print(f"{'exact':15s} grad {np.round(exact, 3)}  (pi = {np.round(pi, 3)}, J = {J:.3f})")
