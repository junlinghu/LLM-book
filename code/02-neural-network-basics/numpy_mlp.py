"""Vectorized one-hidden-layer MLP with a hand-written backward pass (Section 2.7)."""
import numpy as np


def init_params(n_in, n_hidden, n_out, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, np.sqrt(1 / n_hidden), (n_hidden, n_out)),
        "b2": np.zeros(n_out),
    }


def forward(P, X):
    Z1 = X @ P["W1"] + P["b1"]          # (B, H)   bias broadcasts over rows
    H = np.maximum(0, Z1)               # (B, H)
    Z2 = H @ P["W2"] + P["b2"]          # (B, C)   logits
    return Z2, (X, Z1, H)


def softmax_cross_entropy(Z, y):
    """Mean cross-entropy of integer labels y under softmax(Z); also returns dL/dZ."""
    Zs = Z - Z.max(axis=1, keepdims=True)        # stability: subtract the row max
    logsumexp = np.log(np.exp(Zs).sum(axis=1, keepdims=True))
    logp = Zs - logsumexp                        # log-softmax, (B, C)
    B = len(y)
    loss = -logp[np.arange(B), y].mean()
    dZ = np.exp(logp)                            # probabilities
    dZ[np.arange(B), y] -= 1.0                   # p - onehot(y)
    return loss, dZ / B


def backward(P, cache, dZ2):
    X, Z1, H = cache
    grads = {}
    grads["W2"] = H.T @ dZ2                      # (H, B) @ (B, C) -> (H, C)
    grads["b2"] = dZ2.sum(axis=0)                # (C,)
    dH = dZ2 @ P["W2"].T                         # (B, C) @ (C, H) -> (B, H)
    dZ1 = dH * (Z1 > 0)                          # ReLU passes gradient where Z1 > 0
    grads["W1"] = X.T @ dZ1                      # (D, B) @ (B, H) -> (D, H)
    grads["b1"] = dZ1.sum(axis=0)                # (H,)
    for k in P:
        assert grads[k].shape == P[k].shape, k   # every gradient matches its parameter
    return grads


def train_sgd(P, X, y, lr=0.1, epochs=100, batch_size=32, seed=0, X_val=None, y_val=None):
    rng = np.random.default_rng(seed)
    n = len(X)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            Z, cache = forward(P, X[idx])
            loss, dZ = softmax_cross_entropy(Z, y[idx])
            g = backward(P, cache, dZ)
            for k in P:
                P[k] -= lr * g[k]
        full_loss, _ = softmax_cross_entropy(forward(P, X)[0], y)
        history.append(full_loss)
    return P, history
