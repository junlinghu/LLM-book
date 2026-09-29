"""Shared MLP helpers for Chapter 2 figures (original NumPy code)."""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def relu(z):
    return np.maximum(0, z)


def mlp_forward(X, W1, b1, W2, b2, act="tanh"):
    z1 = X @ W1 + b1
    h = np.tanh(z1) if act == "tanh" else relu(z1)
    z2 = h @ W2 + b2
    p = sigmoid(z2.ravel())
    return p, h, z1, z2


def mlp_train(X, y, n_hidden=8, act="tanh", lr=0.5, epochs=800, seed=0, l2=1e-4):
    """Binary-classification MLP with one hidden layer; returns params + loss history."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.normal(0, 1 / np.sqrt(d), (d, n_hidden))
    b1 = np.zeros(n_hidden)
    W2 = rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, 1))
    b2 = np.zeros(1)
    losses = []
    for _ in range(epochs):
        p, h, z1, z2 = mlp_forward(X, W1, b1, W2, b2, act)
        p = np.clip(p, 1e-8, 1 - 1e-8)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * l2 * (
            np.sum(W1**2) + np.sum(W2**2)
        )
        losses.append(loss)
        dz2 = (p - y)[:, None] / n  # (n, 1)
        dW2 = h.T @ dz2 + l2 * W2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ W2.T
        if act == "tanh":
            dz1 = dh * (1 - h**2)
        else:
            dz1 = dh * (z1 > 0)
        dW1 = X.T @ dz1 + l2 * W1
        db1 = dz1.sum(axis=0)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, act=act, losses=losses)


def predict_proba(params, X):
    p, *_ = mlp_forward(X, params["W1"], params["b1"], params["W2"], params["b2"], params["act"])
    return p
