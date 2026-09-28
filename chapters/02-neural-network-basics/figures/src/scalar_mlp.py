"""A one-hidden-layer network built on the Scalar engine (Section 2.6, Lab 4)."""
import math
import random

from scalar_autograd import Scalar


def make_params(n_in, n_hidden, seed=0):
    rng = random.Random(seed)
    W1 = [[Scalar(rng.gauss(0, 1 / math.sqrt(n_in))) for _ in range(n_hidden)] for _ in range(n_in)]
    b1 = [Scalar(0.0) for _ in range(n_hidden)]
    W2 = [Scalar(rng.gauss(0, 1 / math.sqrt(n_hidden))) for _ in range(n_hidden)]
    b2 = Scalar(0.0)
    return W1, b1, W2, b2


def all_params(params):
    W1, b1, W2, b2 = params
    return [w for row in W1 for w in row] + b1 + W2 + [b2]


def logit(params, x):
    """Forward pass for one example x (a list of floats); returns a Scalar logit."""
    W1, b1, W2, b2 = params
    h = []
    for j in range(len(b1)):
        z = b1[j]
        for i in range(len(x)):
            z = z + x[i] * W1[i][j]
        h.append(z.tanh())
    out = b2
    for j in range(len(h)):
        out = out + h[j] * W2[j]
    return out


def bce_from_logit(z, y):
    """Binary cross-entropy -[y log σ(z) + (1-y) log(1-σ(z))], computed stably.

    Uses log(1 + e^z) - y z, rewritten as z + log(1 + e^-z) - y z when z > 0
    so that exp never receives a large positive argument.
    """
    if z.value > 0:
        return z + ((-z).exp() + 1).log() - y * z
    return (z.exp() + 1).log() - y * z


def train(X, Y, n_hidden=8, lr=0.5, steps=200, seed=0, log=None):
    params = make_params(len(X[0]), n_hidden, seed)
    history = []
    for step in range(steps):
        loss = sum((bce_from_logit(logit(params, x), y) for x, y in zip(X, Y)), Scalar(0.0))
        loss = loss * (1.0 / len(X))
        for p in all_params(params):
            p.grad = 0.0                 # 1. zero old gradients
        loss.backward()                  # 2. backpropagate
        for p in all_params(params):
            p.value -= lr * p.grad       # 3. gradient-descent step
        history.append(loss.value)
        if log and step % log == 0:
            print(f"step {step:4d}  loss {loss.value:.4f}")
    return params, history
