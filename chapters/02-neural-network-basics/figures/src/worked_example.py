"""The 2-2-1 worked example used in Sections 2.3 and 2.6 (forward and backward numbers)."""
import numpy as np

x = np.array([1.0, 0.5])
W1 = np.array([[0.5, -0.3],
               [0.8, 0.2]])   # W1[i, j]: weight from input i to hidden unit j
b1 = np.array([0.0, 0.1])
W2 = np.array([1.0, -1.5])    # weight from hidden unit j to the output
b2 = 0.2
y = 1.0


def forward_backward(x=x, W1=W1, b1=b1, W2=W2, b2=b2, y=y):
    z1 = x @ W1 + b1
    h = np.tanh(z1)
    z2 = h @ W2 + b2
    p = 1 / (1 + np.exp(-z2))
    L = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    dz2 = p - y
    dW2 = dz2 * h
    db2 = dz2
    dh = dz2 * W2
    dz1 = dh * (1 - h ** 2)
    dW1 = np.outer(x, dz1)
    db1 = dz1
    return dict(z1=z1, h=h, z2=z2, p=p, L=L, dz2=dz2, dW2=dW2, db2=db2, dh=dh, dz1=dz1, dW1=dW1, db1=db1)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    r = forward_backward()
    for k, v in r.items():
        print(k, v)
    # finite-difference check on every parameter
    eps = 1e-6
    def loss_with(**kw):
        return forward_backward(**kw)["L"]
    for name, arr in [("W1", W1), ("b1", b1), ("W2", W2)]:
        num = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            a1 = arr.copy(); a1[idx] += eps
            a2 = arr.copy(); a2[idx] -= eps
            num[idx] = (loss_with(**{name: a1}) - loss_with(**{name: a2})) / (2 * eps)
        print("numeric d" + name, num)
    print("numeric db2", (loss_with(b2=b2 + eps) - loss_with(b2=b2 - eps)) / (2 * eps))
