"""A tiny scalar reverse-mode autodiff engine written for Chapter 2.

Design: every Scalar remembers the Scalars it was computed from together with
the *local* partial derivative with respect to each of them, evaluated during
the forward pass. Backpropagation then only needs multiplication and addition:
    parent.grad += local_derivative * child.grad
"""
import math


class Scalar:
    __slots__ = ("value", "grad", "inputs", "op")

    def __init__(self, value, inputs=(), op=""):
        self.value = float(value)
        self.grad = 0.0
        self.inputs = inputs      # tuple of (parent Scalar, d self / d parent)
        self.op = op              # operation name, for printing and debugging

    def __repr__(self):
        return f"Scalar({self.value:.4f}, grad={self.grad:.4f}, op='{self.op}')"

    # ---- building blocks -------------------------------------------------
    @staticmethod
    def lift(x):
        return x if isinstance(x, Scalar) else Scalar(x)

    def __add__(self, other):
        other = Scalar.lift(other)
        return Scalar(self.value + other.value, ((self, 1.0), (other, 1.0)), "+")

    def __mul__(self, other):
        other = Scalar.lift(other)
        return Scalar(self.value * other.value,
                      ((self, other.value), (other, self.value)), "*")

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        return Scalar(self.value ** k, ((self, k * self.value ** (k - 1)),), f"^{k}")

    def exp(self):
        e = math.exp(self.value)
        return Scalar(e, ((self, e),), "exp")

    def log(self):
        return Scalar(math.log(self.value), ((self, 1.0 / self.value),), "log")

    def tanh(self):
        t = math.tanh(self.value)
        return Scalar(t, ((self, 1.0 - t * t),), "tanh")

    def relu(self):
        return Scalar(max(0.0, self.value), ((self, float(self.value > 0)),), "relu")

    # ---- derived operations (built from the ones above) ------------------
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Scalar.lift(other))

    def __truediv__(self, other):
        return self * Scalar.lift(other) ** -1

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return Scalar.lift(other) - self

    def __rtruediv__(self, other):
        return Scalar.lift(other) / self

    # ---- reverse pass -------------------------------------------------------
    def backward(self):
        """Fill in .grad for every Scalar that this one depends on."""
        order, visited = [], set()
        stack = [(self, False)]
        while stack:                       # iterative depth-first topological sort
            node, children_done = stack.pop()
            if children_done:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent, _ in node.inputs:
                if id(parent) not in visited:
                    stack.append((parent, False))
        self.grad = 1.0
        for node in reversed(order):       # outputs before inputs
            for parent, local in node.inputs:
                parent.grad += local * node.grad
