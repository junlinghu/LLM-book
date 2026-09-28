"""Check micrograd's backprop gradients against numerical estimates.

For each input we nudge it by a tiny h, see how much the output changes,
and compare that slope with the gradient backprop computed.

Run from this folder:  python gradient_check.py
"""
from micrograd.engine import Value


def f(a, b, c):
    # A small expression that uses +, *, **, /, and relu
    d = a * b + c ** 2
    e = (d / (a + 3)).relu()
    return e * b - a


a, b, c = Value(2.0), Value(3.0), Value(1.5)
out = f(a, b, c)
out.backward()

h = 1e-6
for name, v in [("a", a), ("b", b), ("c", c)]:
    args = {"a": a.data, "b": b.data, "c": c.data}
    args_up = dict(args, **{name: args[name] + h})
    numeric = (f(*map(Value, args_up.values())).data - f(*map(Value, args.values())).data) / h
    status = "OK" if abs(numeric - v.grad) < 1e-4 else "MISMATCH"
    print(f"d(out)/d{name}: backprop {v.grad:+.6f}   numeric {numeric:+.6f}   {status}")
