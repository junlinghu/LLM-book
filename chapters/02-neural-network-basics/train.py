"""Train a tiny neural network with micrograd on four toy examples.

Run from this folder:  python train.py
"""
import random

from micrograd.nn import MLP

random.seed(1337)

# 3 inputs, two hidden layers of 4 ReLU neurons, 1 linear output
model = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # targets

learning_rate = 0.05

for step in range(100):
    # 1. Forward pass: predictions and mean squared error loss
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)) * (1.0 / len(xs))

    # 2. Backward pass: reset gradients, then backpropagate from the loss
    model.zero_grad()
    loss.backward()

    # 3. Update: move every parameter a little against its gradient
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if step % 10 == 0 or step == 99:
        print(f"step {step:3d}  loss {loss.data:.6f}")

print("\nfinal predictions:", [round(model(x).data, 3) for x in xs])
print("targets:          ", ys)
