# Chapter 2: The Basics of Neural Networks

This chapter builds a neural network from scratch and trains it with backpropagation, using **micrograd**, the tiny automatic-differentiation engine Andrej Karpathy wrote to teach these ideas. It follows his lecture [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).

The code in `micrograd/` is copied unchanged from [karpathy/micrograd](https://github.com/karpathy/micrograd) and is used under its MIT license (see `micrograd/LICENSE`).

## Files

| File | What it is |
| --- | --- |
| `micrograd/engine.py` | The `Value` class: a scalar that records how it was computed and knows how to backpropagate gradients. |
| `micrograd/nn.py` | `Neuron`, `Layer`, and `MLP` built on top of `Value`. |
| `gradient_check.py` | Compares backprop gradients with numerical estimates, to show backprop is correct. |
| `train.py` | Trains a small network on four toy examples and prints the loss falling. |

## Run it

No dependencies beyond Python 3.

```bash
cd chapters/02-neural-network-basics
python gradient_check.py
python train.py
```

## How it works

### 1. Every number remembers where it came from

Each `Value` wraps one number. When you combine Values with `+`, `*`, `**`, or `relu()`, the result is a new Value that stores its inputs (`_prev`) and a small function (`_backward`). Running the network forward therefore also builds a graph of every arithmetic step between the inputs and the loss.

### 2. Each operation knows its local derivative

`_backward` applies the chain rule for one operation. Here `out.grad` means how much the final loss changes when `out` changes a little.

- Addition, `out = a + b`: both inputs receive `out.grad` unchanged.
- Multiplication, `out = a * b`: `a` receives `b * out.grad`, and `b` receives `a * out.grad`.
- Power, `out = a ** n`: `a` receives `n * a**(n-1) * out.grad`.
- ReLU: the gradient passes through if the output was positive, and is blocked otherwise.

### 3. `backward()` runs those rules in the right order

It sets the loss's gradient to 1, sorts the graph so every node comes after its inputs (a topological sort), and then calls each node's `_backward` in reverse order. When a node runs, its own gradient is already complete, so it can pass the correct amount on to its inputs.

Gradients are added with `+=` because a value used in several places must collect gradient from every path. That is also why `zero_grad()` resets them before each backward pass.

### 4. Training is forward, backward, update

`train.py` repeats three steps:

1. **Forward:** compute predictions and the mean squared error loss.
2. **Backward:** `model.zero_grad()`, then `loss.backward()` fills in `p.grad` for every weight and bias.
3. **Update:** `p.data -= learning_rate * p.grad` moves each parameter slightly in the direction that lowers the loss.

This is gradient descent. PyTorch does the same thing, operating on tensors instead of one number at a time.

## Example output

```
step   0  loss 4.608049
step  10  loss 0.235384
step  50  loss 0.027020
step  99  loss 0.008141

final predictions: [1.012, -0.844, -1.081, 0.971]
targets:           [1.0, -1.0, -1.0, 1.0]
```

## Exercises

1. Add a `tanh()` method to `Value` (its derivative is `1 - tanh(x)**2`) and use it instead of ReLU.
2. Change the learning rate to 0.5 and to 0.001. What happens to the loss?
3. Remove `model.zero_grad()` from the training loop and explain the result.
