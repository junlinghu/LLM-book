# 3.1 Why Go Deep

Chapter 2 built a neural network with one hidden layer: inputs go through a weight matrix, an activation, and another weight matrix to produce outputs. That architecture is already a *universal approximator*: with enough hidden units, a one-hidden-layer network can approximate any continuous function on a compact set to any accuracy. So why do modern models, including every large language model, use dozens or hundreds of layers instead of one very wide one?

The short answer is that depth and width are not interchangeable in practice. Depth can represent some functions with far fewer units, it builds hierarchical features layer by layer, and the training techniques in this chapter make stacking many layers workable. This section explains what "deep" means, why depth helps, how to count the cost of a deep MLP, and why naively stacking layers often makes training *worse*, which is the problem the rest of the chapter solves.

## From one hidden layer to many

A **multi-layer perceptron (MLP)** is a stack of affine transforms interleaved with nonlinear activations. With $`L`$ hidden layers, the forward pass is

```math
\begin{aligned}
\mathbf{h}_0 &= \mathbf{x}, \\
\mathbf{h}_\ell &= \phi\big(W^{(\ell)} \mathbf{h}_{\ell-1} + \mathbf{b}^{(\ell)}\big), \qquad \ell = 1, \dots, L, \\
\hat{\mathbf{y}} &= W^{(L+1)} \mathbf{h}_L + \mathbf{b}^{(L+1)},
\end{aligned}
```

where each $`W^{(\ell)}`$ is a weight matrix, each $`\mathbf{b}^{(\ell)}`$ is a bias, and $`\phi`$ is an elementwise nonlinearity such as ReLU or tanh (Chapter 2). The network is called **deep** when $`L`$ is large, and **shallow** when $`L`$ is small (often $`L = 1`$). There is no sharp cutoff; "deep" is a relative term that grew from a few layers in the mid-2000s to hundreds of layers in modern residual networks and transformers.

**Depth** is the number of layers. **Width** is the number of units (neurons) in a layer. A network can be deep and narrow, shallow and wide, or deep and wide. Parameter count alone does not tell you which: a shallow network with a huge hidden layer and a deep network with modest width can have the same number of weights but behave very differently under training and represent different families of functions efficiently.

In later practice, "layers" are often residual *blocks* that themselves contain several linear transforms (Section 4). The same intuition applies: stacking many nonlinear stages is what "going deep" means.

## Universal approximation vs. efficient representation

Chapter 2 stated the universal approximation theorem: a single hidden layer with a suitable nonlinearity can approximate any continuous function on a compact domain arbitrarily well, provided the hidden layer is wide enough. That result is an existence theorem. It does not say how many units you need, how to find the weights, or whether a deeper network might need far fewer units for the same accuracy.

Depth can help dramatically for some functions. Telgarsky (2016) proved that for every positive integer $`k`$, there are functions computed by ReLU networks with $`\Theta(k^3)`$ layers and a constant number of units per layer that cannot be approximated by networks with $`O(k)`$ layers unless those networks have exponentially many (order $`2^k`$) units. Deep networks can be exponentially more efficient than shallow ones.

### A sawtooth example

The intuition behind such results is easy to see in one dimension. A ReLU network computes a **piecewise linear** function: a continuous function made of straight segments. Count the segments.

- A one-hidden-layer ReLU network with $`m`$ hidden units on a scalar input computes $`\sum_{i=1}^{m} a_i \,\mathrm{ReLU}(w_i x + b_i) + c`$. Each unit contributes at most one "kink" (at $`x = -b_i/w_i`$), so the function has at most $`m + 1`$ linear pieces. The number of pieces grows **linearly** with the number of units.
- Now consider the "tent" map on $`[0, 1]`$, which rises from 0 to 1 and falls back to 0. It needs only two ReLUs:

```math
t(x) = 2\,\mathrm{ReLU}(x) - 4\,\mathrm{ReLU}(x - \tfrac{1}{2}).
```

Composing the tent map with itself folds the interval again: $`t(t(x))`$ has two teeth, $`t(t(t(x)))`$ has four, and $`k`$ compositions have $`2^{k-1}`$ teeth, that is, $`2^k`$ linear pieces. A network that computes $`k`$ compositions has $`k`$ layers of two ReLUs each: $`2k`$ units in total.

```python
import numpy as np

relu = lambda z: np.maximum(z, 0)
tent = lambda x: 2 * relu(x) - 4 * relu(x - 0.5)

x = np.linspace(0, 1, 200_001)
y = x.copy()
for k in range(1, 7):
    y = tent(y)                                    # one more layer
    slopes = np.sign(np.round(np.diff(y) / np.diff(x), 6))
    print(k, 1 + np.sum(slopes[1:] != slopes[:-1]))  # linear pieces: 2, 4, 8, ..., 64
```

To match the $`2^k`$ pieces of the $`k`$-layer sawtooth, a one-hidden-layer network needs at least $`2^k - 1`$ units. With $`k = 20`$, that is two ReLUs per layer for 20 layers (40 units) against more than a million units in a single layer. Depth multiplies the number of linear regions; width only adds to it. Telgarsky's theorem turns this counting argument into a statement about approximation: a shallow network with too few pieces cannot stay close to a function that oscillates this many times.

The lesson for practitioners is not that every task needs a deep net, but that *width is not a free substitute for depth*. When the target has hierarchical or compositional structure, stacking layers is often the cheaper way to express it. When the target is essentially a smooth low-dimensional map, a shallow network may be enough.

## Hierarchical features

A useful mental model for deep networks is **hierarchical feature learning**. Each layer builds on the representations of the layer below:

- Early layers respond to simple patterns in the raw input (edges in an image, local co-occurrences in embeddings).
- Middle layers combine those into mid-level parts (textures, motifs, short phrases).
- Late layers combine parts into task-specific abstractions (object categories, semantic roles, next-token predictions).

This picture was clearest historically in convolutional networks for vision, where visualizations of filters showed edge detectors in the first layer and more complex patterns deeper in. The same hierarchy appears in MLPs and transformers: each stage is a nonlinear remix of the previous stage's coordinates. Depth lets the model reuse intermediate computations. A shallow net that tried to jump from pixels to class labels in one step would need to rediscover every intermediate regularity inside a single giant weight matrix.

Compositionality is the other side of the same coin. Many useful functions are compositions: $`f = f_L \circ f_{L-1} \circ \cdots \circ f_1`$. A deep network is a parametric family of compositions. If the world (or the data distribution) is itself compositional, depth matches the structure of the problem.

## The deep learning revival

Neural networks with many layers were studied for decades, but until the late 2000s they were hard to train and often lost to other methods on standard benchmarks. Three changes turned depth from a liability into the default:

1. **More data.** Large labeled datasets (and later, unlabeled text at internet scale) gave deep models enough signal to fit many parameters without immediate catastrophic overfitting.
2. **GPUs and better software.** Matrix multiplications map well onto GPUs. Frameworks made it practical to define deep graphs and differentiate them automatically (Chapter 2).
3. **Training techniques.** The bulk of this chapter (initialization, residual connections, normalization, adaptive optimizers, learning-rate schedules, and regularization) is the engineering that made deep stacks trainable. Without those, more data and faster chips alone were not enough. Some of the earliest successes in training deep networks, in the mid-2000s, relied on greedy layer-wise *pretraining*, training one layer at a time with an unsupervised objective before fine-tuning the whole stack. Better activations, initialization, and normalization later made that step unnecessary for most problems: deep networks could be trained end to end from random initialization.

The 2012 ImageNet result of Krizhevsky, Sutskever, and Hinton, whose deep convolutional network trained on GPUs with ReLU activations and dropout won the ImageNet image-classification challenge by a wide margin, made the pattern public: depth plus the right training recipe beat shallower alternatives. Language modeling followed a similar path from shallower nets and n-grams to deep recurrent models and then to deep transformers. The architectural details change; the need for the techniques in this chapter does not.

## Counting parameters and compute in a deep MLP

Before worrying about trainability, it helps to know what a deep MLP *costs*. Suppose every hidden layer has width $`d`$ and the input and output dimensions are also $`d`$ for simplicity (a square MLP). Each hidden layer has a weight matrix of shape $`d \times d`$ and a bias of length $`d`$, so

```math
\text{parameters per hidden layer} = d^2 + d.
```

With $`L`$ hidden layers plus an output layer of the same size,

```math
\text{total parameters} \approx (L+1)(d^2 + d) \approx L d^2
```

for large $`d`$. Roughly, **parameters grow linearly with depth and quadratically with width**.

**Forward-pass compute.** A dense matrix-vector product $`W\mathbf{h}`$ with $`W \in \mathbb{R}^{d \times d}`$ costs about $`2d^2`$ floating-point operations (one multiply and one add per weight). So one forward pass through $`L`$ hidden layers costs about $`2 L d^2`$ FLOPs per example (ignoring activations and biases, which are $`O(Ld)`$). A minibatch of size $`B`$ multiplies that by $`B`$.

**Backward-pass compute.** Backpropagation through a linear layer needs two matrix products of the same size as the forward one: one to compute the gradient with respect to the layer's input ($`W^\top \boldsymbol{\delta}`$), and one to compute the gradient with respect to its weights ($`\boldsymbol{\delta}\mathbf{h}^\top`$). So the backward pass costs about twice the forward pass, and a full training step costs about **three times** a forward pass, roughly $`6 L d^2`$ FLOPs per example, before counting the optimizer's small per-parameter overhead (Section 6).

**Activation memory.** Backpropagation needs the activations of every layer from the forward pass (Chapter 2). For a minibatch of $`B`$ examples, each layer stores on the order of $`B d`$ numbers, so activation memory grows as $`O(L B d)`$: linearly in depth and in batch size. For deep networks trained with large batches, activation memory can exceed the memory for the parameters themselves. Parameters, gradients, optimizer state, and activations together determine whether a model fits on a device, a theme Sections 6 and 9 return to.

**Depth vs. width at fixed parameter budget.** Fix a budget of about $`N`$ weights. A shallow net with one hidden layer of width $`w`$ has roughly $`w \cdot d_{\text{in}} + d_{\text{out}} \cdot w`$ parameters. A deep net with $`L`$ layers of width $`d`$ has roughly $`L d^2`$ parameters. You can spend the same $`N`$ on large $`w`$ or on large $`L`$ (with smaller $`d`$). Empirically, for many tasks, spending on depth (up to a point) yields better sample efficiency and accuracy than spending only on width—but only if training succeeds. That last clause is the catch.

Here is a tiny counter that makes the scaling concrete:

```python
def mlp_params(layer_sizes):
    """Parameter count for an MLP with the given layer widths (input to output)."""
    total = 0
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        total += n_in * n_out + n_out  # weights + biases
    return total

# One hidden layer, width 4096: input 784 (MNIST) -> 4096 -> 10
print(mlp_params([784, 4096, 10]))       # about 3.3e6

# Ten hidden layers, width 512: 784 -> [512]*10 -> 10
sizes = [784] + [512] * 10 + [10]
print(mlp_params(sizes))                 # about 2.8e6
```

The deep network has a similar parameter count (about 2.8 million against 3.3 million) but ten nonlinear stages instead of one. The forward FLOPs per example are roughly twice the parameter count in both cases, so the two networks cost about the same to run. Whether the deep one is better depends on the task, and on whether gradients still reach its early layers.

### What counts as a layer

Conventions for counting depth vary. Some authors count weight layers (so a one-hidden-layer MLP has depth 2); others count hidden layers. In residual architectures (Section 4), people usually count **blocks**, where each block contains two or more weight layers plus a normalization. The count that matters for training difficulty is the length of the longest path of nonlinear transforms a gradient must traverse, which is exactly what residual connections shorten. When comparing depths across papers, check which convention is in use.

## The catch: deeper is not automatically better

If you take the one-hidden-layer MLP from Chapter 2, clone its hidden layer many times, and train with plain SGD and naive initialization, something unpleasant often happens:

- Training loss decreases more slowly as depth grows, or stops decreasing.
- Training accuracy of a deeper network can be *worse* than that of a shallower one, even on the training set. This is the **degradation problem** (Section 4): it is not overfitting, because the deeper model fails to fit the training data that the shallow model fits.
- Activation magnitudes and gradient magnitudes become extremely small or extremely large in some layers.
- Loss curves spike, or training produces NaN values.

These failures share a root cause. Backpropagation multiplies one Jacobian per layer (Section 2). With many layers, that product tends to vanish or explode unless the architecture and initialization carefully control the size of each factor. Depth increases expressive power only if the optimization problem remains solvable.

The rest of this chapter is the toolkit that makes depth work:

| Ingredient | Role | Section |
|---|---|---|
| Diagnosis and gradient clipping | See and tame exploding gradients | 2 |
| Xavier / He initialization | Keep activation and gradient scales stable at the start | 3 |
| Residual connections | Give gradients a shortcut path through depth | 4 |
| Normalization (BatchNorm, LayerNorm, RMSNorm) | Keep layer inputs in a stable range during training | 5 |
| Momentum, Adam, AdamW | Adapt step sizes and smooth noisy gradients | 6 |
| Warmup and decay schedules | Use large steps early and small steps late | 7 |
| Weight decay, dropout, early stopping | Limit overfitting once the net can fit | 8 |
| A practical recipe and debugging checklist | Put the pieces together | 9 |

Section 10 then shows how the same ideas carry to convolutional and recurrent networks—the structured relatives of the MLP that dominated vision and sequence modeling before transformers.

Depth is worth wanting. The next section explains precisely why the chain rule makes it hard to get.

## Key takeaways

- Depth is the number of nonlinear stages; width is the size of each stage. They buy capacity in different ways.
- Universal approximation says one wide hidden layer is enough in principle; depth can represent some functions with far fewer units, as formalized by results such as Telgarsky (2016).
- In one dimension, a ReLU layer adds linear pieces while composing layers multiplies them: $`k`$ layers of two units make a sawtooth with $`2^k`$ pieces.
- Deep networks build hierarchical features by composing simple transforms, matching compositional structure in many tasks.
- The deep learning revival combined more data, GPUs, and the training techniques in this chapter.
- In a square MLP, parameters scale as about $`L d^2`$, a training step costs about $`6Ld^2`$ FLOPs per example, and activation memory grows with depth and batch size.
- Naively increasing $`L`$ often hurts training until initialization, residuals, normalization, and optimizers are in place.

## Further reading

Goodfellow, Ian, et al. *Deep Learning*. Cambridge, MA: MIT Press, 2016. https://www.deeplearningbook.org/.

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet Classification with Deep Convolutional Neural Networks." In *Advances in Neural Information Processing Systems 25*, 2012. https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.

Telgarsky, Matus. "Benefits of Depth in Neural Networks." In *Conference on Learning Theory*, 2016. https://arxiv.org/abs/1602.04485.
