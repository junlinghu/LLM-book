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

Depth can help dramatically for some functions. Telgarsky (2016) proved that there are functions that a deep network with ReLU activations can represent with a polynomial number of units, while any shallow network that approximates them well needs an exponential number of units. The constructions are about highly oscillatory functions whose level sets require many "pieces"; each layer of ReLUs can multiply the number of linear regions, so depth buys an exponential increase in representational power for a linear increase in parameters.

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
3. **Training techniques.** The bulk of this chapter—initialization, residual connections, normalization, adaptive optimizers, learning-rate schedules, and regularization—is the engineering that made deep stacks trainable. Without those, more data and faster chips alone were not enough.

The ImageNet breakthrough with deep convolutional networks around 2012 made the pattern public: depth plus the right training recipe beat shallower alternatives. Language modeling followed a similar path from shallower nets and n-grams to deep recurrent models and then to deep transformers. The architectural details change; the need for the techniques in this chapter does not.

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

**Backward-pass compute.** Backpropagation through a linear layer needs a comparable amount of work to the forward pass (gradients with respect to activations and with respect to weights). A standard rule of thumb is that training costs on the order of **twice to three times** a forward pass per example, before counting the optimizer overhead discussed in Section 6.

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

The deep network has a similar parameter count but ten nonlinear stages instead of one. Whether that helps depends on the task—and on whether gradients still reach the early layers.

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
- Deep networks build hierarchical features by composing simple transforms, matching compositional structure in many tasks.
- The deep learning revival combined more data, GPUs, and the training techniques in this chapter.
- In a square MLP, parameters and FLOPs scale as about $`L d^2`$; naively increasing $`L`$ often hurts training until initialization, residuals, normalization, and optimizers are in place.

## Further reading

Goodfellow, Ian, et al. *Deep Learning*. Cambridge, MA: MIT Press, 2016. https://www.deeplearningbook.org/.

Telgarsky, Matus. "Benefits of Depth in Neural Networks." In *Conference on Learning Theory*, 2016. https://arxiv.org/abs/1602.04485.
