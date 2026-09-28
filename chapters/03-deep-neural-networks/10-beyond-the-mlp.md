# 3.10 Beyond the MLP: Toward Sequence Models

Everything in this chapter so far has used the multi-layer perceptron as its running example. An MLP treats its input as a flat vector: every input coordinate connects to every hidden unit with its own weight, and the network knows nothing about which inputs are near each other or which come first. That generality is also a weakness. Images have spatial structure, and text and audio have sequential structure, and a network that ignores structure must learn it from data, with many more parameters and examples than it would otherwise need.

This section shows how to build structure into networks through **weight sharing**, and surveys the two classic structured architectures: **convolutional networks** for grids such as images, and **recurrent networks** for sequences. Recurrent networks bring us back to this chapter's central theme: they are deep networks in time, and they suffer from vanishing and exploding gradients in an especially severe form. We look at how gated units (LSTMs) help, and at the limits of recurrence that remain even with gating.

## Building structure into networks: weight sharing and inductive bias

Every learning algorithm makes assumptions about what kind of function it should find beyond what the training data alone determines. These assumptions are its **inductive bias**. An MLP's inductive bias is weak: it assumes little beyond smoothness. A model with a stronger, correct inductive bias learns from fewer examples and generalizes better, because it does not have to discover the structure of the problem from scratch.

The main tool for building inductive bias into neural networks is **weight sharing**: using the same parameters in many places. Weight sharing does two things at once:

1. **It encodes an assumption.** Using the same weights at every position of an image says "the same pattern means the same thing wherever it appears." Using the same weights at every time step of a sequence says "the rules for processing the next element don't depend on the absolute position in the sequence."
2. **It reduces parameters.** The parameter count no longer grows with the size of the input. A network with shared weights can process images of any size or sequences of any length with a fixed number of parameters.

Consider an image of $`224 \times 224`$ pixels with 3 color channels, which is about 150,000 inputs. A single fully connected layer with 1,000 hidden units needs about 150 million weights, and it would have to learn separately that a cat's ear in the top-left corner and a cat's ear in the bottom-right are the same kind of thing. Convolution replaces this with a small set of shared filters.

## Convolutional networks in brief

A **convolutional layer** slides a small filter (or *kernel*) across the input and computes a weighted sum at every position. For a single-channel 2-D input $`X`$ and a $`k \times k`$ filter $`K`$, the output at position $`(i, j)`$ is

```math
Y_{ij} = \sum_{a=0}^{k-1} \sum_{b=0}^{k-1} K_{ab} \, X_{i+a,\, j+b} + b_0.
```

(Deep learning libraries implement this operation, technically a cross-correlation, and call it convolution.) A real layer has many input channels and many output channels: each output channel has its own filter spanning all input channels, and the layer's parameters are $`C_{\text{out}} \times C_{\text{in}} \times k \times k`$ weights plus biases, **independent of the image size**.

Two assumptions are built in:

- **Locality.** Each output depends only on a small neighborhood of the input ($`k \times k`$, often $`3 \times 3`$). Nearby pixels are related; distant ones are handled by later layers.
- **Translation equivariance.** Because the same filter is applied at every position, shifting the input shifts the output by the same amount. A detector for vertical edges finds them anywhere in the image.

LeCun et al. (1998) used networks of exactly this kind, trained by backpropagation, for handwritten digit and document recognition. Stacking convolutional layers builds the hierarchy described in Section 1. Each layer sees a slightly larger region of the original image than the one below: after $`L`$ layers of $`3 \times 3`$ filters, each unit's **receptive field** spans $`(2L + 1) \times (2L + 1)`$ pixels. Downsampling operations, such as pooling or strided convolution, reduce the spatial resolution between stages so that receptive fields grow faster and later layers can combine information from across the whole image. Early layers learn edge and color detectors, middle layers textures and parts, and late layers object-level features.

All the techniques from this chapter apply directly to convolutional networks, and several were developed for them. He initialization (Section 3) was derived for deep ReLU convolutional networks; residual connections (Section 4) and batch normalization (Section 5) made very deep convolutional networks trainable, and ResNets combine both. The training recipe of Section 9 carries over almost unchanged.

```python
import torch.nn as nn

class ConvBlock(nn.Module):
    """A residual convolutional block: conv-BN-ReLU-conv-BN, plus identity."""
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
        )
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(x + self.f(x))
```

This block has $`2 \times 9c^2`$ convolution weights (plus normalization parameters), whether the image is $`32 \times 32`$ or $`1024 \times 1024`$.

## Recurrent networks for sequences

Text, speech, and time series are **sequences**: ordered lists $`\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T`$ whose length varies from example to example. A **recurrent neural network (RNN)** processes a sequence one element at a time, carrying a **hidden state** $`\mathbf{h}_t`$ that summarizes everything seen so far:

```math
\mathbf{h}_t = \tanh\big(W_{hh}\,\mathbf{h}_{t-1} + W_{xh}\,\mathbf{x}_t + \mathbf{b}\big), \qquad
\mathbf{y}_t = W_{hy}\,\mathbf{h}_t.
```

The key design choice is weight sharing **across time**: the same matrices $`W_{hh}`$, $`W_{xh}`$, and $`W_{hy}`$ are used at every time step. So an RNN has a fixed number of parameters regardless of sequence length, can process sequences of any length, and applies the same rules at every position. For language modeling, $`\mathbf{x}_t`$ is the embedding of token $`t`$ and $`\mathbf{y}_t`$ gives logits for token $`t+1`$, trained with the cross-entropy loss of Chapter 2 summed over positions.

```python
import torch

def rnn_forward(xs, h0, W_hh, W_xh, W_hy, b):
    h, ys = h0, []
    for x in xs:                                  # one step at a time
        h = torch.tanh(W_hh @ h + W_xh @ x + b)
        ys.append(W_hy @ h)
    return ys, h
```

### Backpropagation through time

To train an RNN, **unroll** it: draw one copy of the network for each time step, connected by the hidden state. The unrolled network is an ordinary feedforward computational graph, $`T`$ layers deep, in which every layer shares the same weights. **Backpropagation through time (BPTT)** is simply backpropagation (Chapter 2) applied to this unrolled graph. Because the weights are shared, the gradient for $`W_{hh}`$ is the sum of its gradients from every time step, which is the "gradients accumulate when a value is used more than once" rule from Chapter 2.

For long sequences, full BPTT is expensive in memory, since activations for every time step must be stored. **Truncated BPTT** splits the sequence into chunks, carries the hidden state forward across chunks, but backpropagates only within each chunk. This bounds memory at the cost of not learning dependencies longer than the chunk.

## Why plain RNNs suffer from vanishing and exploding gradients

An unrolled RNN is a very deep network, $`T`$ steps deep, so everything from Section 2 applies, with an extra twist: every layer uses the **same** weight matrix. The gradient of a loss at step $`T`$ with respect to the hidden state at an earlier step $`k`$ is

```math
\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_k}
=
\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T}
\prod_{t=k+1}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}},
\qquad
\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \mathrm{diag}\big(1 - \mathbf{h}_t^2\big)\, W_{hh}.
```

In a deep MLP, each layer has its own matrix, and their effects can partly average out. In an RNN, the product contains the same $`W_{hh}`$ raised, in effect, to the power $`T - k`$. Ignoring the tanh derivatives, if $`W_{hh}`$'s largest singular value is below 1, the gradient shrinks exponentially with the time gap; if it is above 1, it can grow exponentially. Pascanu et al. (2013) analyzed exactly this and showed that a largest singular value below 1 is sufficient for vanishing gradients, and that above 1 is necessary for explosion. The tanh derivatives, each at most 1, can only push the product further toward vanishing.

The consequences are specific:

- **Exploding gradients** cause sudden, catastrophic updates. Gradient clipping (Section 2) was popularized precisely for RNNs and handles this failure well.
- **Vanishing gradients** are the harder problem. The gradient signal from a loss at step $`T`$ about an input at step $`k`$ decays exponentially with $`T - k`$. Updates are then dominated by short-range dependencies, and the network fails to learn long-range ones, such as matching the subject of a sentence to a verb many words later. Clipping does nothing for vanishing gradients.

## Gated units: the LSTM

The **long short-term memory (LSTM)** network of Hochreiter and Schmidhuber (1997) addresses vanishing gradients by changing how the state is updated. Its central idea is a **cell state** $`\mathbf{c}_t`$ that is updated **additively**, with learned **gates** controlling what is written, kept, and read.

At each step, the LSTM computes three gates, each a vector of values between 0 and 1 produced by a sigmoid, plus a candidate update:

```math
\begin{aligned}
\mathbf{f}_t &= \sigma(W_f [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_f) && \text{(forget gate)} \\
\mathbf{i}_t &= \sigma(W_i [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_i) && \text{(input gate)} \\
\mathbf{o}_t &= \sigma(W_o [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_o) && \text{(output gate)} \\
\tilde{\mathbf{c}}_t &= \tanh(W_c [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_c) && \text{(candidate)}
\end{aligned}
```

and then updates the cell and hidden states:

```math
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t, \qquad
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t).
```

Here $`[\mathbf{h}_{t-1}; \mathbf{x}_t]`$ denotes concatenation. (The forget gate was added to the original LSTM design by Gers et al. (2000); the form above is the standard modern one.)

Why does this help? Look at the cell-state path. Ignoring the gates' own dependence on the state, the Jacobian from one cell state to the next is

```math
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} \approx \mathrm{diag}(\mathbf{f}_t).
```

There is no repeated multiplication by a weight matrix and no squashing nonlinearity along this path. When the forget gate is near 1, the cell state, and the gradient flowing back through it, passes through almost unchanged for many steps. The network can *learn* to keep information (forget gate near 1) or discard it (near 0), depending on the input. This is the same idea as the residual connection of Section 4: an additive update path along which gradients flow without repeated attenuation. The LSTM's cell state was, in effect, a gated residual stream through time, nearly two decades before residual networks.

A common practical detail is to initialize the forget gate's bias to a positive value (such as 1), so that the network starts out remembering by default. The **gated recurrent unit (GRU)** of Cho et al. (2014) is a popular simplification with two gates and no separate cell state, which often performs comparably.

LSTMs made recurrent networks practical for language modeling, machine translation, and speech recognition, and were the dominant sequence models for years.

## The limits of recurrence

Gating mitigates vanishing gradients, but two fundamental limits remain.

**Computation must proceed one step at a time.** The hidden state $`\mathbf{h}_t`$ depends on $`\mathbf{h}_{t-1}`$, which depends on $`\mathbf{h}_{t-2}`$, and so on. Processing a sequence of length $`T`$ requires $`T`$ sequential steps, in both the forward and backward passes, no matter how much parallel hardware is available. Each step is a relatively small matrix-vector operation (or matrix-matrix, across a batch), which uses accelerators poorly compared with the large matrix multiplications of an MLP or convolutional network over many positions at once. Training on long sequences is therefore slow, and it becomes harder to use more hardware to train on more data.

**Information from distant steps is hard to preserve.** Everything the network knows about the past must pass through a fixed-size state vector. To use a word from 500 tokens ago, the network must have kept information about it in its state through 500 updates, while also storing everything else it might need. Gating helps the gradient survive, but the capacity of the state and the difficulty of learning what to keep over very long spans remain. In practice, LSTMs use nearby context much more effectively than distant context.

These two limits—sequential computation and a fixed-size bottleneck between distant positions—are what an architecture for long sequences at scale needs to overcome. An ideal sequence model would let every position access every other position directly, in a way that can be computed in parallel across the whole sequence, while keeping everything this chapter has built: residual paths, normalization, good initialization, adaptive optimizers, and schedules.

## Key takeaways

- Weight sharing builds inductive bias into a network and makes the parameter count independent of input size.
- Convolutional networks share small local filters across positions, giving locality and translation equivariance; He initialization, residual connections, and BatchNorm were developed largely for them.
- Recurrent networks share weights across time steps and carry a hidden state; they are trained by backpropagation through time on the unrolled graph.
- Plain RNNs suffer severely from vanishing and exploding gradients because the same recurrent matrix multiplies the gradient at every step; clipping handles explosion but not vanishing.
- LSTMs add a gated, additively updated cell state, a residual-like path through time that lets gradients survive many steps.
- Recurrence remains limited by sequential computation and by a fixed-size state that must carry all information across long distances.

## Further reading

Cho, Kyunghyun, et al. "Learning Phrase Representations Using RNN Encoder–Decoder for Statistical Machine Translation." In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, 2014. https://arxiv.org/abs/1406.1078.

Gers, Felix A., Jürgen Schmidhuber, and Fred Cummins. "Learning to Forget: Continual Prediction with LSTM." *Neural Computation* 12, no. 10 (2000): 2451–2471. https://doi.org/10.1162/089976600300015015.

Goodfellow, Ian, et al. *Deep Learning*. Cambridge, MA: MIT Press, 2016. Chapters 9 and 10. https://www.deeplearningbook.org/.

Hochreiter, Sepp, and Jürgen Schmidhuber. "Long Short-Term Memory." *Neural Computation* 9, no. 8 (1997): 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735.

LeCun, Yann, Léon Bottou, Yoshua Bengio, and Patrick Haffner. "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE* 86, no. 11 (1998): 2278–2324. https://doi.org/10.1109/5.726791.

Pascanu, Razvan, Tomas Mikolov, and Yoshua Bengio. "On the Difficulty of Training Recurrent Neural Networks." In *Proceedings of the 30th International Conference on Machine Learning*, 2013. https://arxiv.org/abs/1211.5063.
