# 3.5 Normalization Layers

Initialization (Section 3) sets activation scales at step zero, and residual connections (Section 4) give gradients a direct path through depth. But as soon as training starts, weights change, and the distribution of every layer's inputs shifts with them. A layer that was receiving inputs of standard deviation 1 at initialization may be receiving inputs of standard deviation 20 a thousand steps later. **Normalization layers** fix this continuously: they rescale a layer's inputs, on every forward pass, into a standard range, and then let the network learn whatever scale and offset it actually wants.

This section explains the idea, then covers the three normalization layers that matter most in practice: **batch normalization**, **layer normalization**, and **RMSNorm**. It closes with two questions that come up in every modern architecture: why transformers use LayerNorm or RMSNorm rather than BatchNorm, and where in a residual block the normalization should go.

## The idea

A normalization layer takes a set of numbers, subtracts their mean, divides by their standard deviation, and then applies a learned elementwise scale $`\boldsymbol{\gamma}`$ and shift $`\boldsymbol{\beta}`$:

```math
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma \, \hat{x} + \beta.
```

The small constant $`\epsilon`$ (often $`10^{-5}`$ or $`10^{-6}`$) prevents division by zero. The learned $`\gamma`$ and $`\beta`$ matter: without them, normalization would force every layer's output to have mean 0 and variance 1, which could remove information the network needs. With them, the network can undo the normalization entirely if that is best (set $`\gamma = \sigma`$ and $`\beta = \mu`$), but the *default* is a well-scaled signal.

What differs between the normalization layers is **which set of numbers** the mean and variance are computed over. Consider a minibatch of activations arranged as a matrix $`X \in \mathbb{R}^{B \times d}`$: $`B`$ examples, $`d`$ features.

- **BatchNorm** computes statistics down each column: for each feature, over the $`B`$ examples in the batch.
- **LayerNorm** computes statistics along each row: for each example, over its $`d`$ features.
- **RMSNorm** also works along each row but skips the mean.

The practical benefits of normalization, observed consistently across architectures, are:

1. **Faster training.** Normalized networks usually reach a given loss in fewer steps.
2. **Tolerance of larger learning rates.** Because a layer's output scale no longer depends as sensitively on its weight scale, larger steps are less likely to blow up activations.
3. **Less sensitivity to initialization.** A normalized network is more forgiving of imperfect initial scales, since each normalization layer resets the scale.
4. **Smoother optimization.** Normalization tends to make the loss landscape better conditioned, so gradient steps are more predictable.

Exactly *why* normalization helps has been debated. Ioffe and Szegedy motivated batch normalization as reducing "internal covariate shift" (the changing distribution of layer inputs during training). Later work questioned that explanation and argued that the main effect is a smoother loss landscape. For this chapter, the operational picture is enough: normalization keeps each layer's inputs in a stable range, and that makes training faster and more robust.

## Batch normalization

**Batch normalization** (BatchNorm), introduced by Ioffe and Szegedy (2015), normalizes each feature over the minibatch. For feature $`j`$ and a minibatch of $`B`$ examples,

```math
\mu_j = \frac{1}{B} \sum_{i=1}^{B} x_{ij}, \qquad
\sigma_j^2 = \frac{1}{B} \sum_{i=1}^{B} (x_{ij} - \mu_j)^2,
```

```math
y_{ij} = \gamma_j \, \frac{x_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} + \beta_j.
```

There is one $`\gamma_j`$ and one $`\beta_j`$ per feature. In a convolutional network, statistics are computed per channel, pooling over the batch *and* all spatial positions.

### Running statistics at test time

BatchNorm creates a problem at inference. A model deployed to classify one image at a time has no minibatch to compute statistics over, and even with a batch, we do not want a prediction for one example to depend on which other examples happen to share its batch. So BatchNorm behaves differently in training and evaluation:

- **Training:** normalize with the current minibatch's $`\mu_j`$ and $`\sigma_j^2`$, and update **running estimates** with an exponential moving average, for example $`\mu^{\text{run}}_j \leftarrow (1 - m)\, \mu^{\text{run}}_j + m\, \mu_j`$ with momentum $`m = 0.1`$ (PyTorch's default).
- **Evaluation:** normalize with the stored running estimates. Each example is now processed independently and deterministically.

This is one of the main reasons `model.train()` and `model.eval()` exist (Section 8). Forgetting to call `model.eval()` before evaluating a BatchNorm network is a classic bug: predictions then depend on the evaluation batch composition, and running statistics get corrupted by evaluation data.

### Dependence on batch size

BatchNorm's statistics are estimates from $`B`$ samples, so they are noisy when $`B`$ is small. With batches of 2 or 4 examples per device, the per-batch mean and variance fluctuate a lot, the train-time normalization differs from the test-time normalization, and accuracy drops. BatchNorm works best with reasonably large batches (tens of examples or more per statistics group). This dependence is mild for image classifiers trained with large batches and severe in settings with small per-device batches, which include large models where each device can only hold a few examples, and sequence models with variable lengths.

BatchNorm also couples the examples in a batch: the output for example $`i`$ depends on every other example in the batch. That coupling acts as a mild regularizer (the noise in batch statistics resembles a form of data-dependent noise), but it complicates reasoning about per-example behavior and makes distributed training need care (whether statistics are synchronized across devices).

Despite these caveats, BatchNorm was transformative for convolutional networks: it made very deep conv nets such as ResNets train reliably at high learning rates, and it remains standard in many vision models.

## Layer normalization

**Layer normalization** (LayerNorm), introduced by Ba et al. (2016), computes statistics over the features of each example instead of over the batch. For an input vector $`\mathbf{x} \in \mathbb{R}^{d}`$,

```math
\mu = \frac{1}{d} \sum_{j=1}^{d} x_j, \qquad
\sigma^2 = \frac{1}{d} \sum_{j=1}^{d} (x_j - \mu)^2,
```

```math
\mathrm{LayerNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \boldsymbol{\beta},
```

where $`\odot`$ is elementwise multiplication and $`\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^{d}`$ are learned.

The key property is **independence from the batch**. Each example is normalized using only its own features. So:

- The computation is identical in training and evaluation. There are no running statistics and no train/eval difference.
- It works with any batch size, including batch size 1.
- In a sequence model, each position (each token's vector) is normalized independently, so it works for any sequence length, and padding tokens in other sequences have no effect.

Ba et al. designed LayerNorm with recurrent networks in mind, where BatchNorm is awkward because statistics would need to be tracked separately for each time step. LayerNorm later became the standard normalization in transformers.

## RMSNorm

**RMSNorm**, proposed by Zhang and Sennrich (2019), simplifies LayerNorm by dropping the mean subtraction and the shift. It rescales the vector by its root mean square:

```math
\mathrm{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{j=1}^{d} x_j^2 + \epsilon}, \qquad
\mathrm{RMSNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x}}{\mathrm{RMS}(\mathbf{x})}.
```

Zhang and Sennrich hypothesized that the re-scaling part of LayerNorm, not the re-centering, is responsible for most of its benefit, and found that RMSNorm matched LayerNorm's quality on the tasks they tested while being cheaper to compute: it needs one reduction (the sum of squares) instead of two (mean, then variance), and has no $`\boldsymbol{\beta}`$. On accelerators, normalization is limited by memory traffic rather than arithmetic, so saving a pass over the data is a real saving at scale.

RMSNorm is used in many recent LLMs, including the Llama family, while LayerNorm remains common in others. The choice is typically an efficiency decision; both keep the residual stream's inputs to each block in a stable range.

## Normalization from scratch

The chapter's suggested code lab implements all three layers and checks them against PyTorch. Here is a compact version:

```python
import torch
import torch.nn as nn

def batch_norm(x, gamma, beta, eps=1e-5):
    # x: (B, d); statistics over the batch dimension (training mode)
    mu = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, unbiased=False, keepdim=True)
    return gamma * (x - mu) / torch.sqrt(var + eps) + beta

def layer_norm(x, gamma, beta, eps=1e-5):
    # statistics over the feature dimension, per example
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return gamma * (x - mu) / torch.sqrt(var + eps) + beta

def rms_norm(x, gamma, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return gamma * x / rms

B, d = 8, 16
x = torch.randn(B, d) * 3 + 1
g, b = torch.ones(d), torch.zeros(d)

print(torch.allclose(layer_norm(x, g, b), nn.LayerNorm(d)(x), atol=1e-5))
print(torch.allclose(batch_norm(x, g, b), nn.BatchNorm1d(d).train()(x), atol=1e-5))
print(torch.allclose(rms_norm(x, g), nn.RMSNorm(d, eps=1e-6)(x), atol=1e-5))

# BatchNorm output for example 0 depends on the rest of the batch; LayerNorm's does not.
x2 = x.clone(); x2[1:] = torch.randn(B - 1, d) * 10
print(torch.allclose(batch_norm(x, g, b)[0], batch_norm(x2, g, b)[0]))   # False
print(torch.allclose(layer_norm(x, g, b)[0], layer_norm(x2, g, b)[0]))   # True
```

Note the use of the *biased* variance (dividing by $`B`$ or $`d`$, not $`B - 1`$ or $`d - 1`$), which matches what PyTorch's layers use in the forward pass. (`nn.RMSNorm` was added in PyTorch 2.4.) The last two lines demonstrate the batch dependence directly: changing the *other* examples in the batch changes BatchNorm's output for example 0 but leaves LayerNorm's unchanged.

## Why transformers use LayerNorm or RMSNorm, not BatchNorm

Transformers process sequences of token vectors, and the reasons for per-example normalization pile up:

1. **Variable-length sequences.** Batches of text contain sequences of different lengths padded to a common length. BatchNorm statistics over the batch would mix real tokens with padding, or require masking logic, and the statistics would depend on the length distribution of each batch. LayerNorm normalizes each token vector on its own, so padding elsewhere is irrelevant.
2. **Small per-device batches.** Large models often fit only a few sequences per accelerator. BatchNorm's statistics would be noisy unless synchronized across devices, which adds communication. LayerNorm's cost and quality do not depend on batch size.
3. **Identical behavior in training and inference.** During generation, a language model processes one new token at a time, often for a single user. BatchNorm would need running statistics that match that regime; any mismatch between train-time and test-time statistics shifts the model's behavior. LayerNorm and RMSNorm compute exactly the same function in training and inference.
4. **No cross-example interaction.** With BatchNorm, one example's output depends on the others in its batch, which is undesirable when a batch mixes unrelated requests from different users.

For these reasons, LayerNorm became the default in transformers from the start, and RMSNorm is the common cheaper alternative in recent LLMs. BatchNorm remains a strong choice for convolutional networks trained on images with large batches.

## Where to put the norm: post-norm vs. pre-norm

In a residual block (Section 4), there are two natural places for the normalization.

**Post-norm** normalizes after the residual addition:

```math
\mathbf{h}_{\ell+1} = \mathrm{Norm}\big(\mathbf{h}_\ell + F_\ell(\mathbf{h}_\ell)\big).
```

This is the arrangement in the original transformer.

**Pre-norm** normalizes the input to the branch and leaves the residual path untouched:

```math
\mathbf{h}_{\ell+1} = \mathbf{h}_\ell + F_\ell\big(\mathrm{Norm}(\mathbf{h}_\ell)\big).
```

A pre-norm network usually applies one final normalization after the last block, since the residual stream itself is never normalized along the way.

The difference matters for gradient flow. In post-norm, every block's output passes through a normalization, so the identity path from the output back to the input is interrupted $`L`$ times: the gradient must pass through $`L`$ normalization Jacobians. In pre-norm, the residual stream is a pure sum, $`\mathbf{h}_L = \mathbf{h}_0 + \sum_{\ell} F_\ell(\cdot)`$, so there is a clean identity path from the loss all the way back to the input, exactly as in Section 4.

Xiong et al. (2020) analyzed this for transformers. They showed that in post-norm transformers the expected gradients of the parameters near the output are large at initialization, which is why post-norm transformers need a learning-rate **warmup** (Section 7) to train stably. For pre-norm transformers, they showed gradients are well behaved at initialization, and found that pre-norm transformers could be trained without warmup and reached good results faster in their experiments.

In practice, pre-norm is the default in most modern deep transformers, including GPT-style LLMs, because it trains more stably as depth grows. Post-norm is sometimes reported to reach slightly better final quality when it can be trained successfully, and a variety of hybrid arrangements have been proposed, but the stability advantage of pre-norm has made it the usual choice for deep models. Warmup is still commonly used with pre-norm transformers as an extra safeguard.

In code, the two variants differ by one line:

```python
class PostNormBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        return self.norm(x + self.f(x))

class PreNormBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        return x + self.f(self.norm(x))
```

## Key takeaways

- Normalization rescales a layer's inputs to zero mean and unit variance (or unit RMS), then applies a learned scale and shift; it speeds up training and tolerates larger learning rates.
- BatchNorm normalizes each feature over the minibatch, needs running statistics at evaluation time, and degrades with small batches.
- LayerNorm normalizes over each example's features, so it is independent of batch size and sequence length and behaves identically in training and inference.
- RMSNorm drops the mean subtraction and shift, keeping only RMS rescaling; it is cheaper and common in recent LLMs.
- Transformers use LayerNorm or RMSNorm because of variable-length sequences, small per-device batches, and the need for identical train and inference behavior.
- Pre-norm keeps a clean identity path along the residual stream and trains more stably in deep transformers than post-norm.

## Further reading

Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer Normalization." arXiv preprint arXiv:1607.06450, 2016. https://arxiv.org/abs/1607.06450.

Ioffe, Sergey, and Christian Szegedy. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." In *Proceedings of the 32nd International Conference on Machine Learning*, 2015. https://arxiv.org/abs/1502.03167.

Santurkar, Shibani, et al. "How Does Batch Normalization Help Optimization?" In *Advances in Neural Information Processing Systems 31*, 2018. https://arxiv.org/abs/1805.11604.

Xiong, Ruibin, et al. "On Layer Normalization in the Transformer Architecture." In *Proceedings of the 37th International Conference on Machine Learning*, 2020. https://arxiv.org/abs/2002.04745.

Zhang, Biao, and Rico Sennrich. "Root Mean Square Layer Normalization." In *Advances in Neural Information Processing Systems 32*, 2019. https://arxiv.org/abs/1910.07467.
