# 3.3 Initialization

Section 2 showed that deep training fails when activation and gradient scales drift with depth. The first place that drift can be stopped is at step zero: **initialization**, the distribution from which we draw the starting weights. Bad initialization can make a correctly implemented network untrainable; good initialization can keep forward and backward signal variances roughly constant across layers so that optimization has a chance. This section derives the two standard variance-preserving schemes—**Xavier/Glorot** for tanh-like activations and **He/Kaiming** for ReLU—covers normal vs. uniform variants and bias initialization, and notes what frameworks do by default.

## Why the starting weights matter

Two classic failure modes appear before any clever optimizer can help.

**All zeros (or all equal).** If every weight in a layer is the same (for example all zero), every unit in that layer computes the same function of its inputs. The gradients for those units are therefore identical, and symmetry-breaking never occurs: the units stay tied forever. Chapter 2 already flagged this **symmetry** problem. Biases can be zero; weights cannot all be equal.

**Weights too small.** If entries of $`W`$ are tiny, each layer's outputs shrink (for linear or ReLU nets), and after many layers the activations are essentially zero. Gradients shrink with them. Training stalls.

**Weights too large.** If entries are huge, pre-activations land deep in the saturation region of sigmoid or tanh, or become enormous with ReLU. Forward signals explode; gradients either vanish (saturation) or explode (large linear gain). Training diverges or produces NaNs.

We need random weights that break symmetry while keeping the **typical scale** of activations and gradients stable as depth grows. That is a statement about **variance**.

## Variance propagation through one linear layer

Consider a linear layer without bias, $`z_i = \sum_{j=1}^{n_{\text{in}}} W_{ij} x_j`$, and assume:

- the inputs $`x_j`$ are independent, each with mean 0 and variance $`\mathrm{Var}(x)`$;
- the weights $`W_{ij}`$ are independent, each with mean 0 and variance $`\mathrm{Var}(W)`$, and independent of the inputs.

Then each output coordinate has mean 0 and

```math
\mathrm{Var}(z_i)
=
\sum_{j=1}^{n_{\text{in}}} \mathrm{Var}(W_{ij} x_j)
=
n_{\text{in}} \, \mathrm{Var}(W) \, \mathrm{Var}(x).
```

So the output variance scales with $`n_{\text{in}} \, \mathrm{Var}(W)`$. If we want $`\mathrm{Var}(z) \approx \mathrm{Var}(x)`$, we need

```math
\mathrm{Var}(W) = \frac{1}{n_{\text{in}}}.
```

That is the forward half of variance-preserving initialization. The backward half is symmetric. During backpropagation, a gradient $`\delta_j`$ with respect to an input coordinate receives contributions from all $`n_{\text{out}}`$ output gradients through $`W`$. Under analogous independence assumptions,

```math
\mathrm{Var}(\delta^{\text{in}}) = n_{\text{out}} \, \mathrm{Var}(W) \, \mathrm{Var}(\delta^{\text{out}}).
```

Preserving backward variance asks for $`\mathrm{Var}(W) = 1/n_{\text{out}}`$. Forward and backward desiderata disagree when $`n_{\text{in}} \ne n_{\text{out}}`$. Xavier initialization compromises between them; He initialization prioritizes the forward (and ReLU-adjusted) picture used in modern ReLU nets.

These derivations are approximate—inputs are not really independent after the first layer, and nonlinearities complicate the story—but they give scales that work remarkably well in practice.

## Xavier / Glorot initialization

Glorot and Bengio (2010) studied why deep sigmoid and tanh networks were hard to train and proposed initializing so that both forward and backward variances stay orderly. Their compromise is

```math
\mathrm{Var}(W_{ij}) = \frac{2}{n_{\text{in}} + n_{\text{out}}}.
```

**Normal variant.** Draw

```math
W_{ij} \sim \mathcal{N}\!\left(0,\, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right).
```

**Uniform variant.** The uniform distribution on $`[-a, a]`$ has variance $`a^2/3`$, so matching the target variance gives

```math
W_{ij} \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},\,
\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right).
```

Xavier was derived with activations whose derivatives at 0 are about 1 (tanh, softsign) and with the assumption that one is in the linear regime at initialization. It is still a solid default for tanh networks and for some linear layers. For ReLU, it underestimates the variance that the forward pass needs, because ReLU zeros out roughly half the coordinates.

## He / Kaiming initialization

He et al. (2015) redid the variance calculation for **ReLU**. After a ReLU, about half of the pre-activations are set to zero (for symmetric zero-mean pre-activations). That halves the variance:

```math
\mathrm{Var}(\mathrm{ReLU}(z)) \approx \tfrac{1}{2} \mathrm{Var}(z).
```

To keep post-activation variance equal to the input variance, the linear transform must *double* the usual forward variance:

```math
\mathrm{Var}(W_{ij}) = \frac{2}{n_{\text{in}}}.
```

That is **He (Kaiming) initialization**. The factor of 2 compensates for ReLU's half-sparsity. For Leaky ReLU with negative slope $`a`$, the correction becomes $`2 / ((1 + a^2) n_{\text{in}})`$.

**Normal variant:**

```math
W_{ij} \sim \mathcal{N}\!\left(0,\, \frac{2}{n_{\text{in}}}\right).
```

**Uniform variant** (matching variance $`2/n_{\text{in}}`$):

```math
W_{ij} \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}}}},\, \sqrt{\frac{6}{n_{\text{in}}}}\right).
```

He initialization is the standard choice for deep ReLU MLPs and convolutional nets. Combined with BatchNorm or LayerNorm (Section 5), it is forgiving; without normalization, getting the scale right matters more.

### Forward vs. backward modes in frameworks

PyTorch's `torch.nn.init.kaiming_normal_` accepts a `mode` argument:

- `mode='fan_in'` uses $`n_{\text{in}}`$ in the denominator (preserves forward variance)—the usual choice;
- `mode='fan_out'` uses $`n_{\text{out}}`$ (preserves backward variance).

For Xavier, `nn.init.xavier_normal_` always uses the average of fan-in and fan-out. Knowing which mode you are in avoids matching the wrong formula when you reimplement initialization from scratch.

## Normal vs. uniform, and bias initialization

**Normal vs. uniform.** Both are widely used. Glorot and Bengio originally emphasized the uniform form; many modern codebases use truncated or plain Gaussians. For a fixed variance, the difference is usually second-order compared with choosing Xavier vs. He. Truncating a Gaussian (redrawing samples beyond two standard deviations, for example) avoids rare huge weights; some libraries do this by default for "truncated normal" initializers.

**Biases.** Common practice is to initialize biases to **zero**. A constant zero bias does not break symmetry among units because the weights are already random. Sometimes a small positive bias for ReLU layers is used to reduce the fraction of dead units at the start, but zero is the default in PyTorch's `nn.Linear`. Output-layer biases are occasionally set to log class frequencies for imbalanced classification so that the initial predictions match the base rates; that is a task-specific trick, not a general deep-learning rule.

**Output layers.** The final linear layer that produces logits is sometimes initialized with a smaller standard deviation than He would suggest, so that initial predictions are not overconfident. In large residual networks and transformers, residual *branches* are also often scaled down at initialization (Section 4); that is related but distinct from Xavier/He for ordinary layers.

## What frameworks do by default

In **PyTorch**, `nn.Linear` initializes weights from a uniform distribution

```math
W_{ij} \sim \mathcal{U}(-1/\sqrt{n_{\text{in}}},\, 1/\sqrt{n_{\text{in}}}),
```

and biases from the same uniform range (see the `nn.Linear` documentation). The weight variance is

```math
\mathrm{Var}(W) = \frac{1}{3\, n_{\text{in}}},
```

which is neither Xavier nor He exactly; it is closer to a LeCun-style fan-in uniform. For many shallow nets this default is fine. For deep ReLU stacks without normalization, replacing it with explicit He initialization is a common improvement.

```python
import math
import torch
import torch.nn as nn

def init_he_linear(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 10),
)
model.apply(init_he_linear)
```

For tanh networks, use `nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("tanh"))` or the uniform Xavier variant.

**When to override defaults.**

- Deep ReLU MLPs or conv nets: prefer He.
- Deep tanh / sigmoid nets: prefer Xavier.
- Residual branches in very deep nets: consider extra scaling (Section 4).
- Embedding layers and some attention projections in transformers use their own conventions (often small normal initialization); follow the architecture's recipe when reproducing a paper.

## A quick numerical check

The chapter's suggested code lab initializes the same deep network with small random, Xavier, and He weights and tracks activation variance by depth. A minimal version:

```python
import torch
import torch.nn as nn

def track_activation_std(init_fn, depth=30, width=256, activation=nn.ReLU):
    layers = []
    for _ in range(depth):
        lin = nn.Linear(width, width)
        init_fn(lin)
        layers += [lin, activation()]
    net = nn.Sequential(*layers)
    x = torch.randn(128, width)
    stds = []
    with torch.no_grad():
        for layer in net:
            x = layer(x)
            if isinstance(layer, activation):
                stds.append(x.std().item())
    return stds

def small_init(lin):
    nn.init.normal_(lin.weight, std=0.01)
    nn.init.zeros_(lin.bias)

def xavier_init(lin):
    nn.init.xavier_normal_(lin.weight)
    nn.init.zeros_(lin.bias)

def he_init(lin):
    nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
    nn.init.zeros_(lin.bias)

for name, fn in [("small", small_init), ("xavier", xavier_init), ("he", he_init)]:
    stds = track_activation_std(fn)
    print(name, "first", round(stds[0], 3), "last", round(stds[-1], 3))
```

On one run of this code with 30 ReLU layers, the small initialization's activation standard deviation was already about 0.09 after the first layer and had collapsed to zero (at printed precision) by the last. Xavier started at about 0.58 and also collapsed to zero by layer 30: with $`n_{\text{in}} = n_{\text{out}}`$, Xavier's variance is $`1/n_{\text{in}}`$, and each ReLU halves the signal variance, so the standard deviation shrinks by a factor of about $`1/\sqrt{2}`$ per layer, or roughly $`2^{-15}`$ over 30 layers. He initialization stayed near 0.8 from the first layer to the last. Rerun with `activation=nn.Tanh`, Xavier becomes the better match. Running this experiment once builds more intuition than memorizing the formulas.

## Initialization is necessary but not sufficient

Variance-preserving initialization keeps the *start* of training well scaled. As soon as weights update, scales can drift again—especially without residual connections or normalization. In modern deep training, initialization, residuals (Section 4), and normalization (Section 5) work as a package: initialization sets a good starting gain, residuals keep an identity path whose gain is exactly 1, and normalization continuously re-centers and re-scales activations. Skip any one of them in a very deep plain MLP and the failure modes of Section 2 tend to return.

## Key takeaways

- Zero or identical weights preserve symmetry; weights that are too small or too large make signals vanish or explode with depth.
- For a linear layer, $`\mathrm{Var}(z) = n_{\text{in}}\,\mathrm{Var}(W)\,\mathrm{Var}(x)`$; preserving variance fixes $`\mathrm{Var}(W)`$ in terms of fan-in and fan-out.
- Xavier/Glorot uses $`\mathrm{Var}(W) = 2/(n_{\text{in}}+n_{\text{out}})`$ and suits tanh-like activations.
- He/Kaiming uses $`\mathrm{Var}(W) = 2/n_{\text{in}}`$ to compensate for ReLU zeroing about half its inputs.
- Biases are usually zero; framework defaults (such as PyTorch's `nn.Linear` uniform) are not always He or Xavier—override them for deep ReLU stacks.

## Further reading

Glorot, Xavier, and Yoshua Bengio. "Understanding the Difficulty of Training Deep Feedforward Neural Networks." In *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics*, 249–256. PMLR 9, 2010. https://proceedings.mlr.press/v9/glorot10a.html.

He, Kaiming, et al. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." In *Proceedings of the IEEE International Conference on Computer Vision*, 2015. https://arxiv.org/abs/1502.01852.
