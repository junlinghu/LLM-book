# 3.6 Optimizers Beyond SGD

Chapter 2 trained networks with **stochastic gradient descent (SGD)**: compute the gradient of the loss on a minibatch, take a step of size $`\eta`$ against it, repeat. SGD is simple and, for many problems, surprisingly good. But deep networks expose its weaknesses. Their loss surfaces have directions of very different curvature, their gradients are noisy, and their parameters live in layers whose gradient scales can differ by orders of magnitude. The optimizers in this section fix these problems one at a time: **momentum** smooths the path and accelerates consistent directions; **AdaGrad** and **RMSProp** give each parameter its own effective step size; **Adam** combines the two; and **AdamW** fixes how Adam interacts with weight decay. AdamW is the default optimizer for training transformers and LLMs, so understanding it line by line pays off throughout the rest of the book.

Throughout, $`\boldsymbol{\theta}`$ is the vector of all parameters, $`\mathbf{g}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}_t(\boldsymbol{\theta}_{t-1})`$ is the minibatch gradient at step $`t`$, and operations on vectors such as $`\mathbf{g}_t^2`$, square roots, and divisions are elementwise.

## The problems with plain SGD

The SGD update is

```math
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \, \mathbf{g}_t.
```

Three problems show up in deep networks.

**Ill-conditioning: slow along shallow directions, oscillation along steep ones.** Picture a loss surface shaped like a long, narrow valley: steep walls across the valley, a gentle slope along its floor. The gradient points mostly across the valley, toward the nearest wall's bottom, not along the floor toward the minimum. A learning rate large enough to make progress along the floor makes the iterate bounce from wall to wall; a learning rate small enough to stop the bouncing makes progress along the floor painfully slow.

A quadratic makes this exact. For $`\mathcal{L}(\boldsymbol{\theta}) = \tfrac{1}{2} \sum_i \lambda_i \theta_i^2`$ with curvatures $`\lambda_i \gt 0`$, SGD with exact gradients updates each coordinate independently as $`\theta_i \leftarrow (1 - \eta \lambda_i)\,\theta_i`$. Stability requires $`|1 - \eta\lambda_i| \lt 1`$ for every $`i`$, so $`\eta \lt 2/\lambda_{\max}`$. But the slowest coordinate then shrinks by a factor of about $`1 - \eta \lambda_{\min} \approx 1 - 2\lambda_{\min}/\lambda_{\max}`$ per step. When the **condition number** $`\kappa = \lambda_{\max}/\lambda_{\min}`$ is large, convergence takes on the order of $`\kappa`$ steps. Deep networks' loss surfaces are, locally, very badly conditioned.

**Noise.** Minibatch gradients are noisy estimates of the full gradient. Noise helps escape poor regions, but it also makes the path jittery and forces smaller learning rates than the curvature alone would.

**One learning rate for every parameter.** Embedding rows for rare tokens get nonzero gradients only occasionally; biases, normalization gains, and weights in different layers have gradients of very different magnitudes. A single global $`\eta`$ is too large for some parameters and too small for others.

## Momentum

**Momentum** keeps a running average of past gradients, called the *velocity*, and steps along it:

```math
\mathbf{v}_t = \mu \, \mathbf{v}_{t-1} + \mathbf{g}_t, \qquad
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \, \mathbf{v}_t,
```

with momentum coefficient $`\mu`$ typically 0.9. (This is the form PyTorch's `torch.optim.SGD` uses. Another common form scales the gradient by $`1 - \mu`$; the two differ only by a rescaling of $`\eta`$.)

The physical analogy is a heavy ball rolling on the loss surface. Momentum helps both problems of the narrow valley:

- **Consistent directions accelerate.** If the gradient points the same way step after step, the velocity builds up to about $`\mathbf{g}/(1 - \mu)`$, so with $`\mu = 0.9`$ the effective step is about ten times larger along the valley floor.
- **Oscillating directions cancel.** Across the valley, the gradient flips sign from step to step, so successive contributions to the velocity partly cancel and the bouncing is damped.

Momentum also averages out minibatch noise, since the velocity is an exponentially weighted average of roughly the last $`1/(1 - \mu)`$ gradients.

### Nesterov momentum

**Nesterov momentum** evaluates the gradient at a "look-ahead" point, where the momentum is about to carry the parameters, rather than at the current point:

```math
\mathbf{v}_t = \mu \, \mathbf{v}_{t-1} + \nabla \mathcal{L}\big(\boldsymbol{\theta}_{t-1} - \eta \mu \, \mathbf{v}_{t-1}\big), \qquad
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \, \mathbf{v}_t.
```

If the velocity is about to overshoot, the look-ahead gradient already points back, so Nesterov corrects earlier. Sutskever et al. (2013) showed that SGD with momentum (including Nesterov's variant), combined with a well-designed random initialization and a slowly increasing momentum schedule, could train deep and recurrent networks to levels previously thought to require second-order methods. They emphasized that initialization and momentum together matter: neither alone was sufficient. Deep learning frameworks implement Nesterov momentum with an algebraically equivalent rearrangement so that the gradient is still computed at the stored parameters (`torch.optim.SGD(..., momentum=0.9, nesterov=True)`).

## Adaptive learning rates: AdaGrad and RMSProp

Momentum fixes the direction of the step. **Adaptive methods** fix its size, separately for each parameter, by dividing by a measure of that parameter's typical gradient magnitude.

**AdaGrad** (Duchi et al. 2011) accumulates the sum of squared gradients for each parameter and divides by its square root:

```math
\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \qquad
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \, \frac{\mathbf{g}_t}{\sqrt{\mathbf{s}_t} + \epsilon}.
```

Parameters with large or frequent gradients get smaller steps; parameters with small or rare gradients get relatively larger ones. That is well suited to sparse features (such as rare words). But $`\mathbf{s}_t`$ only grows, so AdaGrad's effective learning rate decays toward zero over a long run, which is a problem for training deep networks for many steps.

**RMSProp**, proposed by Hinton in lecture notes rather than a paper, replaces the sum by an exponential moving average, so the denominator tracks the *recent* gradient magnitude instead of the entire history:

```math
\mathbf{s}_t = \rho \, \mathbf{s}_{t-1} + (1 - \rho)\, \mathbf{g}_t^2, \qquad
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \, \frac{\mathbf{g}_t}{\sqrt{\mathbf{s}_t} + \epsilon},
```

with $`\rho`$ typically around 0.9 to 0.99. Dividing by $`\sqrt{\mathbf{s}_t}`$ makes the step approximately invariant to the scale of each parameter's gradient: a parameter whose gradients are consistently 100 times larger does not take 100 times larger steps. In the narrow valley, the steep direction has large gradients and gets its step shrunk, while the shallow direction has small gradients and gets its step enlarged. This acts like a crude, diagonal form of preconditioning.

## Adam

**Adam** (Kingma and Ba 2015) combines momentum with RMSProp-style scaling. It keeps two exponential moving averages per parameter: the *first moment* $`\mathbf{m}_t`$ (a momentum-like average of gradients) and the *second moment* $`\mathbf{v}_t`$ (an average of squared gradients):

```math
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\mathbf{g}_t, \qquad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\mathbf{g}_t^2
```

Both averages start at zero. The update divides the (bias-corrected) first moment by the square root of the (bias-corrected) second moment:

```math
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}, \qquad \hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
```

The defaults proposed by Kingma and Ba are $`\beta_1 = 0.9`$, $`\beta_2 = 0.999`$, and $`\epsilon = 10^{-8}`$.

### Why bias correction

Because $`\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}`$, the moving averages are biased toward zero in the first steps. After one step, $`\mathbf{m}_1 = (1 - \beta_1)\mathbf{g}_1 = 0.1\,\mathbf{g}_1`$, only a tenth of the actual gradient; with $`\beta_2 = 0.999`$, $`\mathbf{v}_1 = 0.001\,\mathbf{g}_1^2`$. More generally, if the gradients had a constant expected value, then

```math
\mathbb{E}[\mathbf{m}_t] = (1 - \beta_1^t)\, \mathbb{E}[\mathbf{g}], \qquad
\mathbb{E}[\mathbf{v}_t] = (1 - \beta_2^t)\, \mathbb{E}[\mathbf{g}^2].
```

Dividing by $`1 - \beta_1^t`$ and $`1 - \beta_2^t`$ removes this bias exactly. The correction matters most for $`\mathbf{v}`$, whose bias persists for thousands of steps with $`\beta_2 = 0.999`$; without it, the denominator would be too small early on and the first steps would be too large. As $`t`$ grows, both correction factors approach 1 and have no effect.

### What Adam's step looks like

Ignoring $`\epsilon`$, the step for each parameter is $`\eta \, \hat{m}/\sqrt{\hat{v}}`$. The ratio $`\hat{m}/\sqrt{\hat{v}}`$ is roughly the gradient's mean divided by its root mean square, a number whose magnitude is at most about 1. So Adam's per-parameter step size is roughly bounded by $`\eta`$, regardless of the gradient's raw scale. When a parameter's gradient is consistent, the ratio is near $`\pm 1`$ and the step is about $`\eta`$; when its gradient is noisy and changes sign, the ratio is small and the step shrinks. This built-in scale invariance is a big reason Adam works across layers of very different gradient magnitudes with one learning rate, and why its learning rates (commonly around $`10^{-4}`$ to $`10^{-3}`$ for small models, lower for large ones) look so different from SGD's.

The role of $`\epsilon`$ is to prevent division by zero and to cap the step for parameters whose gradients are nearly zero. It is usually left at its default, but in low-precision training or for some models a larger value (such as $`10^{-6}`$) is used for stability.

## AdamW: decoupled weight decay

**Weight decay** shrinks weights toward zero each step to regularize the model (Section 8). With plain SGD, adding an L2 penalty $`\tfrac{\lambda}{2}\|\boldsymbol{\theta}\|^2`$ to the loss is the same as weight decay: the penalty's gradient $`\lambda \boldsymbol{\theta}`$ enters the update as $`-\eta\lambda\boldsymbol{\theta}`$, which multiplies the weights by $`1 - \eta\lambda`$ every step.

With Adam, the two are **not** the same. If the L2 penalty's gradient is added to $`\mathbf{g}_t`$, it passes through Adam's adaptive scaling: it is divided by $`\sqrt{\hat{\mathbf{v}}_t}`$ along with everything else. Parameters with large gradient history (large $`\hat{v}`$) get *less* effective decay, and parameters with small gradient history get *more*. The regularization strength then depends on the gradient statistics in a way nobody intended.

Loshchilov and Hutter (2019) proposed **AdamW**, which **decouples** weight decay from the gradient-based update. The decay is applied directly to the weights, outside the adaptive scaling:

```math
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \left( \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \, \boldsymbol{\theta}_{t-1} \right),
```

where $`\mathbf{m}_t`$ and $`\mathbf{v}_t`$ are computed from the loss gradient only, without the penalty term. Every weight now decays at the same relative rate $`\eta\lambda`$ per step, regardless of its gradient history. Loshchilov and Hutter showed that this decoupling made the best weight decay value less dependent on the learning rate and substantially improved Adam's generalization on the image classification tasks they studied, closing much of the gap to well-tuned SGD with momentum.

AdamW is the default optimizer for training transformers and LLMs. Typical LLM settings use $`\beta_1 = 0.9`$, $`\beta_2`$ between 0.95 and 0.999 (0.95 is common in large-scale pretraining, since a shorter averaging window reacts faster to changing gradient statistics), and weight decay around 0.1, often not applied to biases and normalization gains. Note that PyTorch's `AdamW` multiplies the decay by the learning rate, as in the equation above, so the decay also follows the learning-rate schedule (Section 7).

## Optimizers from scratch

The chapter's suggested code lab implements each optimizer and verifies it against `torch.optim`. Here is a compact version for Adam and AdamW, operating on a list of parameter tensors:

```python
import torch

class MyAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr, self.eps, self.wd = lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            p.mul_(1 - self.lr * self.wd)                 # decoupled weight decay
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)    # first moment
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)  # second moment
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p.add_(-self.lr * m_hat / (v_hat.sqrt() + self.eps))

# Check against PyTorch step for step
torch.manual_seed(0)
w1 = torch.randn(5, 3, requires_grad=True)
w2 = w1.detach().clone().requires_grad_(True)
mine = MyAdamW([w1], lr=1e-2, weight_decay=0.1)
ref = torch.optim.AdamW([w2], lr=1e-2, weight_decay=0.1)
x = torch.randn(16, 5)
for _ in range(20):
    for w, opt in [(w1, mine), (w2, ref)]:
        w.grad = None
        loss = (x @ w).pow(2).mean()
        loss.backward()
        opt.step()
print(torch.allclose(w1, w2, atol=1e-6))   # True
```

Setting `weight_decay=0` gives Adam. Dropping the second moment gives momentum SGD (with the $`1 - \beta_1`$ scaling), and dropping the first moment gives RMSProp. Plotting each optimizer's path on a 2-D function with a long, narrow valley, such as the Rosenbrock function $`f(x, y) = (1 - x)^2 + 100\,(y - x^2)^2`$, shows the behaviors described above: SGD zigzags across the valley, momentum overshoots and then speeds along the floor, and RMSProp and Adam take more balanced steps.

## The memory cost

Adam's per-parameter state is not free. For every parameter, Adam stores two extra numbers, $`m`$ and $`v`$, the same size as the parameter itself. So parameters plus optimizer state take about **three times** the memory of the parameters alone (and the gradients add another copy during the backward pass).

For small models this is irrelevant. For large models it is a major cost. A model with $`N`$ parameters stored in 32-bit floats needs $`4N`$ bytes for its weights and another $`8N`$ bytes for Adam's two moments. For a model with 7 billion parameters, that is 28 GB of weights plus 56 GB of optimizer state, before counting gradients and activations. Mixed-precision training (Section 9) typically keeps FP32 master weights and FP32 moments alongside 16-bit working copies, so optimizer state often dominates training memory for large models. This is why much engineering for large-scale training focuses on sharding optimizer state across devices and on optimizer variants with smaller state. By comparison, SGD with momentum stores one extra value per parameter, and plain SGD stores none.

## Hyperparameters that matter most

In rough order of importance:

1. **Learning rate $`\eta`$.** By far the most important. Too high diverges, too low wastes compute. It interacts with the schedule (Section 7) and the batch size. Always tune it first.
2. **$`\beta_1`$.** The momentum coefficient. 0.9 is almost universal and rarely worth tuning.
3. **$`\beta_2`$.** Controls the averaging window for squared gradients, roughly $`1/(1 - \beta_2)`$ steps. The default 0.999 averages over about 1,000 steps; a lower value such as 0.95 or 0.98 averages over fewer and adapts faster when gradient statistics shift, which can make large-scale training more stable against loss spikes.
4. **$`\epsilon`$.** Usually left at the default, but it matters when gradients are tiny or when training in low precision.
5. **Weight decay $`\lambda`$.** A regularization strength (Section 8); typical AdamW values range from 0.01 to 0.1.

When a training run misbehaves, check the learning rate before anything else.

## Key takeaways

- Plain SGD struggles with ill-conditioned loss surfaces (slow along shallow directions, oscillating along steep ones), gradient noise, and a single learning rate for all parameters.
- Momentum averages gradients over time, accelerating consistent directions and damping oscillations; Nesterov momentum evaluates the gradient at a look-ahead point.
- AdaGrad and RMSProp divide each parameter's step by the root of its accumulated or recent squared gradients, giving per-parameter step sizes.
- Adam combines momentum with RMSProp scaling and corrects the bias from zero-initialized moments; its steps are roughly bounded by the learning rate regardless of gradient scale.
- AdamW decouples weight decay from the adaptive update and is the default optimizer for transformers and LLMs.
- Adam's two moments triple the memory of the parameters alone; the learning rate is the most important hyperparameter to tune.

## Further reading

Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization." *Journal of Machine Learning Research* 12, no. 61 (2011): 2121–2159. https://jmlr.org/papers/v12/duchi11a.html.

Kingma, Diederik P., and Jimmy Ba. "Adam: A Method for Stochastic Optimization." In *International Conference on Learning Representations*, 2015. https://arxiv.org/abs/1412.6980.

Loshchilov, Ilya, and Frank Hutter. "Decoupled Weight Decay Regularization." In *International Conference on Learning Representations*, 2019. https://arxiv.org/abs/1711.05101.

Sutskever, Ilya, James Martens, George Dahl, and Geoffrey Hinton. "On the Importance of Initialization and Momentum in Deep Learning." In *Proceedings of the 30th International Conference on Machine Learning*, 1139–1147. PMLR 28, 2013. https://proceedings.mlr.press/v28/sutskever13.html.
