# 3.2 The Trouble with Depth: Vanishing and Exploding Gradients

Section 1 argued that depth is valuable and then warned that stacking layers often makes training worse. This section explains the mechanism. During backpropagation, the gradient with respect to early activations is a product of many Jacobians—one per layer. If those factors are typically smaller than one, the product shrinks exponentially with depth (**vanishing gradients**). If they are typically larger than one, the product grows exponentially (**exploding gradients**). Either failure stops early layers from learning useful features. We will write the product formally, connect it to saturating activations, list the symptoms you see in training curves, show how to diagnose the problem from activation and gradient statistics, and introduce **gradient clipping** as a standard safeguard.

## Backpropagation as a product of Jacobians

Recall from Chapter 2 that reverse-mode differentiation pushes a scalar loss $`\mathcal{L}`$ backward through the computational graph. For a deep MLP with hidden states $`\mathbf{h}_0, \mathbf{h}_1, \dots, \mathbf{h}_L`$, the chain rule gives

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_0}
=
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_L}
\prod_{\ell=1}^{L}
\frac{\partial \mathbf{h}_\ell}{\partial \mathbf{h}_{\ell-1}}.
```

Each factor $`\partial \mathbf{h}_\ell / \partial \mathbf{h}_{\ell-1}`$ is the **Jacobian** of layer $`\ell`$. For a layer $`\mathbf{h}_\ell = \phi(W^{(\ell)} \mathbf{h}_{\ell-1} + \mathbf{b}^{(\ell)})`$, that Jacobian is

```math
\frac{\partial \mathbf{h}_\ell}{\partial \mathbf{h}_{\ell-1}}
=
\mathrm{diag}\big(\phi'(W^{(\ell)} \mathbf{h}_{\ell-1} + \mathbf{b}^{(\ell)})\big)\, W^{(\ell)},
```

a diagonal matrix of activation derivatives times the weight matrix. The gradient for the earliest weights must travel through *all* $`L`$ of these factors. Gradients for late weights travel through fewer factors. So depth creates an asymmetry: late layers see relatively healthy gradients, while early layers see a long product.

A one-dimensional cartoon makes the scaling obvious. Suppose each factor contributes a multiplier of size $`\alpha`$ (thinking in terms of a typical singular value, or of a scalar network). Then the early gradient scales like $`\alpha^L`$.

- If $`\alpha = 0.9`$ and $`L = 50`$, then $`\alpha^L \approx 0.005`$.
- If $`\alpha = 1.1`$ and $`L = 50`$, then $`\alpha^L \approx 117`$.
- If $`\alpha = 0.5`$ and $`L = 20`$, then $`\alpha^L \approx 10^{-6}`$.

Tiny changes in the typical factor become enormous differences after many layers. Training deep networks is largely about keeping those factors near one—through initialization (Section 3), residual shortcuts that add an identity term (Section 4), and normalization that stabilizes activation scales (Section 5).

The same product appears in recurrent networks over time (Section 10) and in any deep composition of maps. Vanishing and exploding gradients are not an MLP-specific bug; they are a general fact about differentiating long chains.

## When do the factors shrink or grow?

Three ingredients set the size of each Jacobian factor:

1. **The weight matrix.** If the singular values of $`W^{(\ell)}`$ are well above 1, repeated multiplication amplifies signals and gradients. If they are well below 1, repeated multiplication attenuates them. Random matrices with the wrong entry variance produce one or the other depending on width (Section 3).
2. **The activation derivative $`\phi'`$.** This is the focus of the next subsection.
3. **The depth $`L`$.** More factors mean more chances to drift away from magnitude 1.

Forward activations obey a related recurrence: if each layer maps $`\mathbf{h}_{\ell-1}`$ to $`\mathbf{h}_\ell`$ with a typical gain other than 1, activation norms explode or vanish with depth even *before* any gradient is computed. Unstable forward signals and unstable backward gradients often appear together, although, as the experiment later in this section shows, not always: saturating activations can keep forward values in a bounded range while gradients vanish.

### A linear warm-up

It helps to strip away the nonlinearity entirely. In a **deep linear network** $`\mathbf{h}_L = W^{(L)} \cdots W^{(1)} \mathbf{x}`$, every layer Jacobian is just its weight matrix, and the gradient with respect to the input is

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L}\, W^{(L)} W^{(L-1)} \cdots W^{(1)}.
```

Suppose each $`W^{(\ell)}`$ is a $`d \times d`$ matrix with independent entries of mean 0 and variance $`\sigma^2`$. Multiplying a vector by such a matrix scales its squared length by about $`d\sigma^2`$ on average (Section 3 derives this). After $`L`$ layers, the squared length is scaled by about $`(d\sigma^2)^L`$. With $`d = 256`$ and entries of standard deviation 0.05, $`d\sigma^2 = 0.64`$, and after 20 layers the squared norm is multiplied by about $`0.64^{20} \approx 10^{-4}`$. With standard deviation 0.08, $`d\sigma^2 \approx 1.64`$, and the factor is about $`2 \times 10^{4}`$. A modest change in weight scale flips the network from vanishing to exploding. Only $`d\sigma^2 \approx 1`$ keeps the product stable, which is exactly the condition that variance-preserving initialization imposes.

Even at $`d\sigma^2 = 1`$, a product of random matrices is not perfectly behaved: the *average* squared norm is preserved, but individual directions are stretched and squeezed, and the spread of the product's singular values widens with depth. This is one reason initialization alone is not enough for very deep networks, and why the identity paths of Section 4 matter.

## Saturating activations make vanishing worse

Chapter 2 introduced sigmoid and tanh, and noted that both **saturate**: for large positive or negative inputs, the function is almost flat and its derivative is almost zero.

For the logistic sigmoid $`\sigma(z) = 1/(1+e^{-z})`$,

```math
\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big) \le \tfrac{1}{4},
```

with equality only at $`z = 0`$. So every sigmoid layer contributes a factor of at most $`1/4`$ from the activation derivative alone, before counting the weights. Stacking many sigmoid layers multiplies many factors $`\le 1/4`$, and early gradients disappear.

Tanh is centered at zero and has $`\tanh'(z) = 1 - \tanh^2(z) \le 1`$, which is kinder than sigmoid, but it still saturates for $`|z|`$ large. Once hidden units spend most of their time in the flat tails, $`\tanh'`$ is near zero and the Jacobian product collapses.

**ReLU** $`\phi(z) = \max(z, 0)`$ has derivative $`1`$ on the positive side and $`0`$ on the negative side. Units that are active pass gradients at full strength through the activation; units that are inactive pass nothing. There is no soft saturation for positive pre-activations. That is why ReLU-family activations (ReLU, Leaky ReLU, GELU, SiLU) made deep feedforward networks much easier to train than sigmoid stacks. They do not solve the weight-scale problem by themselves—an ill-scaled $`W`$ can still explode or vanish—but they remove one reliable source of vanishing.

ReLU has its own failure mode: a **dead** unit. If a unit's pre-activation is negative for every input (for example after a large update pushes its bias far negative), its gradient is zero for every example, and it can never recover. A network with many dead units has lost capacity. Leaky ReLU, which uses a small slope such as 0.01 on the negative side, and smooth variants avoid exact zeros. In practice, with sensible initialization and learning rates, dead units are a manageable nuisance rather than a fundamental obstacle.

GELU and SiLU, common in transformers, are smooth relatives of ReLU: their derivatives are near 1 for large positive inputs and near 0 for large negative inputs, without a hard cutoff. The same qualitative story applies.

## Symptoms in training

Vanishing and exploding gradients show up in recognizable ways:

| Symptom | Typical cause |
|---|---|
| Early layers' weights barely move; late layers fit the training set alone | Vanishing gradients |
| Training loss plateaus far above a shallower net's loss | Vanishing, or degradation without residuals |
| Loss spikes suddenly, then may recover or diverge | Exploding gradients or an oversized learning rate |
| Loss becomes NaN | Exploding activations or gradients overflowing float range |
| Activation means/variances shrink or blow up with depth at initialization | Bad initialization scale |

A plateau is ambiguous—it can also mean the learning rate is too small, the data are mislabeled, or the model is under-capacity—so you should combine loss curves with the layer-wise diagnostics below rather than guessing from the loss alone.

Exploding gradients are especially dangerous with recurrent nets and with attention over long sequences, but deep MLPs with large initial weights exhibit them too. A single batch with an outsized gradient can corrupt the weights enough that recovery is impossible without a restart or a lower learning rate.

## Diagnosing with activation and gradient statistics

The most informative plots for a deep net at the start of training (and periodically later) are **layer-wise histograms or summary statistics** of activations and of gradients.

**Activations.** After a forward pass on a minibatch, record for each layer the mean and standard deviation of the pre-activations (values before $`\phi`$) and of the post-activations (values after $`\phi`$). Healthy ReLU networks often show post-activation standard deviations of order 1 across depth at initialization, if He initialization is used (Section 3). If the standard deviation collapses toward 0 by layer 20, later layers receive nearly constant inputs and cannot represent much. If it grows to $`10^3`$ or more, you are in the exploding regime.

**Gradients.** After `loss.backward()`, record the L2 norm of the gradient for each parameter tensor, or the mean absolute gradient per layer. With vanishing gradients, early-layer gradient norms are orders of magnitude smaller than late-layer norms. With exploding gradients, some norms are huge and may already contain Inf.

A few lines of PyTorch are enough to see the problem. The function below builds a deep MLP, runs one forward and backward pass, and returns the gradient norm of each hidden layer's weight matrix:

Notebook: [3.2-diagnosing-with-activation-and-gradient-statistics.ipynb](../../code/03-deep-neural-networks/3.2-diagnosing-with-activation-and-gradient-statistics.ipynb)

On one run with this seed, the 20-layer sigmoid network's first-layer gradient norm was about $`3 \times 10^{-13}`$, twelve orders of magnitude smaller than its last hidden layer's (about $`0.2`$): the early layers receive essentially no learning signal. The ReLU network with He initialization (Section 3) had gradient norms of the same order, between about 0.5 and 0.7, in its first and last layers. Note that the sigmoid network's forward activations look unremarkable in this experiment (their spread stays roughly constant across depth), so plotting activations alone would not have revealed the problem. Always look at both activations and gradients.

Wrapping the same idea in a loop over depths (5, 20, 50) and activations (sigmoid, tanh, ReLU), and plotting the per-layer norms on a log scale, is the first suggested code lab for this chapter.

Plotting these statistics over training steps also catches failures that appear only after the weights move: a network can look fine at step 0 and explode at step 500 when a learning-rate schedule jumps or a bad batch arrives.

## Gradient clipping

Even with good initialization and ReLU, gradient norms can occasionally spike—noisy batches, sharp loss regions, or the start of training before normalization statistics settle. **Gradient clipping** is a simple insurance policy: if the gradient's global norm exceeds a threshold $`\tau`$, rescale it so the norm equals $`\tau`$.

**Global norm clipping** (the usual default) concatenates all parameter gradients into one conceptual vector $`\mathbf{g}`$ and replaces it by

```math
\mathbf{g} \leftarrow
\begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\| \le \tau, \\
\tau \, \mathbf{g} / \|\mathbf{g}\| & \text{if } \|\mathbf{g}\| \gt \tau.
\end{cases}
```

The *direction* of the update is unchanged; only the step length is capped. That matters: clipping is not the same as reducing the learning rate globally, because it only fires when the norm is large.

Pascanu et al. (2013) analyzed clipping in the context of training recurrent networks, where exploding gradients are chronic. Their geometric picture is useful for any deep network: the loss surface can contain "cliffs," narrow regions where the loss changes very steeply. A gradient step taken on a cliff is huge and can throw the parameters far away, undoing much of the progress so far. Clipping limits the step to a bounded length, so the parameters move by at most $`\eta\tau`$ in one step no matter how steep the cliff. The same operation is now routine in large-scale training of transformers and LLMs: a typical recipe clips the global gradient norm to a constant such as 1.0 before the optimizer step.

In PyTorch:

Notebook: [3.2-gradient-clipping.ipynb](../../code/03-deep-neural-networks/3.2-gradient-clipping.ipynb)

**Element-wise clipping** (`clip_grad_value_`) caps each gradient entry independently. It is less common as a primary stabilizer because it changes the direction of $`\mathbf{g}`$, but it appears in some older recipes.

Clipping does not fix a systematically vanishing gradient: if every early-layer gradient is near zero, capping the global norm does nothing helpful. Clipping also does not replace a learning rate that is far too large; it only blunts the worst steps. Treat it as a safeguard alongside the structural fixes in Sections 3–5, not as a cure for a misscaled network.

### Choosing the threshold

There is no universal threshold. A value of 1.0 is a common default for transformers trained with Adam. A more principled approach is to run for a while without clipping (or with a very loose threshold), log the gradient norm, and set $`\tau`$ somewhat above its typical value, so that clipping fires only on outliers. Since gradient norms usually fall as training progresses, a threshold chosen early will fire less and less often later in training.

Clipping interacts with adaptive optimizers (Section 6). Adam normalizes each parameter's step by a running estimate of its gradient magnitude, so a single large gradient has a bounded effect on the step size already. But a huge gradient also inflates Adam's running estimates, which then shrink subsequent steps for many iterations. Clipping before the optimizer step protects those estimates too.

### What clipping looks like in the logs

Logging $`\|\mathbf{g}\|`$ before clipping is useful. If the pre-clip norm is almost always below $`\tau`$, clipping is inactive and you may be fine (or $`\tau`$ may be too high to matter). If the pre-clip norm is almost always far above $`\tau`$, every step is being rescaled and you are effectively training with a smaller step size than you think—often a sign that the learning rate, initialization, or residual/normalization setup needs attention.

## Looking ahead within this chapter

Vanishing and exploding gradients are the diagnosis. The next sections are the treatment: initialize so that a typical Jacobian factor starts near 1 (Section 3), add identity shortcuts so the product contains factors of the form $`I + \partial F`$ rather than only $`\partial F`$ (Section 4), and normalize so activation scales cannot wander as far (Section 5). Adaptive optimizers and schedules (Sections 6–7) then make the remaining optimization landscape easier to traverse, and clipping remains on as a last line of defense.

## Key takeaways

- The gradient for early layers is a product of $`L`$ Jacobians; factors smaller than 1 vanish exponentially and factors larger than 1 explode exponentially.
- Saturating activations (sigmoid, tanh) multiply in extra factors $`\lt 1`$; ReLU-family activations avoid soft saturation on the positive side.
- Symptoms include idle early layers, loss plateaus, loss spikes, and NaNs; diagnose with per-layer activation means/variances and gradient norms.
- In a deep linear network the per-layer gain $`d\sigma^2`$ decides everything: below 1 vanishes, above 1 explodes, exponentially in depth.
- Global-norm gradient clipping rescales large gradients without changing their direction and is standard in deep and LLM training.
- Clipping is a safeguard, not a substitute for initialization, residuals, and normalization.

## Further reading

Goodfellow, Ian, et al. *Deep Learning*. Cambridge, MA: MIT Press, 2016. Chapters 8 and 10. https://www.deeplearningbook.org/.

Pascanu, Razvan, et al. "On the Difficulty of Training Recurrent Neural Networks." In *Proceedings of the 30th International Conference on Machine Learning*, 2013. https://arxiv.org/abs/1211.5063.
