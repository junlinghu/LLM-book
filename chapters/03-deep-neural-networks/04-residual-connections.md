# 3.4 Residual Connections

Even with ReLU activations and careful initialization, very deep *plain* networks—stacks of layers with no shortcuts—run into a surprising obstacle: as depth increases, **training** error can get *worse*, not better. The network has more capacity, yet it fits the training set more poorly. He et al. (2016) called this the **degradation problem** and introduced **residual connections** (or *skip connections*) as the fix. Residual blocks are now ubiquitous: they appear in nearly every deep vision backbone and, in the form of the residual stream, in every standard transformer. This section explains degradation, defines the residual block, shows why the identity path helps gradients and optimization, covers projection shortcuts when shapes change, introduces the residual-stream view, and discusses scaling residual branches at initialization.

## The degradation problem

Consider a plain feedforward stack of depth $`L`$ and a deeper plain stack of depth $`L' \gt L`$. The deeper net can *represent* anything the shallower net can: set the extra layers to compute the identity and copy the shallow solution. In principle, deeper should be at least as good on the training set. In practice, He et al. observed that deep plain networks had *higher* training error than shallower ones on ImageNet and CIFAR. That cannot be explained by overfitting—overfitting would mean lower training error and higher test error. The optimizer is simply failing to find solutions that the architecture can represent.

Degradation is the empirical face of the difficulties in Section 2. Without an explicit identity path, the network must learn near-identity maps by composing many nonlinear layers whose Jacobians stay well conditioned. Optimization does not reliably do that when depth is large. The residual reformulation makes "do nothing" an easy default.

## The residual block

A **residual block** computes

```math
\mathbf{h}_{\ell+1} = \mathbf{h}_\ell + F_\ell(\mathbf{h}_\ell),
```

where $`F_\ell`$ is a learned residual function—typically one or two linear (or convolutional) layers with nonlinearity and, in modern designs, normalization (Section 5). The block's job is to learn a *change* to the input, not an entirely new representation from scratch. If the best thing to do is pass the input through unchanged, the block can learn $`F_\ell \approx 0`$.

Two equivalent ways to read the equation:

1. **Additive update.** The layer outputs an increment that is added to the running representation.
2. **Identity plus learned branch.** Rewrite as $`\mathbf{h}_{\ell+1} = \big(I + F_\ell\big)(\mathbf{h}_\ell)`$ when shapes match: every block is a perturbation of the identity.

A minimal residual MLP block in code:

Notebook: [3.4-the-residual-block.ipynb](../../code/03-deep-neural-networks/3.4-the-residual-block.ipynb)

Stacking many such blocks gives a deep residual MLP. The chapter's suggested code lab trains plain vs. residual MLPs of increasing depth on the same task: plain nets degrade; residual nets keep fitting.

### Pre-activation variants

He et al. later popularized arrangements where normalization and activation come *before* the weight layers inside $`F`$ (**pre-activation** residual units). Transformers use a closely related idea under the names **pre-norm** and **post-norm** (Section 5). The additive identity path remains the essential ingredient.

## Why residuals help

### A direct path for gradients

Differentiate the residual recurrence with respect to an earlier hidden state. Unrolling $`\mathbf{h}_{L} = \mathbf{h}_{\ell} + \sum_{k=\ell}^{L-1} F_k(\mathbf{h}_k)`$ (along the identity path, treating intermediate dependencies loosely for intuition) shows that $`\partial \mathbf{h}_L / \partial \mathbf{h}_\ell`$ contains an **identity term** plus terms that involve products of residual Jacobians. More carefully, the Jacobian of one block is

```math
\frac{\partial \mathbf{h}_{\ell+1}}{\partial \mathbf{h}_\ell}
=
I + \frac{\partial F_\ell}{\partial \mathbf{h}_\ell}.
```

Compared with a plain layer Jacobian $`\partial F_\ell / \partial \mathbf{h}_\ell`$ alone, the $`I`$ keeps a singular value near 1 even when $`\partial F_\ell / \partial \mathbf{h}_\ell`$ is small. Gradients have a highway through depth that does not require every residual branch to be well scaled. This is the most cited reason residuals tame vanishing gradients.

### Easy to start near the identity

At initialization, if the residual branch $`F_\ell`$ outputs values near zero (small weights, or explicit scaling—see below), each block begins close to the identity. Training can then gradually grow the perturbations that help the loss. A plain stack does not offer that default: every layer must immediately compute something useful and pass a well-scaled signal onward.

### Ensemble and unrolled views

Expanding the product of block Jacobians $`\prod_\ell (I + \partial F_\ell/\partial \mathbf{h}_\ell)`$ gives a sum over $`2^L`$ terms, one for each subset of blocks: the identity term (skip every block), terms that pass through one block, terms that pass through two, and so on. Veit et al. (2016) interpreted residual networks as ensembles of these many paths of different lengths and found experimentally that, in the networks they studied, most of the gradient during training came from relatively short paths, and that deleting individual blocks at test time degraded performance only mildly, unlike in a plain network. Residual networks have also been interpreted as iterative refinement of a representation. Those views are helpful conceptually; the practical takeaway is the same: depth becomes an iterative update to a shared representation rather than a single long chain of irreversible transforms.

## Projection shortcuts when shapes differ

The identity add $`\mathbf{h} + F(\mathbf{h})`$ requires $`F(\mathbf{h})`$ to have the same shape as $`\mathbf{h}`$. Architectures often change width (or, in conv nets, spatial resolution) between stages. Then one uses a **projection shortcut**:

```math
\mathbf{h}_{\ell+1} = W_s \mathbf{h}_\ell + F_\ell(\mathbf{h}_\ell),
```

where $`W_s`$ is a learned linear map (often a $`1\times 1`$ convolution in vision) that matches dimensions. He et al. compared identity shortcuts (with zero-padding or similar when possible) to projections and found that identity shortcuts are preferred when shapes already match—fewer parameters, less risk of the shortcut itself becoming a hard-to-train plain layer. Use projections only where the shape change forces them.

In MLPs that keep width constant across depth, pure identity shortcuts suffice throughout.

## The residual stream view

A productive way to think about a stack of residual blocks is as a **residual stream**: a running sum that every block reads from and writes into.

```math
\mathbf{h}_{0} \xrightarrow[+F_0]{} \mathbf{h}_{1} \xrightarrow[+F_1]{} \mathbf{h}_{2} \xrightarrow[+F_2]{} \cdots \xrightarrow[+F_{L-1}]{} \mathbf{h}_{L}.
```

Each block:

1. Reads the current stream $`\mathbf{h}_\ell`$ (often after a normalization step).
2. Computes an update $`F_\ell(\mathbf{h}_\ell)`$ in a branch.
3. Adds the update back into the stream.

Information from early layers can flow to the output along the stream without being overwritten by intermediate layers—later blocks *add* features rather than replacing the entire state. Gradients flow backward along the same stream. Transformer practitioners lean on this picture heavily: attention blocks and MLP blocks are alternating writers to a shared residual stream of width $`d`$.

This view also clarifies **depth**: adding a block adds another writer. It does not insert a mandatory bottleneck that every bit of information must pass through without a bypass.

## Initializing residual branches

Residual connections fix the gradient path, but they create a new forward-pass problem: every block *adds* to the stream, so the stream's variance grows with depth. Suppose a branch's output has variance $`g^2`$ times its input's variance and is roughly uncorrelated with it. Then

```math
\mathrm{Var}(\mathbf{h}_{\ell+1}) \approx \mathrm{Var}(\mathbf{h}_\ell) + g^2 \, \mathrm{Var}(\mathbf{h}_\ell) = (1 + g^2)\, \mathrm{Var}(\mathbf{h}_\ell),
```

and after $`L`$ blocks the variance is multiplied by $`(1 + g^2)^L`$. If each branch is initialized with He-style weights so that it preserves variance on its own ($`g \approx 1`$), the stream's variance doubles with every block. In one run of the experiment below with width 256, the stream's standard deviation grew from 1 at the input to about 4 after 4 blocks, about 300 after 16 blocks, and about $`6 \times 10^9`$ after 64 blocks. The identity path does not help here; it is what carries the growing values forward.

The fix is to make each branch start small. If the branch gain is $`g^2 = 1/L`$, the total factor is

```math
\left(1 + \frac{1}{L}\right)^{L} \lt e \approx 2.72,
```

bounded no matter how deep the network is. In the same experiment, scaling the last layer of each branch by $`1/\sqrt{L}`$ kept the standard deviation between about 1.5 and 1.7 for 4, 16, and 64 blocks, close to $`\sqrt{e} \approx 1.65`$. Two common countermeasures follow:

1. **Small branch initialization.** Scale down the initial weights of the last layer inside each residual branch, typically by a factor proportional to $`1/\sqrt{L}`$, so that the total added variance stays bounded at initialization. GPT-2, for example, scaled the initial weights of the layers that write into the residual stream by $`1/\sqrt{N}`$, where $`N`$ is the number of residual layers. Some methods go further and initialize the last layer of each branch to exactly zero, so that every block starts as the identity.
2. **Normalization before the branch.** Pre-norm architectures (Section 5) normalize the stream before feeding it into $`F`$, so each branch sees well-scaled inputs even when the stream's raw magnitude grows, and its output no longer scales with the stream.

Large language models combine both habits: pre-norm (or RMSNorm) residual blocks, and scaled-down initialization of the projections that write back into the stream.

A simple illustration for an MLP residual stack:

Notebook: [3.4-initializing-residual-branches.ipynb](../../code/03-deep-neural-networks/3.4-initializing-residual-branches.ipynb)

Exact scaling conventions differ across codebases; the principle is to keep the residual stream from exploding as depth increases *at initialization*, then let learning grow the updates as needed. You can check the numbers quoted above by stacking these blocks with and without the scaling line and printing the standard deviation of the stream on random inputs:

Deleting the line that multiplies by `depth_hint ** -0.5` reproduces the explosive growth.

## Plain vs. residual: what to expect in a lab

Train two families of classifiers on the same data (for example Fashion-MNIST with an MLP):

- Plain: `Linear → ReLU` repeated $`L`$ times, then a linear head.
- Residual: `ResidualMLPBlock` repeated $`L`$ times, then a head.

Use He initialization, the same optimizer, and the same learning rate. As $`L`$ grows (try 5, 20, 50), the plain net's training accuracy typically drops or its loss becomes unstable, while the residual net continues to train. Adding LayerNorm inside each block (Section 5) widens the stable range further. That experiment is the empirical core of this section; the equations explain why the curves look the way they do.

## Residuals do not remove the need for everything else

Residual connections are necessary for today's deepest models, but they are not a license to ignore initialization, normalization, or learning-rate choice. Extremely deep residual nets can still diverge if the learning rate is too high or if branch scales are wild. Think of residuals as changing the inductive bias and the gradient graph so that the other tools in this chapter have a well-conditioned problem to work on.

## Key takeaways

- Deeper plain networks can show *higher* training error than shallower ones (degradation); this is an optimization failure, not overfitting.
- A residual block learns an additive update: $`\mathbf{h}_{\ell+1} = \mathbf{h}_\ell + F_\ell(\mathbf{h}_\ell)`$, so the identity is the default.
- The block Jacobian $`I + \partial F_\ell / \partial \mathbf{h}_\ell`$ keeps a direct gradient path through depth.
- Projection shortcuts match shapes when width or resolution changes; prefer identity shortcuts when shapes already match.
- The residual stream is a shared running sum that every block reads from and writes to.
- Unscaled branches multiply the stream's variance by about $`(1+g^2)`$ per block; scaling branch outputs by about $`1/\sqrt{L}`$ at initialization keeps it bounded for any depth.

## Further reading

He, Kaiming, et al. "Deep Residual Learning for Image Recognition." In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016. https://arxiv.org/abs/1512.03385.

He, Kaiming, et al. "Identity Mappings in Deep Residual Networks." In *European Conference on Computer Vision*, 2016. https://arxiv.org/abs/1603.05027.

Radford, Alec, et al. "Language Models Are Unsupervised Multitask Learners." OpenAI technical report, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

Veit, Andreas, Michael Wilber, and Serge Belongie. "Residual Networks Behave Like Ensembles of Relatively Shallow Networks." In *Advances in Neural Information Processing Systems 29*, 2016. https://arxiv.org/abs/1605.06431.
