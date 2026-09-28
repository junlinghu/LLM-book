# Chapter 3: Deep Neural Networks

Chapter 2 built a neural network with one hidden layer and trained it with plain gradient descent. Modern models, LLMs included, stack dozens or hundreds of layers, and naively stacking more layers makes training fail: gradients vanish or explode, and optimization stalls. This chapter covers the ideas that make deep networks trainable: careful initialization, residual connections, normalization, adaptive optimizers such as Adam and AdamW, learning-rate schedules, and regularization. Together they form a practical recipe for training deep networks, and the chapter closes by showing how the same ideas extend to networks built for images and sequences: convolutional and recurrent networks.

## Learning goals

- Explain why depth helps, and why deep networks are harder to train than shallow ones.
- Diagnose vanishing and exploding gradients by monitoring activation and gradient statistics, and use gradient clipping.
- Derive variance-preserving initialization (Xavier/Glorot and He/Kaiming) and explain when to use each.
- Explain how residual connections and normalization layers (BatchNorm, LayerNorm, RMSNorm) keep deep networks trainable, and why transformers use LayerNorm or RMSNorm.
- Implement SGD with momentum, RMSProp, Adam, and AdamW from scratch, and explain what each one fixes.
- Choose a learning-rate schedule with warmup and decay, and regularize a deep network with weight decay and dropout.
- Train a deep network end to end with a practical recipe and a debugging checklist.

## Outline

### 1. [Why go deep](01-why-go-deep.md)
- From one hidden layer to many: what "deep" means and how depth differs from width
- Universal approximation (Chapter 2) says one wide layer is enough in principle; depth can represent some functions with far fewer units (for example Telgarsky 2016)
- Hierarchical features: each layer builds on the representations of the layer below
- The deep learning revival: more data, GPUs, and the training techniques in this chapter
- Counting parameters and compute in a deep MLP
- The catch: naively stacking layers often makes training worse, not better

### 2. [The trouble with depth: vanishing and exploding gradients](02-the-trouble-with-depth.md)
- Backpropagation through $`L`$ layers multiplies $`L`$ Jacobians:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_L} \prod_{l=1}^{L} \frac{\partial \mathbf{h}_l}{\partial \mathbf{h}_{l-1}}
```

- If typical factors are smaller than 1, gradients shrink exponentially with depth (vanishing); if larger than 1, they grow exponentially (exploding)
- Saturating activations (sigmoid, tanh) make vanishing gradients worse; why ReLU-family activations help
- Symptoms: early layers stop learning, loss plateaus, loss spikes, or NaN values
- Diagnosing: plot activation means and variances and gradient norms layer by layer
- **Gradient clipping**: rescale the gradient when its norm exceeds a threshold, a standard safeguard in LLM training (Pascanu et al. 2013)

### 3. [Initialization](03-initialization.md)
- Why the starting weights matter: all-zero weights keep units identical (symmetry, Chapter 2), and weights that are too large or too small make signals explode or vanish
- Variance propagation through one linear layer: the output variance scales with $`n_{\text{in}} \, \mathrm{Var}(W)`$
- **Xavier/Glorot initialization** for tanh-like activations: balance the forward and backward signals

```math
\mathrm{Var}(W_{ij}) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
```

- **He/Kaiming initialization** for ReLU: compensate for ReLU zeroing half of its inputs

```math
\mathrm{Var}(W_{ij}) = \frac{2}{n_{\text{in}}}
```

- Normal vs. uniform variants, and bias initialization
- Framework defaults (what PyTorch's `nn.Linear` uses) and when to override them

### 4. [Residual connections](04-residual-connections.md)
- The degradation problem: deeper plain networks can have higher training error than shallower ones (He et al. 2016)
- The residual block: learn a change to the input instead of a whole new representation

```math
\mathbf{h}_{l+1} = \mathbf{h}_l + F_l(\mathbf{h}_l)
```

- Why it helps: the identity path gives gradients a direct route back to early layers, and a block can start close to the identity
- Projection shortcuts when input and output shapes differ
- The "residual stream" view: each block reads from and writes to a shared running sum, so information and gradients flow through the whole network along one path
- Initializing residual branches: scaling down the initial weights of layers that add into the running sum keeps its variance from growing with depth, a practice common in large models such as LLMs

### 5. [Normalization layers](05-normalization-layers.md)
- The idea: keep each layer's inputs in a stable range so training is faster and less sensitive to the learning rate
- **Batch normalization**: normalize each feature over the minibatch, then apply a learned scale and shift; running statistics at test time; dependence on batch size (Ioffe and Szegedy 2015)
- **Layer normalization**: normalize over the features of each example, independent of the batch and the sequence length (Ba et al. 2016)

```math
\mathrm{LayerNorm}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \boldsymbol{\beta}
```

- **RMSNorm**: drop the mean subtraction and the shift, keeping only rescaling by the root mean square; cheaper, and common in recent LLMs (Zhang and Sennrich 2019)
- Why transformers use LayerNorm or RMSNorm rather than BatchNorm: variable-length sequences, small per-device batches, and identical behavior in training and inference
- Where to put the norm: post-norm vs. pre-norm residual blocks, and why pre-norm trains more stably in deep transformers (Xiong et al. 2020)

### 6. [Optimizers beyond SGD](06-optimizers-beyond-sgd.md)
- The problems with plain SGD: slow progress along shallow directions, oscillation along steep ones, and one learning rate for every parameter
- **Momentum**: a running average of gradients smooths the path and speeds up consistent directions; Nesterov momentum (Sutskever et al. 2013)
- **Adaptive learning rates**: AdaGrad and RMSProp scale each parameter's step by a running average of its squared gradients
- **Adam**: momentum plus RMSProp, with bias correction for the early steps (Kingma and Ba 2015)

```math
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\mathbf{g}_t, \qquad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\mathbf{g}_t^2
```

```math
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \, \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}, \qquad \hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
```

- **AdamW**: decouple weight decay from the adaptive update; the default optimizer for training transformers and LLMs (Loshchilov and Hutter 2019)
- The memory cost: Adam keeps two extra values (first and second moments) for every parameter, so parameters plus optimizer state take about three times the memory of the parameters alone, a major cost for large models
- Hyperparameters that matter most: the learning rate first, then $`\beta_1`$, $`\beta_2`$, $`\epsilon`$, and weight decay

### 7. [Learning-rate schedules](07-learning-rate-schedules.md)
- Why a constant learning rate is rarely best: large steps early for fast progress, small steps late to settle
- **Warmup**: start small and ramp up to avoid early instability, especially with Adam and in transformers
- **Decay schedules**: step decay, linear decay, and cosine decay (Loshchilov and Hutter 2017)
- The warmup-then-cosine (or warmup-then-linear) schedule used in most LLM pretraining
- Finding a good peak learning rate: a learning-rate range test, and watching for divergence
- Batch size and learning rate interact: larger batches usually tolerate larger learning rates, up to a point

### 8. [Regularization in deep networks](08-regularization-in-deep-networks.md)
- Overfitting revisited (Chapter 2): large networks can memorize their training data
- **Weight decay**: L2 penalty vs. decoupled decay, and why the difference matters for Adam
- **Dropout**: randomly zero activations during training and rescale; turn it off at evaluation time (Srivastava et al. 2014)
- Early stopping and data augmentation
- Why many LLMs use little or no dropout during pretraining: a single pass over a huge dataset overfits little
- Train mode vs. evaluation mode (`model.train()` and `model.eval()`) and the layers that behave differently

### 9. [Training deep networks in practice](09-training-deep-networks-in-practice.md)
- A baseline recipe: He or Xavier initialization, residual connections, normalization, AdamW, warmup plus decay, gradient clipping
- **Mixed-precision training**: FP16 or BF16 arithmetic with FP32 master weights, and loss scaling for FP16 (Micikevicius et al. 2018)
- A debugging checklist: overfit a single batch first, check the initial loss against its expected value, monitor gradient norms and the update-to-weight ratio, and watch for loss spikes and NaN values
- Reproducibility: seeds, logging, and checkpointing
- Scaling up the recipe: larger models and batches, and keeping training stable with gradient clipping, warmup, and monitoring for loss spikes

### 10. [Beyond the MLP: toward sequence models](10-beyond-the-mlp.md)
- Building structure into networks: weight sharing and inductive bias
- Convolutional networks in brief: local filters shared across positions
- Recurrent networks for sequences: the same weights applied at each time step, and backpropagation through time
- Why plain RNNs suffer badly from vanishing and exploding gradients over long sequences, and how gated units (LSTM) help (Hochreiter and Schmidhuber 1997)
- The limits of recurrence: computation must proceed one time step at a time, and information from distant steps is hard to preserve even with gating

## Suggested code labs

1. **Watch gradients vanish and explode.** Build deep MLPs (for example 5, 20, and 50 layers) with sigmoid, tanh, and ReLU activations. Plot activation statistics and gradient norms layer by layer at initialization and after a few training steps.
2. **Initialization shoot-out.** Initialize the same deep ReLU and tanh networks with small random, Xavier, and He weights. Track how the activation variance changes with depth, then train each one and compare loss curves.
3. **Plain vs. residual networks.** Train deep plain MLPs and residual MLPs of increasing depth on the same classification task. Show that the plain network gets worse with depth while the residual one keeps training.
4. **Normalization from scratch.** Implement BatchNorm, LayerNorm, and RMSNorm in NumPy or PyTorch, check them against `nn.BatchNorm1d`, `nn.LayerNorm`, and `nn.RMSNorm`, and show how BatchNorm's output depends on the batch while LayerNorm's does not.
5. **Optimizers from scratch.** Implement SGD, SGD with momentum, RMSProp, Adam, and AdamW. Plot their paths on a 2-D test function with a long, narrow valley, then train the same small network with each one and verify your implementations match `torch.optim` step for step.
6. **Learning-rate schedules and range test.** Run a learning-rate range test to find a peak learning rate, then compare constant, step, cosine, and warmup-plus-cosine schedules on the same model.
7. **Regularization.** Take a network that overfits a small dataset and compare no regularization, weight decay, dropout, and early stopping using train and validation curves. Confirm that dropout is off in evaluation mode.
8. **Put it all together.** Train a deep residual MLP with normalization, AdamW, warmup plus cosine decay, gradient clipping, and mixed precision on a standard image dataset (for example Fashion-MNIST). Walk through the debugging checklist, then run an ablation that removes one ingredient at a time.

## Key takeaways

- Depth makes networks more expressive, but gradients multiplied across many layers tend to vanish or explode.
- Variance-preserving initialization, residual connections, and normalization are what make deep networks trainable.
- Adam and AdamW adapt the step size per parameter, and AdamW with warmup and decay is the standard for transformers.
- Weight decay, dropout, and early stopping control overfitting; gradient clipping and mixed precision keep large-scale training stable and affordable.
- The same ingredients carry across architectures: MLPs, convolutional networks, and recurrent networks all depend on good initialization, normalization, adaptive optimizers, and regularization.

## Further reading

Ba, Jimmy Lei, et al. "Layer Normalization." arXiv preprint arXiv:1607.06450, 2016. https://arxiv.org/abs/1607.06450.

Glorot, Xavier, et al. "Understanding the Difficulty of Training Deep Feedforward Neural Networks." In *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics*, 249–256. PMLR 9, 2010. https://proceedings.mlr.press/v9/glorot10a.html.

Goodfellow, Ian, et al. *Deep Learning*. Cambridge, MA: MIT Press, 2016. https://www.deeplearningbook.org/.

He, Kaiming, et al. "Deep Residual Learning for Image Recognition." In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016. https://arxiv.org/abs/1512.03385.

He, Kaiming, et al. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification." In *Proceedings of the IEEE International Conference on Computer Vision*, 2015. https://arxiv.org/abs/1502.01852.

Hochreiter, Sepp, et al. "Long Short-Term Memory." *Neural Computation* 9, no. 8 (1997): 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735.

Ioffe, Sergey, et al. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." In *Proceedings of the 32nd International Conference on Machine Learning*, 2015. https://arxiv.org/abs/1502.03167.

Kingma, Diederik P., et al. "Adam: A Method for Stochastic Optimization." In *International Conference on Learning Representations*, 2015. https://arxiv.org/abs/1412.6980.

Loshchilov, Ilya, et al. "Decoupled Weight Decay Regularization." In *International Conference on Learning Representations*, 2019. https://arxiv.org/abs/1711.05101.

Loshchilov, Ilya, et al. "SGDR: Stochastic Gradient Descent with Warm Restarts." In *International Conference on Learning Representations*, 2017. https://arxiv.org/abs/1608.03983.

Micikevicius, Paulius, et al. "Mixed Precision Training." In *International Conference on Learning Representations*, 2018. https://arxiv.org/abs/1710.03740.

Pascanu, Razvan, et al. "On the Difficulty of Training Recurrent Neural Networks." In *Proceedings of the 30th International Conference on Machine Learning*, 2013. https://arxiv.org/abs/1211.5063.

Srivastava, Nitish, et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *Journal of Machine Learning Research* 15, no. 56 (2014): 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html.

Sutskever, Ilya, et al. "On the Importance of Initialization and Momentum in Deep Learning." In *Proceedings of the 30th International Conference on Machine Learning*, 1139–1147. PMLR 28, 2013. https://proceedings.mlr.press/v28/sutskever13.html.

Telgarsky, Matus. "Benefits of Depth in Neural Networks." In *Conference on Learning Theory*, 2016. https://arxiv.org/abs/1602.04485.

Xiong, Ruibin, et al. "On Layer Normalization in the Transformer Architecture." In *Proceedings of the 37th International Conference on Machine Learning*, 2020. https://arxiv.org/abs/2002.04745.

Zhang, Biao, et al. "Root Mean Square Layer Normalization." In *Advances in Neural Information Processing Systems 32*, 2019. https://arxiv.org/abs/1910.07467.
