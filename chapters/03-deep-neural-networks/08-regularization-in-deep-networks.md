# 3.8 Regularization in Deep Networks

The previous sections were about getting a deep network to fit its training data at all. This section is about the opposite problem: once a deep network *can* fit its training data, it may fit it too well. **Regularization** is any technique that reduces the gap between training performance and performance on new data. We revisit overfitting, then cover the regularizers that matter most for deep networks: **weight decay** (and why the L2-penalty and decoupled forms differ under Adam), **dropout**, **early stopping**, and **data augmentation**. We then explain why many LLMs use little or no dropout during pretraining, and close with the practical detail that ties several of these together: the difference between training mode and evaluation mode.

## Overfitting revisited

Chapter 2 introduced **overfitting**: a model that performs well on its training data but poorly on held-out data, because it has learned the particulars of the training examples rather than the underlying pattern. The diagnostic is the pair of curves every training loop should record: training loss and validation loss. When training loss keeps falling while validation loss flattens and then rises, the model is overfitting.

Deep networks are especially capable of overfitting because they have so many parameters. Zhang et al. (2017) demonstrated this vividly: standard image-classification networks could reach zero training error even when the labels in the training set were replaced by random labels, which is pure memorization, since random labels have no pattern to learn. The same networks trained on the real labels generalized well. So a large network's capacity is enough to memorize; whether it generalizes depends on the data, the architecture, the optimizer, and the regularization.

A useful organizing principle is that regularizers act in different places:

| Regularizer | Acts on | Mechanism |
|---|---|---|
| Weight decay | Parameters | Pulls weights toward zero each step |
| Dropout | Activations | Randomly removes units during training |
| Early stopping | Training duration | Stops before the model memorizes |
| Data augmentation | Data | Enlarges the training set with label-preserving transforms |

They are complementary, and practical recipes often combine several.

## Weight decay

**Weight decay** shrinks every weight toward zero by a small fraction at each step, which discourages the model from relying on large weights. Large weights let a network produce sharp, highly specific functions; keeping weights small biases the network toward smoother functions that tend to generalize better.

### L2 penalty vs. decoupled decay

There are two ways to implement this idea.

**L2 regularization** adds a penalty on the squared norm of the weights to the loss:

```math
\mathcal{L}_{\text{reg}}(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta}) + \frac{\lambda}{2}\|\boldsymbol{\theta}\|^2,
\qquad
\nabla \mathcal{L}_{\text{reg}} = \nabla \mathcal{L} + \lambda \boldsymbol{\theta}.
```

**Decoupled weight decay** leaves the loss alone and shrinks the weights directly in the update:

```math
\boldsymbol{\theta}_t = (1 - \eta\lambda)\,\boldsymbol{\theta}_{t-1} - \eta \, \Delta_t,
```

where $`\Delta_t`$ is whatever update direction the optimizer computes from the loss gradient.

For **plain SGD**, the two are identical: substituting the L2 gradient into the SGD update gives $`\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta(\nabla\mathcal{L} + \lambda\boldsymbol{\theta}_{t-1}) = (1 - \eta\lambda)\boldsymbol{\theta}_{t-1} - \eta\nabla\mathcal{L}`$. This is why the two names were used interchangeably for years.

For **Adam**, they are different, as Section 6 explained. With L2 regularization, the penalty gradient $`\lambda\boldsymbol{\theta}`$ is added to the loss gradient and then divided by $`\sqrt{\hat{\mathbf{v}}_t}`$. Weights whose loss gradients have been large get a large denominator and hence *weak* regularization; weights with small gradient history get *strong* regularization. That is not what the practitioner intended, and Loshchilov and Hutter (2019) showed it made L2 regularization much less effective with Adam than with SGD. **AdamW** applies decoupled decay, so every weight shrinks by the same factor $`1 - \eta\lambda`$ per step regardless of its gradient history.

The practical upshot:

- With Adam-family optimizers, use **AdamW** (decoupled decay), not Adam with an L2 term. In PyTorch, `torch.optim.Adam(..., weight_decay=λ)` implements the L2 version, while `torch.optim.AdamW(..., weight_decay=λ)` implements decoupled decay. The argument has the same name but a different meaning.
- Common practice excludes **biases and normalization gains** ($`\boldsymbol{\gamma}`$, $`\boldsymbol{\beta}`$) from weight decay. These parameters are few, and shrinking a normalization gain toward zero changes the scale of the whole layer rather than simplifying the function. Embeddings are handled differently across codebases.

Notebook: [3.8-l2-penalty-vs-decoupled-decay.ipynb](../../code/03-deep-neural-networks/3.8-l2-penalty-vs-decoupled-decay.ipynb)

## Dropout

**Dropout**, introduced by Srivastava et al. (2014), randomly sets a fraction of a layer's activations to zero at each training step. For each unit independently, with dropout rate $`p`$:

```math
r_i \sim \mathrm{Bernoulli}(1 - p), \qquad \tilde{h}_i = \frac{r_i \, h_i}{1 - p}.
```

Each unit is kept with probability $`1 - p`$ and zeroed with probability $`p`$. The division by $`1 - p`$, called **inverted dropout**, keeps the expected value of each activation unchanged: $`\mathbb{E}[\tilde{h}_i] = h_i`$. Because the rescaling happens during training, nothing needs to change at evaluation time: dropout is simply **turned off** and activations pass through unmodified. (The original paper instead scaled the weights down at test time; the inverted form, used by all modern frameworks, is equivalent in expectation and more convenient.)

Why does randomly removing units help? Two complementary explanations:

1. **Preventing co-adaptation.** Without dropout, a unit can rely on specific other units to correct its mistakes, forming fragile, highly specific combinations that fit the training data. With dropout, any unit might vanish at any step, so each unit must be useful in many contexts. This pushes the network toward more robust, redundant features.
2. **Implicit ensembling.** Each training step uses a different random subnetwork. A network with $`n`$ droppable units has $`2^n`$ possible subnetworks, all sharing weights. Evaluating the full network with dropout off approximates averaging the predictions of this huge ensemble.

Typical dropout rates are 0.1 to 0.5. Srivastava et al. used rates around 0.5 for hidden layers of fully connected networks and lower rates for inputs. Transformers commonly use lower rates (such as 0.1) when they use dropout at all. Dropout slows convergence, since each step trains only part of the network, so a model with dropout typically needs more steps to reach the same training loss.

Dropout from scratch is two lines:

Notebook: [3.8-dropout.ipynb](../../code/03-deep-neural-networks/3.8-dropout.ipynb)

## Early stopping

**Early stopping** monitors validation loss during training and keeps the checkpoint with the best validation loss, stopping training once validation loss has not improved for a set number of evaluations (the *patience*).

Early stopping is effective and nearly free, since it only requires saving checkpoints and evaluating periodically. It can be viewed as a regularizer on training duration: with small initial weights and a limited number of steps, the parameters cannot travel far from initialization, which has an effect similar to an L2 penalty. For simple linear models trained with gradient descent, this similarity can be made precise.

Two cautions. First, the validation set is now being used to make a decision, so the final validation score is slightly optimistic; report performance on a separate test set. Second, with a learning-rate schedule (Section 7), validation loss often drops sharply late in training as the learning rate decays, so stopping at the first plateau can be premature. Early stopping works best with enough patience, or combined with a schedule designed for the planned training length.

## Data augmentation

The most direct way to reduce overfitting is more data. **Data augmentation** creates additional training examples by applying transformations that preserve the label. For images, standard augmentations include random crops, horizontal flips, small rotations, and color jitter: a flipped photo of a cat is still a cat. Augmentation teaches the network invariances that we know in advance the task has, and it effectively enlarges the training set at no labeling cost.

Augmentation for text is harder, because small changes to text can change its meaning, and there is no universal set of label-preserving transformations comparable to image flips. Techniques such as back-translation (translating a sentence to another language and back) and synonym replacement exist, but they are used much less than in vision. For language modeling, the most effective "augmentation" has simply been more and more diverse text.

## Why many LLMs use little or no dropout during pretraining

The regularizers above were developed for a regime where the model sees each training example many times: dozens or hundreds of epochs over a fixed dataset. Overfitting in that regime is a central concern.

Large language model pretraining is usually in a different regime. The training corpus is so large that each example is seen about **once**, or at most a few times. A model that sees each example once has no opportunity to memorize it in the way a model training for 100 epochs does, so training loss and validation loss stay close together throughout training. In this regime, the model is typically limited by its capacity and compute, not by overfitting, and a regularizer that slows learning, as dropout does, costs more than it saves.

Consequently, many recent LLMs are pretrained with dropout set to zero or very low. Weight decay (via AdamW) is typically still used, partly as a regularizer and partly because it helps keep weight norms under control over long training runs. Dropout reappears more often in fine-tuning, where the datasets are much smaller and multiple epochs are common, so the classic overfitting concern returns.

The lesson generalizes beyond LLMs: how much regularization a model needs depends on the ratio of data to capacity and on how many times each example is seen. Monitor the gap between training and validation loss rather than applying a fixed recipe.

## Train mode vs. evaluation mode

Two layers in this chapter behave differently in training and evaluation:

| Layer | Training mode | Evaluation mode |
|---|---|---|
| Dropout | Randomly zeros activations and rescales by $`1/(1-p)`$ | Identity (no dropout) |
| BatchNorm | Normalizes with the current batch's statistics; updates running statistics | Normalizes with the stored running statistics; no updates |
| LayerNorm, RMSNorm | Same computation | Same computation |

PyTorch tracks which mode each module is in with a flag set by `model.train()` and `model.eval()`. These calls do not start training or evaluation; they only set the flag, which propagates to every submodule.

Notebook: [3.8-train-mode-vs-evaluation-mode.ipynb](../../code/03-deep-neural-networks/3.8-train-mode-vs-evaluation-mode.ipynb)

`model.eval()` and `torch.no_grad()` are independent and do different things: the first changes layer behavior, the second disables gradient tracking to save memory and time. Evaluation usually needs both.

Common bugs from mixing up modes:

- **Evaluating in training mode.** Validation loss is noisy (dropout is active) and worse than it should be; BatchNorm's running statistics are updated with validation data.
- **Training in evaluation mode** (forgetting to switch back after a validation pass). Dropout is silently disabled and BatchNorm stops updating its statistics, so the model trains without its intended regularization.

A quick check confirms dropout is off in evaluation mode: run the same input through the model twice in `eval()` mode and verify the outputs are identical, then do the same in `train()` mode and verify they differ.

## Key takeaways

- Deep networks have enough capacity to memorize training data, even random labels; regularization narrows the gap between training and validation performance.
- L2 regularization and weight decay are equivalent for SGD but not for Adam; AdamW's decoupled decay applies the same shrinkage to every weight and is the right choice for Adam-family optimizers.
- Dropout randomly zeros activations during training and rescales the survivors; it is turned off at evaluation, and it discourages co-adaptation and approximates an ensemble.
- Early stopping and data augmentation are simple and effective; augmentation is most powerful where label-preserving transformations are known, as in vision.
- Many LLMs use little or no dropout in pretraining because each example is seen about once, so overfitting is minor; weight decay is usually kept.
- `model.train()` and `model.eval()` switch dropout and BatchNorm between their two behaviors; LayerNorm and RMSNorm behave the same in both.

## Further reading

Loshchilov, Ilya, and Frank Hutter. "Decoupled Weight Decay Regularization." In *International Conference on Learning Representations*, 2019. https://arxiv.org/abs/1711.05101.

Srivastava, Nitish, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *Journal of Machine Learning Research* 15, no. 56 (2014): 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html.

Zhang, Chiyuan, et al. "Understanding Deep Learning Requires Rethinking Generalization." In *International Conference on Learning Representations*, 2017. https://arxiv.org/abs/1611.03530.
