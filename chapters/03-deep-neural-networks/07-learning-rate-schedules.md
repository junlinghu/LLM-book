# 3.7 Learning-Rate Schedules

Section 6 ended with a rule: when training misbehaves, check the learning rate first. This section adds a refinement: the best learning rate is usually not a single number but a **schedule**, a function $`\eta_t`$ of the training step. Almost every serious training run, from small image classifiers to LLM pretraining, changes its learning rate over time. We look at why a constant learning rate is rarely best, at **warmup** at the start of training, at **decay schedules** (step, linear, and cosine) for the rest of it, at the warmup-then-decay schedules used in LLM pretraining, at how to find a good peak learning rate, and at how the learning rate interacts with batch size.

## Why a constant learning rate is rarely best

Think about what the learning rate has to do at different stages of training.

**Early in training**, the parameters are far from any good solution. Large steps make fast progress, and the noise they inject helps the optimizer move across the loss landscape instead of settling into the first basin it finds.

**Late in training**, the parameters are near a good region. Now large steps hurt. With stochastic gradients, SGD does not converge to a point at a constant learning rate; it bounces around the minimum in a region whose size grows with $`\eta`$ and with the gradient noise. In the simplest quadratic picture, the steady-state fluctuation of the parameters around the minimum is proportional to $`\eta`$ times the gradient noise variance. Shrinking $`\eta`$ shrinks that "noise ball" and lets the loss settle lower.

A constant learning rate forces a compromise: large enough to make early progress and small enough to settle late, which means it is not ideal for either phase. A familiar signature in loss curves is that training loss drops noticeably right after the learning rate is reduced: the optimizer was bouncing in a noise ball, and the smaller step lets it descend into it.

A schedule has two parts that address different problems: **warmup** at the start and **decay** for the rest.

## Warmup

**Warmup** starts with a small learning rate and increases it, usually linearly, to its peak value $`\eta_{\max}`$ over the first $`T_w`$ steps:

```math
\eta_t = \eta_{\max} \cdot \frac{t}{T_w}, \qquad t \le T_w.
```

Why start small, when the argument above says large steps are best early? Because the very first steps are special:

- **The initial gradients can be large and poorly aligned.** At initialization, a network's outputs are essentially random and the loss is high. Gradients can be large, and a full-size step in their direction can push the parameters into a region from which training does not recover, causing a loss spike or divergence.
- **Adam's second-moment estimates are unreliable at first.** In the first steps, $`\hat{\mathbf{v}}_t`$ is estimated from very few gradients. Bias correction removes the systematic bias (Section 6), but the estimate is still noisy. Parameters whose early gradients happened to be small get a small denominator and hence large steps. A small learning rate during warmup limits the damage while the moment estimates settle.
- **Transformers are especially sensitive.** Section 5 described Xiong et al.'s analysis showing that post-norm transformers have large gradients near the output at initialization, which is why they need warmup to train at all. Pre-norm transformers are more stable but warmup remains standard as a safeguard.

Warmup is typically short compared with the full run: from a few hundred to a few thousand steps, or a small percentage of total training. The exact length is rarely critical as long as it is not zero for a sensitive model.

## Decay schedules

After warmup, the learning rate decreases over the rest of training. Let $`T`$ be the total number of steps, and write the progress through the decay phase as $`s = (t - T_w)/(T - T_w) \in [0, 1]`$. The common choices are:

**Step decay.** Multiply the learning rate by a factor (often 0.1) at a few fixed points, such as at 50% and 75% of training:

```math
\eta_t = \eta_{\max} \cdot \gamma^{\lfloor t / T_{\text{step}} \rfloor}.
```

Step decay was the standard for training image classifiers such as ResNets with SGD. Its drawback is that it introduces extra hyperparameters (when to step and by how much), and the abrupt changes produce abrupt changes in the loss curve.

**Linear decay.** Decrease linearly from $`\eta_{\max}`$ to a final value $`\eta_{\min}`$ (often 0 or a small fraction of the peak):

```math
\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min})\,(1 - s).
```

**Cosine decay.** Follow half a cosine wave from $`\eta_{\max}`$ down to $`\eta_{\min}`$:

```math
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\big(1 + \cos(\pi s)\big).
```

Cosine decay was popularized by Loshchilov and Hutter (2017) in SGDR ("stochastic gradient descent with warm restarts"), which repeatedly decayed the learning rate with a cosine and then reset it to the peak. In practice, a single cosine cycle without restarts became the widely used variant. Compared with linear decay, cosine stays near the peak longer at the start, decays fastest in the middle, and flattens out near the end, spending more steps at small learning rates. In many experiments linear and cosine perform similarly; both are much smoother than step decay and have only one or two hyperparameters beyond the peak.

**Inverse square root decay**, $`\eta_t \propto 1/\sqrt{t}`$ after warmup, was used in the original transformer paper. It never reaches zero, which makes it convenient when the total training length is not known in advance.

## Warmup plus decay for LLM pretraining

Most LLM pretraining runs use **linear warmup followed by cosine decay** (or linear decay) to a small final learning rate, often around 10% of the peak or lower. In code:

Notebook: [3.7-warmup-plus-decay-for-llm-pretraining.ipynb](../../code/03-deep-neural-networks/3.7-warmup-plus-decay-for-llm-pretraining.ipynb)

A consequence of cosine decay is that the schedule depends on the total number of steps $`T`$. The learning rate at step 50,000 is different in a 100,000-step run than in a 200,000-step run. So a model trained with a cosine schedule to $`T`$ steps cannot simply be "trained longer" without re-warming or re-planning the schedule, and a checkpoint taken mid-run is not equivalent to a finished shorter run. This has motivated schedules that hold the learning rate constant for most of training and then decay quickly at the end (sometimes called warmup-stable-decay), so that a run can be extended or branched from any point in the constant phase. The underlying principle is the same as above: large steps for progress, then a decay phase to settle.

## Finding a good peak learning rate

The peak learning rate matters more than the shape of the schedule. Two practical tools help find it.

### The learning-rate range test

A **learning-rate range test**, popularized by Smith (2017), trains the model briefly while increasing the learning rate exponentially from a tiny value (for example $`10^{-7}`$) to a large one (for example $`10`$) over a few hundred steps, recording the loss at each step. Plot loss against learning rate on a log scale. The typical curve has three regions:

1. At very small learning rates, the loss barely moves.
2. In a middle range, the loss falls steadily; the steepest descent is somewhere in here.
3. At large learning rates, the loss stops falling and then shoots up as training diverges.

A reasonable peak learning rate lies somewhat below the point where the loss is lowest, often about an order of magnitude below where divergence begins. The test is cheap, costing a few hundred steps, and gives a principled starting point.

Notebook: [3.7-the-learning-rate-range-test.ipynb](../../code/03-deep-neural-networks/3.7-the-learning-rate-range-test.ipynb)

Remember to reinitialize the model after the test; it has been trained, and at the end damaged, by the sweep.

### Watching for divergence

The range test gives a starting point, not a guarantee. The learning rate that is stable in the first few hundred steps may not be stable later, and at larger scale the maximum stable learning rate typically decreases. During a real run, watch for:

- **Loss spikes** that recover slowly or not at all.
- **Gradient norms** that grow steadily or spike (Section 2); clipping that fires on almost every step is a warning sign.
- **NaN or Inf** values.

If these appear, the usual first fixes are a lower peak learning rate, a longer warmup, or both. A short sweep of a few peak values (for example, a factor of 2 or 3 apart) on a shortened run is often the most reliable way to choose.

## Batch size and learning rate interact

The batch size $`B`$ controls how noisy each gradient is: averaging over $`B`$ examples reduces the variance of the gradient estimate by a factor of $`B`$. Larger batches therefore give more accurate gradients and usually tolerate larger learning rates.

For SGD, a widely used heuristic is the **linear scaling rule**, which Goyal et al. (2017) used with warmup to train an ImageNet classifier with minibatches of 8,192 images: when the batch size is multiplied by $`k`$, multiply the learning rate by $`k`$ as well (usually with a warmup). The intuition is that $`k`$ steps of SGD with batch $`B`$ and learning rate $`\eta`$ move the parameters about as far as one step with batch $`kB`$ and learning rate $`k\eta`$, as long as the gradient does not change much over those steps. For Adam, whose step size is roughly normalized (Section 6), a square-root scaling rule ($`\eta \propto \sqrt{B}`$) is sometimes suggested instead. Neither rule is exact; both are starting points for tuning.

These rules hold only **up to a point**. McCandlish et al. (2018) studied this limit empirically and related it to a measurable "gradient noise scale." Beyond some batch size, increasing $`B`$ further no longer lets you take proportionally larger steps: the gradient estimate is already accurate, and the limit is the curvature of the loss surface rather than the noise. Past that point, larger batches cost more compute per step without reducing the number of steps much, so the total compute to reach a given loss rises. The batch size at which this happens (McCandlish et al. call it the *critical batch size*) depends on the task and, in their experiments, tends to grow as training progresses and the loss falls. In practice, large training runs choose a batch size large enough to use hardware efficiently but not far beyond the point of diminishing returns, and some increase the batch size during training.

## Key takeaways

- A constant learning rate is a compromise; large steps help early progress, small steps let the loss settle late in training.
- Warmup ramps the learning rate up over the first steps to avoid early instability from large initial gradients and unreliable Adam statistics; transformers are especially sensitive.
- Step, linear, and cosine decay reduce the learning rate over training; cosine (Loshchilov and Hutter 2017) and linear are smooth, low-hyperparameter choices.
- Most LLM pretraining uses linear warmup followed by cosine or linear decay to a small final learning rate.
- A learning-rate range test gives a starting peak value; watch loss spikes and gradient norms for divergence during the real run.
- Larger batches usually tolerate larger learning rates, roughly linearly for SGD, up to a point beyond which larger batches give diminishing returns.

## Further reading

Goyal, Priya, et al. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv preprint arXiv:1706.02677, 2017. https://arxiv.org/abs/1706.02677.

Loshchilov, Ilya, and Frank Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts." In *International Conference on Learning Representations*, 2017. https://arxiv.org/abs/1608.03983.

McCandlish, Sam, et al. "An Empirical Model of Large-Batch Training." arXiv preprint arXiv:1812.06162, 2018. https://arxiv.org/abs/1812.06162.

Smith, Leslie N. "Cyclical Learning Rates for Training Neural Networks." In *IEEE Winter Conference on Applications of Computer Vision*, 2017. https://arxiv.org/abs/1506.01186.

Xiong, Ruibin, et al. "On Layer Normalization in the Transformer Architecture." In *Proceedings of the 37th International Conference on Machine Learning*, 2020. https://arxiv.org/abs/2002.04745.
