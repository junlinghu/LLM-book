# 14.3 Direct Preference Optimization (DPO)

PPO spends a sampling loop and four models on an optimization problem that has a closed form. Section 11.6 already wrote the solution down: under the KL-regularized objective, the best policy is the reference model reweighted by the exponentiated reward. Direct Preference Optimization (DPO) inverts that solution (Rafailov et al. 2023). Instead of learning a reward model and then chasing it with RL, it writes the reward in terms of the policy and fits the policy to the preference pairs directly. The result is a classification loss. There is no reward model, no value model, and no sampling during training.

## The optimal policy, again

Start from the objective of Section 11.6, for a single prompt, with the reward $`r(y)`$ treated as fixed:

```math
\max_{\pi}\ \mathbb{E}_{y \sim \pi}\left[ r(y) \right] - \beta\, D_{\mathrm{KL}}\left[ \pi \,\|\, \pi_{\mathrm{ref}} \right].
```

Section 11.6 showed that the unique maximizer is

```math
\pi^*(y) = \frac{1}{Z}\, \pi_{\mathrm{ref}}(y)\, \exp\!\left(\frac{r(y)}{\beta}\right), \qquad Z = \sum_{y} \pi_{\mathrm{ref}}(y)\, \exp\!\left(\frac{r(y)}{\beta}\right).
```

The partition function $`Z`$ sums over every possible response, so the formula is not something we can compute. DPO's move is to solve it for the reward instead of for the policy. Take the logarithm and rearrange:

```math
r(y) = \beta \log \frac{\pi^*(y)}{\pi_{\mathrm{ref}}(y)} + \beta \log Z.
```

The optimal policy is an implicit reward model. The first term is a log-ratio of two distributions we can evaluate: the policy we are training, and the frozen reference. The second term depends on the prompt, through $`Z`$, but not on which response we are scoring.

## The reward cancels in a comparison

Preference data, as in Section 11.5, come in triples $`(x, y_w, y_l)`$: a prompt, a chosen response, and a rejected one. The Bradley-Terry model says the probability that $`y_w`$ wins is a sigmoid of the reward gap,

```math
P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big).
```

Substitute the expression for $`r`$. The two copies of $`\beta \log Z(x)`$ cancel:

```math
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}.
```

The impossible sum $`Z(x)`$ is gone. The preference probability under the optimal policy is

```math
P^*(y_w \succ y_l \mid x) = \sigma\left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \right).
```

This is the same Bradley-Terry loss as Section 11.5, with a reward that is no longer a separate network. Define the **implicit reward** of a policy $`\pi_{\theta}`$ by

```math
\hat{r}_{\theta}(x, y) = \beta \log \frac{\pi_{\theta}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}.
```

Then $`\hat{r}_{\theta}`$ is exactly the reward whose KL-regularized optimum is $`\pi_{\theta}`$, up to a prompt-level shift that comparisons ignore. "Your language model is secretly a reward model" is this identity, not a metaphor.

## The DPO loss

Maximum likelihood on a dataset of pairs asks $`\pi_{\theta}`$ to make the observed winners probable under the Bradley-Terry model above. The loss on one pair is the negative log-probability of the observed preference, and the training objective is its expectation:

```math
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\,\mathbb{E}_{(x, y_w, y_l)}\left[ \log \sigma\Big( \hat{r}_{\theta}(x, y_w) - \hat{r}_{\theta}(x, y_l) \Big) \right].
```

Written out in log-probabilities, which is the form to implement:

```math
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\,\mathbb{E}\left[ \log \sigma\left( \beta \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \right) \right].
```

Both log-probabilities are sums over tokens, as in every language-model loss in this book, and the prompt tokens are not part of either sum. The reference log-probabilities are constants. They can be computed once, before training, and stored with the dataset. After that, a DPO step is two forward passes of the policy (chosen and rejected), a sigmoid, and a backward pass. It looks like supervised fine-tuning.

The gradient of the loss with respect to the policy parameters has a useful form. Let $`\Delta_{\theta} = \hat{r}_{\theta}(x, y_w) - \hat{r}_{\theta}(x, y_l)`$ be the implicit margin. Then

```math
\nabla_{\theta} \mathcal{L}_{\mathrm{DPO}} = -\,\mathbb{E}\left[ \beta\, \sigma\big(-\Delta_{\theta}\big)\, \big( \nabla_{\theta} \log \pi_{\theta}(y_w \mid x) - \nabla_{\theta} \log \pi_{\theta}(y_l \mid x) \big) \right].
```

The weight $`\sigma(-\Delta_{\theta})`$ is the probability the current implicit reward assigns to the *wrong* order. A pair the model already ranks correctly, by a wide margin, contributes almost nothing. A pair it ranks backwards contributes a weight near one. The update raises the log-probability of the chosen response and lowers the log-probability of the rejected one, in proportion to how wrong the ranking is. This is the same weighting the reward-model loss uses in Section 11.5. DPO has absorbed that loss into the policy.

A numerical pair shows the quantities before any update. Suppose $`\beta = 0.1`$, the policy's log-probabilities are $`\log \pi(y_w) = -1.20`$ and $`\log \pi(y_l) = -0.80`$, and the reference's are $`-1.00`$ and $`-1.10`$. The policy currently finds the rejected response more likely than the chosen one. The implicit rewards are

```math
\hat{r}(y_w) = 0.1 \times (-1.20 - (-1.00)) = -0.020, \qquad \hat{r}(y_l) = 0.1 \times (-0.80 - (-1.10)) = 0.030.
```

The margin is $`-0.050`$, so the model assigns probability $`\sigma(-0.050) \approx 0.488`$ to the observed preference. The loss is $`-\log \sigma(-0.050) \approx 0.718`$, a little worse than the $`\log 2 \approx 0.693`$ of a tie, and the gradient weight is $`\sigma(0.050) \approx 0.512`$. [Code 14.3.1](#code-1431-the-dpo-loss-on-one-pair) computes these values and then walks a three-way softmax, initialized at a tie, for a few steps of gradient descent. The chosen response's probability rises, the rejected response's falls, and the implicit margin grows from 0 to about 0.025 in five steps of size 0.5. The third response, which appears in neither side of the pair, barely moves. That last fact is not always true, and the weakness section explains when it fails.

## The role of beta and the reference model

The coefficient $\beta$ is the same coefficient as in the KL-regularized objective. In the closed form it is a temperature: large $\beta$ keeps $`\pi^*`$ close to the reference, small $\beta$ lets the reward dominate. In the loss it scales the log-ratio gap before the sigmoid. A large $\beta$ makes the sigmoid saturate as soon as the log-ratios differ by a little, so the weight $`\sigma(-\Delta)`$ collapses and learning stops while the policy is still near the reference. A small $\beta$ keeps the weight large even after the policy has moved, and the optimizer keeps pushing the chosen and rejected responses apart. The DPO paper's experiments use values around $`\beta = 0.1`$, which became the usual default. It is not portable across datasets any more than the RLHF $\beta$ was. A dataset of noisy, nearly tied pairs needs a gentler setting than a dataset of obvious preferences, and the right check is the same one as in Section 11.6: how far the policy's log-probabilities have moved from the reference, not the value of $\beta$ in isolation.

The reference model is the policy's anchor, and it is usually the SFT model the policy was initialized from. At initialization, $`\pi_{\theta} = \pi_{\mathrm{ref}}`$, every implicit reward is zero, and the loss on every pair is $`\log 2`$. Training moves the policy off that point. The reference is what makes "move" a well-defined direction. Without it, the loss would be a softmax over two responses with nothing saying how the rest of the vocabulary should behave, which is the gap the reference-free losses in Section 4 fill in other ways.

Two practical consequences follow. First, the policy should start at the reference. Initializing DPO from a base model, with a reference that is also the base model, asks the loss to teach instruction following and preferences at once; it can, but the pairs have to carry all of that, and the usual recipe is SFT first. Second, because $`\pi_{\mathrm{ref}}`$ is frozen, its log-probabilities on a fixed offline dataset do not change. Caching them turns DPO into a one-model fine-tune. On the memory accounting of Section 11.7, that is the policy's 112 GB and nothing else, against roughly 250 GB for PPO with a separate 7B critic, reward model, and reference.

## What DPO is good at

The appeal is the list of things the training loop does not contain.

**It is a supervised loss.** There is no environment, no rollout, no ratio between successive policies, and no value target. The tools of Chapter 9 (masking, packing, learning-rate schedules, LoRA) apply without translation. A run that diverges looks like a fine-tune that diverges, not like an RL run whose reward, KL, clip fraction, and value loss must be read together.

**It does not sample during training.** The pairs are given. Generation, which dominated PPO's iteration time in Section 11.7, happens only if and when someone builds the dataset. For a research group that already has preference data, this is the difference between a fine-tune and a distributed RL system.

**It does not train a reward model.** The policy is the reward model. There is no second Bradley-Terry fit whose errors the policy can then exploit as a proxy, in the sense of a separate network that stays frozen while the policy drifts off its training distribution. This does not mean DPO cannot overfit. It means the overfit, when it happens, is the policy overfitting the pairs, which is the weakness below.

Rafailov et al. (2023) showed that this loss can match PPO-based RLHF on the summarization, dialogue, and sentiment tasks in their experiments, at a small fraction of the implementation cost. Later controlled comparisons are more mixed, and Section 11.8 already reported them: Xu et al. (2024) and Ivison et al. (2024) found that a well-tuned PPO can outperform DPO, and that the preference data often matter more than the choice of algorithm. DPO's claim is not that it dominates the reward-KL frontier. Its claim is that the frontier is reachable without the RL machinery, on data that were collected from a policy close to the one being trained.

## Where DPO is weak

**The data are offline.** The derivation says that $`\pi_{\theta}`$ is the optimum for the reward implicit in the preferences, but the preferences were collected from some other policy: the SFT model, a previous checkpoint, or a stronger model entirely. If $`\pi_{\theta}`$ moves far from the policy that produced $`y_w`$ and $`y_l`$, it is fitting a reward on a distribution it no longer samples, which is the same off-policy gap PPO was designed to avoid. Xu et al. (2024) identify this distribution shift as a main reason DPO stalls or loses to PPO on tasks where the policy needs to wander, including code. The symptom is a training loss that falls while evaluations on freshly sampled answers do not. Section 4's online and iterative variants regenerate the pairs from the current model so the derivation's assumption holds again.

**The loss trusts the pairs.** There is no reward model whose calibration can be inspected, and there is no KL budget plotted against a gold score in the sense of Gao et al. (2023), because there is no separate score. Label noise, length bias, and sycophancy in the comparisons (Section 1) go straight into the policy. A pair that prefers a long, agreeable, wrong answer teaches the model to produce one. Filtering the dataset is the corresponding lever: DPO has fewer training tricks than PPO and fewer excuses for bad labels.

**Both likelihoods can fall.** The loss depends on the margin $`\hat{r}(y_w) - \hat{r}(y_l)`$, not on either reward alone. Any change that lowers the rejected response's probability more than the chosen response's improves the loss, even if the chosen response also becomes less likely. In the three-way softmax of [Code 14.3.1](#code-1431-the-dpo-loss-on-one-pair) this did not happen, because the chosen and rejected responses had independent logits and the gradient had no reason to touch the third one. Real responses share parameters.

[Code 14.3.2](#code-1432-likelihood-displacement-when-responses-share-a-direction) is a small model of that sharing. Three responses get scores from vectors dotted with one parameter vector. The chosen and rejected vectors point almost the same way, and a third vector points elsewhere. Gradient descent on the DPO loss increases the margin, as it should: after seven steps the log-ratio gap $`h`$ has moved from 0 to about 0.030. Over those same steps the log-probability of the chosen response moves from about $`-0.987`$ to about $`-0.999`$, and the rejected response's from about $`-1.002`$ to about $`-1.043`$. Both fell. The third response's log-probability rose, from about $`-1.347`$ to about $`-1.275`$. The optimizer satisfied the pairwise constraint by draining probability from the pair into a response that was in neither label.

Razin et al. (2025) call this *likelihood displacement* and show that it is not a toy pathology. When the chosen and rejected responses induce similar hidden states, the update that separates them can move probability onto a third response with a different meaning. In one of their experiments, preference pairs whose two sides were both refusals reduced Llama-3-8B-Instruct's refusal rate on SORRY-Bench from 74.4% to 33.4%: the model became less likely to say either refusal, and more likely to comply. Their practical warning matches the arithmetic. Pairs in which $`y_w`$ and $`y_l`$ are near-paraphrases are exactly the pairs DPO is most willing to satisfy by abandoning both. An SFT term on the chosen response, a larger $\beta$, or simply dropping pairs whose two sides are too alike, all limit the damage. Section 4's IPO loss faces a related version of the same issue, a margin with no finite target, and fixes that version differently.

## Code for this section

### Code 14.3.1: The DPO loss on one pair

The first half evaluates the implicit rewards on the numerical pair from this section. The second half optimizes a three-way softmax (chosen, rejected, other) from a tie, for five steps. The reference is the initial policy, so the implicit margin starts at zero.

Notebook: [14.3.1-the-dpo-loss-on-one-pair.ipynb](../../code/14-more-rl-methods-for-llms/14.3.1-the-dpo-loss-on-one-pair.ipynb)

### Code 14.3.2: Likelihood displacement when responses share a direction

The chosen and rejected responses have nearly aligned score vectors, and a third response points somewhere else. The DPO margin rises while both labeled responses lose probability.

Notebook: [14.3.2-likelihood-displacement-when-responses-share-a-direction.ipynb](../../code/14-more-rl-methods-for-llms/14.3.2-likelihood-displacement-when-responses-share-a-direction.ipynb)

## Key takeaways

- The KL-regularized optimum is $`\pi^* \propto \pi_{\mathrm{ref}} \exp(r / \beta)`$. Solving for $`r`$ gives $`r(y) = \beta \log(\pi^*(y) / \pi_{\mathrm{ref}}(y)) + \beta \log Z`$.
- In a Bradley-Terry comparison the partition function cancels, so the preference probability depends on the policy only through the implicit reward $`\hat{r}_{\theta}(x, y) = \beta \log(\pi_{\theta}(y \mid x) / \pi_{\mathrm{ref}}(y \mid x))`$.
- The DPO loss is binary cross-entropy on that margin. Pairs the model already ranks correctly get a small gradient; pairs it ranks backwards get a large one.
- $\beta$ is the KL temperature. The reference is usually the SFT model, and its log-probabilities on a fixed dataset can be cached, leaving one network to train.
- DPO is offline and only constrains a margin. The policy can leave the distribution the pairs came from, and the probabilities of both the chosen and the rejected response can fall together.

## Further reading

Ivison, Hamish, et al. "Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback." In *Advances in Neural Information Processing Systems 37*, 2024. https://arxiv.org/abs/2406.09279.

Rafailov, Rafael, et al. "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2305.18290.

Razin, Noam, et al. "Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization." In *International Conference on Learning Representations*, 2025. https://arxiv.org/abs/2410.08847.

Xu, Shusheng, et al. "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study." In *Proceedings of the 41st International Conference on Machine Learning*, 2024. https://arxiv.org/abs/2404.10719.
