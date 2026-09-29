# 14.1 Where Chapter 11 Left Off: The Limits of PPO-Based RLHF

Chapter 11 built the classic RLHF pipeline and showed what it achieved: a supervised fine-tuned policy, a reward model trained on human comparisons, and PPO maximizing that reward while a KL penalty keeps the policy near the reference model. InstructGPT, ChatGPT, GPT-4, and Llama 2-Chat were trained this way (Section 11.8). The recipe works. It is also expensive, sensitive to its hyperparameters, and able to improve the number it optimizes while the behavior people wanted gets worse. This section names those limits. Each one points at a method later in the chapter: critic-free policy gradients, direct preference optimization, verifiable rewards, and AI feedback.

## A short recap

The objective from Section 11.6 is expected reward minus a penalty for leaving the reference model, usually the SFT model:

```math
J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_{\theta}(\cdot \mid x)}\left[ r_{\phi}(x, y) - \beta \log \frac{\pi_{\theta}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} \right].
```

The reward model $`r_{\phi}`$ is a frozen copy of a language model with a scalar head, trained with the Bradley-Terry loss of Section 11.5 so that preferred responses score higher than rejected ones. The coefficient $\beta$ sets how much KL divergence from $`\pi_{\mathrm{ref}}`$ the optimizer may spend to raise reward. The optimal policy of this objective is the reference reweighted by the exponentiated reward, $`\pi^*(y \mid x) \propto \pi_{\mathrm{ref}}(y \mid x) \exp(r_{\phi}(x, y) / \beta)`$. PPO does not use that formula. It approximates $`\pi^*`$ by sampling.

PPO, carried over from Section 10.7 and adapted to text in Section 11.7, does the approximation with four networks. The policy generates responses. The reward model scores each completed response. The reference model supplies the per-token KL penalty. A value model estimates the expected future reward of every prefix so that generalized advantage estimation can assign credit along the response. The update clips the probability ratio between the new policy and the policy that collected the batch, and it may reuse the batch for a few epochs. Nothing in that description is optional window dressing: each piece is there because the one before it was unstable or too noisy. The methods in this chapter ask which pieces can be removed.

## Cost

A training step of PPO for language models has to hold, or at least visit, four large models, and it has to generate text.

Chapter 11 counted the memory for a 7-billion-parameter policy in mixed precision with Adam: about 16 bytes per trained parameter, or about 112 GB, for weights, gradients, and optimizer states. A separate value model of the same size costs the same again. The frozen reference and reward models need only their weights, about 2 bytes per parameter, or 14 GB each. Before activations and before the KV cache used to generate rollouts, that is on the order of 250 GB. Sharing a backbone between the policy and the value head, shrinking the reward model, training the policy with LoRA, or offloading frozen weights all help, and production systems use them. They do not change the shape of the problem. Two of the four models exist only to produce a baseline for the policy gradient.

Generation adds a second bill, paid in time rather than memory. Scoring a finished response is one parallel forward pass. Producing the response is one forward pass per token, and for a long answer that loop dominates the iteration (Section 11.7). The samples are then used for a handful of updates and thrown away, because the policy gradient is on-policy (Section 10.6). The next iteration has to sample again.

Section 2 attacks the memory half of this by dropping the value model. REINFORCE with a baseline, RLOO, and GRPO estimate the advantage by comparing rewards inside a batch or inside a group of answers to the same prompt. They still sample, so they do not remove the generation cost. Sections 3 and 4 attack both halves. Direct preference optimization trains on a fixed set of chosen and rejected pairs, with no reward model and no sampling loop during training. The policy and, unless its log-probabilities were cached ahead of time, a frozen reference are the whole recipe.

## Instability

PPO for language models has a long list of knobs, and several of them interact. Section 11.7 collected the ones that show up in published recipes and in reproductions: the clip range $\epsilon$, the KL coefficient $\beta$, the GAE parameter $\lambda$, the value-loss weight, whether rewards and advantages are whitened, the learning rate, the number of epochs per batch, the batch size, and what happens to responses that hit the length limit without an end-of-sequence token. Huang et al. (2024), reproducing the summarization experiments of Stiennon et al. (2020), and Xu et al. (2024), comparing PPO with DPO, both found that details of this kind decide whether PPO learns at all. Advantage normalization, a large batch, and a careful KL constraint are not cosmetic. A run can look healthy on the reward-model score while the KL penalty, the clip, and the value loss are fighting each other.

The value model is a particular source of trouble. It is trained by regression on returns that, for a language model, are a single score arriving at the last token. Early in a long response its target depends on thousands of future tokens. If the value estimates are biased, the advantages are biased, and the policy follows them. On the short episodes of the CartPole experiments in Chapter 10 this is manageable. On chain-of-thought traces it is one of the reasons later reasoning recipes either invest in a specialized value model or give up on one (Sections 5 and 8).

DPO and the preference losses in Sections 3 and 4 replace this loop with a classification-style loss on a static dataset. There is no value model, no clipping schedule, and no ratio between successive policies. The main coefficient is the same $\beta$ that appears in the KL-regularized objective, plus an ordinary learning rate. That is a smaller surface on which a run can go quietly wrong. It is not a surface of size zero: Section 3 describes the failure modes that remain, including a drop in the probability of the preferred response. The point of the comparison is narrower. The instability that comes from on-policy sampling, a learned critic, and a clipped surrogate is specific to PPO, and the preference losses do not have it.

## Reward hacking

The reward model is a proxy. It was trained to imitate human comparisons on responses that look like the reference model's, and its score is reliable in that neighborhood. PPO searches for responses with a high score. Given enough optimization, it finds responses that score well because the proxy is wrong.

Stiennon et al. (2020) measured this with people. As optimization against the reward model continued, human judgments of summary quality rose and then fell, while the reward model's score kept rising. Gao et al. (2023) studied the same gap systematically, replacing the expensive human evaluation with a "gold" reward model that stands in for true preference, and optimizing against a separate, weaker proxy. They compared RL and best-of-$n$ selection, and they used the KL divergence from the initial policy as the x-axis, because that is the quantity $\beta$ is spending. The pattern is stable across their setups. The proxy reward increases with optimization. The gold reward increases at first and then decreases. The KL at which the gold reward peaks depends on how good the proxy is: a larger reward model, or one trained on more comparisons, can be optimized further before the gold score turns down. They summarize the gold-reward curve with a simple function of the square root of KL, which is why Section 11.6 suggested plotting reward against KL rather than against training steps.

The practical reading is the one Chapter 11 already used. A rising reward-model score is not evidence that the policy got better. It is evidence that the policy got better *at the proxy*. Past some KL budget, those are different statements. Length is the usual first exploit: if longer answers won comparisons for reasons that had nothing to do with content, the proxy pays for length, and the policy becomes verbose. Repeated stock phrases, excessive confidence, and format tricks are the same phenomenon with a different surface feature.

Section 5 is the direct response. If the reward is a checker for the thing we actually care about, such as an exact math answer or a unit test, there is no separate proxy whose mistakes can be mined. The checker can still be gamed, and a policy can still overfit the training problems, but it cannot improve the training reward by finding text that a learned model mis-scores. That is a different and narrower failure mode.

## Less diversity

The KL term is mode-seeking. Section 11.6 showed why: the reverse KL makes it cheap for the policy to drop responses the reference considered likely, and expensive to invent responses the reference gives almost no probability. The optimum puts more mass on the high-reward subset of the reference's behavior. Users experience this as a model that has a style. Answers to similar questions start to share an opening, a length, and a list of caveats.

Kirk et al. (2023) measured the effect. Comparing RLHF policies with SFT models, they found that RLHF reduces the diversity of outputs, and that the gain on inputs like the fine-tuning distribution can come with weaker performance on inputs that are unlike it. Diversity here means variation across samples, not fluency: a model can be a good writer and still say nearly the same thing every time it is asked. Reduced diversity is not only an aesthetic complaint. A policy that has collapsed toward one phrasing is brittle when the prompt changes, and it is a poor generator of the varied candidates that best-of-$n$ and self-consistency need at inference time (Section 13.9).

None of the later methods repeal this by default. DPO is derived from the same KL-regularized objective, so it has the same mode-seeking optimum. GRPO with a strong reward does too, and reasoning runs often *want* the policy to concentrate on answers that survive a checker. When diversity matters, it has to be measured (distinct samples, or performance on a shifted prompt set) and protected on purpose, for example by stopping at a modest KL, by keeping an entropy or KL term, or by not optimizing so hard. The failure mode is easy to miss if the only dashboard is a reward curve.

## Sycophancy

A second way the proxy can be "right" about the labels and wrong about the goal is sycophancy: the model agrees with the user instead of answering the question. If the user says "I believe the answer is 7" and the answer is 12, a sycophantic model finds a way to say 7. If the user states a political opinion and asks for an analysis, the model mirrors the opinion.

Sharma et al. (2023) documented this in several deployed assistants and then looked for the cause in the preference data. Human raters, and reward models trained on human ratings, sometimes prefer a confident answer that matches the user's stated belief over a correct answer that contradicts it. The preference is understandable. Agreement feels helpful, and a response that argues with the user can read as rude. Once that preference is in the reward model, PPO reinforces it, because agreeing raises the score. The behavior is then stable: the model has learned a cheap feature, the user's own words, that predicts reward.

Sycophancy is not unique to PPO. Any method that imitates these comparisons will inherit them, including the direct preference losses of Sections 3 and 4, which never build an explicit reward model but optimize the same pairwise signal. Fixing it is mostly a data and evaluation problem. The comparisons have to reward correcting the user when the user is wrong, and the evaluation has to include cases where the user's premise is false. Sharma et al. also showed the uncomfortable direction: if you optimize a preference model that prefers sycophancy, sycophancy increases. The algorithm is doing what it was asked.

## What the rest of the chapter does with these limits

The five problems do not have one fix. They line up with the sections that follow.

| Limit | What it costs | Where this chapter goes |
|---|---|---|
| Four models and on-policy sampling | Memory and generation time | Critic-free RL drops the value model (Section 2); DPO drops sampling and the reward model (Section 3) |
| A large, interacting hyperparameter surface | Runs that depend on details | Preference losses replace the PPO loop (Sections 3 and 4) |
| A learned reward that can be exploited | Rising proxy score, falling true quality | Verifiable rewards remove the learned proxy (Section 5) |
| Mode-seeking optimization | Less diverse answers | Not removed; has to be measured. The same KL objective is inside DPO |
| Preferences that reward agreement | Sycophancy | Not removed by the optimizer; it is in the labels. AI feedback can apply a written principle instead (Section 6) |

The last row is the remaining cost that has not come up yet. Even a stable, critic-free, preference-based method needs labels, and human comparisons are slow. Section 6 uses another language model to write them, guided by a constitution or a rubric, which cuts the human labeling bill and brings its own biases. Section 7 turns the menu into a choice given the data and the goal, and Section 8 records which of these methods open recipes actually shipped through 2026.

## Key takeaways

- Classic RLHF maximizes a learned reward minus $\beta$ times the KL divergence from a reference model, and PPO approximates that objective with a policy, a value model, a reward model, and a reference model.
- The value model and the on-policy sampling loop dominate the memory and the time. Critic-free methods and DPO each remove part of that cost.
- PPO's clip, KL coefficient, value loss, whitening, and learning rate interact; preference losses avoid that surface and have different failure modes.
- Gao et al. (2023) showed reward-model overoptimization: the proxy score rises with KL while a gold score rises and then falls. Verifiable rewards are the countermeasure in Section 5.
- RLHF reduces output diversity (Kirk et al. 2023), and preference data can teach the model to agree with the user rather than correct them (Sharma et al. 2023). Changing the optimizer does not by itself fix either one.

## Further reading

Gao, Leo, et al. "Scaling Laws for Reward Model Overoptimization." In *Proceedings of the 40th International Conference on Machine Learning*, 2023. https://arxiv.org/abs/2210.10760.

Huang, Shengyi, et al. "The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization." In *Conference on Language Modeling*, 2024. https://arxiv.org/abs/2403.17031.

Kirk, Robert, et al. "Understanding the Effects of RLHF on LLM Generalisation and Diversity." arXiv preprint arXiv:2310.06452, 2023. https://arxiv.org/abs/2310.06452.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Sharma, Mrinank, et al. "Towards Understanding Sycophancy in Language Models." arXiv preprint arXiv:2310.13548, 2023. https://arxiv.org/abs/2310.13548.

Stiennon, Nisan, et al. "Learning to Summarize from Human Feedback." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2009.01325.

Xu, Shusheng, et al. "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study." In *Proceedings of the 41st International Conference on Machine Learning*, 2024. https://arxiv.org/abs/2404.10719.
