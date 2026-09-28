# 11.7 Choosing a Method

The previous sections introduced more algorithms than a project should run. This section is the selection rule. It has three parts: a table of what each method consumes and what it costs, two recipes that cover most of the goals in this book, and the curves that tell you the run you picked is learning the wrong thing.

## What each method needs

The columns that actually decide are the data you have, the number of large models you must keep in memory, and whether training samples from the policy. Stability in the last column is relative to PPO as Section 1 described it, not a promise that the loss cannot overfit.

| Method | Data | Large models in memory | Samples during training? | Stability | Typical use |
|---|---|---|---|---|---|
| PPO (Ch. 10) | Prompts, plus a reward model fit on pairs | Policy, value, reward, reference | Yes | Lowest; many interacting knobs | Classic RLHF; also RLVR when a critic is wanted (Tulu 3) |
| REINFORCE with a baseline | Prompts and a scorer | Policy, plus the scorer and an optional reference | Yes | Better than an untuned PPO; the step size can still collapse it | A simple on-policy baseline |
| RLOO | Prompts and a scorer; $`K`$ samples each | Same as REINFORCE | Yes, $`K`$ per prompt | The leave-one-out baseline removes one source of variance | Preference or reward optimization without a critic |
| GRPO | Prompts and a scorer; a group per prompt | Policy, reference, scorer | Yes, $`G`$ per prompt | Clip helps; tied groups and length need watching | Reasoning, math, code, agents |
| DPO | Chosen/rejected pairs | Policy and reference; the reference can be cached | No | High, until the pairs are far off-policy | Cheap alignment on existing pairs |
| IPO | Pairs | Same as DPO | No | High; the finite margin resists a kind of overfitting DPO has | Pairs that are clean and nearly deterministic |
| KTO | Binary labels, one response each | Policy and reference | No | High | Thumbs up and thumbs down, without pairs |
| ORPO | Pairs | Policy only | No | High; $`\lambda`$ has to balance SFT against the odds ratio | One stage, no reference model |
| SimPO | Pairs | Policy only | No | High; length is normalized explicitly | Chat, when DPO outputs grow too long |
| Online or iterative DPO | Prompts and a labeler | Policy, reference, labeler | Yes, to build the next pairs | The loss is stable; the labeler can still be gamed | When offline pairs no longer match the policy |
| Rejection sampling FT | Prompts and a scorer | The sampler and the scorer, then an SFT run | Yes, once, to build the dataset | The stability of SFT | A baseline, and a stage inside larger recipes |
| RLVR (any of the RL rows) | Problems with checkable answers | Whatever the RL method needs, minus a learned reward | Yes | Depends on the RL method; the reward itself is not a learned proxy | Math, code, format, instruction constraints |

Memory is the Section 10.7 accounting, repeated so the table can be used alone. A trained 7B model with Adam in mixed precision is about 112 GB before activations. A frozen 7B model is about 14 GB of weights. PPO with a separate critic, reward model, and reference is near 250 GB. GRPO with a frozen reference and a frozen reward model is near 140 GB, and GRPO with a programmatic verifier is near 126 GB. Offline DPO with cached reference log-probabilities is the policy alone, 112 GB. These numbers move with LoRA, sharding, and optimizer choice. The ordering does not.

A scorer in the table is whatever produces $`R`$: a reward model, a verifier, or a judge. It counts as a large model only when it is one.

## Two recipes

Most goals in this book fall into one of two shapes.

**Chat quality, and you have or can build preference pairs.** Supervised fine-tuning first, then DPO. SFT teaches the model to answer in the format the pairs assume; DPO then teaches it which of two answers to prefer. Use IPO instead of DPO if a pilot run drives the rejected log-probability down without a matching gain on a held-out judge. Use KTO if the feedback is thumbs rather than pairs. Use SimPO or ORPO if a second copy of the weights is the constraint, or if length is already growing under DPO. Move to online DPO, or to PPO or RLOO against a reward model, only when a held-out evaluation shows the offline policy has stopped matching freshly sampled answers. This is the recipe behind a large share of the chat models in Section 8, often with rejection sampling as an extra stage.

**Reasoning, and the answers can be checked.** Supervised fine-tuning on demonstrations that already contain long chains of thought, then GRPO (or RLOO) with a verifier. The SFT stage is optional in principle: R1-Zero skipped it (Section 5). It is useful in practice when the base model rarely samples a correct answer, so that every group is a tie at zero and RL has no gradient, or when the traces RL discovers are correct but unreadable. Keep the reward as a checker for as long as the task allows. Add a rubric-based judge, as in Section 6, only for the part of the data a checker cannot see. Add a KL penalty if the policy should stay near the SFT model; drop it, as several of the Section 8 systems do, if the behavior you want is a long trace the reference model almost never produced. The reverse KL makes that trace expensive (Section 10.6).

A third pattern is the composition of the first two, and it is what Tulu 3 and Olmo 3 report: SFT, then a preference loss such as DPO for general chat behavior, then RLVR for tasks with checkers. The preference stage and the verifier stage are not substitutes. Each is aimed at the labels it can see.

If you are unsure which recipe you are in, look at the data rather than at the papers. Pairs, and no programmatic notion of correct: the first recipe. A gold answer or a test suite: the second. A pile of unpaired thumbs: KTO, or spend the budget to turn a subset of them into pairs and use the first recipe. A reward model someone else trained, and no access to its training pairs: RLOO or GRPO against that reward, with a KL term and a held-out check that is not the reward itself.

## Diagnosing a run

The training loss, or the training reward, is the number the optimizer moves. It is not the number you are trying to move. Four curves, read together, catch the failures in this chapter. Log them against training steps and, for any KL-regularized method, against the KL divergence from the reference.

**The reward and a held-out evaluation.** For a learned reward, plot the reward-model score and a metric the reward model does not control: a human win rate, or a judge that was not the training labeler, on a fixed prompt set. The overoptimization curve of Gao et al. (2023) is a reward that keeps rising while the held-out metric rises and then falls. Stop at the peak of the held-out metric, or lower $\beta$'s willingness to chase the reward. For RLVR the two curves should be the same function on different problems. Training accuracy rising while held-out accuracy is flat is memorization. Both rising is the run you wanted.

**The KL divergence from the reference.** This is the x-axis on which two runs with different $\beta$ can be compared (Section 10.6). A KL that stays near zero means the constraint is too tight or the learning rate is too small; the policy is still the SFT model. A KL that jumps in a single iteration means the update is too large: in PPO or GRPO the clip is not binding, and in DPO the learning rate or $\beta$ is letting one batch move the log-probabilities a long way. For reasoning runs that intentionally drop the KL term, there is no reference curve to watch, and the held-out accuracy and the length have to carry that job.

**Response length.** Plot the mean number of tokens next to the reward. A length that tracks the reward, on a task whose true answers are not longer, is the length exploit of Section 10.5. DPO has a version of it through the sum of log-ratios (Section 4); SimPO and the length normalizations of Section 8 are the corresponding fixes. On a reasoning run, length *should* rise if the model is doing more useful work per answer. The distinguishing measurement is held-out accuracy per token, or accuracy at a fixed token budget. Longer and more accurate is test-time compute. Longer and equally accurate is padding.

**The pair statistics, for DPO and its variants.** Plot the mean log-probability of the chosen response, the mean log-probability of the rejected response, and the margin between the implicit rewards. The margin should increase, and the training accuracy of the pairwise ranking should increase. If both log-probabilities fall, you are looking at the likelihood displacement of Section 3: the margin is being purchased by abandoning both responses. That is the moment to raise $\beta$, add an SFT term on the chosen response, or drop pairs whose two sides are very similar. If the margin increases and a freshly sampled evaluation does not, the dataset is off-policy relative to the model you have now, and it is time for a round of online labels.

GRPO adds two diagnostics of its own. The fraction of groups in which every reward is equal should not be most of the batch; those groups contribute a zero advantage, and a run can look busy while the effective batch is a handful of prompts. The standard deviation of rewards within a group tells you whether standardization is amplifying the near-tie prompts, which is the bias Dr. GRPO removes. And the clip fraction means what it meant in Section 9.7. Near zero, the clip never binds and larger steps are available. Very high, most tokens are contributing no gradient and the update is wasting the batch.

Read generations. A curve cannot show that every answer now begins with the same sentence, or that the model agrees with a false premise the user planted, or that a correct boxed answer is attached to a nonsense derivation. A fixed set of prompts, sampled every few iterations and read, catches sycophancy, collapse of diversity, and verifier exploits that a scalar reward is defined to miss.

## A short decision procedure

1. If the task has a checker, start with SFT on any demonstrations you have, then GRPO or RLOO against the checker. Do not train a reward model for a quantity a program already computes.
2. If the task has pairs and no checker, start with SFT, then offline DPO at $\beta$ near $`0.1`$. Cache the reference log-probabilities.
3. If that DPO run's chosen log-probability falls, switch to IPO or add an SFT term. If the answers get longer and worse, try SimPO. If you never had pairs, use KTO.
4. If offline training stalls while the model still looks under-optimized on fresh samples, regenerate pairs from the current policy and continue, or move to GRPO against a reward model with an explicit KL term.
5. In every case, stop on the held-out metric, not on the training reward, and keep a small human audit on the failure modes of Section 1: hacking, diversity, and sycophancy.

Section 8 records which of these choices the public recipes actually made.

## Key takeaways

- Choose on the data. Pairs point to DPO and its variants, binary labels to KTO, checkable answers to GRPO or RLOO with a verifier, and a single scored sample to rejection sampling.
- The usual chat recipe is SFT then DPO. The usual reasoning recipe is SFT on long traces, then group-relative RL against a checker. Systems that need both run them as stages.
- Read reward against a held-out metric, KL against the reference, response length, and, for DPO, the chosen and rejected log-probabilities. A rising training reward is not one of the success criteria.
- Groups with zero advantage, a falling chosen likelihood, and length that rises while accuracy does not, are the three cheapest signs that the run has left the objective you meant.

## Further reading

Gao, Leo, et al. "Scaling Laws for Reward Model Overoptimization." In *Proceedings of the 40th International Conference on Machine Learning*, 2023. https://arxiv.org/abs/2210.10760.

Lambert, Nathan, et al. "Tulu 3: Pushing Frontiers in Open Language Model Post-Training." arXiv preprint arXiv:2411.15124, 2024. https://arxiv.org/abs/2411.15124.

Rafailov, Rafael, et al. "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2305.18290.
