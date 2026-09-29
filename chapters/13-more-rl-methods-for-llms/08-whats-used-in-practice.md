# 13.8 What's Used in Practice

Sections 13.2 through 13.6 described the methods. This section records which of them public reports actually used, as of September 2026. The picture is more specific than "PPO was replaced by DPO." PPO is no longer the default optimizer in open post-training reports. DPO did not take its place as the final stage. The usual final stage, when a report names one, is critic-free group-relative RL, most often GRPO or a close variant, with a checker where the task allows one and a judge where it does not. DPO remains a cheap stage before or after that RL, not a substitute for it.

The table attributes an algorithm to a model only when the developers named it. Closed labs often confirm that they ran large-scale RL and do not say which optimizer. Those rows say so, and the last part of the section explains why guessing further is not a technical claim.

## The shift away from PPO

From 2019 through 2023, the named optimizer in a post-training report was PPO. Section 10.8 covers that period: InstructGPT, ChatGPT, GPT-4's documented RLHF, Llama 2-Chat, and Anthropic's published helpful-and-harmless research. The reports from late 2024 onward, where they name an algorithm, mostly describe a group baseline instead of a learned value model. The reasons are the ones Section 1 listed. A separate critic is a second trained model of policy size, long chain-of-thought traces make the value targets late and noisy, and a group of verifiable scores already contains a baseline. GRPO, introduced for math by Shao et al. (2024) and used for DeepSeek-R1 (DeepSeek-AI 2025), is the version that spread. RLOO and other REINFORCE-style updates are the same idea with less clipping (Ahmadian et al. 2024).

"Most open reports" is not "all reports," and it is not "PPO does not work." Section 10.8 cited controlled comparisons in which a tuned PPO beat DPO. The shift is in what teams ship when they write the recipe down, not in a proof that a value model is dominated.

## PPO has not disappeared

Three lines of public evidence keep a value model in the story.

Classic RLHF used PPO, and nothing in this chapter rewrites that. InstructGPT, ChatGPT, GPT-4, and Llama 2-Chat are documented as PPO-based RLHF in Section 10.8. Anyone reproducing those systems is reproducing PPO.

Verifiable rewards were also trained with PPO before GRPO became the default. Tulu 3's final stage is RLVR with PPO (Lambert et al. 2024). OLMo 2 follows that recipe, with PPO for the 7B and 13B models, the value function initialized from the reward models, and GRPO for the 1B and 32B models (Team OLMo 2024). A checker does not require a critic. These systems used one anyway, and they are part of the record.

Value-based PPO was then reworked for long chain-of-thought rather than abandoned. VAPO is a PPO variant that keeps a value model and is aimed at the critic's failure modes on long traces: value bias, mixed sequence lengths, and sparse rewards (Yue et al. 2025). Chapter 10 notes the result they report, a score of 60.4 on AIME 2024 from a Qwen2.5-32B base model. ByteDance's Seed1.5-Thinking reasoning model uses a PPO-style actor-critic with a value model, drawing on VAPO and on the critic-free method DAPO (ByteDance Seed 2025). For a team that wants credit assignment inside a long trace, a value model is still a live option. It is no longer the option a recipe uses without comment.

## DPO's place

DPO shows up in the table as a stage, usually offline, usually next to SFT, and usually not as the run that teaches a model to solve harder math. Llama 3 chose rejection sampling, SFT, and DPO over PPO-style RL, describing that choice as more stable and easier to scale (Grattafiori et al. 2024). Qwen2.5 runs offline DPO and then online GRPO with a reward model (Qwen Team 2024). Tulu 3 runs length-normalized DPO and then RLVR with PPO (Lambert et al. 2024). Llama 4's report describes lightweight SFT, an online RL stage whose algorithm is not named, and then lightweight DPO (Meta AI 2025). Olmo 3 runs SFT, a DPO variant it calls delta learning, and then RLVR (Team Olmo 2025).

The pattern is consistent with Section 7. DPO is the cheap way to use a file of preferences. It is stable, it does not sample, and it does not need a value model. It is a poor fit, used alone, for a task whose signal is a checker and whose useful behavior is a long trace the preference dataset never contained. Reports that need both behaviors run both stages.

## A snapshot of named recipes

The table lists publicly documented post-training methods. "Not named" means the report confirms RL, or a preference stage, and does not name the policy-gradient algorithm. Sources are the technical reports and blogs cited in the chapter's further reading.

| Model | Year | Post-training RL method(s) | Source |
|---|---|---|---|
| Llama 3 | 2024 | Rejection sampling, SFT, and DPO; chose these over PPO-style RL as more stable and easier to scale | Grattafiori et al. 2024 |
| Qwen2.5 | 2024 | Offline DPO, then online GRPO with a reward model | Qwen Team 2024 |
| Tulu 3 | 2024 | SFT, length-normalized DPO, then RLVR trained with PPO | Lambert et al. 2024 |
| OLMo 2 | 2024 | Tulu 3 recipe; RLVR with PPO (7B, 13B) and GRPO (1B, 32B) | Team OLMo 2024 |
| DeepSeek-V3 | 2024 | GRPO with rule-based and model-based reward models | DeepSeek-AI 2024 |
| DeepSeek-R1 | 2025 | GRPO with rule-based (verifiable) rewards; R1-Zero skips SFT | DeepSeek-AI 2025 |
| Kimi k1.5 | 2025 | Variant of online mirror descent (critic-free policy gradient) | Kimi Team 2025 |
| Gemma 3 | 2025 | Distillation, then RL based on improved BOND, WARM, and WARP | Gemma Team 2025 |
| Llama 4 | 2025 | Lightweight SFT, online RL, then lightweight DPO; RL algorithm not named | Meta AI 2025 |
| Seed1.5-Thinking (ByteDance) | 2025 | PPO-style actor-critic with a value model, using techniques from VAPO and DAPO | ByteDance Seed 2025 |
| Qwen3 | 2025 | GRPO for reasoning RL; on-policy distillation for small models; GSPO credited for later Qwen3 models | Yang et al. 2025; Zheng et al. 2025 |
| Magistral (Mistral) | 2025 | Modified GRPO: no KL term, length-normalized loss, simplified advantages | Mistral AI 2025 |
| MiniMax-M1 | 2025 | **CISPO** (clips importance-sampling weights instead of token updates) | MiniMax 2025 |
| Gemini 2.5 | 2025 | RL with verifiable rewards and model-based generative rewards; algorithm not disclosed | Comanici et al. 2025 |
| Kimi K2 | 2025 | K1.5 policy optimization; verifiable rewards plus self-critique rubric reward | Kimi Team 2025 |
| GLM-4.5 | 2025 | GRPO without the KL term | GLM-4.5 Team 2025 |
| gpt-oss (OpenAI) | 2025 | "CoT RL techniques" similar to o3; algorithm not named | OpenAI 2025 |
| DeepSeek-V3.2 | 2025 | GRPO scaled up (unbiased KL estimate, off-policy sequence masking) in one mixed RL stage | DeepSeek-AI 2025 |
| Olmo 3 | 2025 | SFT, DPO (delta learning), then RLVR with OlmoRL (GRPO with DAPO and Dr. GRPO fixes) | Team Olmo 2025 |
| Nemotron 3 Nano (NVIDIA) | 2025 | Synchronous GRPO with masked importance sampling | NVIDIA 2025 |
| Kimi K2.5 | 2026 | Critic-free clipped policy gradient with mean-reward baseline; parallel-agent RL (PARL); generative reward models | Kimi Team 2026 |
| GLM-5 | 2026 | GRPO with IcePop; reasoning, agentic, then general RL; on-policy cross-stage distillation | GLM-5 Team 2026 |
| Qwen3.5 | 2026 | Large-scale asynchronous RL across agent environments; algorithm not named in release blog | Qwen Team 2026 |
| Muse Spark (Meta) | 2026 | Large-scale RL with a thinking-time penalty; algorithm not disclosed | Meta AI 2026 |
| DeepSeek-V4 | 2026 | Domain experts trained with SFT then GRPO, merged by on-policy distillation (reverse KL) | DeepSeek-AI 2026 |
| MiniMax-M2 series | 2026 | CISPO, adapted to agentic RL | Chen et al. 2026 |

Two entries are easy to misread. Gemma 3's report describes distillation followed by RL based on improved BOND, WARM, and WARP; it does not describe PPO or GRPO (Gemma Team 2025). Kimi k1.5 describes a critic-free online policy gradient related to mirror descent, not a value model (Kimi Team 2025). Both are outside the PPO recipe. They are not GRPO either, and the table keeps the name the authors used.

## The GRPO family

Once GRPO was the starting point, the papers of 2025 and 2026 changed one term of it at a time. The names below are the ones the table's reports cite. They share a group of sampled responses and a baseline computed from that group. They differ in the clip, the average, and whether a KL term is present.

**DAPO** (Yu et al. 2025) makes four changes, aimed at long chain-of-thought runs.

- *Clip-higher.* PPO's interval $`[1 - \epsilon,\ 1 + \epsilon]`$ is split into $`[1 - \epsilon_{\mathrm{low}},\ 1 + \epsilon_{\mathrm{high}}]`$ with $`\epsilon_{\mathrm{high}} > \epsilon_{\mathrm{low}}`$. In the DAPO runs, $`\epsilon_{\mathrm{low}} = 0.2`$ and $`\epsilon_{\mathrm{high}} = 0.28`$. The lower clip still stops the policy from crushing a token's probability in one update. The upper clip is looser, so a token that is currently rare, which is the usual state of a useful reasoning step the model almost never samples, can be raised by more than the old $`1.2`$ ceiling. Yu et al. report that the symmetric clip was driving entropy down until the policy stopped exploring.
- *Dynamic sampling.* Groups with accuracy 0 or 1 have zero advantage. DAPO keeps sampling until the batch is filled with groups that are mixed, so every optimization step is spent on prompts that produce a gradient.
- *Token-level loss.* The original GRPO objective averages tokens inside a response and then averages responses, so each response has equal weight regardless of length. DAPO averages over tokens in the batch. A long response contributes more terms.
- *Overlong reward shaping.* Truncating a response at the length limit and then scoring the fragment adds noise, because the checker is looking at an unfinished answer. DAPO masks truncated samples or applies a soft penalty that grows as the response enters a punishment interval before the hard limit, so the policy is told to finish rather than told that an unfinished answer was wrong.

The same report gives a concrete scale for the recipe: with Qwen2.5-32B as the base model, their system reaches 50 points on AIME 2024. That number is a property of one open recipe, not of DAPO in isolation, and the paper's own ablations are the right place to see which of the four changes moved it.

**Dr. GRPO** (Liu et al. 2025) removes two normalizers from the GRPO objective: the division by the per-response length, and the division by the group's reward standard deviation. The second one is a difficulty bias, and a small group shows it. Take eight samples and a binary reward. If seven are correct, the group standard deviation is about $`0.331`$, a correct sample's centered reward is $`+0.125`$, and its z-score advantage is about $`+0.38`$. The single wrong sample gets a z-score of about $`-2.65`$. If only one of the eight is correct, that correct sample's z-score is about $`+2.65`$, against a centered reward of $`+0.875`$. Relative to a balanced group, whose correct samples have z-score $`+1`$ and centered reward $`+0.5`$, standardization amplifies the prompts whose rewards barely vary. Liu et al. (2025) also argue that the length normalizer interacts with this advantage so that incorrect responses grow during R1-Zero-style training. Their fix is to subtract the group mean and not divide, and to normalize the token average by a constant such as the generation budget rather than by the realized length, which recovers an unbiased policy gradient with a Monte Carlo baseline. Olmo 3's OlmoRL is GRPO with the DAPO and Dr. GRPO fixes applied (Team Olmo 2025).

**VAPO** is the value-based counterpart already mentioned. Where Dr. GRPO and DAPO delete or repair the critic-free baseline, VAPO keeps the critic and changes how it is trained for long, sparse traces (Yue et al. 2025). Seed1.5-Thinking is the public model that says it used that line together with DAPO.

**CISPO** (MiniMax 2025) changes what the clip does to the gradient. In PPO and GRPO, once a token's ratio leaves the clip interval in the direction the advantage wants, the minimum selects a constant and the gradient on that token is zero. CISPO clips the importance weight, stops the gradient from flowing through the weight, and multiplies the policy's log-probability by that constant. A common form caps the weight from above:

```math
\mathcal{L}_{\mathrm{CISPO}} = -\,\mathbb{E}\left[ \mathrm{sg}\big(\min(\rho_t,\ 1 + \epsilon)\big)\, \hat{A}_t\, \log \pi_{\theta}(a_t \mid s_t) \right].
```

A wild ratio cannot dominate the step, because the coefficient is capped. The coefficient does not depend on $\theta$, so every token still receives a gradient through $`\log \pi_{\theta}`$, including tokens that PPO would have dropped. MiniMax-M1 uses this for long generations; the MiniMax-M2 series adapts it to agentic RL (Chen et al. 2026).

**GSPO** (Zheng et al. 2025) moves the importance ratio from the token to the sequence. The ratio of a response is the length-normalized likelihood ratio

```math
s_i(\theta) = \left( \frac{\pi_{\theta}(y_i \mid x)}{\pi_{\theta_{\mathrm{old}}}(y_i \mid x)} \right)^{1/|y_i|} = \exp\left( \frac{1}{|y_i|} \sum_{t} \log \frac{\pi_{\theta}(y_{i,t} \mid x, y_{i,\lt t})}{\pi_{\theta_{\mathrm{old}}}(y_{i,t} \mid x, y_{i,\lt t})} \right),
```

and the clip is applied to $`s_i`$, once per response, not once per token. Length normalization is what makes one clip threshold meaningful for short and long answers. Without it, a sequence of 100 tokens each only $`0.01`$ nats more likely under the new policy has a raw likelihood ratio of $`e^{1} \approx 2.72`$, while the length-normalized ratio is $`e^{0.01} \approx 1.01`$. Token-level ratios have a different problem on mixture-of-experts models: a token's probability jumps when its expert routing changes between the sampler and the trainer, and a single noisy token can clip or explode the update. The sequence likelihood is an average over the whole response, so it moves less when one expert assignment flickers. Zheng et al. (2025) report that this stabilizes RL on MoE models, and the Qwen3 line credits GSPO for later models (Yang et al. 2025).

A few modifications show up so often that they are part of the recipe rather than a named algorithm.

- **Drop the KL term.** Magistral and GLM-4.5 both say they run GRPO without it, and DAPO-style reasoning runs often do the same. Section 10.6's reverse KL forbids probability mass where the reference has almost none. A long chain of thought is exactly such a region for a base or SFT reference. Removing the penalty lets the trace grow. It also removes the term that was limiting hacking of a learned reward, which is safe only while the reward is a checker.
- **Change the length normalization.** Token-level averages, sequence-level averages, and a constant budget are three different answers to "should a long wrong answer contribute more gradient than a short wrong answer?" Dr. GRPO, DAPO, and Magistral each pick one, and they do not pick the same one.
- **Skip zero-advantage groups.** If every sample in the group got the same reward, there is nothing to learn. Filtering them, or resampling until the group is mixed, is DAPO's dynamic sampling and is now a common default.
- **Correct the gap between the sampler and the trainer.** Generation often runs in an inference engine and training in another, and the two do not compute identical log-probabilities. A ratio built from inconsistent log-probabilities is not the PPO ratio. Reports describe masked or truncated importance weights, and off-policy sequence masking, to keep tokens whose sampler and trainer disagree from entering the update (DeepSeek-AI 2025, on V3.2; NVIDIA 2025). GSPO's sequence-level ratio is another way to make that gap less sharp.

## What closed labs do not say

OpenAI's o1 and GPT-5 reports, Anthropic's Claude reports, Google's Gemini reports, xAI's Grok 4 report, and Meta's Muse Spark post confirm large-scale reinforcement learning, human feedback, or AI and critic feedback. Gemini 2.5's report describes RL with verifiable rewards and model-based generative rewards and does not name the optimizer (Comanici et al. 2025). The gpt-oss model card describes chain-of-thought RL techniques similar to o3 and does not name them either (OpenAI 2025). Muse Spark describes large-scale RL with a penalty on thinking time and does not disclose the algorithm (Meta AI 2026). A claim that any of these systems "uses GRPO" or "uses PPO," beyond the sentences the lab wrote, is a guess. This chapter does not make it. The methods in the table are the ones a reader can implement from a paper. The frontier labs are evidence that large-scale RL and verifiable rewards are in use. They are not evidence about the clip function.

## The recipe that the reports converge on

Read down the 2025 and 2026 rows and the same stages recur, whether or not the clip is the one from Shao et al. (2024).

1. **Supervised fine-tuning**, often on long chain-of-thought traces, and often distilled from a stronger model or from a previous RL run. R1-Zero is the deliberate exception that skips this step.
2. **An optional preference stage**, usually DPO or a length-normalized relative, on pairs that encode chat behavior a checker cannot see.
3. **A large GRPO-style RL stage.** Verifiable rewards for math, code, and agent tasks with tests. Reward models or rubric-based LLM judges for open-ended tasks. A group baseline, a clip, and some subset of the fixes above. The KL term is present when the run must not drift, and absent when the run must discover long traces.
4. **On-policy distillation** to merge specialists or to train a smaller model. Qwen3, GLM-5, and DeepSeek-V4 all describe a version of this: train domain experts or a larger teacher, then distill into the model that will be served, sampling from the student rather than imitating a fixed offline dataset (Yang et al. 2025; GLM-5 Team 2026; DeepSeek-AI 2026). DeepSeek-V4 describes that distillation loss as a reverse KL.

That is the practical answer to Section 7's decision procedure, as of this writing. Use a checker when you have one, a group baseline instead of a critic unless you have a reason to keep the critic, DPO where you have pairs and a budget, and a held-out evaluation that is not the training reward. The names will keep changing. The split between a programmatic reward and a judged reward, and the decision to compare several samples rather than to train a value model, is what the public recipes have in common.

## Key takeaways

- Since late 2024, open post-training reports that name an RL algorithm mostly name GRPO or a critic-free variant, not PPO with a learned value model.
- PPO remains documented for classic RLHF, for Tulu 3 and OLMo 2's RLVR, and for value-based reasoning methods such as VAPO and Seed1.5-Thinking.
- DPO stays in the recipe as a stable preference stage. It is rarely the stage that trains verifiable reasoning.
- DAPO, Dr. GRPO, CISPO, and GSPO edit GRPO's clip, its normalizers, or the level at which the importance ratio is computed. Dropping the KL term, skipping tied groups, and correcting sampler-trainer log-probability mismatch are now ordinary.
- Closed labs confirm large-scale RL and, in several cases, verifiable rewards. They do not confirm a specific optimizer, and this chapter does not invent one.
- The common shape is SFT, optional DPO, then group-relative RL with checkers and judges, followed increasingly by on-policy distillation.

## Further reading

ByteDance Seed. "Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning." arXiv preprint arXiv:2504.13914, 2025. https://arxiv.org/abs/2504.13914.

Chen, Aili, et al. "The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence." arXiv preprint arXiv:2605.26494, 2026. https://arxiv.org/abs/2605.26494.

Comanici, Gheorghe, et al. "Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities." arXiv preprint arXiv:2507.06261, 2025. https://arxiv.org/abs/2507.06261.

DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv preprint arXiv:2501.12948, 2025. https://arxiv.org/abs/2501.12948.

DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv preprint arXiv:2412.19437, 2024. https://arxiv.org/abs/2412.19437.

DeepSeek-AI. "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models." arXiv preprint arXiv:2512.02556, 2025. https://arxiv.org/abs/2512.02556.

DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." arXiv preprint arXiv:2606.19348, 2026. https://arxiv.org/abs/2606.19348.

Gemma Team. "Gemma 3 Technical Report." arXiv preprint arXiv:2503.19786, 2025. https://arxiv.org/abs/2503.19786.

GLM-4.5 Team. "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models." arXiv preprint arXiv:2508.06471, 2025. https://arxiv.org/abs/2508.06471.

GLM-5 Team. "GLM-5: From Vibe Coding to Agentic Engineering." arXiv preprint arXiv:2602.15763, 2026. https://arxiv.org/abs/2602.15763.

Grattafiori, Aaron, et al. "The Llama 3 Herd of Models." arXiv preprint arXiv:2407.21783, 2024. https://arxiv.org/abs/2407.21783.

Kimi Team. "Kimi k1.5: Scaling Reinforcement Learning with LLMs." arXiv preprint arXiv:2501.12599, 2025. https://arxiv.org/abs/2501.12599.

Kimi Team. "Kimi K2: Open Agentic Intelligence." arXiv preprint arXiv:2507.20534, 2025. https://arxiv.org/abs/2507.20534.

Kimi Team. "Kimi K2.5: Visual Agentic Intelligence." arXiv preprint arXiv:2602.02276, 2026. https://arxiv.org/abs/2602.02276.

Lambert, Nathan, et al. "Tulu 3: Pushing Frontiers in Open Language Model Post-Training." arXiv preprint arXiv:2411.15124, 2024. https://arxiv.org/abs/2411.15124.

Liu, Zichen, et al. "Understanding R1-Zero-Like Training: A Critical Perspective." arXiv preprint arXiv:2503.20783, 2025. https://arxiv.org/abs/2503.20783.

Meta AI. "Introducing Muse Spark: Scaling Towards Personal Superintelligence." April 8, 2026. https://ai.meta.com/blog/introducing-muse-spark-msl/.

Meta AI. "The Llama 4 Herd: The Beginning of a New Era of Natively Multimodal AI Innovation." April 5, 2025. https://ai.meta.com/blog/llama-4-multimodal-intelligence/.

MiniMax. "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention." arXiv preprint arXiv:2506.13585, 2025. https://arxiv.org/abs/2506.13585.

Mistral AI. "Magistral." arXiv preprint arXiv:2506.10910, 2025. https://arxiv.org/abs/2506.10910.

NVIDIA. "Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning." arXiv preprint arXiv:2512.20848, 2025. https://arxiv.org/abs/2512.20848.

OpenAI. "gpt-oss-120b & gpt-oss-20b Model Card." arXiv preprint arXiv:2508.10925, 2025. https://arxiv.org/abs/2508.10925.

Qwen Team. "Qwen2.5 Technical Report." arXiv preprint arXiv:2412.15115, 2024. https://arxiv.org/abs/2412.15115.

Qwen Team. "Qwen3.5: Towards Native Multimodal Agents." February 16, 2026. https://qwen.ai/blog?id=qwen3.5.

Team OLMo. "2 OLMo 2 Furious." arXiv preprint arXiv:2501.00656, 2024. https://arxiv.org/abs/2501.00656.

Team Olmo. "Olmo 3." arXiv preprint arXiv:2512.13961, 2025. https://arxiv.org/abs/2512.13961.

Yang, An, et al. "Qwen3 Technical Report." arXiv preprint arXiv:2505.09388, 2025. https://arxiv.org/abs/2505.09388.

Yu, Qiying, et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." arXiv preprint arXiv:2503.14476, 2025. https://arxiv.org/abs/2503.14476.

Yue, Yu, et al. "VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks." arXiv preprint arXiv:2504.05118, 2025. https://arxiv.org/abs/2504.05118.

Zheng, Chujie, et al. "Group Sequence Policy Optimization." arXiv preprint arXiv:2507.18071, 2025. https://arxiv.org/abs/2507.18071.
