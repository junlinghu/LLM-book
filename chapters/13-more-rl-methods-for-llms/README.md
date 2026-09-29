# Chapter 13: More RL Methods for LLMs (PPO Variants, DPO, etc.)

Chapter 10 covered the classic RLHF pipeline (a reward model and PPO) and the models it produced. This chapter starts with that pipeline's limitations and then surveys the methods that came after it: variants that make PPO cheaper or more stable, methods that learn directly from preferences without a separate reward model or RL loop, and methods that use verifiable rewards to train reasoning.

## Outline

### 1. [Where Chapter 10 left off: the limits of PPO-based RLHF](01-limits-of-ppo-based-rlhf.md)
- A short recap of the RLHF objective and PPO (see Chapters 9 and 10)
- **Cost**: four models in memory and slow sampling, which critic-free RL (Section 2) reduces
- **Instability**: training is sensitive to hyperparameters, which DPO and its variants avoid (Sections 3 and 4)
- **Reward hacking**: the reward model's score keeps rising while real quality falls (Gao et al. 2023), which verifiable rewards resist (Section 5)
- **Less diversity**: RLHF makes answers less varied than SFT does (Kirk et al. 2023)
- **Sycophancy**: the model learns to agree with the user instead of being correct (Sharma et al. 2023)
- How AI feedback (Section 6) cuts the cost of human labels

### 2. [RL without a value model](02-rl-without-a-value-model.md)
- Why the critic is expensive for LLMs
- **REINFORCE with baseline**: the simplest policy gradient for sequences
- **RLOO (REINFORCE Leave-One-Out)**: sample several answers and use the others as the baseline
- **GRPO (Group Relative Policy Optimization)**: normalize rewards within a group of answers to the same prompt, removing the value model
- How these compare with PPO in memory, stability, and results

### 3. [Direct Preference Optimization (DPO)](03-direct-preference-optimization.md)
- Deriving the closed-form optimal policy for the KL-regularized objective
- Rewriting the reward in terms of the policy: the "implicit reward model"
- The DPO loss on chosen and rejected pairs
- The role of beta and the reference model
- Strengths: simple, stable, cheap (no sampling during training)
- Weaknesses: offline data only, sensitivity to data quality, and likelihood of both responses can fall

### 4. [DPO variants and other preference methods](04-dpo-variants-and-other-preference-methods.md)
- **IPO**: fixes DPO's tendency to overfit deterministic preferences
- **KTO**: learns from single thumbs-up or thumbs-down labels instead of pairs
- **ORPO**: combines SFT and preference optimization in one step, without a reference model
- **SimPO**: length-normalized, reference-free objective
- **Online and iterative DPO**: generating fresh pairs with the current model
- Rejection sampling fine-tuning (best-of-n then SFT) as a simple baseline

### 5. [Reinforcement learning with verifiable rewards (RLVR)](05-reinforcement-learning-with-verifiable-rewards.md)
- Replacing a learned reward model with a checker: correct math answers, passing unit tests, valid format
- Why verifiable rewards resist reward hacking better than learned ones
- Training reasoning models with GRPO-style RL (for example the DeepSeek-R1 recipe)
- Emergent behaviors: longer chains of thought, self-checking
- Outcome rewards vs. process rewards (process reward models that score each step)

### 6. [AI feedback and self-improvement](06-ai-feedback-and-self-improvement.md)
- RLAIF: using an LLM instead of humans to label preferences
- Constitutional AI: critiques and revisions guided by written principles
- Self-rewarding and self-play approaches

### 7. [Choosing a method](07-choosing-a-method.md)
- A comparison table: data needed, models in memory, compute, stability, and typical use
- Common recipes: SFT then DPO for chat quality; SFT then GRPO with verifiable rewards for reasoning
- Diagnosing training: reward curves, KL divergence, response length, and evaluation scores

### 8. [What's used in practice](08-whats-used-in-practice.md)
- A snapshot of publicly documented post-training recipes, as of September 2026
- The shift away from PPO: most open reports since late 2024 use critic-free, group-baseline RL (mostly **GRPO** and its variants) instead of PPO with a learned value model
- PPO has not disappeared: classic RLHF used it (InstructGPT, ChatGPT, GPT-4, and Llama 2-Chat; see Chapter 10, Section 8), Tulu 3 and OLMo 2 ran RLVR with PPO, and ByteDance Seed1.5-Thinking uses a value-based PPO-style method
- DPO lives on as a cheap, stable stage before or after RL (Llama 3, Qwen2.5, Tulu 3, Llama 4, Olmo 3), not as a replacement for RL

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

- New GRPO-family algorithms (2025-2026):
  - **DAPO**: decoupled clipping (clip-higher), dynamic sampling, token-level loss, overlong reward shaping
  - **Dr. GRPO**: removes a GRPO bias that inflates response length
  - **VAPO**: value-based PPO variant for long chain-of-thought reasoning
  - **CISPO**: clips importance-sampling weights, so every token still gets a gradient
  - **GSPO**: sequence-level importance ratios and clipping; stabilizes RL on mixture-of-experts models
  - Common tweaks: drop the KL term, token-level loss normalization, filter groups with zero advantage, correct training-inference mismatch
- Closed labs disclose little: OpenAI (o1, GPT-5), Anthropic (Claude), Google (Gemini), xAI (Grok 4), and Meta (Muse Spark) confirm large-scale RL, human feedback, or AI/critic feedback, but do not name their policy-optimization algorithm; any claim about their exact algorithm is a guess
- The common modern recipe: SFT (often distilled long chain-of-thought data), optional DPO, then large-scale GRPO-style RL with verifiable rewards for math, code, and agents plus reward models or rubric-based LLM judges for open-ended tasks; on-policy distillation increasingly merges domain experts or trains smaller models (Qwen3, GLM-5, DeepSeek-V4)

## Suggested code labs

1. **Observe reward hacking.** Using the REINFORCE setup from Chapter 10 (Lab 1) or the PPO setup from Chapter 10 (Lab 3), train against a deliberately flawed reward (for example one that rewards length, or a target word the model can simply repeat) and watch the model exploit it. Then increase the KL penalty and compare.
2. **Implement the DPO loss from scratch.** Write the loss in PyTorch using log-probabilities from a policy and a frozen reference model. Test it on a toy batch of chosen and rejected responses and confirm the implicit reward margin grows during training.
3. **Fine-tune a small model with DPO.** Train a small open model (for example a 0.5B-parameter model) on a small preference dataset. Compare its outputs with the SFT model on the same prompts, and track the chosen and rejected log-probabilities.
4. **Implement GRPO on a math task.** For each prompt, sample a group of answers, score each with a verifiable reward (does the final number match?), compute group-relative advantages, and update the policy. Plot accuracy and average response length over training.
5. **Compare DPO, KTO, and SimPO.** Train the same base model with each loss on the same data, then compare win rates using an LLM judge and check response length for signs of length bias.
6. **GRPO vs. RLOO.** Train the same model on the same math task with both methods and compare accuracy, stability, and training cost.

## Key takeaways

- PPO-based RLHF is expensive and fragile, and it can over-optimize a learned reward (reward hacking, reduced diversity, sycophancy); most newer methods simplify or fix some part of it.
- DPO turns preference learning into a simple classification-style loss with no reward model or sampling.
- GRPO removes the value model by comparing answers within a group, which makes RL on LLMs much cheaper.
- Verifiable rewards are the key ingredient behind recent reasoning models.
- The best method depends on your data (pairs, single labels, or checkable answers) and your goal.

## Further reading

Ahmadian, Arash, et al. "Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs." In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2024. https://arxiv.org/abs/2402.14740.

Anthropic. "System Card: Claude Opus 4 & Claude Sonnet 4." May 2025. https://www-cdn.anthropic.com/4263b940cabb546aa0e3283f35b686f4f3b2ff47.pdf.

Bai, Yuntao, et al. "Constitutional AI: Harmlessness from AI Feedback." arXiv preprint arXiv:2212.08073, 2022. https://arxiv.org/abs/2212.08073.

ByteDance Seed. "Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning." arXiv preprint arXiv:2504.13914, 2025. https://arxiv.org/abs/2504.13914.

Chen, Aili, et al. "The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence." arXiv preprint arXiv:2605.26494, 2026. https://arxiv.org/abs/2605.26494.

Comanici, Gheorghe, et al. "Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities." arXiv preprint arXiv:2507.06261, 2025. https://arxiv.org/abs/2507.06261.

DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv preprint arXiv:2501.12948, 2025. https://arxiv.org/abs/2501.12948.

DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv preprint arXiv:2412.19437, 2024. https://arxiv.org/abs/2412.19437.

DeepSeek-AI. "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models." arXiv preprint arXiv:2512.02556, 2025. https://arxiv.org/abs/2512.02556.

DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." arXiv preprint arXiv:2606.19348, 2026. https://arxiv.org/abs/2606.19348.

Gao, Leo, et al. "Scaling Laws for Reward Model Overoptimization." In *Proceedings of the 40th International Conference on Machine Learning*, 2023. https://arxiv.org/abs/2210.10760.

Gemma Team. "Gemma 3 Technical Report." arXiv preprint arXiv:2503.19786, 2025. https://arxiv.org/abs/2503.19786.

GLM-4.5 Team. "GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models." arXiv preprint arXiv:2508.06471, 2025. https://arxiv.org/abs/2508.06471.

GLM-5 Team. "GLM-5: From Vibe Coding to Agentic Engineering." arXiv preprint arXiv:2602.15763, 2026. https://arxiv.org/abs/2602.15763.

Grattafiori, Aaron, et al. "The Llama 3 Herd of Models." arXiv preprint arXiv:2407.21783, 2024. https://arxiv.org/abs/2407.21783.

Kimi Team. "Kimi k1.5: Scaling Reinforcement Learning with LLMs." arXiv preprint arXiv:2501.12599, 2025. https://arxiv.org/abs/2501.12599.

Kimi Team. "Kimi K2: Open Agentic Intelligence." arXiv preprint arXiv:2507.20534, 2025. https://arxiv.org/abs/2507.20534.

Kimi Team. "Kimi K2.5: Visual Agentic Intelligence." arXiv preprint arXiv:2602.02276, 2026. https://arxiv.org/abs/2602.02276.

Kirk, Robert, et al. "Understanding the Effects of RLHF on LLM Generalisation and Diversity." arXiv preprint arXiv:2310.06452, 2023. https://arxiv.org/abs/2310.06452.

Lambert, Nathan, et al. "Tulu 3: Pushing Frontiers in Open Language Model Post-Training." arXiv preprint arXiv:2411.15124, 2024. https://arxiv.org/abs/2411.15124.

Liu, Zichen, et al. "Understanding R1-Zero-Like Training: A Critical Perspective." arXiv preprint arXiv:2503.20783, 2025. https://arxiv.org/abs/2503.20783.

Meta AI. "Introducing Muse Spark: Scaling Towards Personal Superintelligence." April 8, 2026. https://ai.meta.com/blog/introducing-muse-spark-msl/.

Meta AI. "The Llama 4 Herd: The Beginning of a New Era of Natively Multimodal AI Innovation." April 5, 2025. https://ai.meta.com/blog/llama-4-multimodal-intelligence/.

MiniMax. "MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention." arXiv preprint arXiv:2506.13585, 2025. https://arxiv.org/abs/2506.13585.

Mistral AI. "Magistral." arXiv preprint arXiv:2506.10910, 2025. https://arxiv.org/abs/2506.10910.

NVIDIA. "Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning." arXiv preprint arXiv:2512.20848, 2025. https://arxiv.org/abs/2512.20848.

OpenAI. "GPT-5 System Card." August 13, 2025. https://cdn.openai.com/gpt-5-system-card.pdf.

OpenAI. "gpt-oss-120b & gpt-oss-20b Model Card." arXiv preprint arXiv:2508.10925, 2025. https://arxiv.org/abs/2508.10925.

OpenAI. "Learning to Reason with LLMs." September 12, 2024. https://openai.com/index/learning-to-reason-with-llms/.

Qwen Team. "Qwen2.5 Technical Report." arXiv preprint arXiv:2412.15115, 2024. https://arxiv.org/abs/2412.15115.

Qwen Team. "Qwen3.5: Towards Native Multimodal Agents." February 16, 2026. https://qwen.ai/blog?id=qwen3.5.

Rafailov, Rafael, et al. "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2305.18290.

Sharma, Mrinank, et al. "Towards Understanding Sycophancy in Language Models." arXiv preprint arXiv:2310.13548, 2023. https://arxiv.org/abs/2310.13548.

Shao, Zhihong, et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv preprint arXiv:2402.03300, 2024. https://arxiv.org/abs/2402.03300.

Team OLMo. "2 OLMo 2 Furious." arXiv preprint arXiv:2501.00656, 2024. https://arxiv.org/abs/2501.00656.

Team Olmo. "Olmo 3." arXiv preprint arXiv:2512.13961, 2025. https://arxiv.org/abs/2512.13961.

xAI. "Grok 4." July 9, 2025. https://x.ai/news/grok-4.

Yang, An, et al. "Qwen3 Technical Report." arXiv preprint arXiv:2505.09388, 2025. https://arxiv.org/abs/2505.09388.

Yu, Qiying, et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." arXiv preprint arXiv:2503.14476, 2025. https://arxiv.org/abs/2503.14476.

Yue, Yu, et al. "VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks." arXiv preprint arXiv:2504.05118, 2025. https://arxiv.org/abs/2504.05118.

Zheng, Chujie, et al. "Group Sequence Policy Optimization." arXiv preprint arXiv:2507.18071, 2025. https://arxiv.org/abs/2507.18071.
