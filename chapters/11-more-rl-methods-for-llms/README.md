# Chapter 11: More RL Methods for LLMs (PPO Variants, DPO, etc.)

Chapter 10 covered the classic RLHF pipeline: a reward model and PPO. This chapter surveys the methods that came after it: variants that make PPO cheaper or more stable, methods that learn directly from preferences without a separate reward model or RL loop, and methods that use verifiable rewards to train reasoning.

## Learning goals

- Explain how critic-free methods (REINFORCE, RLOO, GRPO) simplify PPO.
- Derive DPO from the RLHF objective and explain why it needs no reward model or sampling loop.
- Compare the main preference-optimization variants and when to use each.
- Explain GRPO and reinforcement learning with verifiable rewards (RLVR), and why they matter for reasoning models.
- Choose a method given your data, compute budget, and goal.

## Outline

### 1. Where Chapter 10 left off
- A short recap of the KL-regularized RLHF objective and the PPO pipeline (see Chapter 10)
- The costs this chapter addresses: the value model, the reward model, online sampling, and instability
- A map of the chapter: simpler RL, direct preference methods, verifiable rewards, and AI feedback

### 2. RL without a value model
- Why the critic is expensive for LLMs
- **REINFORCE with baseline**: the simplest policy gradient for sequences
- **RLOO (REINFORCE Leave-One-Out)**: sample several answers and use the others as the baseline
- **GRPO (Group Relative Policy Optimization)**: normalize rewards within a group of answers to the same prompt, removing the value model
- How these compare with PPO in memory, stability, and results

### 3. Direct Preference Optimization (DPO)
- Deriving the closed-form optimal policy for the KL-regularized objective
- Rewriting the reward in terms of the policy: the "implicit reward model"
- The DPO loss on chosen and rejected pairs
- The role of beta and the reference model
- Strengths: simple, stable, cheap (no sampling during training)
- Weaknesses: offline data only, sensitivity to data quality, and likelihood of both responses can fall

### 4. DPO variants and other preference methods
- **IPO**: fixes DPO's tendency to overfit deterministic preferences
- **KTO**: learns from single thumbs-up or thumbs-down labels instead of pairs
- **ORPO**: combines SFT and preference optimization in one step, without a reference model
- **SimPO**: length-normalized, reference-free objective
- **Online and iterative DPO**: generating fresh pairs with the current model
- Rejection sampling fine-tuning (best-of-n then SFT) as a simple baseline

### 5. Reinforcement learning with verifiable rewards (RLVR)
- Replacing a learned reward model with a checker: correct math answers, passing unit tests, valid format
- Why verifiable rewards resist reward hacking better than learned ones
- Training reasoning models with GRPO-style RL (for example the DeepSeek-R1 recipe)
- Emergent behaviors: longer chains of thought, self-checking
- Outcome rewards vs. process rewards (process reward models that score each step)

### 6. AI feedback and self-improvement
- RLAIF: using an LLM instead of humans to label preferences
- Constitutional AI: critiques and revisions guided by written principles
- Self-rewarding and self-play approaches

### 7. Choosing a method
- A comparison table: data needed, models in memory, compute, stability, and typical use
- Common recipes: SFT then DPO for chat quality; SFT then GRPO with verifiable rewards for reasoning
- Diagnosing training: reward curves, KL divergence, response length, and evaluation scores

## Suggested code labs

1. **Implement the DPO loss from scratch.** Write the loss in PyTorch using log-probabilities from a policy and a frozen reference model. Test it on a toy batch of chosen and rejected responses and confirm the implicit reward margin grows during training.
2. **Fine-tune a small model with DPO.** Train a small open model (for example a 0.5B-parameter model) on a small preference dataset. Compare its outputs with the SFT model on the same prompts, and track the chosen and rejected log-probabilities.
3. **Implement GRPO on a math task.** For each prompt, sample a group of answers, score each with a verifiable reward (does the final number match?), compute group-relative advantages, and update the policy. Plot accuracy and average response length over training.
4. **Compare DPO, KTO, and SimPO.** Train the same base model with each loss on the same data, then compare win rates using an LLM judge and check response length for signs of length bias.
5. **GRPO vs. RLOO.** Train the same model on the same math task with both methods and compare accuracy, stability, and training cost.

## Key takeaways

- PPO works but is expensive and fragile; most newer methods simplify some part of it.
- DPO turns preference learning into a simple classification-style loss with no reward model or sampling.
- GRPO removes the value model by comparing answers within a group, which makes RL on LLMs much cheaper.
- Verifiable rewards are the key ingredient behind recent reasoning models.
- The best method depends on your data (pairs, single labels, or checkable answers) and your goal.

## Further reading

Ahmadian, Arash, et al. "Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs." In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2024. https://arxiv.org/abs/2402.14740.

DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv preprint arXiv:2501.12948, 2025. https://arxiv.org/abs/2501.12948.

Rafailov, Rafael, et al. "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2305.18290.

Shao, Zhihong, et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv preprint arXiv:2402.03300, 2024. https://arxiv.org/abs/2402.03300.
