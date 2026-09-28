# Chapter 10: RLHF

Reinforcement Learning from Human Feedback (RLHF) is the method that turned instruction-tuned models into helpful chat assistants. This chapter builds the classic pipeline step by step: collect human preferences, train a reward model, and optimize the language model against it with PPO while keeping it close to its starting point.

## Learning goals

- Explain why supervised fine-tuning alone is not enough, and what RLHF adds.
- Describe the three-stage pipeline: SFT, reward model, then RL.
- Train a reward model from pairwise preferences using the Bradley-Terry loss.
- Write the KL-regularized RLHF objective and explain each term.
- Explain PPO for language models in detail: rollouts, advantages, the clipped objective, and the value model.
- Recognize reward hacking and the practical costs that motivate the methods in Chapter 11.

## Outline

### 1. Why RLHF
- What SFT teaches (imitate demonstrations) and what it cannot (rank good vs. better answers)
- Preferences are easier for people to give than perfect demonstrations
- Goals: helpful, honest, and harmless
- A short history: from Christiano et al. (2017) to InstructGPT and ChatGPT

### 2. The RLHF pipeline
- **Stage 1: SFT** produces the starting policy and the reference model
- **Stage 2: reward model** learns to score responses from human preferences
- **Stage 3: RL** optimizes the policy to earn high reward
- Framing text generation as RL: prompt is the state, each token is an action, the full response is an episode, and reward arrives at the end

### 3. Collecting human preference data
- Pairwise comparisons vs. rankings vs. ratings
- Labeling guidelines and annotator agreement
- Public datasets (for example Anthropic HH-RLHF, UltraFeedback)

### 4. Reward modeling
- Architecture: a language model with a scalar output head
- The Bradley-Terry model and the pairwise loss: -log σ(r(chosen) − r(rejected))
- Evaluating a reward model: accuracy on held-out preference pairs
- Known weaknesses: length bias, spurious features, overconfidence

### 5. The RLHF objective
- Maximize expected reward while staying close to the reference model
- The KL penalty and the coefficient beta: why it prevents the model from drifting into gibberish that fools the reward model
- Per-token KL vs. per-sequence reward

### 6. PPO in detail
- Policy gradient basics: increase the probability of actions that led to high reward
- Baselines and advantages: comparing a result with what was expected
- The value model (critic) and generalized advantage estimation (GAE)
- The clipped surrogate objective and why it stabilizes updates
- The PPO training loop for LLMs: generate rollouts, score them, compute advantages, run several update epochs
- Practical tricks: reward normalization and whitening, adaptive KL control, learning-rate and batch-size choices

### 7. What goes wrong
- **Four models in memory**: policy, reference, reward model, and value model
- Unstable training and sensitivity to hyperparameters
- **Reward hacking and over-optimization**: rising reward with falling real quality
- Mode collapse and reduced diversity
- Sycophancy and confident-sounding answers
- These costs are the motivation for the methods in Chapter 11

## Suggested code labs

1. **Train a reward model.** Fine-tune a small model with a scalar head on a preference dataset using the Bradley-Terry loss. Report its accuracy on held-out pairs and check whether it prefers longer answers.
2. **Implement PPO on a toy task.** Write the clipped objective, GAE, and value loss from scratch for a simple environment (for example CartPole) to see each piece working before applying it to text.
3. **Run PPO on a small language model.** Using the reward model from Lab 1, optimize a small SFT model with PPO. Track reward, KL divergence from the reference model, and response length over training.
4. **Explore the KL coefficient.** Repeat Lab 3 with several beta values and compare how far the model drifts, how high the reward goes, and how the outputs read.
5. **Observe reward hacking.** Train against a deliberately flawed reward (for example one that rewards length) and watch the model exploit it. Then increase the KL penalty and compare.

## Key takeaways

- RLHF teaches a model to prefer better answers, not just to imitate good ones.
- The reward model turns human preferences into a number the RL step can optimize.
- The KL penalty keeps the policy close to the SFT model so it doesn't exploit the reward model.
- PPO works, but it is expensive, fragile, and vulnerable to reward hacking.

## Further reading

Bai, Yuntao, et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." arXiv preprint arXiv:2204.05862, 2022. https://arxiv.org/abs/2204.05862.

Christiano, Paul F., et al. "Deep Reinforcement Learning from Human Preferences." In *Advances in Neural Information Processing Systems 30*, 2017. https://arxiv.org/abs/1706.03741.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Schulman, John, et al. "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347.

Stiennon, Nisan, et al. "Learning to Summarize from Human Feedback." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2009.01325.
