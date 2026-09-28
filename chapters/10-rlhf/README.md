# Chapter 10: RLHF

Reinforcement Learning from Human Feedback (RLHF) is the method that turned instruction-tuned models into helpful chat assistants. This chapter builds on the RL basics from Chapter 9 (MDPs, policy gradients, baselines, and advantages). It first casts language generation as an RL problem, then builds the classic pipeline step by step: collect human preferences, train a reward model, and optimize the language model against it with PPO while keeping it close to its starting point.

## Learning goals

- Explain why supervised fine-tuning alone is not enough, and what RLHF adds.
- Frame text generation as RL: the token-level MDP, the language model as a policy, sparse sequence-level reward, and credit assignment.
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

### 2. Language generation as RL
- Builds on the RL vocabulary of Chapter 9: MDPs, policies, returns, and policy gradients
- The token-level MDP: the state is the prompt plus the tokens so far, the action is the next token, transitions are deterministic, and the full response is one episode
- The language model as a policy: its softmax output is a stochastic policy over the vocabulary
- Sparse, sequence-level reward: one score for the whole response, from a reward model, a human, or a checker, arriving only at the end
- Two views: the whole response as one action (a contextual bandit) vs. each token as an action (an MDP)
- Credit assignment: which tokens deserve credit for a good or bad reward
- Roadmap: the rest of this chapter supplies the reward (a reward model trained on human preferences), a KL penalty to a reference model (Section 6), and the optimizer (PPO, Section 7)

### 3. The RLHF pipeline
- **Stage 1: SFT** produces the starting policy and the reference model
- **Stage 2: reward model** learns to score responses from human preferences
- **Stage 3: RL** optimizes the policy to earn high reward

### 4. Collecting human preference data
- Pairwise comparisons vs. rankings vs. ratings
- Labeling guidelines and annotator agreement
- Public datasets (for example Anthropic HH-RLHF, UltraFeedback)

### 5. Reward modeling
- Architecture: a language model with a scalar output head
- The Bradley-Terry model and the pairwise loss: -log σ(r(chosen) − r(rejected))
- Evaluating a reward model: accuracy on held-out preference pairs
- Known weaknesses: length bias, spurious features, overconfidence

### 6. The RLHF objective
- Maximize expected reward while staying close to the reference model
- Why a KL penalty to the reference model: keep outputs fluent, avoid drifting into gibberish that fools the reward model, and preserve what pretraining and SFT learned
- The coefficient beta: trading reward against drift
- Per-token KL vs. per-sequence reward

### 7. PPO in detail
- Starting point: REINFORCE, baselines, and advantages from Chapter 9 (Section 7), now applied per token
- The value model (critic) and generalized advantage estimation (GAE)
- The clipped surrogate objective and why it stabilizes updates
- The PPO training loop for LLMs: generate rollouts, score them, compute advantages, run several update epochs
- Practical tricks: reward normalization and whitening, adaptive KL control, learning-rate and batch-size choices

### 8. What goes wrong
- **Four models in memory**: policy, reference, reward model, and value model
- Unstable training and sensitivity to hyperparameters
- **Reward hacking and over-optimization**: rising reward with falling real quality
- Mode collapse and reduced diversity
- Sycophancy and confident-sounding answers
- These costs are the motivation for the methods in Chapter 11

## Suggested code labs

1. **A language model as a policy.** Take a tiny pretrained language model (for example a small GPT-2 variant) and fine-tune it with REINFORCE toward a simple, checkable reward, such as including a target word or producing positive sentiment by a small classifier. Train with and without a KL penalty to the original model and compare rewards, KL divergence, and outputs.
2. **Train a reward model.** Fine-tune a small model with a scalar head on a preference dataset using the Bradley-Terry loss. Report its accuracy on held-out pairs and check whether it prefers longer answers.
3. **Implement PPO on a toy task.** Extend your REINFORCE-with-baseline CartPole agent from Chapter 9 (Lab 4) with GAE, the clipped objective, and a value loss, to see each piece working before applying it to text.
4. **Run PPO on a small language model.** Using the reward model from Lab 2, optimize a small SFT model with PPO. Track reward, KL divergence from the reference model, and response length over training.
5. **Explore the KL coefficient.** Repeat Lab 4 with several beta values and compare how far the model drifts, how high the reward goes, and how the outputs read.
6. **Observe reward hacking.** Using the REINFORCE setup from Lab 1 or the PPO setup from Lab 4, train against a deliberately flawed reward (for example one that rewards length, or a target word the model can simply repeat) and watch the model exploit it. Then increase the KL penalty and compare.

## Key takeaways

- RLHF teaches a model to prefer better answers, not just to imitate good ones.
- Text generation is an RL problem with a huge action space, deterministic transitions, and sparse sequence-level rewards, so RLHF relies on the baselines and advantages from Chapter 9 plus a KL penalty.
- The reward model turns human preferences into a number the RL step can optimize.
- The KL penalty keeps the policy close to the SFT model so it doesn't exploit the reward model.
- PPO works, but it is expensive, fragile, and vulnerable to reward hacking.

## Further reading

Bai, Yuntao, et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." arXiv preprint arXiv:2204.05862, 2022. https://arxiv.org/abs/2204.05862.

Christiano, Paul F., et al. "Deep Reinforcement Learning from Human Preferences." In *Advances in Neural Information Processing Systems 30*, 2017. https://arxiv.org/abs/1706.03741.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Schulman, John, et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv preprint arXiv:1506.02438, 2015. https://arxiv.org/abs/1506.02438.

Schulman, John, et al. "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347.

Stiennon, Nisan, et al. "Learning to Summarize from Human Feedback." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2009.01325.
