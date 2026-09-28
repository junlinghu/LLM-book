# Chapter 10: RLHF

Chapter 8 ended with a supervised fine-tuned (SFT) model: it follows instructions by imitating demonstrations, but it has no way to learn that one acceptable answer is better than another. Reinforcement Learning from Human Feedback (RLHF) picks up from there and is the method that turned instruction-tuned models into helpful chat assistants. This chapter is about RL for language models specifically. It takes the general methods from Chapter 9 (policy gradients and PPO) as given, casts text generation as an RL problem, and builds the classic pipeline step by step: collect human preferences, train a reward model, and optimize the SFT model against it while keeping it close to where it started.

## Learning goals

- Explain what SFT cannot teach and what RLHF adds on top of it.
- Frame text generation as RL: the token-level MDP, the language model as a policy, sparse sequence-level reward, and credit assignment.
- Describe the three-stage pipeline: SFT, reward model, then RL.
- Train a reward model from pairwise preferences using the Bradley-Terry loss.
- Write the KL-regularized RLHF objective and explain each term.
- Explain what changes when PPO is applied to a language model: the four models, token-level KL reward shaping, rollout generation, and the costs.
- Recognize reward hacking and the practical costs that motivate the methods in Chapter 11.

## Outline

### 1. From SFT to RLHF
- Where Chapter 8 left off: an SFT model that imitates demonstrations
- What SFT cannot teach: telling good answers from better ones, avoiding mistakes it never saw, or going beyond the quality of its demonstrations
- Preferences are easier for people to give than perfect demonstrations
- Goals: helpful, honest, and harmless
- A short history: from Christiano et al. (2017) and Stiennon et al. (2020) to InstructGPT and ChatGPT

### 2. Language generation as RL
- The token-level MDP: the state is the prompt plus the tokens so far, the action is the next token, transitions are deterministic, and the full response is one episode
- The language model as a policy: its softmax output is a stochastic policy over the vocabulary
- Sparse, sequence-level reward: one score for the whole response, from a reward model, a human, or a checker, arriving only at the end
- Two views: the whole response as one action (a contextual bandit) vs. each token as an action (an MDP)
- Credit assignment: which tokens deserve credit for a good or bad reward
- Why policy gradients (Chapter 9, Sections 6 and 7) rather than value-based methods: the action space is the whole vocabulary at every step

### 3. The RLHF pipeline
- **Stage 1: SFT** (Chapter 8) produces the starting policy and the frozen reference model
- **Stage 2: reward model** learns to score responses from human preferences
- **Stage 3: RL** optimizes the policy to earn high reward while staying close to the reference

### 4. Collecting human preference data
- Pairwise comparisons vs. rankings vs. ratings
- Labeling guidelines and annotator agreement
- Public datasets (for example Anthropic HH-RLHF, UltraFeedback)

### 5. Reward modeling
- Architecture: a language model, usually initialized from the SFT model, with a scalar output head
- The Bradley-Terry model and the pairwise loss on a chosen response and a rejected one

```math
\mathcal{L}_{\mathrm{RM}} = -\log \sigma\left( r_{\phi}(x, y_{\mathrm{chosen}}) - r_{\phi}(x, y_{\mathrm{rejected}}) \right)
```

- Evaluating a reward model: accuracy on held-out preference pairs
- Known weaknesses: length bias, spurious features, overconfidence

### 6. The RLHF objective
- Maximize expected reward while staying close to the SFT reference model

```math
\max_{\theta}\ \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_{\theta}(\cdot \mid x)}\left[ r_{\phi}(x, y) \right] - \beta\, \mathbb{D}_{\mathrm{KL}}\left[ \pi_{\theta}(\cdot \mid x) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid x) \right]
```

- Why a KL penalty to the reference model: keep outputs fluent, avoid drifting into gibberish that fools the reward model, and preserve what pretraining and SFT learned
- The coefficient beta: trading reward against drift, and adaptive KL control that tunes beta toward a target KL

### 7. PPO for language models
- The algorithm is the PPO of Chapter 9 (Section 7); this section covers only what changes for text
- **Four models**: the policy (actor), a value model (critic) with a per-token value head, the frozen reward model, and the frozen reference model
- Rollouts are generated responses to a batch of prompts; generation often dominates training time
- Token-level reward shaping: a KL penalty at every token, with the reward model's score added at the last token

```math
r_t = -\beta \log \frac{\pi_{\theta}(y_t \mid x, y_{\lt t})}{\pi_{\mathrm{ref}}(y_t \mid x, y_{\lt t})} + \mathbf{1}[t = T]\, r_{\phi}(x, y)
```

- Per-token advantages from GAE on this shaped reward, then the usual clipped update over a few epochs
- LLM-specific practice: reward normalization and whitening, handling responses that never end, small learning rates, and large prompt batches
- Memory and compute: four large models, and a loop that alternates between generation and training

### 8. What goes wrong
- Cost and complexity: four models in memory and slow rollout generation
- Unstable training and sensitivity to hyperparameters
- **Reward hacking and over-optimization**: rising reward with falling real quality
- Mode collapse and reduced diversity
- Sycophancy and confident-sounding answers
- These costs are the motivation for the methods in Chapter 11

## Suggested code labs

1. **A language model as a policy.** Take a tiny pretrained language model (for example a small GPT-2 variant) and fine-tune it with REINFORCE toward a simple, checkable reward, such as including a target word or producing positive sentiment by a small classifier. Train with and without a KL penalty to the original model and compare rewards, KL divergence, and outputs.
2. **Train a reward model.** Fine-tune a small model with a scalar head on a preference dataset using the Bradley-Terry loss. Report its accuracy on held-out pairs and check whether it prefers longer answers.
3. **Run PPO on a small language model.** Adapt your PPO from Chapter 9 (Lab 5), or use a library such as TRL, to optimize a small SFT model against the reward model from Lab 2 with per-token KL shaping. Track reward, KL divergence from the reference model, and response length over training.
4. **Explore the KL coefficient.** Repeat Lab 3 with several beta values and compare how far the model drifts, how high the reward goes, and how the outputs read.
5. **Observe reward hacking.** Using the REINFORCE setup from Lab 1 or the PPO setup from Lab 3, train against a deliberately flawed reward (for example one that rewards length, or a target word the model can simply repeat) and watch the model exploit it. Then increase the KL penalty and compare.

## Key takeaways

- RLHF picks up where SFT leaves off: it teaches a model to prefer better answers, not just to imitate good ones.
- Text generation is an RL problem with a huge action space, deterministic transitions, and sparse sequence-level rewards.
- The reward model turns human preferences into a number the RL step can optimize.
- The KL penalty keeps the policy close to the SFT model so it doesn't exploit the reward model.
- PPO for LLMs is Chapter 9's PPO plus four models, per-token KL shaping, and expensive generation; it works, but it is costly, fragile, and vulnerable to reward hacking.

## Further reading

Bai, Yuntao, et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." arXiv preprint arXiv:2204.05862, 2022. https://arxiv.org/abs/2204.05862.

Christiano, Paul F., et al. "Deep Reinforcement Learning from Human Preferences." In *Advances in Neural Information Processing Systems 30*, 2017. https://arxiv.org/abs/1706.03741.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Stiennon, Nisan, et al. "Learning to Summarize from Human Feedback." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2009.01325.

Ziegler, Daniel M., et al. "Fine-Tuning Language Models from Human Preferences." arXiv preprint arXiv:1909.08593, 2019. https://arxiv.org/abs/1909.08593.
