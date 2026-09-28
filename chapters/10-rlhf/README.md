# Chapter 10: RLHF

Chapter 8 ended with a supervised fine-tuned (SFT) model: it follows instructions by imitating demonstrations, but it has no way to learn that one acceptable answer is better than another. Reinforcement Learning from Human Feedback (RLHF) picks up from there and is the method that turned instruction-tuned models into helpful chat assistants. This chapter is about RL for language models specifically. It takes the general methods from Chapter 9 (policy gradients and PPO) as given, casts text generation as an RL problem, and builds the classic pipeline step by step: collect human preferences, train a reward model, and optimize the SFT model against it while keeping it close to where it started. It ends with what this recipe achieved: the models built with PPO-based RLHF from 2019 to 2026, from InstructGPT and ChatGPT to GPT-4 and Llama 2-Chat.

## Learning goals

- Explain what SFT cannot teach and what RLHF adds on top of it.
- Frame text generation as RL: the token-level MDP, the language model as a policy, sparse sequence-level reward, and credit assignment.
- Describe the three-stage pipeline: SFT, reward model, then RL.
- Train a reward model from pairwise preferences using the Bradley-Terry loss.
- Write the KL-regularized RLHF objective and explain each term.
- Explain what changes when PPO is applied to a language model: the four models, token-level KL reward shaping, rollout generation, and the costs.
- Describe what PPO-based RLHF achieved in practice and the models it produced.

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

### 8. PPO-based RLHF in practice (2019–2026)
- GPT-3 itself was not trained with RLHF: it was a pretrained model used through few-shot prompting (Brown et al. 2020). InstructGPT fine-tuned GPT-3 with SFT and then PPO, and ChatGPT applied the same methods to a model from the GPT-3.5 series
- Early demonstrations: PPO fine-tuning of GPT-2 from human comparisons, then summaries preferred over human-written references
- Instruction following: a small RLHF model beat a much larger pretrained one, and was more truthful and less toxic
- Assistants: ChatGPT, Anthropic's helpful-and-harmless assistant, and the Claude models that followed (trained with RLHF, algorithm not named)
- Frontier and open models: GPT-4 used PPO with extra rule-based rewards for safety, and Llama 2-Chat combined rejection-sampling fine-tuning with PPO
- A close relative: DeepMind's Sparrow used the same RLHF recipe with an A2C optimizer instead of PPO, adding rule-specific reward models and cited evidence (Glaese et al. 2022)
- PPO holds up in controlled comparisons: tuned carefully, it matched or beat DPO and other alignment methods on dialogue, math, and code (Xu et al. 2024; Ivison et al. 2024)
- After 2024: PPO remains in use for RL with verifiable rewards (Tulu 3, OLMo 2) and in value-based variants (VAPO, Seed1.5-Thinking); closed reasoning models from o1 onward do not disclose their algorithm. Chapter 11 (Section 8) covers the shift toward DPO and GRPO-style methods

| Model | What PPO-based RLHF achieved |
|---|---|
| GPT-2 fine-tuned from human preferences | Learned stylistic continuation (positive sentiment, descriptive text) from only 5,000 human comparisons (Ziegler et al. 2019) |
| Summarization policies (TL;DR) | Summaries preferred over human references and over much larger supervised models; transferred to CNN/DM news without news-specific training (Stiennon et al. 2020) |
| InstructGPT | Outputs of the 1.3B model preferred to those of 175B GPT-3; more truthful, less toxic, with minimal regressions on public NLP benchmarks (Ouyang et al. 2022) |
| Helpful-and-harmless assistant | Improved almost all NLP evaluations for large models (an "alignment bonus"), stayed compatible with coding and summarization skills, and was updated weekly with fresh feedback (Bai et al. 2022) |
| ChatGPT | Conversational assistant fine-tuned from a GPT-3.5 model with InstructGPT's methods, using PPO over several iterations (OpenAI 2022) |
| GPT-4 | PPO against a reward model, plus rule-based reward models during PPO; with the other safety steps, 82% less likely than GPT-3.5 to respond to requests for disallowed content (OpenAI 2023) |
| Llama 2-Chat | Rejection-sampling fine-tuning, then PPO; outperformed open-source chat models on most benchmarks and in human evaluations of helpfulness and safety (Touvron et al. 2023) |
| Claude 2 | Trained with RLHF and Constitutional AI (which has an RL phase); improved over Claude 1.3 in helpfulness and honesty; RL algorithm not named (Anthropic 2023) |

## Suggested code labs

1. **A language model as a policy.** Take a tiny pretrained language model (for example a small GPT-2 variant) and fine-tune it with REINFORCE toward a simple, checkable reward, such as including a target word or producing positive sentiment by a small classifier. Train with and without a KL penalty to the original model and compare rewards, KL divergence, and outputs.
2. **Train a reward model.** Fine-tune a small model with a scalar head on a preference dataset using the Bradley-Terry loss. Report its accuracy on held-out pairs and check whether it prefers longer answers.
3. **Run PPO on a small language model.** Adapt your PPO from Chapter 9 (Lab 5), or use a library such as TRL, to optimize a small SFT model against the reward model from Lab 2 with per-token KL shaping. Track reward, KL divergence from the reference model, and response length over training.
4. **Explore the KL coefficient.** Repeat Lab 3 with several beta values and compare how far the model drifts, how high the reward goes, and how the outputs read.

## Key takeaways

- RLHF picks up where SFT leaves off: it teaches a model to prefer better answers, not just to imitate good ones.
- Text generation is an RL problem with a huge action space, deterministic transitions, and sparse sequence-level rewards.
- The reward model turns human preferences into a number the RL step can optimize.
- The KL penalty keeps the policy close to the SFT model so it doesn't exploit the reward model.
- PPO for LLMs is Chapter 9's PPO plus four models, per-token KL shaping, and expensive generation.
- PPO-based RLHF produced InstructGPT, ChatGPT, GPT-4, and Llama 2-Chat, and a small RLHF model can beat a much larger pretrained one; Chapter 11 covers its limitations and the methods that followed.

## Further reading

Anthropic. "Model Card and Evaluations for Claude Models." July 2023. https://www-cdn.anthropic.com/bd2a28d2535bfb0494cc8e2a3bf135d2e7523226.pdf.

Bai, Yuntao, et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." arXiv preprint arXiv:2204.05862, 2022. https://arxiv.org/abs/2204.05862.

Brown, Tom B., et al. "Language Models Are Few-Shot Learners." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2005.14165.

Christiano, Paul F., et al. "Deep Reinforcement Learning from Human Preferences." In *Advances in Neural Information Processing Systems 30*, 2017. https://arxiv.org/abs/1706.03741.

Glaese, Amelia, et al. "Improving Alignment of Dialogue Agents via Targeted Human Judgements." arXiv preprint arXiv:2209.14375, 2022. https://arxiv.org/abs/2209.14375.

Ivison, Hamish, et al. "Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback." In *Advances in Neural Information Processing Systems 37*, 2024. https://arxiv.org/abs/2406.09279.

OpenAI. "GPT-4 Technical Report." arXiv preprint arXiv:2303.08774, 2023. https://arxiv.org/abs/2303.08774.

OpenAI. "Introducing ChatGPT." November 30, 2022. https://openai.com/index/chatgpt/.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Stiennon, Nisan, et al. "Learning to Summarize from Human Feedback." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2009.01325.

Touvron, Hugo, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv preprint arXiv:2307.09288, 2023. https://arxiv.org/abs/2307.09288.

Xu, Shusheng, et al. "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study." In *Proceedings of the 41st International Conference on Machine Learning*, 2024. https://arxiv.org/abs/2404.10719.

Ziegler, Daniel M., et al. "Fine-Tuning Language Models from Human Preferences." arXiv preprint arXiv:1909.08593, 2019. https://arxiv.org/abs/1909.08593.
