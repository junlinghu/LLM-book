# 11.8 PPO-Based RLHF in Practice (2019–2026)

Sections 3 through 7 built the classic RLHF recipe: SFT, a reward model trained on human preferences, and PPO against that reward model with a KL penalty to the SFT model. This section looks at what the recipe achieved. It traces the models that were trained with PPO-based RLHF, from the first experiments on GPT-2 in 2019 to the chat assistants and frontier models of 2022 and 2023, notes a close relative that used a different RL optimizer, summarizes controlled comparisons of PPO with newer methods, and describes where PPO still appears after 2024. Throughout, we attribute an algorithm to a model only when its developers have said which one they used.

## A note on GPT-3, InstructGPT, and ChatGPT

It is common to read that "GPT-3 was trained with RLHF." It was not. GPT-3 was a pretrained language model, 175 billion parameters trained on next-token prediction, and its paper studied how to use it through few-shot prompting: placing a few examples of a task in the context and letting the model continue the pattern (Brown et al. 2020; Section 8.6). No human preferences were involved.

RLHF entered with **InstructGPT**, which started from GPT-3 models and fine-tuned them first with SFT on labeler demonstrations and then with PPO against a reward model trained on labeler rankings (Ouyang et al. 2022). **ChatGPT** applied the same methods to a model from the GPT-3.5 series, with "slight differences in the data collection setup" (OpenAI 2022). The distinction matters because the capabilities people attribute to "GPT-3" in conversation mostly came from this post-training, not from pretraining alone.

## Early demonstrations

**Stylistic continuation and summarization with GPT-2 (2019).** Ziegler et al. (2019) fine-tuned the 774-million-parameter GPT-2 with PPO against reward models trained on human choices among four samples, with a KL penalty to the original model (Section 6). For stylistic continuation, writing with positive sentiment or vivid description, about 5,000 human comparisons sufficed; labelers preferred the fine-tuned model's continuations 86% of the time over the zero-shot model's and 77% of the time over a model fine-tuned against a supervised sentiment classifier. For summarization, trained on 60,000 comparisons, the models learned to copy whole sentences from the input while skipping irrelevant preamble, which scored well with labelers but, the authors noted, may have exploited labelers' reliance on simple heuristics.

**Summaries preferred over human references (2020).** Stiennon et al. (2020) scaled the approach to summarizing Reddit posts (the TL;DR dataset) with 1.3- and 6.7-billion-parameter models and 64,832 comparisons collected with close attention to labeler quality. Their RLHF policies significantly outperformed both the human-written reference summaries and much larger models fine-tuned with supervised learning alone. Without any news-specific fine-tuning, the same policies produced summaries of CNN/Daily Mail news articles nearly as good as the human references. The paper also documented over-optimization: pushed too far against a reward model, the policy's summaries got worse by human judgment while the reward kept rising (Section 6). Several details that became standard, including a value model with separate parameters initialized from the reward model and normalizing the reward model's offset, come from this work.

## Instruction following

InstructGPT (Ouyang et al. 2022) showed that the recipe works for general instructions, not just one task. Its headline result is about scale: labelers preferred outputs of the 1.3-billion-parameter InstructGPT model to those of the 175-billion-parameter GPT-3, despite the more than 100-fold difference in size. The 175B InstructGPT model's outputs were preferred to 175B GPT-3's 85 ± 3% of the time, and to few-shot-prompted GPT-3's 71 ± 4% of the time.

The improvements went beyond general preference. On TruthfulQA, InstructGPT gave truthful and informative answers about twice as often as GPT-3. When prompted to be respectful, it generated about 25% fewer toxic outputs than GPT-3 on RealToxicityPrompts, although it did not improve on the bias benchmarks the authors measured. The first RLHF models regressed on some public NLP benchmarks such as SQuAD and DROP; mixing pretraining gradients into PPO (PPO-ptx, Section 3) reduced these regressions to a minimum without hurting labeler preference.

InstructGPT's lesson was that post-training could matter more than a hundredfold increase in parameters for the qualities users notice. It also established the concrete hyperparameters that later work started from: a 6B reward and value model, $\beta = 0.02$, 512 prompts per PPO batch, no discounting, and a clip ratio of 0.2 (Section 7).

## Assistants

**ChatGPT (2022).** OpenAI's announcement describes the recipe in a few sentences (OpenAI 2022). For SFT, human AI trainers wrote conversations playing both the user and the assistant, mixed with the InstructGPT data converted to dialogue format. For the reward model, trainers ranked several alternative completions of model-written messages sampled from their conversations with the chatbot. The model was then fine-tuned with PPO against these reward models, and "we performed several iterations of this process." The same announcement lists limitations that this chapter has explained: the model was "often excessively verbose," which OpenAI attributed to trainers preferring longer answers that look more comprehensive and to "well-known over-optimization issues" (Sections 4 and 5).

**Anthropic's helpful and harmless assistant (2022).** Bai et al. (2022) trained assistants of up to 52 billion parameters with preference modeling and PPO, using separate helpfulness and harmlessness (red-teaming) data collected from crowdworkers in open-ended conversations. They found that RLHF improved performance on almost all the NLP evaluations they ran for their larger models. Smaller models paid an "alignment tax," while the 13B and 52B models did better on zero-shot evaluations after RLHF and the same on few-shot ones, an "alignment bonus." RLHF was also compatible with specialized skills: it improved the programming evaluations of models first fine-tuned on code, and combining it with summarization training cost nothing on either. They ran *iterated online* RLHF, updating preference models and policies on a roughly weekly cadence with fresh feedback, and they reported the roughly linear relation between reward and the square root of KL divergence discussed in Section 6.

**The Claude models (2023 onward).** Anthropic's Claude models were trained with RLHF and Constitutional AI. The model card for Claude 2 states that Claude models are "trained via unsupervised learning, RLHF, and Constitutional AI (including both a supervised and Reinforcement Learning (RL) phase)," and reports that Claude 2 improved over Claude 1.3 on helpfulness and honesty in human preference evaluations while scoring similarly on harmlessness (Anthropic 2023). The model card does not name the RL algorithm. Anthropic's published research on RLHF used PPO, but that does not tell us what was used for Claude, so we do not attribute Claude to PPO. Constitutional AI, which replaces some human labels with AI feedback, is covered in Chapter 14 (Section 6).

## Frontier and open models

**GPT-4 (2023).** The GPT-4 technical report says that GPT-4's behavior was fine-tuned with RLHF and gives some detail about the safety part of the recipe (OpenAI 2023). Besides an additional set of safety-relevant RLHF training prompts, OpenAI used **rule-based reward models (RBRMs)**: zero-shot GPT-4 classifiers that take a prompt, the policy's output, and a human-written rubric, for example asking whether the output is a refusal in the desired style, a refusal in an undesired (evasive) style, or contains disallowed content, and that "provide an additional reward signal to the GPT-4 policy model during PPO fine-tuning on a subset of training prompts." The report credits these mitigations, together with its other safety work, with decreasing the model's tendency to respond to requests for disallowed content by 82% compared with GPT-3.5.

**Llama 2-Chat (2023).** Meta's Llama 2 paper is the most detailed public description of PPO-based RLHF at scale (Touvron et al. 2023). Meta collected over 1.4 million binary comparisons of its own, trained separate helpfulness and safety reward models with the margin loss of Section 5, and combined them with a rule that favors the safety score on potentially unsafe prompts. RLHF ran in five rounds, RLHF-V1 to V5, with fresh preference data collected on the latest model before each. The early rounds used only rejection-sampling fine-tuning (sampling several responses, keeping the best by reward, and fine-tuning on it); from RLHF-V4 on, rejection sampling was followed by PPO. PPO used a batch of 512 prompts, a clip threshold of 0.2, a learning rate of $`10^{-6}`$, $\beta = 0.01$ for the 7B and 13B models and $\beta = 0.005$ for the 34B and 70B models, and whitened reward scores. The resulting Llama 2-Chat models outperformed open-source chat models on most benchmarks the authors tested and, in their human evaluations of helpfulness and safety, appeared to be a suitable substitute for some closed-source models.

## A close relative: Sparrow

DeepMind's **Sparrow** (Glaese et al. 2022), an information-seeking dialogue agent built on a 70-billion-parameter model, used the same overall RLHF recipe, human preference judgments distilled into reward models followed by RL, but optimized the policy with a synchronous advantage actor-critic (A2C) algorithm instead of PPO. It added two ideas. First, it broke good behavior down into natural-language rules (for example, not pretending to have a human identity) and asked raters about each rule separately, which allowed rule-conditional reward models. Second, it had the agent retrieve and cite evidence for factual claims, and raters judged responses together with that evidence. The authors report that the evidence supported the sampled response 78% of the time on factual questions, and that under adversarial probing Sparrow violated its rules only 8% of the time. Sparrow is a useful reminder that "RLHF" names the recipe of learning a reward from human judgments and optimizing it with RL; PPO is the most common optimizer, not the only one.

## Does PPO hold up?

From 2023, simpler methods, especially DPO (Chapter 14), became popular because they avoid the four models and the sampling loop of Section 7. Did that mean PPO was worse? Two controlled studies in 2024 suggest that, when tuned carefully, it was not.

- Xu et al. (2024) compared DPO and PPO on dialogue and code generation. They argued that DPO is sensitive to distribution shift between the policy's outputs and the preference data, identified advantage normalization, large batch sizes, and an exponential moving average of the reference model as critical for PPO, and found that PPO consistently outperformed DPO in their experiments, achieving state-of-the-art results on the CodeContests competitive programming benchmark.
- Ivison et al. (2024) disentangled four aspects of learning from preferences: the preference data, the learning algorithm, the reward model, and the prompts used for policy training. Better preference data gave the largest improvements, followed by the choice of algorithm. PPO outperformed DPO by up to 2.5% in math and 1.2% in general domains.

Neither study says PPO is always the right choice. Both say its reputation for being hard to use comes from its many interacting details (Section 7), not from a ceiling on what it can achieve.

## After 2024

The center of gravity has since shifted. Many open post-training recipes since late 2024 use DPO as a cheap offline stage and critic-free, group-baseline RL methods such as GRPO for online RL, a shift that Chapter 14 (Section 8) documents. But PPO has not disappeared:

- **RL with verifiable rewards.** Tulu 3 used PPO for its final stage of reinforcement learning with verifiable rewards (RLVR), where the reward is a check of the answer, such as a correct math result or satisfied instruction constraints, instead of a learned reward model (Lambert et al. 2024). OLMo 2 followed the Tulu 3 recipe, running RLVR with PPO for its 7B and 13B models, with PPO's value function initialized from the corresponding reward models, and with GRPO for its 1B and 32B models (Team OLMo 2024).
- **Value-based variants for reasoning.** VAPO is a value-model-based PPO variant for long chain-of-thought reasoning that addresses value-model bias, heterogeneous sequence lengths, and sparse rewards; with a Qwen2.5-32B base model it reported a score of 60.4 on AIME 2024 (Yue et al. 2025). ByteDance's Seed1.5-Thinking reasoning model was trained with techniques from VAPO and its critic-free sibling DAPO (ByteDance Seed 2025).
- **Closed reasoning models.** OpenAI's o1 and the reasoning models that followed are trained with large-scale RL, but their developers do not disclose the policy-optimization algorithm. Any claim that they use PPO, or anything else, is a guess.

## Summary table

The table below summarizes what PPO-based RLHF achieved in the models discussed in this section, as reported by their developers. (The Claude 2 row records that the model was trained with RLHF and Constitutional AI; its RL algorithm is not named.)

| Model | What PPO-based RLHF achieved | Source |
|---|---|---|
| GPT-2 fine-tuned from human preferences | Learned stylistic continuation (positive sentiment, descriptive text) from only 5,000 human comparisons | Ziegler et al. 2019 |
| Summarization policies (TL;DR) | Summaries preferred over human references and over much larger supervised models; transferred to CNN/DM news without news-specific training | Stiennon et al. 2020 |
| InstructGPT | Outputs of the 1.3B model preferred to those of 175B GPT-3; more truthful, less toxic, with minimal regressions on public NLP benchmarks | Ouyang et al. 2022 |
| Helpful-and-harmless assistant | Improved almost all NLP evaluations for large models (an "alignment bonus"), stayed compatible with coding and summarization skills, and was updated weekly with fresh feedback | Bai et al. 2022 |
| ChatGPT | Conversational assistant fine-tuned from a GPT-3.5 model with InstructGPT's methods, using PPO over several iterations | OpenAI 2022 |
| GPT-4 | PPO against a reward model, plus rule-based reward models during PPO; with the other safety steps, 82% less likely than GPT-3.5 to respond to requests for disallowed content | OpenAI 2023 |
| Llama 2-Chat | Rejection-sampling fine-tuning, then PPO; outperformed open-source chat models on most benchmarks and in human evaluations of helpfulness and safety | Touvron et al. 2023 |
| Claude 2 | Trained with RLHF and Constitutional AI (which has an RL phase); improved over Claude 1.3 in helpfulness and honesty; RL algorithm not named | Anthropic 2023 |

## What the record shows

A few patterns run through these results.

- **Post-training beats scale on what users notice.** A 1.3B InstructGPT model was preferred to 175B GPT-3, and summarization policies beat much larger supervised models. Pretraining supplies the knowledge; RLHF decides how it is used.
- **The alignment tax can be small, or negative, for large models.** InstructGPT reduced benchmark regressions by mixing in pretraining gradients, and Bai et al. found an alignment bonus for their larger models.
- **The recipe is iterative.** ChatGPT, Anthropic's assistant, and Llama 2-Chat all repeated the loop of collecting preferences on the latest model, updating the reward model, and running RL again.
- **Safety became part of the reward.** Separate safety reward models (Llama 2), rule-based reward models (GPT-4), and rule-conditional reward models (Sparrow) all extend the single scalar reward of Section 5 with explicit safety signals.
- **The same weaknesses keep appearing.** Verbosity (ChatGPT), over-optimization (Stiennon et al.), and exploitation of labeler heuristics (Ziegler et al.) are the failure modes that Sections 5 and 6 predicted.

Those weaknesses, together with PPO's cost and complexity, are where Chapter 14 begins: its first section examines the limits of PPO-based RLHF, and the rest of the chapter covers the critic-free, direct, verifiable, and AI-feedback methods that followed.

## Key takeaways

- GPT-3 was not trained with RLHF; InstructGPT fine-tuned GPT-3 with SFT and PPO, and ChatGPT applied the same methods to a GPT-3.5 model over several iterations.
- The early demonstrations were GPT-2 stylistic continuation from about 5,000 comparisons and TL;DR summaries preferred over human references.
- InstructGPT's 1.3B model was preferred to 175B GPT-3, and was more truthful and less toxic, with regressions on NLP benchmarks reduced by PPO-ptx.
- Anthropic's HH assistant, GPT-4 (with rule-based reward models during PPO), and Llama 2-Chat (rejection sampling, then PPO) extended the recipe to assistants and frontier models; Claude models were trained with RLHF, but their RL algorithm is not disclosed.
- Sparrow used the same recipe with A2C instead of PPO; carefully tuned PPO matched or beat DPO in controlled comparisons.
- After 2024, PPO remains in use for RLVR (Tulu 3, OLMo 2) and in value-based variants (VAPO, Seed1.5-Thinking), while most open recipes moved to DPO and GRPO-style methods (Chapter 14).

## Further reading

Anthropic. "Model Card and Evaluations for Claude Models." July 2023. https://www-cdn.anthropic.com/bd2a28d2535bfb0494cc8e2a3bf135d2e7523226.pdf.

Bai, Yuntao, et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." arXiv preprint arXiv:2204.05862, 2022. https://arxiv.org/abs/2204.05862.

Brown, Tom B., et al. "Language Models Are Few-Shot Learners." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2005.14165.

ByteDance Seed. "Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning." arXiv preprint arXiv:2504.13914, 2025. https://arxiv.org/abs/2504.13914.

Glaese, Amelia, et al. "Improving Alignment of Dialogue Agents via Targeted Human Judgements." arXiv preprint arXiv:2209.14375, 2022. https://arxiv.org/abs/2209.14375.

Ivison, Hamish, et al. "Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback." In *Advances in Neural Information Processing Systems 37*, 2024. https://arxiv.org/abs/2406.09279.

Lambert, Nathan, et al. "Tülu 3: Pushing Frontiers in Open Language Model Post-Training." arXiv preprint arXiv:2411.15124, 2024. https://arxiv.org/abs/2411.15124.

OpenAI. "GPT-4 Technical Report." arXiv preprint arXiv:2303.08774, 2023. https://arxiv.org/abs/2303.08774.

OpenAI. "Introducing ChatGPT." November 30, 2022. https://openai.com/index/chatgpt/.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Stiennon, Nisan, et al. "Learning to Summarize from Human Feedback." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2009.01325.

Team OLMo. "2 OLMo 2 Furious." arXiv preprint arXiv:2501.00656, 2024. https://arxiv.org/abs/2501.00656.

Touvron, Hugo, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv preprint arXiv:2307.09288, 2023. https://arxiv.org/abs/2307.09288.

Xu, Shusheng, et al. "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study." In *Proceedings of the 41st International Conference on Machine Learning*, 2024. https://arxiv.org/abs/2404.10719.

Yue, Yu, et al. "VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks." arXiv preprint arXiv:2504.05118, 2025. https://arxiv.org/abs/2504.05118.

Ziegler, Daniel M., et al. "Fine-Tuning Language Models from Human Preferences." arXiv preprint arXiv:1909.08593, 2019. https://arxiv.org/abs/1909.08593.
