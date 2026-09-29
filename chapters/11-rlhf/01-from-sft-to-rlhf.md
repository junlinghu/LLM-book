# 11.1 From SFT to RLHF

Chapter 9 ended with a model that can hold a conversation. Supervised fine-tuning (SFT) took a pretrained model, which only continues text, and trained it on demonstrations of an assistant answering prompts. The result follows instructions, uses the chat template, and stops when its turn is over. But SFT has a blind spot built into its loss: it only ever tells the model "produce this," and never "this answer is better than that one." This section explains what imitation cannot teach, why human preferences are a practical way to fill the gap, what we want the resulting model to be, and how the idea of learning from human feedback grew from a robotics experiment in 2017 into the method behind ChatGPT.

## Where Chapter 9 left off

Recall the SFT objective. Given a dataset of prompts $x$ and demonstrations $y$, the model minimizes the cross-entropy of the demonstration tokens,

```math
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\,\mathbb{E}_{(x, y) \sim \mathcal{D}_{\mathrm{demo}}} \sum_{t=1}^{T} \log \pi_{\theta}(y_t \mid x, y_{\lt t}),
```

where we now write the language model as $`\pi_{\theta}`$, anticipating its role as a policy in Section 2. Every term in this sum pushes probability toward a token that a person wrote. The loss has no term for what the model itself would have written, and no way to express that one response, though acceptable, is worse than another.

This is a remarkably effective recipe. InstructGPT's SFT stage used only about 13,000 training prompts (Ouyang et al. 2022), and Chapter 9 showed that a small, clean, diverse set of demonstrations goes a long way. Yet the same authors found that humans preferred the outputs of their RL-trained models to those of the SFT model they started from, at every model size they tried. Something was left on the table. What?

## What SFT cannot teach

### Telling good answers from better ones

For most prompts there are many acceptable responses. An explanation of photosynthesis can be short or long, formal or friendly, with or without an example. A demonstration picks one, and SFT pushes the model toward it. With enough demonstrations, the model learns the *distribution* of responses people write, including their variation in quality. It learns to produce text that looks like the demonstrations on average; it does not learn which features of a response make people like it more.

Put differently, SFT treats every demonstration as equally good and every other response as equally bad. Real quality is graded. A response can be correct but unhelpfully terse, or helpful but slightly wrong, and people have consistent opinions about which of two such responses they would rather receive. That comparative signal is exactly what the SFT loss throws away.

### Avoiding mistakes it never saw

SFT shows the model only positive examples. It never sees a response that invents a citation, misreads the question, or answers a harmful request, labeled as such. A model trained this way can reduce the probability of bad responses only indirectly, by increasing the probability of good ones.

The problem is sharpened by *teacher forcing* (Section 7.3). During SFT, every prediction is conditioned on a prefix written by a person. At inference, the model conditions on its own earlier tokens. Once it has generated something slightly off, say a wrong intermediate step, it is in a state that never appeared in the demonstrations, and nothing in training told it how to recover. This mismatch between training on reference prefixes and generating from its own is often called *exposure bias*. Training on the model's *own* samples, and scoring them, attacks the mismatch directly: the model learns from the states it actually visits.

### Going beyond the demonstrations

Imitation is capped by the quality of what is imitated. If demonstrations are written by people who are rushed, or who are not experts in every topic, the model learns their level of quality. For hard tasks such as long proofs, subtle code, or careful summaries of long documents, excellent demonstrations are slow and expensive to write. Chapter 9 also noted that fine-tuning on facts the model does not already know tends to teach it to state things confidently rather than to know them (Gekhman et al. 2024). A training signal that can reward responses *better* than any single demonstration, and penalize confident errors, would remove the cap.

## Preferences are easier to give than demonstrations

The key practical observation behind RLHF is that people can often *recognize* a good answer more easily than they can *write* one. Reading two summaries of a long article and saying which one is more accurate takes a minute or two. Writing an excellent summary from scratch takes much longer. Judging which of two programs is cleaner is easier than writing the cleanest program. This asymmetry between producing and evaluating is familiar from everyday life: most of us can tell which of two restaurants we prefer without being able to cook either meal.

Comparisons have further advantages:

- **They are relative.** Asking "which is better?" avoids the calibration problems of absolute scores. Two labelers may disagree about whether a response deserves a 6 or a 7 out of 10, yet agree about which of two responses is better. Section 4 compares comparisons, rankings, and ratings in detail.
- **They include negative information.** Every comparison says something about what *not* to do. The rejected response is often a realistic mistake that the model itself made.
- **They are on-distribution.** The responses being compared are sampled from the model being trained, so the feedback concerns exactly the kinds of outputs the model produces, including its characteristic errors.
- **They scale beyond the labelers' own ability.** A labeler who could not write a response of a certain quality can still often recognize it, so the model can in principle be pushed past the level of its demonstrations.

Comparisons are not free of problems. People disagree, they are swayed by length and confident tone, and they can be fooled by answers that sound right. Section 5 and Chapter 14 return to these weaknesses. But comparisons are cheap, plentiful, and informative, and they can be turned into a training signal.

That turning is the whole job of RLHF. A comparison cannot be differentiated like a cross-entropy loss. Instead, RLHF first fits a *reward model* that predicts which responses people prefer (Section 5), then uses reinforcement learning to change the language model so that its own samples score highly under that reward model (Sections 6 and 7). Reinforcement learning is the natural tool because the signal arrives only after the model has generated a complete response, and because the model must learn from its own samples rather than from fixed targets, exactly the setting that Chapter 10 prepared.

## Goals: helpful, honest, and harmless

What should the preferences reward? Askell et al. (2021) proposed a short answer that stuck: an assistant should be **helpful, honest, and harmless** ("HHH").

- **Helpful**: it tries to do what the user asked, answers the actual question, asks for clarification when the request is ambiguous, and is efficient with the user's time.
- **Honest**: it gives accurate information, expresses appropriate uncertainty, and does not deceive. Honesty is hard to reward directly, because a labeler often cannot check whether a confident statement is true.
- **Harmless**: it avoids causing harm to the user or others, for example by declining to help with dangerous activities, while not being needlessly preachy or evasive.

Ouyang et al. (2022) adopted this framing for InstructGPT, and Bai et al. (2022) trained Anthropic's assistant with separate helpfulness and harmlessness preference data. These goals are not always compatible. A maximally helpful model follows every instruction, including harmful ones; a maximally harmless model refuses anything borderline. Bai et al. found that the two objectives are in tension, which they could measure both in their preference models and in the trained policies. In practice, labeling guidelines spell out how to trade them off, and later systems train separate reward models for different goals, as Llama 2-Chat did for helpfulness and safety (Section 8).

The HHH goals also show why a single scalar reward is an approximation. "Better" depends on the person, the task, and the context. RLHF compresses a population of labelers' judgments, filtered through written guidelines, into one number per response. Much of the craft of RLHF, and many of its failure modes, come from that compression.

## A short history

Learning from human preferences did not start with language models.

**Robots and Atari (2017).** Christiano et al. (2017) asked whether an RL agent could learn a task for which nobody writes down a reward function. People watched pairs of short video clips of the agent's behavior and said which clip was closer to the goal. A reward model was fit to these comparisons with a Bradley-Terry model (Section 5), and a standard deep RL algorithm optimized the learned reward while more comparisons were collected. The method solved Atari games and simulated robot locomotion tasks with feedback on less than 1% of the agent's interactions, and it taught behaviors with no obvious reward function: a simulated Hopper robot learned to do backflips from 900 queries answered in less than an hour. Every ingredient of modern RLHF, including pairwise comparisons, a learned reward model, and RL against it, is already here.

**Language models (2019).** Ziegler et al. (2019) applied the idea to the 774-million-parameter GPT-2. Labelers chose the best of four continuations of a passage. After fitting a reward model, the authors fine-tuned GPT-2 with PPO (Section 10.7) and, crucially, added a penalty on the KL divergence between the fine-tuned model and the original one, which is the objective of Section 6. With about 5,000 comparisons the model learned stylistic continuation, such as writing positively or descriptively. On summarization the models learned to copy relevant sentences from the article, which scored well with labelers but, as the authors noted, may have exploited the fact that labelers relied on simple heuristics. This was an early glimpse of a theme that Chapter 14 develops: a policy optimizes what the feedback *rewards*, not what the designers *meant*.

**Summarization (2020).** Stiennon et al. (2020) scaled the approach up and made it work convincingly. They collected over 64,000 comparisons of summaries of Reddit posts, took care that labelers agreed with the researchers, and trained models of up to 6.7 billion parameters. Summaries from their RL-trained policies were preferred to the human-written reference summaries and to those of much larger models trained with supervised learning alone, and the policies transferred to news articles without news-specific training. The paper also showed the danger: optimizing too hard against the reward model eventually produced summaries that the reward model loved and people did not.

**Instruction following (2022).** Ouyang et al. (2022) turned the recipe into a general assistant. Starting from GPT-3, they trained an SFT model on labeler demonstrations, a reward model on labeler rankings of model outputs for prompts submitted to the OpenAI API, and then a policy with PPO against the reward model. The resulting InstructGPT models were strongly preferred to GPT-3: outputs from the 1.3-billion-parameter InstructGPT model were preferred to those of the 175-billion-parameter GPT-3, a model more than 100 times larger. The InstructGPT paper is the clearest description of the three-stage pipeline that this chapter builds (Section 3).

**ChatGPT (2022).** In November 2022, OpenAI released ChatGPT, trained with "the same methods as InstructGPT, but with slight differences in the data collection setup," in its creators' words, starting from a model in the GPT-3.5 series (OpenAI 2022). Its reception made RLHF the default final stage of building a chat model. In the same year, Anthropic published its own helpful and harmless assistant trained with RLHF (Bai et al. 2022), and DeepMind described Sparrow (Glaese et al. 2022). Section 8 tells the rest of the story, up to 2026.

What changed between 2017 and 2022 was mostly the policy. With a strong pretrained model and an SFT stage to start from, RL no longer had to discover good behavior from scratch. It only had to *choose* among behaviors the model could already produce, shifting probability toward the responses people prefer. That is why a relatively small amount of preference data, collected on tens of thousands of prompts, could change a model so visibly.

To do that choosing, we need to see text generation through the lens of Chapter 10. What is the state, what is the action, and where does the reward come from when a model writes a paragraph?

## Key takeaways

- SFT maximizes the likelihood of demonstrations; it cannot express that one acceptable response is better than another, never sees labeled mistakes, trains only on human-written prefixes, and is capped by the quality of its demonstrations.
- People can compare responses more easily and consistently than they can write perfect ones, and comparisons carry negative information about the model's own outputs.
- RLHF turns comparisons into a reward model and then uses reinforcement learning to make the model's own samples score highly.
- The usual goals are helpful, honest, and harmless; they can conflict, and a single scalar reward is only an approximation of them.
- The method grew from learning Atari and robot behaviors from clip comparisons (Christiano et al. 2017), to GPT-2 and summarization (Ziegler et al. 2019; Stiennon et al. 2020), to InstructGPT and ChatGPT (Ouyang et al. 2022; OpenAI 2022).

## Further reading

Askell, Amanda, et al. "A General Language Assistant as a Laboratory for Alignment." arXiv preprint arXiv:2112.00861, 2021. https://arxiv.org/abs/2112.00861.

Bai, Yuntao, et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." arXiv preprint arXiv:2204.05862, 2022. https://arxiv.org/abs/2204.05862.

Christiano, Paul F., et al. "Deep Reinforcement Learning from Human Preferences." In *Advances in Neural Information Processing Systems 30*, 2017. https://arxiv.org/abs/1706.03741.

Gekhman, Zorik, et al. "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 2024. https://arxiv.org/abs/2405.05904.

Glaese, Amelia, et al. "Improving Alignment of Dialogue Agents via Targeted Human Judgements." arXiv preprint arXiv:2209.14375, 2022. https://arxiv.org/abs/2209.14375.

OpenAI. "Introducing ChatGPT." November 30, 2022. https://openai.com/index/chatgpt/.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Stiennon, Nisan, et al. "Learning to Summarize from Human Feedback." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2009.01325.

Ziegler, Daniel M., et al. "Fine-Tuning Language Models from Human Preferences." arXiv preprint arXiv:1909.08593, 2019. https://arxiv.org/abs/1909.08593.
