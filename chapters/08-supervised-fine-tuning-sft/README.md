# Chapter 8: Supervised Fine-Tuning (SFT)

Chapter 7 ended with a base model that continues text. Asked a question, it may answer, or it may continue with more questions, because it was trained to predict what comes next in web pages and books, not to help the person typing. **Supervised fine-tuning** closes much of that gap with a simple idea: collect examples of the behavior we want, each a prompt paired with a good response, and keep training the pretrained model on them with the same next-token loss. Only two things change. The loss is computed on the response tokens, not on the prompt, and the text is laid out in a fixed conversation format with special tokens marking who is speaking (Chapter 5). The idea grew out of task fine-tuning of pretrained models (Radford et al. 2018), was scaled up by training on many tasks phrased as instructions (Wei et al. 2022a; Sanh et al. 2022; Chung et al. 2024), and became the first stage of turning GPT-3 into an assistant that follows instructions (Ouyang et al. 2022). This chapter defines the SFT objective and its loss masking, formats conversations with chat templates, looks at where instruction data comes from and why quality matters more than quantity, sets out a training recipe and its memory cost, introduces parameter-efficient fine-tuning with LoRA, and ends with how to evaluate an SFT model and what imitation alone cannot teach.

## Learning goals

- Explain what a base model lacks as an assistant, and what supervised fine-tuning on demonstrations adds.
- Write the SFT loss as next-token cross-entropy restricted to response tokens, and implement loss masking for single-turn and multi-turn conversations.
- Format conversations with a chat template, and explain why the same template must be used in training and at inference.
- Compare the main sources of instruction data (human demonstrations, templated NLP tasks, and model-generated data), and explain why a small, high-quality, diverse dataset can go a long way.
- Choose a fine-tuning recipe (learning rate, epochs, packing, and mixing in other data), and estimate the memory needed for full fine-tuning.
- Derive LoRA, count its trainable parameters, and explain how QLoRA fits large models on a single GPU.
- Evaluate an SFT model, and describe the risks of fine-tuning: forgetting, hallucination, and weakened safety behavior.

## Outline

### 1. From base model to assistant
- What a base model does with a request: it continues the text in whatever way its training data makes likely, which may or may not be an answer (Section 7.6)
- Prompting helps only so far: few-shot examples (Section 7.6) take up context and still leave the model imitating a document rather than responding to a user
- **Supervised fine-tuning (SFT)**: continue training the pretrained model on demonstrations, pairs of a prompt and a desired response, so that responding becomes the most likely continuation
- A short history: fine-tuning a pretrained model for one task at a time (Radford et al. 2018; BERT in Section 6.12); **instruction tuning** on many NLP tasks rewritten as natural-language instructions, which improves zero-shot performance on unseen tasks (FLAN, a 137-billion-parameter model tuned on more than 60 datasets, Wei et al. 2022a; T0, Sanh et al. 2022; Super-NaturalInstructions, Wang et al. 2022); and demonstrations of open-ended assistant behavior written by people (Ouyang et al. 2022)
- The InstructGPT result: after SFT and further training on human preferences, outputs of a 1.3-billion-parameter model were preferred by labelers over those of the 175-billion-parameter GPT-3 (Ouyang et al. 2022)
- The **superficial alignment hypothesis**: most knowledge and ability come from pretraining, and SFT mainly teaches the format and style of responses; LIMA fine-tuned a 65-billion-parameter LLaMA model on only 1,000 carefully chosen examples and produced strong responses (Zhou et al. 2023a)
- Where SFT sits in the pipeline: after pretraining, and before any training on preferences or rewards, which starts from the SFT model

### 2. The SFT objective
- A training example is a prompt $`\mathbf{x}`$ and a response $`\mathbf{y} = (y_1, \dots, y_m)`$; the model is trained to maximize the probability of the response given the prompt, the chain rule of Section 7.1 applied to the response only

```math
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{m} \log p_\theta(y_t \mid \mathbf{x}, y_{\lt t})
```

- Implementation: concatenate prompt and response into one sequence and run the ordinary shifted next-token loss of Section 7.1, but set the labels of prompt positions to an ignore value (commonly $`-100`$, Section 5.9) so they contribute nothing
- Why mask the prompt: the model should learn to produce responses, not to imitate users; prompts may be templated or repetitive, and training on them spends model capacity on the wrong text
- Multi-turn conversations: mask every user and system turn, and train on every assistant turn, each conditioned on the whole conversation before it
- Ending the response: include the end-of-turn token in the loss, so the model learns when to stop (Section 5.6)
- Averaging the loss over a batch: dividing by the number of response tokens in the batch weights every token equally, while averaging per example first weights every example equally and so gives short responses more influence per token; the choice changes what the model learns, especially with packing and gradient accumulation
- SFT is teacher forcing (Section 6.9): the model always conditions on the reference response, never on its own earlier outputs, which is the root of a limitation discussed in Section 7

### 3. Formatting conversations
- The chat template (Section 5.6): system, user, and assistant turns flattened into one token sequence, with special tokens marking the start and end of each turn and the speaker's role
- Adding role tokens to a base model's vocabulary: new rows in the embedding and output matrices, initialized from existing embeddings rather than at random (Section 5.9)
- The template is part of the model: at inference, prompts must be formatted exactly as in training and end with an open assistant turn, or quality drops quietly
- System prompts: a first turn that sets the assistant's behavior, trained by including varied system prompts in the data
- Beyond plain text: tool calls and tool results, and structured outputs such as JSON, are represented as further turns with their own markers, and the model learns them by SFT on examples
- Special-token safety: user content must never be able to produce role tokens (Section 5.6)
- Long conversations: truncating from the start while keeping the system prompt, and dropping examples that do not fit rather than cutting a response in the middle

### 4. Instruction data
- **Human-written demonstrations**: labelers write responses to real or invented prompts; InstructGPT's SFT set had about 13,000 training prompts (Ouyang et al. 2022), and Llama 2's SFT stage used 27,540 high-quality annotations, after its authors set aside millions of third-party examples in favor of fewer, higher-quality ones (Touvron et al. 2023)
- **Templated NLP datasets**: existing labeled datasets rewritten with instruction templates; the Flan collection scaled this to 1,836 tasks and showed that mixing zero-shot, few-shot, and chain-of-thought templates helps (Chung et al. 2024; Longpre et al. 2023)
- **Model-generated data**: Self-Instruct bootstraps 52,000 instructions from 175 human-written seed tasks by prompting a model to write new tasks and answers, then filtering them (Wang et al. 2023); generating training data with a stronger teacher model is a form of sequence-level knowledge distillation (Kim and Rush 2016)
- The limits of imitation: a small model fine-tuned on a stronger model's outputs copies its style far better than its factual accuracy or ability (Gudibande et al. 2023)
- **Reasoning traces**: responses that show step-by-step reasoning (Wei et al. 2022b) teach the model to reason before answering; DeepSeek-R1's authors fine-tuned smaller open models on about 800,000 samples curated with DeepSeek-R1, and the distilled models gained strong reasoning ability from SFT alone (DeepSeek-AI 2025)
- **Quality, diversity, and mixture**: a small, diverse, carefully checked set can beat a large noisy one (Zhou et al. 2023a); open recipes such as Tulu 3 balance skills (chat, math, code, safety, and instruction following) by curating and mixing sources (Lambert et al. 2024)
- Cleaning and hygiene: deduplication, filtering of wrong or unsafe responses, removing evaluation prompts to avoid contamination (Section 7.3), and respecting the licenses and terms of use of the data and of any model used to generate it

### 5. The training recipe
- Start from the pretrained weights and use the Chapter 7 training loop with a smaller learning rate, a short warmup, and a decaying schedule (Chapter 3), for a few epochs over a dataset that is tiny compared with the pretraining corpus
- Batching: pad each example to the longest in its batch, or pack several examples into one block (Section 7.3) with a block-diagonal attention mask so examples cannot attend to each other (Section 6.4; Krell et al. 2021) and with loss masks carried along
- Watching for overfitting: validation loss on held-out responses can start rising after an epoch or two, while response quality judged by people may still improve, so check samples and task metrics rather than loss alone (Ouyang et al. 2022)
- **Catastrophic forgetting**: training only on the new data can erode abilities learned in pretraining (Kirkpatrick et al. 2017); remedies include a lower learning rate, fewer steps, and mixing some pretraining-style data or earlier tasks into the SFT data
- The memory cost of full fine-tuning with Adam and mixed precision: 2 bytes per parameter for BF16 weights, 2 for gradients, and 12 for the FP32 master copy and Adam's two moments, about 16 bytes per parameter before activations (Rajbhandari et al. 2020; Chapter 3), so a 7-billion-parameter model needs about 112 GB for these states alone
- Reducing memory: gradient checkpointing to trade compute for activation memory, gradient accumulation, and sharding optimizer states and parameters across devices (Rajbhandari et al. 2020), or training far fewer parameters (Section 6)

### 6. Parameter-efficient fine-tuning
- The idea: freeze the pretrained weights and train a small number of new parameters, which cuts optimizer memory, makes checkpoints small, and lets one base model serve many fine-tuned variants
- Early approaches: **adapters**, small bottleneck layers inserted into each block (Houlsby et al. 2019); **prefix tuning** and **prompt tuning**, trained vectors prepended to the keys and values or to the input embeddings (Li and Liang 2021; Lester et al. 2021)
- **LoRA** (Hu et al. 2022): keep a pretrained weight matrix $`W_0 \in \mathbb{R}^{d \times k}`$ frozen and learn a low-rank update, with rank $`r`$ much smaller than $`d`$ and $`k`$

```math
W = W_0 + \frac{\alpha}{r} BA, \qquad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}
```

- Initialization: $`A`$ random and $`B = 0`$, so the fine-tuned model starts exactly equal to the base model; $`\alpha`$ is a scale that keeps the update size roughly independent of $`r`$
- Counting parameters: for a $`4096 \times 4096`$ projection, full fine-tuning trains 16,777,216 weights, while LoRA with $`r = 8`$ trains $`8 \cdot (4096 + 4096) = 65{,}536`$, about 0.4%; for GPT-3 175B, the LoRA paper reported 10,000 times fewer trainable parameters and 3 times less GPU memory than full fine-tuning with Adam (Hu et al. 2022)
- Where to apply it: the attention projections of Section 6.5, and often the FFN matrices too; after training, $`BA`$ can be merged into $`W_0`$, so inference costs nothing extra
- **QLoRA** (Dettmers et al. 2023): store the frozen base model in 4-bit precision and train LoRA adapters on top in higher precision, which let its authors fine-tune a 65-billion-parameter model on a single 48 GB GPU
- LoRA versus full fine-tuning: LoRA usually learns less from large or very different data, such as new domains of code or math, but also forgets less of what the base model knew (Biderman et al. 2024)

### 7. Evaluating SFT models and the limits of imitation
- Held-out loss on responses measures fit to the demonstrations, not helpfulness; compare checkpoints with task metrics and samples instead
- Instruction-following checks that can be verified by code, such as "answer in fewer than 100 words" or "use exactly three bullet points"; IFEval collects about 500 prompts with 25 types of verifiable instructions (Zhou et al. 2023b)
- Pairwise comparison of two models' responses to the same prompts, by people or by a strong LLM acting as judge, with known biases such as favoring longer answers and the first response shown (Zheng et al. 2023)
- Regression checks on the base model's abilities: perplexity on held-out text (Section 7.1) and few-shot benchmarks, to measure forgetting
- **Hallucination**: fine-tuning on facts the base model does not already know is learned slowly and makes the model more likely to state falsehoods (Gekhman et al. 2024), consistent with the view that SFT teaches format rather than knowledge
- **Safety can be undone by fine-tuning**: fine-tuning GPT-3.5 Turbo on only 10 adversarially designed examples, at a cost below $0.20, removed much of its refusal behavior, and even benign fine-tuning data weakened it (Qi et al. 2024)
- What imitation cannot teach: SFT shows the model only good examples, never which of two responses is better or which mistakes to avoid; the model is trained on reference text but must condition on its own outputs at inference; and good demonstrations are expensive to write for hard tasks, while it is often easier for people to judge responses than to write them. These limits motivate training on comparisons and rewards, which starts from the SFT model

## Suggested code labs

1. **A chat data pipeline with loss masking.** Add system, user, and assistant role tokens to your Chapter 5 tokenizer (or use a Hugging Face tokenizer's chat template), and write a function that turns a multi-turn conversation into input IDs and labels, with every non-assistant token set to $`-100`$ and the end-of-turn token kept in the loss. Add packing with a block-diagonal attention mask and per-example position IDs. Write tests: decoding the unmasked labels must give back exactly the assistant turns, and a packed batch must produce the same per-token losses as the same examples run one at a time. Every later fine-tuning lab reuses this pipeline.
2. **Instruction-tune your mini GPT.** Starting from the model you pretrained in Chapter 7, build a synthetic instruction dataset with checkable answers (for example, "reverse this word", "add these numbers", "sort these letters", written with several phrasings each), and fine-tune the model on it with the Lab 1 pipeline. Measure exact-match accuracy on held-out prompts and held-out phrasings, and compare with the base model prompted few-shot. Ablate loss masking (loss on all tokens versus response only) and the number of epochs, and track perplexity on held-out pretraining text to measure forgetting.
3. **LoRA from scratch.** Write a LoRA wrapper for `nn.Linear` with rank $`r`$, scale $`\alpha`$, and $`B`$ initialized to zero. Check that the wrapped model's outputs equal the base model's before training, count the trainable parameters, and check that merging $`BA`$ into the weights gives the same outputs as the unmerged adapter. Repeat Lab 2 with LoRA on the attention projections for several ranks, and compare accuracy, forgetting, trainable parameters, and peak memory with full fine-tuning.
4. **Fine-tune an open base model into a chat assistant.** Take a small open base model (for example, one with about 0.5 billion parameters) and fine-tune it, fully or with LoRA, on a few thousand examples from a public instruction dataset formatted with its chat template. Compare the base and fine-tuned models on a fixed set of held-out prompts: verifiable instruction-following checks in the style of IFEval, and pairwise comparisons by an LLM judge. As an ablation, train on 1,000 carefully filtered examples versus a larger unfiltered set. Save the fine-tuned checkpoint and its evaluation prompts; they are the starting point for training on preferences and rewards.

## Key takeaways

- A base model continues text; supervised fine-tuning on prompt-response demonstrations makes responding the most likely continuation.
- The SFT loss is the pretraining loss restricted to response tokens: prompt and user tokens are masked out, and the end-of-turn token is kept so the model learns to stop.
- The chat template is part of the model, and must be identical in training and at inference.
- Instruction data comes from people, from templated NLP tasks, and from stronger models; quality and diversity matter more than size, and imitating a stronger model copies its style more easily than its ability.
- Full fine-tuning needs about 16 bytes per parameter for weights, gradients, and Adam states; LoRA trains a low-rank update with a tiny fraction of the parameters and can be merged into the weights afterward.
- SFT can cause forgetting, encourage hallucination when it teaches new facts, and weaken safety behavior, so evaluate beyond held-out loss.
- SFT learns only from good examples; it cannot tell the model which of two responses is better, which motivates training on comparisons and rewards.

## Further reading

Biderman, Dan, et al. "LoRA Learns Less and Forgets Less." *Transactions on Machine Learning Research*, 2024. https://arxiv.org/abs/2405.09673.

Chung, Hyung Won, et al. "Scaling Instruction-Finetuned Language Models." *Journal of Machine Learning Research* 25, no. 70 (2024): 1–53. https://arxiv.org/abs/2210.11416.

DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." *Nature* 645 (2025): 633–638. https://arxiv.org/abs/2501.12948.

Dettmers, Tim, et al. "QLoRA: Efficient Finetuning of Quantized LLMs." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2305.14314.

Gekhman, Zorik, et al. "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 2024. https://arxiv.org/abs/2405.05904.

Gudibande, Arnav, et al. "The False Promise of Imitating Proprietary LLMs." arXiv preprint arXiv:2305.15717, 2023. https://arxiv.org/abs/2305.15717.

Houlsby, Neil, et al. "Parameter-Efficient Transfer Learning for NLP." In *Proceedings of the 36th International Conference on Machine Learning*, 2019. https://arxiv.org/abs/1902.00751.

Hu, Edward J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." In *International Conference on Learning Representations*, 2022. https://arxiv.org/abs/2106.09685.

Kim, Yoon, and Alexander M. Rush. "Sequence-Level Knowledge Distillation." In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, 2016. https://arxiv.org/abs/1606.07947.

Kirkpatrick, James, et al. "Overcoming Catastrophic Forgetting in Neural Networks." *Proceedings of the National Academy of Sciences* 114, no. 13 (2017): 3521–3526. https://arxiv.org/abs/1612.00796.

Krell, Mario Michael, et al. "Efficient Sequence Packing without Cross-Contamination: Accelerating Large Language Models without Impacting Performance." arXiv preprint arXiv:2107.02027, 2021. https://arxiv.org/abs/2107.02027.

Lambert, Nathan, et al. "Tulu 3: Pushing Frontiers in Open Language Model Post-Training." arXiv preprint arXiv:2411.15124, 2024. https://arxiv.org/abs/2411.15124.

Lester, Brian, Rami Al-Rfou, and Noah Constant. "The Power of Scale for Parameter-Efficient Prompt Tuning." In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, 2021. https://arxiv.org/abs/2104.08691.

Li, Xiang Lisa, and Percy Liang. "Prefix-Tuning: Optimizing Continuous Prompts for Generation." In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics*, 2021. https://arxiv.org/abs/2101.00190.

Longpre, Shayne, et al. "The Flan Collection: Designing Data and Methods for Effective Instruction Tuning." In *Proceedings of the 40th International Conference on Machine Learning*, 2023. https://arxiv.org/abs/2301.13688.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Qi, Xiangyu, et al. "Fine-Tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!" In *International Conference on Learning Representations*, 2024. https://arxiv.org/abs/2310.03693.

Radford, Alec, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. "Improving Language Understanding by Generative Pre-Training." OpenAI technical report, 2018. https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.

Rajbhandari, Samyam, et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC20)*, 2020. https://arxiv.org/abs/1910.02054.

Sanh, Victor, et al. "Multitask Prompted Training Enables Zero-Shot Task Generalization." In *International Conference on Learning Representations*, 2022. https://arxiv.org/abs/2110.08207.

Touvron, Hugo, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv preprint arXiv:2307.09288, 2023. https://arxiv.org/abs/2307.09288.

Wang, Yizhong, et al. "Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks." In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, 2022. https://arxiv.org/abs/2204.07705.

Wang, Yizhong, et al. "Self-Instruct: Aligning Language Models with Self-Generated Instructions." In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics*, 2023. https://arxiv.org/abs/2212.10560.

Wei, Jason, et al. "Finetuned Language Models Are Zero-Shot Learners." In *International Conference on Learning Representations*, 2022a. https://arxiv.org/abs/2109.01652.

Wei, Jason, et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." In *Advances in Neural Information Processing Systems 35*, 2022b. https://arxiv.org/abs/2201.11903.

Zheng, Lianmin, et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2306.05685.

Zhou, Chunting, et al. "LIMA: Less Is More for Alignment." In *Advances in Neural Information Processing Systems 36*, 2023a. https://arxiv.org/abs/2305.11206.

Zhou, Jeffrey, et al. "Instruction-Following Evaluation for Large Language Models." arXiv preprint arXiv:2311.07911, 2023b. https://arxiv.org/abs/2311.07911.
