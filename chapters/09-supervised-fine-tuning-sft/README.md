# Chapter 9: Supervised Fine-Tuning (SFT)

Chapter 8 ended with a base model that continues text. Asked a question, it may answer, or it may continue with more questions, because it was trained to predict what comes next in web pages and books, not to help the person typing. **Supervised fine-tuning** closes much of that gap with a simple idea: collect examples of the behavior we want, each a prompt paired with a good response, and keep training the pretrained model on them with the same next-token loss. Only two things change. The loss is computed on the response tokens, not on the prompt, and the text is laid out in a fixed conversation format with special tokens marking who is speaking (Chapter 5). The idea grew out of task fine-tuning of pretrained models (Radford et al. 2018), was scaled up by training on many tasks phrased as instructions (Wei et al. 2022a; Sanh et al. 2022; Chung et al. 2024), and became the first stage of turning GPT-3 into an assistant that follows instructions (Ouyang et al. 2022). This chapter defines the SFT objective and its loss masking, formats conversations with chat templates, looks at where instruction data comes from and why quality matters more than quantity, sets out a training recipe and its memory cost, introduces parameter-efficient fine-tuning with LoRA, and ends with how to evaluate an SFT model and what imitation alone cannot teach.

## Sections

1. **[From Base Model to Assistant](01-from-base-model-to-assistant.md)**

   What SmolLM2-135M and its instruction-tuned version do with the same requests, why few-shot prompting helps only so far, what supervised fine-tuning on demonstrations adds, a short history from task fine-tuning to FLAN and InstructGPT, the superficial alignment hypothesis, and where SFT sits in the training pipeline.

2. **[The SFT Objective](02-the-sft-objective.md)**

   Next-token cross-entropy restricted to response tokens, loss masks built turn by turn for a real multi-turn conversation, why the prompt is masked and the end-of-turn token is kept, how averaging over tokens or over examples changes the batch loss, and teacher forcing and exposure bias.

3. **[Formatting Conversations](03-formatting-conversations.md)**

   The same conversation in three chat templates, the generation prompt, the measured cost of prompting a model in the wrong format, adding role tokens to a vocabulary, keeping user text from producing special tokens, system prompts, tool calls and structured output, and truncating long conversations.

4. **[Instruction Data](04-instruction-data.md)**

   Human demonstrations, templated NLP datasets in the style of FLAN, model-generated data with Self-Instruct's similarity filter, the limits of imitating a stronger model, reasoning traces, data mixtures, and deduplication, 13-gram decontamination, and licenses, each with a small worked example.

5. **[The Training Recipe](05-the-training-recipe.md)**

   Fine-tuning SmolLM2-135M on a synthetic instruction dataset on a CPU (from 0 to about 60 percent exact match, with held-out perplexity rising from 41.7 to 47.0), padding versus packing with a block-diagonal mask, why validation loss can mislead, catastrophic forgetting, and the 16-bytes-per-parameter memory bill of full fine-tuning.

6. **[Parameter-Efficient Fine-Tuning](06-parameter-efficient-fine-tuning.md)**

   Adapters, prefix and prompt tuning, the LoRA update and its initialization, a parameter count for SmolLM2 checked against code, LoRA written from scratch and compared with full fine-tuning at three ranks, merging adapters into the weights, QLoRA, and why LoRA learns less and forgets less.

7. **[Evaluating SFT Models and the Limits of Imitation](07-evaluating-sft-models-and-the-limits-of-imitation.md)**

   Why held-out loss is not helpfulness, verifiable instruction checks in the style of IFEval, pairwise comparison and the biases of LLM judges (with a small judge whose verdict flips when the answers are swapped), regression checks, hallucination and fragile safety after fine-tuning, and what learning from demonstrations cannot teach.

## Suggested code labs

1. **A chat data pipeline with loss masking.** Add system, user, and assistant role tokens to your Chapter 5 tokenizer (or use a Hugging Face tokenizer's chat template), and write a function that turns a multi-turn conversation into input IDs and labels, with every non-assistant token set to $`-100`$ and the end-of-turn token kept in the loss. Add packing with a block-diagonal attention mask and per-example position IDs. Write tests: decoding the unmasked labels must give back exactly the assistant turns, and a packed batch must produce the same per-token losses as the same examples run one at a time. Every later fine-tuning lab reuses this pipeline.
2. **Instruction-tune your mini GPT.** Starting from the model you pretrained in Chapter 8, build a synthetic instruction dataset with checkable answers (for example, "reverse this word", "add these numbers", "sort these letters", written with several phrasings each), and fine-tune the model on it with the Lab 1 pipeline. Measure exact-match accuracy on held-out prompts and held-out phrasings, and compare with the base model prompted few-shot. Ablate loss masking (loss on all tokens versus response only) and the number of epochs, and track perplexity on held-out pretraining text to measure forgetting.
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
