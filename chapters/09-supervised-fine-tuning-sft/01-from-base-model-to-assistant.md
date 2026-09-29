# 9.1 From Base Model to Assistant

Chapter 8 ended with a pretrained language model: a decoder-only Transformer that has read an enormous amount of text and learned to predict the next token. Such a **base model** knows a great deal. It can complete a sentence about the French Revolution, continue a Python function, or finish a recipe. What it does not do reliably is the thing people most want from a language model, which is to *respond* to a request. This section looks at that gap, explains why prompting alone does not close it, and introduces the idea that does: **supervised fine-tuning (SFT)**, continued training on examples of the behavior we want. The rest of the chapter develops SFT in detail.

## What a base model does with a request

A base model is trained to continue documents. Given a prompt, it produces whatever text is most likely to follow in its training data, which is mostly web pages, books, and code. A question can be followed by many things in a document: an answer, but also another question on a quiz page, a list of related questions on a forum, or a sentence that uses the question as a heading. The model has no notion that a person is waiting for an answer.

We can see this directly with a small open model. SmolLM2-135M is a 135-million-parameter decoder-only model pretrained on 2 trillion tokens of web text, educational text, and code (Allal et al. 2025). Its authors also released SmolLM2-135M-Instruct, built from the same base model by supervised fine-tuning on a mixture of public and curated instruction datasets, followed by a stage of preference optimization. Both models are small enough to run on a laptop CPU, which is what every experiment in this chapter uses. The table shows the first 40 tokens each model generates, with greedy decoding, for three requests. The base model receives the request as plain text; the instruction-tuned model receives it in its chat format, which Section 3 explains (the code is in Appendix A.1).

| Request | Base model | Instruction-tuned model |
|---|---|---|
| What is the capital of France? | "The capital of France is Paris. What is the capital of France? The capital of France is Paris. What is the capital of France? The capital…" | "The capital of France is Paris." and then stops |
| Give me three tips for staying focused while studying. | "First, try to focus on one thing at a time. If you are studying for a test, try to focus on one question at a time. If you are studying for a presentation,…" | "1. Create a Dedicated Study Space: Choose a quiet and comfortable place where you can focus without distractions. This could be a library, a coffee shop, or even a peaceful garden. 2…" |
| Translate 'good morning' into Spanish. | "The Spanish word for good morning is 'morir'. The word for good night is 'suena'. The word for bad morning is 'morir'. The…" | "'Good morning' es 'Hola' in español." and then stops |

Three differences stand out. First, the base model knows that Paris is the capital of France, but it does not stop: it produces the answer and then keeps writing what looks like a repetitive FAQ page, because nothing in its training taught it that a response has an end. The instruction-tuned model answers in one sentence and then emits its end-of-turn token, `<|im_end|>`, after 8 tokens. Given the same chat-formatted prompt, the base model never produces that token within 60 tokens. Second, the tips request shows a difference in form: the instruction-tuned model produces a numbered list, the format a person asking for "three tips" expects, while the base model writes a loose paragraph that drifts. Third, neither model translates correctly. The base model invents Spanish words, and the instruction-tuned model gives a fluent but wrong answer ("Hola" means "hello"; "good morning" is "buenos días"). Fine-tuning changed how the model responds, not how much Spanish it knows. That observation, that SFT mainly shapes format and behavior rather than adding knowledge, will come up repeatedly in this chapter.

## Prompting helps only so far

Before fine-tuning, one can try to coax a base model into answering with a better prompt. Section 8.6 described **in-context learning**: examples of a task placed in the prompt steer the model toward continuing the pattern (Brown et al. 2020). For question answering, a prompt can begin with a few question-answer pairs and end with the new question.

With three examples ("Q: What is the capital of Italy? A: Rome." and similar lines for Germany and Spain), the SmolLM2 base model answers "Paris." for France and "Tokyo." for Japan (Appendix A.2). That works, but it has costs and limits:

- **It costs context.** The three examples take 47 tokens, while the question alone takes 12. Every request pays for the examples, and a realistic assistant would need examples covering many kinds of requests.
- **It still imitates a document.** After "Paris." the model continues with a new line starting "Q: What is the capital of England", because the most likely continuation of a list of questions and answers is another question. The application has to cut the output at the right place.
- **It does not generalize across kinds of requests.** Given the tips request in the same format, the base model answers "Read the book." and then continues with invented questions about the capitals of the United States and the United Kingdom, copying the pattern of the examples rather than the intent of the request.

Prompt engineering can push a base model further than this, especially a large one, and it remains useful. But it treats the symptom. The model is still predicting documents, and every application has to rediscover how to disguise its request as the beginning of the right kind of document.

## Supervised fine-tuning

The direct fix is to change the model so that responding *is* the most likely continuation. **Supervised fine-tuning** does this by continuing to train the pretrained model on **demonstrations**: pairs of a prompt and a desired response, written or checked by people or produced by another model. The training objective is the same next-token cross-entropy used in pretraining (Chapter 8), with two changes that Sections 2 and 3 cover in detail:

- The loss is computed only on the response tokens. The model is taught to produce responses, not to imitate prompts.
- Every example is laid out in a fixed conversation format, a **chat template**, in which special tokens mark where each turn starts and ends and who is speaking.

After fine-tuning on enough demonstrations, a prompt formatted as a user turn is followed, with high probability, by an assistant turn that answers it and ends with an end-of-turn token. Nothing about the architecture changes. The same weights are adjusted by gradient descent, usually for a tiny fraction of the compute spent on pretraining.

The word "supervised" distinguishes this stage from pretraining, which is often called self-supervised because its labels come from the text itself. In SFT, each example comes with a target that someone chose deliberately.

## A short history

Fine-tuning a pretrained model is older than chat assistants. The first GPT paper (Radford et al. 2018) pretrained a decoder-only Transformer on unlabeled text and then fine-tuned a copy of it for each task, such as textual entailment or question answering, with a small task-specific output layer. BERT (Devlin et al. 2019; Section 7.6) did the same with an encoder-only model. Each fine-tuned model did one task.

The next step was to fine-tune one model on many tasks at once, each phrased as a natural-language instruction. FLAN (Wei et al. 2022a) took a 137-billion-parameter pretrained model and fine-tuned it on more than 60 existing NLP datasets, each rewritten with templates such as "Is the sentiment of this review positive or negative?". The fine-tuned model performed much better than the base model on *unseen* tasks given only an instruction, which showed that training on instructions teaches something general: how to follow an instruction. T0 (Sanh et al. 2022) reached a similar conclusion with a different collection of prompted datasets, and Super-NaturalInstructions (Wang et al. 2022) assembled more than 1,600 tasks, each with a written definition. This approach became known as **instruction tuning**, and scaling it up, with more tasks and larger models, kept improving results (Chung et al. 2024).

Templated NLP datasets have a limitation: their tasks are academic benchmarks, not the open-ended requests people send to an assistant ("help me write an email to my landlord", "explain this error message"). InstructGPT (Ouyang et al. 2022) addressed this by having human labelers write demonstrations for prompts that real users had submitted to the OpenAI API, and for prompts the labelers wrote themselves. The InstructGPT pipeline fine-tuned GPT-3 on these demonstrations and then trained it further on human judgments of which responses were better. The result was striking: labelers preferred the outputs of a 1.3-billion-parameter InstructGPT model to those of the 175-billion-parameter GPT-3, a model more than 100 times larger. Most chat assistants since then have been built on the same foundation: a pretrained model, supervised fine-tuning on demonstrations, and further training on preferences or rewards.

## The superficial alignment hypothesis

How much does SFT teach? The LIMA study (Zhou et al. 2023a) fine-tuned a 65-billion-parameter LLaMA model on only 1,000 carefully selected prompt-response pairs, with no preference training at all. In the authors' human evaluation, LIMA's responses were often judged equivalent or preferred to those of assistants trained with far more data and effort. The authors proposed the **superficial alignment hypothesis**: almost all of a model's knowledge and abilities are learned during pretraining, and alignment through fine-tuning mainly teaches which *format and style* to use when interacting with users.

The hypothesis is a useful mental model, with caveats. It fits the SmolLM2 translation example: fine-tuning taught the model to answer briefly and stop, but did not teach it Spanish. It also fits evidence, covered in Section 7, that fine-tuning a model on facts it did not already know is learned slowly and makes it more likely to state falsehoods. On the other hand, fine-tuning can teach genuinely new skills when the demonstrations contain them, for example step-by-step reasoning distilled from a stronger model (Section 4), and a thousand examples are enough only when they are of very high quality and the base model is strong. A reasonable summary is that SFT is very effective at shaping *how* a model uses what it knows, and much less effective at changing *what* it knows.

## Where SFT sits in the pipeline

The standard pipeline for building an assistant has three kinds of training, in order:

1. **Pretraining** on a very large corpus with the next-token objective (Chapter 8). This takes almost all of the compute and produces the model's knowledge and general abilities.
2. **Supervised fine-tuning** on demonstrations. This produces a model that follows instructions, uses the chat format, and stops when it has answered.
3. **Training on preferences or rewards**, in which the model learns from comparisons of its own responses or from automatic checks, rather than from fixed demonstrations. This stage starts from the SFT model, and the SFT model often also serves as the reference that the later stage is not allowed to drift too far from.

SFT is the bridge between the first and the third stage, and it is often the stage with the largest visible effect on behavior. It is also where many important product decisions are made: which chat format the model uses, what its default tone is, how it handles tool calls and structured output, and how long its answers tend to be.

## What the rest of the chapter covers

The rest of the chapter follows the pipeline of an SFT project from the loss function to evaluation.

Section 2 defines the **SFT objective**: next-token cross-entropy restricted to the response tokens, how loss masks are built for single-turn and multi-turn conversations, and why the choice of how to average the loss over a batch matters. Section 3 covers **chat templates**, the formats that turn a conversation into one token sequence, and measures how much a model's predictions suffer when it is prompted in the wrong format. Section 4 turns to **instruction data**: human demonstrations, templated NLP datasets, data generated by other models, reasoning traces, and the filtering and deduplication that make data useful. Section 5 puts the pieces into a **training recipe**, fine-tunes SmolLM2-135M on a small synthetic instruction dataset on a CPU, and works out the memory cost of full fine-tuning. Section 6 introduces **parameter-efficient fine-tuning**, especially LoRA, implements it from scratch, and compares it with full fine-tuning on the same task. Section 7 covers **evaluation**, the risks of fine-tuning, and what learning from demonstrations alone cannot teach.

## Key takeaways

- A base model continues documents. It may answer a question, but it does not know that a response should be helpful, well formatted, or finished.
- Few-shot prompting can steer a base model, but it costs context, still imitates a document, and does not generalize well across kinds of requests.
- Supervised fine-tuning continues training on prompt-response demonstrations, with the loss on the response tokens and a fixed chat format, so that responding becomes the most likely continuation.
- Instruction tuning on many tasks phrased as instructions teaches a general ability to follow instructions (FLAN, T0, Super-NaturalInstructions), and demonstrations of open-ended assistant behavior made InstructGPT's 1.3-billion-parameter model preferred to the 175-billion-parameter GPT-3.
- The superficial alignment hypothesis holds that pretraining provides most knowledge and SFT mainly teaches format and style; SFT shapes how a model uses what it knows far more than what it knows.
- SFT sits between pretraining and training on preferences or rewards, and its model is the starting point for that later stage.

## Further reading

Allal, Loubna Ben, et al. "SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model." arXiv preprint arXiv:2502.02737, 2025. https://arxiv.org/abs/2502.02737.

Brown, Tom B., et al. "Language Models Are Few-Shot Learners." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2005.14165.

Chung, Hyung Won, et al. "Scaling Instruction-Finetuned Language Models." *Journal of Machine Learning Research* 25, no. 70 (2024): 1–53. https://arxiv.org/abs/2210.11416.

Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics*, 4171–4186, 2019. https://arxiv.org/abs/1810.04805.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Radford, Alec, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. "Improving Language Understanding by Generative Pre-Training." OpenAI technical report, 2018. https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.

Sanh, Victor, et al. "Multitask Prompted Training Enables Zero-Shot Task Generalization." In *International Conference on Learning Representations*, 2022. https://arxiv.org/abs/2110.08207.

Wang, Yizhong, et al. "Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks." In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, 2022. https://arxiv.org/abs/2204.07705.

Wei, Jason, et al. "Finetuned Language Models Are Zero-Shot Learners." In *International Conference on Learning Representations*, 2022a. https://arxiv.org/abs/2109.01652.

Zhou, Chunting, et al. "LIMA: Less Is More for Alignment." In *Advances in Neural Information Processing Systems 36*, 2023a. https://arxiv.org/abs/2305.11206.

## Appendix: Code for Section 9.1

These listings reproduce the examples in this section. Run them in order in one Python session; they need the `torch` and `transformers` packages, download the models from the Hugging Face Hub on first use, and run on a CPU.

### A.1 Base and instruction-tuned models on the same requests

Load SmolLM2-135M and its instruction-tuned version, and answer three requests with greedy decoding.

Notebook: [9.1-A.1-base-and-instruction-tuned-models-on-the-same-requests.ipynb](../../code/09-supervised-fine-tuning-sft/9.1-A.1-base-and-instruction-tuned-models-on-the-same-requests.ipynb)

### A.2 Few-shot prompting the base model

Turn the base model into a question answerer with examples in the prompt, and count what the examples cost.

Notebook: [9.1-A.2-few-shot-prompting-the-base-model.ipynb](../../code/09-supervised-fine-tuning-sft/9.1-A.2-few-shot-prompting-the-base-model.ipynb)

