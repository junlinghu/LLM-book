# 11.1 Why Evaluation Is Hard

By this point in the book you know how to build a large language model: tokenize text, train a transformer to predict the next token, fine-tune it on demonstrations, and then shape its behavior with reinforcement learning from human feedback, DPO, or GRPO. Every one of those steps had a number attached to it. Pretraining minimized cross-entropy loss. SFT minimized loss on demonstrations. RLHF maximized a learned reward. It is tempting to think that evaluating the finished model is just a matter of reading off one more number.

It is not. Evaluation is one of the hardest and least settled parts of building LLMs, and many practitioners would say it is the part that most often goes wrong. This section explains why. We look at three gaps: the gap between training loss and usefulness, the gap between offline measurements and what happens when real people use a model, and the gap that opens up whenever a measurement becomes a target. The rest of the chapter builds tools for narrowing each of them.

## What makes LLMs different from classic ML models

In a classic supervised learning problem, evaluation is comparatively easy. Suppose you train an image classifier on ten categories. There is one correct label per image, a held-out test set drawn from the same distribution as the training set, and an obvious metric: accuracy. You might add precision, recall, or a confusion matrix, but everyone agrees on what "correct" means.

A modern LLM breaks almost every one of these assumptions:

- **Outputs are open-ended.** Ask a model to "explain why the sky is blue," and there are thousands of acceptable answers that share almost no words. An exact-match metric would call all but one of them wrong.
- **The task distribution is enormous.** The same model is expected to write code, translate poetry, solve olympiad math, summarize legal contracts, and hold a casual conversation. No test set covers all of that.
- **Quality has many dimensions.** A response can be correct but rude, helpful but unsafe, fluent but false, or concise but incomplete. Different users weigh these differently.
- **The training data is (nearly) the whole internet.** A "held-out" test set may not be held out at all, because its questions, or close paraphrases, may have been scraped into pretraining data. Section 2 treats this problem of *contamination* in detail.
- **Behavior depends on how you ask.** The same model can score very differently depending on prompt wording, the number of few-shot examples, whether it is allowed to reason step by step, and the decoding settings. Sclar et al. found that meaning-preserving changes to prompt *formatting* alone (such as separators and capitalization) can swing few-shot accuracy by tens of points on some tasks for some open models.

Because of this, "How good is this model?" is not a well-posed question. A better question is: *How good is this model at this task, for these users, under these conditions, measured this way?* Much of the craft of evaluation is making each part of that sentence explicit.

## Training loss is not the same as usefulness

The pretraining objective is the average negative log-likelihood of the next token. It is an excellent training signal: it is cheap to compute, it is smooth, and it improves predictably with scale, as the scaling-law work of Kaplan et al. showed. Lower loss does correlate with better downstream ability, and loss curves are the first thing any team looks at during pretraining. So why not use it as the evaluation?

**Loss averages over every token, but usefulness depends on a few.** Consider the sentence "The capital of Australia is Canberra." Most of its tokens ("The", "capital", "of", "is") are nearly free to predict for any competent model. Almost all the information that matters is in the single token (or pair of tokens) for "Canberra." A model that gets the easy tokens slightly more right can lower its average loss while still naming Sydney. Loss measures how well a model models *text*; users care about whether it gets the few critical tokens right.

**Small loss differences can hide large capability differences.** Suppose model A has a validation loss of 2.10 nats per token and model B has 2.05. In perplexity terms (Section 2 defines perplexity as the exponentiated loss), that is $e^{2.10} \approx 8.17$ versus $e^{2.05} \approx 7.77$, a difference of about 5 percent. It is hard to know from that number alone whether B can now solve a class of math problems that A could not, or whether B simply got better at predicting boilerplate in web pages. Downstream accuracy on a task often changes much more sharply than loss does, because a task may need many steps to all be correct at once.

**Post-training deliberately moves the model away from the pretraining distribution.** After SFT and RLHF, a chat model no longer tries to imitate an arbitrary web page; it tries to answer the user helpfully. Its likelihood on held-out web text often gets *worse* even as it becomes far more useful. It may also lose calibration: the GPT-4 technical report showed that the pretrained model's confidence on multiple-choice questions matched its accuracy well, and that this calibration was reduced after post-training. So a loss-based metric can move in the opposite direction from the quality we actually care about.

**Loss says nothing about behavior that is not in the text distribution.** Refusing a dangerous request, admitting uncertainty, following a formatting instruction, and calling a tool correctly are all behaviors that a pure next-token loss on web data does not directly measure.

None of this means loss is useless. It is the right metric for comparing pretraining runs on the same data with the same tokenizer, for detecting training instabilities, and for fitting scaling laws. It is the wrong metric for answering "Is this chatbot good?"

## Offline metrics vs. online product metrics

Evaluation happens in two very different settings.

**Offline evaluation** runs a model on a fixed dataset and computes a score: accuracy on a benchmark, perplexity on a held-out corpus, win rate against a baseline judged by humans or by another LLM. Offline evaluation is cheap (relatively), fast, reproducible, and safe; you can run it on a checkpoint that no user will ever see. Almost everything in Sections 2 through 7 of this chapter is offline evaluation.

**Online evaluation** measures what happens when real users interact with a deployed model. Typical online metrics include:

| Online metric | What it captures | Main weakness |
|---|---|---|
| Thumbs-up / thumbs-down rate | Explicit user satisfaction | Few users vote; voters are not representative |
| Regeneration or edit rate | Implicit dissatisfaction with a response | Users regenerate for many reasons |
| Task completion (e.g., code accepted, ticket resolved) | Real usefulness for a workflow | Hard to define; confounded by task mix |
| Retention and session length | Long-term value | Slow, noisy; can reward addictive rather than helpful behavior |
| A/B test win rate between two model versions | Causal effect of a model change | Requires traffic and time; risk of shipping a worse model to some users |

Online metrics are closer to what we ultimately care about, but they are noisy, slow, expensive, and sometimes misleading in their own ways. For example, an assistant that flatters users may get more thumbs-up while giving worse advice; this is related to the *sycophancy* problem discussed in Chapter 10 and in Section 4 of this chapter. Online metrics also cannot be collected for capabilities you are afraid to deploy, such as the ability to assist with a cyberattack, which is why safety evaluation (Section 7) is almost entirely offline.

In practice, teams use offline evaluations as a fast, cheap filter during development, and then confirm promising candidates with a smaller number of online experiments. A key question is how well the offline metric *predicts* the online one. If a new offline benchmark score goes up but users are no happier, the benchmark is measuring something other than what you care about. Checking this correlation from time to time is part of maintaining an evaluation suite.

## Validity: are we measuring what we think we are measuring?

Social scientists and psychometricians have thought about measurement for a century, and some of their vocabulary is useful here.

- **Construct validity** asks whether a test measures the abstract ability it claims to measure. A benchmark called "reasoning" that can be solved by memorizing answers has poor construct validity.
- **Content validity** asks whether the test covers the full range of the ability. A "coding" benchmark made entirely of short, self-contained Python functions says little about maintaining a large codebase.
- **Reliability** asks whether the test gives consistent results when repeated. If a model's score changes by five points when you rerun it with a different random seed or a slightly different prompt, the test has low reliability, and small differences between models are meaningless (Section 8).

Raji et al. argue that many popular AI benchmarks are presented as measuring "general" capabilities when they actually cover a narrow, idiosyncratic slice of tasks, the "everything in the whole wide world benchmark" problem. Keeping validity questions in mind is the best defense against overclaiming.

## Goodhart's law: optimizing a benchmark vs. improving the model

The economist Charles Goodhart observed that statistical regularities tend to break down once they are used for control. The popular paraphrase is: *When a measure becomes a target, it ceases to be a good measure.* Machine learning is a field built on optimizing numbers, so it is especially vulnerable.

Manheim and Garrabrant distinguish several variants of Goodhart's law. Translated to LLMs, they look like this:

- **Regressional Goodhart.** Any metric is an imperfect proxy: $\text{metric} = \text{true quality} + \text{noise}$. If you pick the model with the highest benchmark score out of many candidates, you are partly selecting for lucky noise, so the winner's true quality is likely lower than its score suggests. This is why a model chosen by searching over many checkpoints or prompts on a test set will disappoint on fresh data.
- **Extremal Goodhart.** The relationship between metric and quality holds for typical models but breaks down at the extremes. A reward model trained on ordinary responses may assign very high scores to strange, unusually long responses it never saw during training. Gao et al. measured this *reward overoptimization* directly: as a policy is optimized harder against a learned reward model, the learned reward keeps rising while the "gold" reward first rises and then falls.
- **Causal Goodhart.** The metric correlates with quality for non-causal reasons, so intervening on the metric does not improve quality. Longer answers correlate with helpfulness in human preference data, but making answers longer does not make them more helpful. Section 6 shows how LLM judges exhibit exactly this verbosity bias.
- **Adversarial Goodhart.** Someone optimizes the metric deliberately, for example by training on the test set, tuning prompts on the test set, or submitting many private variants to a leaderboard and publishing only the best. Singh et al.'s study "The Leaderboard Illusion" documents how undisclosed private testing on Chatbot Arena let some providers test many variants and disclose selectively, which biases the public rankings.

A useful way to think about all of this: a benchmark is a sample of problems meant to stand in for a much larger population of problems we care about. Every time you look at the benchmark score and make a decision (choose a checkpoint, tweak data, adjust a prompt), you leak a little information about that specific sample into the model. After enough decisions, you have fit the sample rather than the population. This is the same overfitting you learned about in Chapter 2, but the "training" is being done by the engineers rather than by gradient descent.

### A worked example

Imagine a team fine-tuning a model to do well on a 500-problem math benchmark. They try 40 variants of their data mix, and for each they evaluate on the benchmark and keep the best one. Suppose, to make the point, that all 40 variants have *exactly* the same true accuracy of 60 percent. Each measured score is still noisy: with 500 problems, the standard error of an accuracy near 0.6 is $\sqrt{0.6 \times 0.4 / 500} \approx 0.022$, or about 2.2 points. The maximum of 40 noisy draws will typically land around two standard errors above the mean, so the "winning" variant might report roughly 64 to 65 percent. The team concludes that their best data mix gained four or five points. On a fresh set of problems, it scores 60 percent, like all the others.

The fix is procedural rather than mathematical: keep a separate development set for making decisions, touch the final test set rarely, and report how many configurations were tried. Section 8 returns to these practices.

## Benchmarks saturate, and then they mislead

A benchmark is most informative when models score in its middle range. When all frontier models score above 90 percent, the remaining differences are dominated by label errors, ambiguous questions, and noise. MMLU is the canonical example. It was very hard for models when it was released in 2020, and by the time the Humanity's Last Exam paper appeared in 2025, its authors noted that frontier LLMs were scoring over 90 percent on it. Gema et al. estimated that about 6.5 percent of MMLU questions contain errors, which puts a ceiling on how meaningful scores near the top can be. The field responds by building harder successors (MMLU-Pro, GPQA, Humanity's Last Exam), each of which will eventually saturate too. Section 3 discusses this life cycle.

Saturation is not just about difficulty. Benchmarks also age: the facts they test go stale, the software versions they depend on change, and above all their contents leak onto the web and into training data. A benchmark that was a clean test in 2021 may be partly memorized training data by 2026.

## Who is the evaluation for?

Finally, evaluation is hard because different people want different things from it:

- A **researcher** wants to know whether a new method works, which calls for controlled comparisons with everything else held fixed.
- A **model developer** wants to know whether a checkpoint is ready to ship, which calls for a broad regression suite and safety checks.
- A **product team** wants to know whether the model helps their users with their specific tasks, which calls for task-specific tests built from real usage.
- A **regulator or safety team** wants to know whether the model can cause serious harm, which calls for adversarial testing and capability evaluations.
- A **customer choosing between models** wants a fair comparison, which calls for standardized, transparent settings.

A single leaderboard number serves none of these audiences well. The rest of this chapter gives you the pieces to build the right evaluation for each: intrinsic metrics (Section 2), standard benchmarks (Section 3), the special problem of hallucination (Section 4), human evaluation (Section 5), LLM judges (Section 6), safety evaluation (Section 7), and honest reporting (Section 8).

## Key takeaways

- LLM evaluation is hard because outputs are open-ended, tasks are diverse, quality is multidimensional, training data overlaps with test data, and scores depend on prompting details.
- Training loss is a good signal for pretraining but a poor measure of usefulness; post-training can make loss worse while making the model better.
- Offline metrics are cheap and reproducible; online metrics are closer to real value but noisy and slow. Check that the first predicts the second.
- Goodhart's law is everywhere: every decision made by looking at a test set leaks information into the model, and leaderboards invite selective reporting.
- Benchmarks saturate and age; a score near the ceiling of an old benchmark says little.

## Further reading

Gao, Leo, et al. "Scaling Laws for Reward Model Overoptimization." arXiv preprint arXiv:2210.10760, 2022. https://arxiv.org/abs/2210.10760.

Kaplan, Jared, et al. "Scaling Laws for Neural Language Models." arXiv preprint arXiv:2001.08361, 2020. https://arxiv.org/abs/2001.08361.

Liang, Percy, et al. "Holistic Evaluation of Language Models." arXiv preprint arXiv:2211.09110, 2022. https://arxiv.org/abs/2211.09110.

Manheim, David, et al. "Categorizing Variants of Goodhart's Law." arXiv preprint arXiv:1803.04585, 2018. https://arxiv.org/abs/1803.04585.

OpenAI. "GPT-4 Technical Report." arXiv preprint arXiv:2303.08774, 2023. https://arxiv.org/abs/2303.08774.

Raji, Inioluwa Deborah, et al. "AI and the Everything in the Whole Wide World Benchmark." arXiv preprint arXiv:2111.15366, 2021. https://arxiv.org/abs/2111.15366.

Sclar, Melanie, et al. "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I Learned to Start Worrying about Prompt Formatting." arXiv preprint arXiv:2310.11324, 2023. https://arxiv.org/abs/2310.11324.

Singh, Shivalika, et al. "The Leaderboard Illusion." arXiv preprint arXiv:2504.20879, 2025. https://arxiv.org/abs/2504.20879.
