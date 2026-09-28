# Chapter 12: Evaluation

This chapter covers how to measure whether an LLM is good, and how to detect and reduce one of its most important failures: hallucination.

## Learning goals

- Choose metrics that match the task (language modeling, chat, code, safety).
- Read common benchmarks without overclaiming.
- Explain what hallucination is, why LLMs hallucinate, and how to measure and reduce it.
- Design a small, honest evaluation for a new model or fine-tune.

## Outline

### 1. Why evaluation is hard
- Training loss is not the same as usefulness
- Offline metrics vs. online product metrics
- Goodhart's law: optimizing a benchmark vs. improving the model

### 2. Intrinsic metrics
- **Perplexity and next-token loss**: what they measure and what they miss
- **Held-out data**: contamination, leakage, and train/test overlap

### 3. Benchmarks and task suites
- Knowledge and reasoning: MMLU, ARC, HellaSwag
- Math: GSM8K, MATH
- Code: HumanEval, MBPP
- Chat and preference: MT-Bench, Arena-style Elo ratings, win rates
- Reading a leaderboard critically

### 4. Hallucination
- **What it is**: fluent, confident output that is false or unsupported
- **Types**: factual errors, fabricated citations and quotes, unfaithfulness to a given source (in summarization or RAG), and reasoning errors
- **Why it happens**:
  - Next-token prediction rewards plausible text, not true text
  - Gaps, errors, and staleness in pretraining data
  - Decoding randomness
  - Fine-tuning and RLHF that reward confident, helpful-sounding answers
- **Measuring it**:
  - Factuality benchmarks (for example TruthfulQA, SimpleQA-style short-answer QA)
  - Faithfulness checks against a source document
  - Claim decomposition and verification (split an answer into claims, check each one)
  - Self-consistency: sample several answers and check whether they agree
  - Calibration: does stated confidence match accuracy?
- **Reducing it**:
  - Retrieval-augmented generation (grounding answers in documents)
  - Training the model to abstain or say "I don't know"
  - Citations and verifiable outputs
  - Lower temperature and constrained decoding for factual tasks
  - Verifier models and post-hoc fact-checking

### 5. Human evaluation
- Rubrics and pairwise comparisons
- Inter-annotator agreement
- Cost, bias, and annotator expertise

### 6. LLM-as-a-judge
- Reference-based vs. reference-free scoring
- Known biases: position, verbosity, self-preference
- When you still need humans

### 7. Safety evaluation
- Toxicity and harmful content
- Jailbreak robustness
- Over-refusal

### 8. Reporting results honestly
- Confidence intervals and multiple seeds
- Avoiding cherry-picked examples
- An evaluation checklist for a new model or fine-tune

## Suggested code labs

1. Compute perplexity on a short held-out text with a small open model.
2. Build a tiny factual QA set, run a model on it, and measure its hallucination rate.
3. Use self-consistency (sample five answers) to flag likely hallucinations.
4. Score a handful of answers with a rubric and with an LLM judge, and compare how often they agree.

## Key takeaways

- No single number summarizes an LLM; pick the metric for the job.
- Hallucination comes from how LLMs are trained, so it must be measured, not assumed away.
- Grounding, abstention, and verification reduce hallucination; none eliminates it.
- Treat benchmarks as evidence, not proof.
