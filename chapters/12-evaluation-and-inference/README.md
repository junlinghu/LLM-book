# Chapter 12: Evaluation and Inference

This chapter covers how to measure an LLM and how to run it at generation time. Evaluation answers whether the model is any good. Inference is how you turn a trained model into text people can use.

## Learning goals

- Choose metrics that match the task (language modeling, chat, code, safety).
- Read common academic and industry benchmarks without overclaiming.
- Understand decoding algorithms and the knobs that change creativity vs. reliability.
- Explain the main systems ideas that make serving LLMs fast and affordable.

## Outline

### 1. Why evaluation and inference belong together
- Training loss is not the same as usefulness
- Offline metrics vs. online product metrics
- The train–eval–serve loop

### 2. Evaluating language models
- **Perplexity and next-token loss** — what they measure and what they miss
- **Held-out data** — contamination, leakage, and train/test overlap
- **Task suites** — MMLU, HellaSwag, ARC, GSM8K, HumanEval / MBPP
- **Chat and preference evals** — MT-Bench, Arena-style Elo, win rates
- **Human evaluation** — rubrics, inter-annotator agreement, cost and bias
- **Safety and refusal** — toxicity, jailbreaks, over-refusal
- **How to report results** — confidence intervals, multiple seeds, no cherry-picking

### 3. Automatic judges and their limits
- LLM-as-a-judge (G-Eval-style scoring)
- Reference-based vs. reference-free scoring
- Bias toward verbose or familiar styles
- When you still need humans

### 4. Inference: from logits to text
- Autoregressive generation, one token at a time
- **Greedy decoding** vs. **sampling**
- **Temperature**, **top-k**, **top-p (nucleus)**
- **Repetition penalties** and stop sequences
- Beam search (when it helps, when it hurts)
- Structured / constrained decoding (JSON, tools, grammars)

### 5. Making inference fast
- **KV cache** — what is stored and why it matters
- Prefill vs. decode phases
- Batching and continuous batching
- Speculative decoding (draft + verify)
- Quantization for serving (INT8 / INT4, AWQ / GPTQ-style ideas)
- Context length, RoPE / attention cost, and long-context tradeoffs

### 6. Serving APIs and practical usage
- Chat templates and special tokens
- System / user / assistant roles
- Streaming tokens to the client
- Latency vs. throughput; tokens per second
- Cost models (input vs. output tokens)

### 7. Putting it together
- An evaluation checklist for a new model or fine-tune
- A small end-to-end example: score a model on a tiny benchmark, then generate with different decoding settings
- Common failure modes (good bench scores, bad product; high temperature + tools)

## Suggested code labs

1. Compute perplexity on a short held-out text with a small open model.
2. Run the same prompt under greedy, temperature, and top-p; compare outputs.
3. Time prefill vs. decode, and show the effect of a KV cache (conceptually or with a tiny implementation).
4. Score a handful of model answers with a rubric and with an LLM judge; compare agreement.

## Key takeaways

- Pick the metric for the job; no single number summarizes an LLM.
- Decoding parameters change behavior as much as model choice for many tasks.
- Serving performance is dominated by memory movement and the KV cache, not just FLOPs.
- Treat benchmarks as evidence, not proof.

## Further reading

- Papers and blog posts on nucleus sampling, KV caching, and LLM evaluation suites (to be linked as the chapter is written)
