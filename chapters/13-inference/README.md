# Chapter 13: Inference

This chapter covers how a trained LLM generates text and what computing it takes: the math, memory, and hardware behind every token, and the techniques that make serving fast and affordable.

## Learning goals

- Explain how autoregressive generation turns logits into text.
- Estimate the compute and memory an LLM needs to generate a token.
- Understand why inference is usually limited by memory bandwidth, not raw compute.
- Explain the main techniques for speeding up and scaling inference.

## Outline

### 1. [From logits to text](01-from-logits-to-text.md)
- Autoregressive generation, one token at a time
- Greedy decoding vs. sampling
- Temperature, top-k, and top-p (nucleus) sampling
- Repetition penalties and stop sequences
- Beam search, and constrained decoding (JSON, grammars, tool calls)

### 2. [The compute of a forward pass](02-the-compute-of-a-forward-pass.md)
- Counting FLOPs: roughly 2 × (number of parameters) per token
- Where the compute goes: attention vs. feed-forward (MLP) layers
- Attention cost grows with context length

### 3. [Prefill and decode](03-prefill-and-decode.md)
- **Prefill**: processing the whole prompt in parallel, which is compute-bound
- **Decode**: generating one token at a time, which is memory-bandwidth-bound
- Time to first token vs. time per output token

### 4. [The KV cache](04-the-kv-cache.md)
- What is stored (keys and values for every layer and past token) and why
- Computing its size: layers × heads × head dimension × sequence length × batch × bytes
- Memory pressure at long context and large batch
- Reducing it: multi-query and grouped-query attention, PagedAttention

### 5. [Hardware for inference](05-hardware-for-inference.md)
- GPUs and accelerators: compute (FLOPs) vs. memory bandwidth vs. memory capacity
- Arithmetic intensity and the roofline model
- Why bigger batches improve hardware utilization

### 6. [Making inference faster and cheaper](06-making-inference-faster-and-cheaper.md)
- **Quantization**: FP16/BF16, INT8, INT4, and weight-only vs. activation quantization
- **Batching**: static batching and continuous batching
- **Speculative decoding**: a small draft model proposes, the big model verifies
- **Kernel optimizations**: FlashAttention and fused operations
- **Model-level options**: distillation, mixture-of-experts, smaller models

### 7. [Scaling across devices](07-scaling-across-devices.md)
- Tensor parallelism, pipeline parallelism, and expert parallelism
- Communication costs between GPUs

### 8. [Serving systems](08-serving-systems.md)
- Inference engines (for example vLLM, TensorRT-LLM, llama.cpp)
- Latency vs. throughput, and tokens per second
- Streaming responses
- Cost per million tokens: input vs. output tokens

### 9. [Test-time compute](09-test-time-compute.md)
- Spending more computation at inference to get better answers
- Chain of thought, self-consistency, and search over candidate answers
- The tradeoff between answer quality and latency or cost

## Suggested code labs

1. Implement greedy, temperature, and top-p sampling over a model's logits and compare outputs.
2. Write a tiny KV cache for a toy attention layer and measure the speedup.
3. Compute the KV cache size and FLOPs per token for a real model configuration.
4. Quantize a small model to INT8 and compare speed, memory, and output quality.

## Key takeaways

- Decode speed is usually limited by moving weights and KV cache through memory, not by FLOPs.
- The KV cache trades memory for speed and often sets the limit on context length and batch size.
- Quantization, batching, and speculative decoding are the main levers for cost and latency.
- Spending more compute at test time is a new axis for improving answer quality.
