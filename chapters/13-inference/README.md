# Chapter 13: Inference

Inference is what happens every time someone uses a language model: a trained model reads a prompt and generates a response, one token at a time. Pretraining (Chapter 7) and fine-tuning (Chapters 8 through 11) change the model's weights; inference keeps the weights fixed and only runs the model forward. Training happens a few times, in large offline jobs where total throughput is what matters. Inference happens on every request, for every user, for as long as the model is deployed, and usually someone is waiting for the answer.

That is why inference matters. For a widely used model, the cost of serving it can rival or exceed the cost of training it, so every saving per token is multiplied across every request. Latency shapes the user experience: how soon the first word appears and how quickly the rest streams in. Serving many users at once is a systems problem of memory, batching, and scheduling across many accelerators. And inference is now also a way to get better answers: reasoning models and other test-time compute methods improve quality by generating more tokens, so efficient inference affects quality as well as cost.

This chapter explains what generating a token takes and how to make it fast and affordable. [Section 1](01-from-logits-to-text.md) shows how logits become text through greedy decoding, sampling, beam search, and constrained decoding. [Section 2](02-the-compute-of-a-forward-pass.md) counts the compute of a forward pass, and [Section 3](03-prefill-and-decode.md) explains why processing the prompt is limited by compute while generating tokens is limited by memory bandwidth. [Section 4](04-the-kv-cache.md) introduces the KV cache, which often limits context length and batch size, and [Section 5](05-hardware-for-inference.md) connects all this to hardware with the roofline model. [Section 6](06-making-inference-faster-and-cheaper.md) covers the main levers for speed and cost: quantization, batching, speculative decoding, fast kernels, and model-level choices. [Section 7](07-scaling-across-devices.md) splits a model across devices, [Section 8](08-serving-systems.md) puts the pieces together in serving systems and explains per-token pricing, and [Section 9](09-test-time-compute.md) looks at spending more compute at inference time to get better answers.

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
