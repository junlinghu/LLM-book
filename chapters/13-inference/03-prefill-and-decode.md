# 13.3 Prefill and Decode

When you send a prompt to a chat model, two very different computations happen. First the model reads your whole prompt. Then it writes its answer one token at a time. These two phases, called **prefill** and **decode**, stress the hardware in opposite ways. Prefill is a large, parallel computation that keeps the arithmetic units busy. Decode is a long chain of small computations that spend most of their time waiting for data to arrive from memory. Understanding this split explains why the first token of a response takes a different amount of time than the rest, why output tokens cost more than input tokens (Section 8), and why almost every optimization in this chapter targets one phase or the other.

## The two phases

Consider a prompt of $m$ tokens and a response of $g$ generated tokens. Section 1's decoding loop can be split as follows.

**Prefill.** The model runs a single forward pass over all $m$ prompt tokens at once. Because the prompt is known in advance, there is no sequential dependency between its positions: with a causal mask, position $t$ only needs positions $1, \dots, t$, all of which are available. The pass produces two things: the keys and values for every prompt position in every layer, which are stored in the KV cache (Section 4), and the logits at the last position, from which the first output token is chosen.

**Decode.** Each subsequent step feeds in *one* new token, the one just generated. The model computes that token's query, key, and value in each layer, appends the key and value to the cache, attends over the cache, and produces logits for the next token. This repeats $g - 1$ times (or until a stop condition).

In code, using a model interface that accepts and returns a cache:

```python
def generate_with_cache(model, prompt_ids, choose_token, max_new_tokens):
    # Prefill: one parallel pass over the whole prompt.
    logits, cache = model.forward(prompt_ids, cache=None)   # logits: [m, V]
    next_id = choose_token(logits[-1])
    output = [next_id]
    # Decode: one token per pass, reusing the cache.
    for _ in range(max_new_tokens - 1):
        logits, cache = model.forward([next_id], cache=cache)  # logits: [1, V]
        next_id = choose_token(logits[-1])
        output.append(next_id)
    return output
```

Both phases run the same network with the same weights. What differs is the *shape* of the work: prefill multiplies each weight matrix by a matrix of $m$ token vectors, while decode multiplies it by a single vector (or, with batching, by one vector per active request).

## Prefill is compute-bound

From Section 2, the prefill pass costs about $2N$ FLOPs per prompt token for the weights, plus about $2Ldm^2$ for attention. For an 8-billion-parameter model and a 2,000-token prompt, that is on the order of

$$
2 \times 8 \times 10^9 \times 2000 \approx 3 \times 10^{13} \text{ FLOPs},
$$

or 30 teraFLOPs, most of it in large matrix multiplies of shape $(m \times d) \cdot (d \times d')$.

How long does this take? To answer, we need two properties of the hardware (Section 5 covers them in depth): its peak arithmetic rate $P$ in FLOP/s, and its memory bandwidth $B$ in bytes per second, the rate at which it can read data from its main memory (HBM on a GPU). For concreteness, we use a *hypothetical accelerator* with

$$
P = 1000 \text{ TFLOP/s (16-bit)}, \qquad B = 3 \text{ TB/s}, \qquad \text{80 GB of memory}.
$$

These round numbers are in the same range as current data-center GPUs; they are chosen for easy arithmetic, not to describe a particular product.

A forward pass must, at minimum, (1) perform its FLOPs and (2) read every weight from memory at least once. Storing the 8B model in a 16-bit format (BF16) takes 16 GB. So the two lower bounds for our prefill are:

$$
t_{\text{compute}} \ge \frac{3 \times 10^{13}}{10^{15}} = 30 \text{ ms}, \qquad t_{\text{memory}} \ge \frac{16 \times 10^{9}}{3 \times 10^{12}} \approx 5.3 \text{ ms}.
$$

The arithmetic takes much longer than reading the weights. Each weight, once loaded, is reused for all 2,000 prompt tokens, so the loading cost is amortized. Prefill is **compute-bound**: its speed is set by the FLOP/s of the hardware. In practice kernels do not reach peak FLOP/s, so real prefill takes longer than this bound, but the bottleneck is arithmetic.

The same calculation shows when prefill stops being compute-bound. The FLOPs grow with $m$ while the weight bytes do not. The two bounds are equal when $2Nm / P = 2N / B$ (16-bit weights take 2 bytes per parameter), that is, when

$$
m = \frac{P}{B} \approx 333 \text{ tokens}
$$

for our hypothetical accelerator. Prompts much longer than a few hundred tokens are compute-bound; very short prompts behave more like decode. The ratio $P/B$ reappears in Section 5 as the *ridge point* of the roofline model.

## Decode is memory-bandwidth-bound

Now consider a decode step for a single request. It processes one token, so it performs about $2N \approx 16$ GFLOPs of weight arithmetic. But it still has to read all 16 GB of weights, because every weight participates in computing the next token. The bounds are now:

$$
t_{\text{compute}} \ge \frac{1.6 \times 10^{10}}{10^{15}} = 0.016 \text{ ms}, \qquad t_{\text{memory}} \ge \frac{16 \times 10^{9}}{3 \times 10^{12}} \approx 5.3 \text{ ms}.
$$

Reading the weights takes over 300 times longer than the arithmetic. The accelerator's arithmetic units sit idle almost all the time, waiting for data. Decode is **memory-bandwidth-bound**: its speed is set by how fast bytes can be moved from memory to the compute units, not by how fast they can be multiplied.

This gives a simple and important estimate. For a single sequence, the time per output token is at least

$$
t_{\text{token}} \gtrsim \frac{\text{bytes of weights} + \text{bytes of KV cache read}}{B}.
$$

For our 8B model at short context, that is about 5.3 ms per token, or at most about 190 tokens per second for one user, however many FLOP/s the chip has. Real systems achieve some fraction of peak bandwidth and have other overheads, so actual speeds are lower. The estimate is still extremely useful because it tells you which changes will help:

- **Fewer bytes per weight.** Storing weights in 8 or 4 bits (quantization, Section 6) cuts the bytes to read and speeds up decode nearly proportionally, even though the FLOPs are unchanged.
- **Faster memory.** A chip with more bandwidth decodes faster even with the same FLOP/s.
- **More tokens per weight read.** If one read of the weights can serve many tokens, the cost is amortized. Batching many requests together (Sections 5 and 6) does this across users; speculative decoding (Section 6) does it within one user's sequence by verifying several guessed tokens in one pass.
- **More FLOP/s does not help** a single-stream decode, as long as it stays memory-bound.

At long context the KV cache adds to the bytes. Each decode step must read the keys and values of every earlier position in every layer. Section 4 shows that for long sequences or large batches, the KV cache read can rival or exceed the weight read.

## Time to first token vs. time per output token

The two phases map directly onto the latency a user experiences. Serving systems report several metrics:

- **Time to first token (TTFT):** the time from when a request arrives until the first output token is available. It includes any waiting in a queue, tokenization, the prefill pass, and sampling the first token. TTFT grows with prompt length, roughly linearly for moderate prompts (weight FLOPs) and faster for very long prompts (quadratic attention FLOPs).
- **Time per output token (TPOT):** the average time between consecutive output tokens after the first, also called *inter-token latency* (ITL) when measured per gap. TPOT is dominated by decode and is set mainly by memory bandwidth, model size, context length, and how many other requests share the same batch.
- **End-to-end latency:** the total time for a response of $g$ tokens,

$$
T_{\text{total}} \approx \text{TTFT} + (g - 1) \cdot \text{TPOT}.
$$

Which metric matters depends on the application. For a chat interface that streams tokens as they are generated (Section 8), TTFT determines how responsive the system feels, and TPOT only needs to be faster than a person reads. For a coding agent that must finish a long generation before acting, end-to-end latency matters most, and TPOT dominates it. For summarizing a long document into a short answer, prefill (and thus TTFT) dominates; for writing a long essay from a short prompt, decode dominates.

A small calculator makes the tradeoffs concrete:

```python
def latency_estimate(n_params, bytes_per_param, prompt_len, gen_len,
                     peak_flops=1e15, bandwidth=3e12, mfu=0.5, mbu=0.7):
    """Rough single-request latency using compute and memory lower bounds.
    mfu/mbu: fraction of peak FLOP/s and peak bandwidth actually achieved (assumed)."""
    weight_bytes = n_params * bytes_per_param
    prefill_compute = 2 * n_params * prompt_len / (peak_flops * mfu)
    prefill_memory = weight_bytes / (bandwidth * mbu)
    ttft = max(prefill_compute, prefill_memory)
    tpot = weight_bytes / (bandwidth * mbu)      # ignores KV-cache reads at short context
    return ttft, tpot, ttft + (gen_len - 1) * tpot

ttft, tpot, total = latency_estimate(8e9, 2, prompt_len=2000, gen_len=500)
print(f"TTFT {ttft*1e3:.0f} ms, TPOT {tpot*1e3:.1f} ms, total {total:.2f} s")
# TTFT 64 ms, TPOT 7.6 ms, total 3.87 s
```

The utilization factors `mfu` (model FLOPs utilization) and `mbu` (memory bandwidth utilization) are assumptions for illustration. Even so, the example shows the typical pattern: a 2,000-token prompt is processed in tens of milliseconds, while 500 output tokens take several seconds. Per token, decode is far more expensive than prefill.

## How serving systems exploit the split

Because the phases have different bottlenecks, serving systems treat them differently. We return to these ideas in Sections 6 and 8, but three are worth previewing.

**Prefix caching.** Many requests share a prefix: the same system prompt, the same few-shot examples, or the earlier turns of a conversation. The KV cache for a shared prefix is identical across requests, so a system can keep it and skip that part of prefill. This reduces TTFT and compute for multi-turn chat and agent workloads. SGLang's RadixAttention, for example, organizes cached prefixes in a radix tree so that any request can reuse the longest matching cached prefix. Many API providers pass the savings on as discounted prices for cached input tokens.

**Chunked prefill.** When a long prompt arrives at a server that is already decoding for other users, running its whole prefill at once stalls everyone else's decode for the duration, producing a visible hiccup in their token stream. Systems such as Sarathi-Serve split a long prefill into chunks and mix each chunk into a batch together with ongoing decode steps. The compute-hungry prefill chunks use arithmetic units that decode leaves idle, and the decode steps keep flowing.

**Prefill-decode disaggregation.** A more radical approach runs the two phases on different machines. Prefill servers process prompts and transfer the resulting KV cache to decode servers, which generate the outputs. Each pool can then be sized and configured for its own bottleneck and latency target, at the cost of moving the KV cache over the network. DistServe studied this design and showed that it lets a system meet separate TTFT and TPOT targets more efficiently than colocating both phases.

## Key takeaways

- Inference has two phases: prefill processes the prompt in one parallel pass and builds the KV cache; decode generates one token per pass.
- Prefill reuses each weight across all prompt tokens, so it is compute-bound for prompts longer than a few hundred tokens.
- Decode reads every weight to produce one token per sequence, so it is memory-bandwidth-bound: time per token is roughly (weight bytes + KV bytes read) divided by memory bandwidth.
- TTFT is dominated by queueing and prefill; TPOT is dominated by decode; end-to-end latency is about TTFT plus $(g-1)$ times TPOT.
- Prefix caching, chunked prefill, and prefill-decode disaggregation are system designs that exploit the different bottlenecks of the two phases.

## Further reading

Agrawal, Amey, et al. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." In *18th USENIX Symposium on Operating Systems Design and Implementation*, 2024. https://arxiv.org/abs/2403.02310.

Pope, Reiner, et al. "Efficiently Scaling Transformer Inference." In *Proceedings of Machine Learning and Systems 5*, 2023. https://arxiv.org/abs/2211.05102.

Zheng, Lianmin, et al. "SGLang: Efficient Execution of Structured Language Model Programs." arXiv preprint arXiv:2312.07104, 2023. https://arxiv.org/abs/2312.07104.

Zhong, Yinmin, et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving." In *18th USENIX Symposium on Operating Systems Design and Implementation*, 2024. https://arxiv.org/abs/2401.09670.
