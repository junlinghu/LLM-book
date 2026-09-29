# 12.8 Serving Systems

The previous sections described individual techniques: sampling, KV caching, batching, quantization, speculative decoding, fast kernels, and parallelism. A **serving system** (or *inference engine*) combines them into software that accepts requests from many users, schedules them onto hardware, and streams back tokens. This section describes what such a system does, surveys widely used engines, defines the metrics used to judge them, explains how responses are streamed, and works out where the cost per million tokens comes from.

## Anatomy of an inference engine

A request to a chat API passes through a pipeline of components:

1. **API server.** Accepts HTTP requests, typically in an OpenAI-compatible chat-completions format that has become a de facto standard, validates parameters, and applies the model's chat template (Chapter 8) to turn a list of messages into a single token sequence.
2. **Tokenizer.** Converts text to token IDs (Chapter 5). This runs on the CPU and is usually cheap, but it must not block the GPU.
3. **Scheduler.** Maintains the queue of waiting requests and the set of running ones, and decides at every iteration which sequences to run, which to admit, and which to preempt (continuous batching, Section 6). It enforces limits on the number of tokens per batch and may prioritize requests.
4. **KV cache manager.** Allocates and frees cache blocks, tracks block tables, and reuses cached prefixes (PagedAttention and prefix caching, Sections 3 and 4).
5. **Model executor.** Runs the forward pass using optimized kernels (Section 6), possibly across several devices (Section 7).
6. **Sampler.** Applies penalties, temperature, top-k, top-p, and grammar masks to the logits and chooses tokens (Section 1).
7. **Detokenizer and streamer.** Converts new tokens back to text, checks stop strings, and sends text increments to the client.

Well-engineered systems keep the GPU busy by overlapping CPU work (scheduling, tokenization, grammar masks, detokenization) with GPU work, and by keeping these components from blocking one another.

## Inference engines

Many engines implement this design, with different emphases. Three widely used examples illustrate the range.

**vLLM** is an open-source serving engine that originated at UC Berkeley alongside the PagedAttention paper (Kwon et al.). It pioneered paged KV cache management and combines it with continuous batching, prefix caching, quantized formats, speculative decoding, tensor and pipeline parallelism, and an OpenAI-compatible API server. It supports a wide range of open models and several hardware platforms, which has made it a common default for self-hosted serving.

**TensorRT-LLM** is NVIDIA's open-source library for LLM inference on NVIDIA GPUs. It builds on NVIDIA's TensorRT compiler and hand-optimized kernels, supports in-flight (continuous) batching, paged KV caches, FP8 and INT4 quantization, speculative decoding, and multi-GPU parallelism, and is designed to extract high performance from NVIDIA hardware. It is often deployed behind NVIDIA's Triton Inference Server.

**llama.cpp** is an open-source C/C++ implementation focused on running models efficiently on commodity hardware: laptop and desktop CPUs, Apple Silicon (whose unified memory lets the GPU use most of the system RAM), and consumer GPUs. It introduced the GGUF file format and a family of quantization types (from 8-bit down to 2-bit variants), and supports grammar-constrained decoding. It is the engine underneath many local-LLM desktop applications and is designed for single-user or small-scale use rather than large-scale serving.

Other notable engines include SGLang, which introduced RadixAttention for automatic prefix reuse and a frontend language for structured LLM programs (Zheng et al.), and Hugging Face's Text Generation Inference (TGI). The landscape changes quickly, and relative performance depends heavily on the model, hardware, and workload, so published comparisons should be read with the same skepticism Chapter 11 recommends for benchmarks. Benchmark your own workload before choosing.

Running an engine is usually a single command. For example, vLLM can serve a model from the Hugging Face Hub behind an OpenAI-compatible endpoint:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192
```

## Latency vs. throughput, and tokens per second

### Metrics

A serving system is judged on two kinds of metrics that pull in opposite directions.

**Latency** metrics describe the experience of one request (Section 3):

- **Time to first token (TTFT):** queueing plus prefill plus sampling of the first token.
- **Time per output token (TPOT)** or **inter-token latency (ITL):** the gap between successive tokens during decode. Its inverse is the **per-user tokens per second**.
- **End-to-end latency:** $\text{TTFT} + (g - 1) \cdot \text{TPOT}$ for $g$ output tokens.

**Throughput** metrics describe the whole system:

- **Output tokens per second**, summed over all concurrent requests, often normalized per GPU.
- **Requests per second** for a given mix of prompt and output lengths.

Latency must be reported as a *distribution*, not just a mean. Tail latencies (the 90th, 95th, or 99th percentile) matter because a user who waits ten seconds for one response in twenty remembers it. A common way to combine the two views is **goodput**: the rate of requests completed *within* a service-level objective (SLO), such as TTFT under 500 ms and TPOT under 50 ms for 99 percent of requests. Zhong et al. used goodput as the objective for DistServe; it captures the fact that throughput achieved by violating latency targets is not useful.

### The tradeoff

Section 5 showed why the tradeoff exists: larger batches read the weights once for more tokens, raising throughput, but each decode step takes longer, lowering per-user speed. Long prefills in the batch also delay other users' tokens. So a system can move along a curve:

- At small batch sizes, users get fast tokens, but the hardware is underutilized and each token is expensive.
- At large batch sizes, the hardware is well utilized and tokens are cheap, but each user's tokens arrive more slowly and TTFT grows as requests wait for capacity.

Operators pick a point on this curve by limiting the maximum batch size or tokens per batch, choosing parallelism (Section 7), and sometimes running separate deployments of the same model for latency-sensitive and batch (offline) traffic. Offline batch APIs that return results hours later can run at the high-throughput end of the curve, which is why they are commonly offered at a discount.

A useful identity from queueing theory, **Little's law**, connects these quantities. In a stable system, the average number of requests in the system equals the arrival rate times the average time each spends there:

$$
\text{concurrency} = \text{request rate} \times \text{latency}.
$$

If a service receives 20 requests per second and each takes 6 seconds end to end, about 120 requests are in flight at any moment, and the system needs enough KV cache memory and batch capacity for all of them. When the arrival rate approaches the system's capacity, queues grow and latency rises sharply, so real systems run below full capacity and scale out (add replicas) based on load.

### What "tokens per second" means

"Tokens per second" is quoted in at least three different senses: per-user decode speed ($1 / \text{TPOT}$), aggregate output throughput of a server, and total throughput including prompt tokens processed. These can differ by orders of magnitude for the same system. Tokens are also not comparable across models with different tokenizers: a model whose tokenizer produces fewer tokens for the same text does less work per word. When comparing, ask which definition is used, on which hardware, with what batch size, prompt length, and output length.

## Streaming responses

For interactive use, systems **stream** output: they send each token (or small group of tokens) to the client as soon as it is generated rather than waiting for the full response. The user starts reading after TTFT instead of after the end-to-end latency. Streaming does not make generation faster, but for chat it transforms perceived responsiveness, and it lets a user or client cancel a bad response early.

Most HTTP APIs stream using **server-sent events (SSE)**: the server keeps the connection open and writes a sequence of `data: {...}` lines, each carrying a JSON chunk with the next text increment, followed by a final marker. A client using the OpenAI Python library against any compatible server, such as the vLLM server started above, looks like this:

```python
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed-locally")

start = time.perf_counter()
first, n_chunks = None, 0
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain the KV cache in two sentences."}],
    temperature=0.7,
    max_tokens=200,
    stream=True,
)
for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta.content
    if delta:
        first = first or time.perf_counter()
        n_chunks += 1
        print(delta, end="", flush=True)
end = time.perf_counter()
print(f"\nTTFT {first - start:.3f} s; {n_chunks} chunks in {end - first:.2f} s after the first")
```

Measuring TTFT and the time between chunks from the client side, as here, captures network and queueing delays that server-side metrics miss.

Streaming raises a few implementation issues:

- **Incremental detokenization.** A token does not always correspond to a complete piece of text. With byte-level BPE (Chapter 5), a multi-byte UTF-8 character such as an emoji may be split across several tokens, and decoding a partial byte sequence produces a replacement character. Some tokenizers also add or remove spaces depending on neighboring tokens. The streamer therefore decodes a window of recent tokens and emits only the text that is stable:

```python
def stream_text(token_ids, tokenizer):
    """Yield text increments, holding back incomplete UTF-8 characters."""
    ids, emitted = [], ""
    for tid in token_ids:
        ids.append(tid)
        text = tokenizer.decode(ids)       # real engines decode only a recent window
        if text.endswith("\ufffd"):        # replacement char: incomplete multi-byte sequence
            continue
        yield text[len(emitted):]
        emitted = text
```

- **Stop strings.** As Section 1 explained, text that might be the start of a stop string must be held back until the system knows whether the full stop string follows.
- **Cancellation.** If the client disconnects, the server should abort the request promptly and free its KV cache; otherwise it keeps spending GPU time on tokens nobody will read.
- **Structured output and tool calls.** A partial JSON object is not valid JSON. Clients that need to act on structured output must either wait for the end or use an incremental parser; APIs often stream tool-call arguments as string fragments.

## Cost per million tokens: input vs. output tokens

Commercial LLM APIs usually price usage per million tokens, with separate prices for **input** (prompt) tokens and **output** (generated) tokens, and output tokens are typically priced several times higher than input tokens. The cost model of this chapter explains why.

### Where the cost comes from

At bottom, a provider pays for accelerator time (plus power, networking, and operations). If a device costs $c$ dollars per hour and processes $r$ tokens per second of some kind, the cost per million such tokens is

$$
\text{cost per million tokens} = \frac{c}{3600 \cdot r} \times 10^6.
$$

The key question is $r$, and it is very different for the two kinds of token:

- **Input tokens are processed in prefill**, which is compute-bound and highly parallel (Section 3). A device can process tens of thousands of prompt tokens per second for a mid-sized model, using its arithmetic units efficiently.
- **Output tokens are produced in decode**, which is memory-bound. Even with large batches, the device produces far fewer output tokens per second, and each output token also occupies KV cache memory for the duration of the request.

A rough calculation for an 8B model on our hypothetical accelerator, priced at a hypothetical \$2.50 per hour:

```python
def cost_per_million(gpu_dollars_per_hour, tokens_per_second):
    tokens_per_hour = tokens_per_second * 3600
    return gpu_dollars_per_hour / tokens_per_hour * 1e6

gpu_price = 2.50                      # hypothetical $/hour for one accelerator
n_params = 8e9
# Prefill: compute-bound. Assume 50% of 1,000 TFLOP/s is achieved.
prefill_tps = 0.5 * 1e15 / (2 * n_params)
# Decode: batch of 128 at ~2,000 tokens of context; idealized 7,750 tok/s, assume half is achieved.
decode_tps = 0.5 * 7750

print(f"input  (prefill): {prefill_tps:>8,.0f} tok/s -> ${cost_per_million(gpu_price, prefill_tps):.3f} per M tokens")
print(f"output (decode):  {decode_tps:>8,.0f} tok/s -> ${cost_per_million(gpu_price, decode_tps):.3f} per M tokens")
```

It prints:

```
input  (prefill):   31,250 tok/s -> $0.022 per M tokens
output (decode):     3,875 tok/s -> $0.179 per M tokens
```

Under these assumptions, an output token costs about eight times as much to produce as an input token. The exact ratio depends on the model, hardware, context lengths, batch sizes, and utilization, and the absolute numbers here say nothing about any real provider's costs, which also include idle capacity, redundancy, networking, staff, and margin. But the direction is robust: output tokens are more expensive because decode uses the hardware less efficiently.

### Other factors in pricing

- **Context length.** Long prompts cost more than proportionally in prefill (quadratic attention, Section 2), and long contexts make each decode step slower and consume KV memory, reducing batch size. Some providers charge more per token above a context-length threshold.
- **Cached input.** If a prompt prefix is already in the prefix cache (Section 3), its prefill is skipped. Many providers offer lower prices for cached input tokens, which rewards applications that keep a stable prefix, such as a fixed system prompt, at the start of every request.
- **Batch APIs.** Latency-insensitive jobs let the provider run at large batch sizes and fill idle capacity, and are commonly priced lower.
- **Reasoning tokens.** Models that think before answering (Section 9) generate many hidden output tokens that are billed as output, so the cost of a request can be dominated by tokens the user never sees.
- **Model size and architecture.** Cost scales roughly with active parameters for compute and with total parameters and KV size for memory, which is why small, quantized, GQA, and MoE models (Sections 4 and 6) are cheaper to serve.

For an application developer, a useful rule follows directly: to estimate cost, count input and output tokens separately, multiply by their prices, and remember that shortening outputs usually saves more than shortening prompts.

## Key takeaways

- A serving system combines an API server, tokenizer, scheduler, KV cache manager, model executor, sampler, and streamer, overlapping CPU and GPU work.
- vLLM (paged KV cache and continuous batching), TensorRT-LLM (optimized NVIDIA kernels and compilation), and llama.cpp (efficient local inference with quantized GGUF models) represent different points in the design space; benchmark your own workload.
- Latency (TTFT, TPOT, tail percentiles) and throughput (aggregate tokens per second) trade off through batch size; goodput under a latency SLO combines them, and Little's law links request rate, latency, and concurrency.
- Streaming sends tokens as they are generated over server-sent events; it requires careful incremental detokenization, stop-string handling, and prompt cancellation.
- Output tokens cost more than input tokens because decode is memory-bound while prefill is compute-bound; prefix caching, batch processing, and smaller models reduce cost.

## Further reading

Agrawal, Amey, et al. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." In *18th USENIX Symposium on Operating Systems Design and Implementation*, 2024. https://arxiv.org/abs/2403.02310.

Gerganov, Georgi, et al. "llama.cpp: LLM Inference in C/C++." GitHub repository. https://github.com/ggml-org/llama.cpp.

Kwon, Woosuk, et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." In *Proceedings of the 29th Symposium on Operating Systems Principles*, 2023. https://arxiv.org/abs/2309.06180.

NVIDIA. "TensorRT-LLM." GitHub repository. https://github.com/NVIDIA/TensorRT-LLM.

vLLM Project. "vLLM: A High-Throughput and Memory-Efficient Inference and Serving Engine for LLMs." GitHub repository. https://github.com/vllm-project/vllm.

Yu, Gyeong-In, et al. "Orca: A Distributed Serving System for Transformer-Based Generative Models." In *16th USENIX Symposium on Operating Systems Design and Implementation*, 2022.

Zheng, Lianmin, et al. "SGLang: Efficient Execution of Structured Language Model Programs." arXiv preprint arXiv:2312.07104, 2023. https://arxiv.org/abs/2312.07104.

Zhong, Yinmin, et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving." In *18th USENIX Symposium on Operating Systems Design and Implementation*, 2024. https://arxiv.org/abs/2401.09670.
