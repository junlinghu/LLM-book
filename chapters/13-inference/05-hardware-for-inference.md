# 13.5 Hardware for Inference

The previous sections kept arriving at the same conclusion: whether an inference workload is fast depends less on how many FLOPs it needs than on how those FLOPs relate to the bytes it must move. This section makes that idea precise. We look at the three hardware numbers that matter for LLM inference, introduce *arithmetic intensity* and the *roofline model* as tools for deciding which one limits a given computation, and use them to explain why batching many requests together is the central trick of efficient serving.

## GPUs and accelerators: compute, bandwidth, and capacity

Most LLM inference runs on GPUs. Other accelerators, such as Google's TPUs, AMD's Instinct GPUs, and various custom inference chips, differ in details but share the same basic structure, and so do the CPUs and unified-memory laptops that run small models locally. At the level of detail we need, an accelerator has:

- Many parallel **compute units** (streaming multiprocessors on NVIDIA GPUs), each containing specialized matrix-multiply units (*tensor cores* on NVIDIA hardware, *matrix units* on TPUs) that perform small dense matrix multiplies very quickly.
- A small amount of very fast **on-chip memory**: registers, shared memory or scratchpad, and caches, totaling tens of megabytes at most.
- A large **off-chip main memory**, usually high-bandwidth memory (HBM) stacked next to the chip, holding tens to a few hundred gigabytes.
- **Interconnects** to other accelerators (such as NVLink between GPUs in a server) and to the host CPU (such as PCIe), which matter when a model is split across devices (Section 7).

For inference, three numbers summarize a device:

1. **Peak compute** $P$, in FLOP/s. Modern accelerators reach hundreds of teraFLOP/s to a few petaFLOP/s on dense 16-bit matrix multiplies, and roughly double that for each halving of precision (8-bit, 4-bit) when the hardware supports it. Peak compute is only reached by large, well-shaped matrix multiplies on the tensor cores; everything else runs much slower.
2. **Memory bandwidth** $B$, in bytes per second: how fast data moves between HBM and the compute units. Current data-center accelerators offer a few terabytes per second.
3. **Memory capacity** $M$, in bytes: how much fits in HBM. This bounds the model weights plus KV cache plus activations that can live on one device.

As in Section 3, we use a hypothetical accelerator with $P = 1000$ TFLOP/s (16-bit), $B = 3$ TB/s, and $M = 80$ GB as a running example. These round numbers are in the range of current data-center GPUs; always look up the vendor's specifications for real hardware, and note that quoted peak FLOP/s sometimes assume sparsity features that dense LLM inference does not use.

Each of the three numbers limits inference in a different way:

- **Capacity** decides *whether* a model fits at all, and how much KV cache (and hence how many concurrent tokens) fits alongside it. A 70B-parameter model in BF16 needs about 140 GB for its weights alone, so it cannot run on a single 80 GB device without quantization (Section 6) or splitting it across devices (Section 7).
- **Bandwidth** sets the speed of decode, which reads all the weights for every step (Section 3).
- **Compute** sets the speed of prefill and of decode at very large batch sizes.

A striking trend of the last decade is that peak compute has grown much faster than memory bandwidth. Each generation of accelerators can perform many more FLOPs per byte of memory traffic than the last. This widening gap, often called the *memory wall*, is a major reason LLM inference is so often memory-bound, and why techniques that reduce bytes moved (quantization, GQA, FlashAttention, batching) are so valuable.

## Arithmetic intensity and the roofline model

### Arithmetic intensity

The **arithmetic intensity** $I$ of a computation is the number of FLOPs it performs per byte moved to or from main memory:

$$
I = \frac{\text{FLOPs}}{\text{bytes moved}}.
$$

The hardware has a matching ratio, the number of FLOPs it *can* perform per byte it *can* move:

$$
I^\star = \frac{P}{B}.
$$

For our hypothetical accelerator, $I^\star = 10^{15} / (3 \times 10^{12}) \approx 333$ FLOPs per byte. If a computation has $I \lt I^\star$, the compute units finish their work faster than memory can feed them and must wait: the computation is **memory-bound**. If $I \gt I^\star$, memory delivers data faster than it can be consumed: the computation is **compute-bound**.

### The roofline model

The roofline model of Williams, Waterman, and Patterson turns this into a single formula for the best achievable performance:

$$
\text{attainable FLOP/s} = \min\big(P, \; I \cdot B\big).
$$

Plotted on log-log axes with intensity on the horizontal axis, attainable performance rises along a diagonal line of slope 1 (the *bandwidth roof*, $I \cdot B$) until it hits a horizontal line (the *compute roof*, $P$). The corner where they meet, at $I = I^\star$, is the **ridge point**. Computations to the left of the ridge are memory-bound; to the right, compute-bound. The model is deliberately simple, ignoring caches, latency, and kernel launch overheads, but it is remarkably good at telling you which resource to optimize.

Equivalently, the time of a computation is bounded below by

$$
t \ge \max\left(\frac{\text{FLOPs}}{P}, \; \frac{\text{bytes}}{B}\right),
$$

which is exactly the calculation we did for prefill and decode in Section 3.

### The intensity of a matrix multiply

Consider multiplying a batch of $b$ token vectors, $X \in \mathbb{R}^{b \times d}$, by a weight matrix $W \in \mathbb{R}^{d \times d'}$, with all values in 16-bit (2 bytes). The multiply performs $2bdd'$ FLOPs. At minimum it must read $W$ ($2dd'$ bytes), read $X$ ($2bd$ bytes), and write the output ($2bd'$ bytes):

$$
I = \frac{2 b d d'}{2(dd' + bd + bd')}.
$$

When $b$ is much smaller than $d$ and $d'$, as it is in decode, the weight term dominates the denominator and

$$
I \approx b \quad \text{FLOPs per byte}.
$$

This single approximation explains a great deal. A decode step with one sequence ($b = 1$) has an intensity of about 1 FLOP per byte, more than 300 times below the ridge point of our accelerator: the compute units are almost entirely idle. Prefill of a prompt with $m$ tokens has $b = m$, and for prompts of more than a few hundred tokens it is compute-bound. (With 8-bit weights, each weight takes 1 byte and the intensity doubles to about $2b$; this is one way quantization helps.)

### The intensity of attention during decode

Attention during decode behaves differently. For one sequence at context length $n$, one layer reads $2 n h_{kv} d_h$ cached key and value elements and performs about $4 n h d_h$ FLOPs (Section 2). With 16-bit elements,

$$
I_{\text{attn}} \approx \frac{4 n h d_h}{2 \cdot 2 n h_{kv} d_h} = \frac{h}{h_{kv}}.
$$

For multi-head attention this is 1 FLOP per byte, and it does *not* improve with batch size, because each sequence attends to its own cache: doubling the batch doubles both the FLOPs and the bytes. Grouped-query attention raises it to the group size $h / h_{kv}$ (4 for Llama 3 8B), which is still far below the ridge point. Attention in decode is memory-bound essentially always, and the only ways to speed it up are to read fewer bytes (smaller or quantized caches, Section 4) or to read them more efficiently (better kernels, Section 6).

## Why bigger batches improve hardware utilization

Now we can see why serving systems work so hard to batch requests together. Suppose $b$ sequences are decoding simultaneously. In one step, the system reads the weights *once* and uses them for all $b$ tokens. The time for one decode step is approximately

$$
t_{\text{step}}(b) \approx \max\left( \frac{b \cdot 2N + b \cdot 4Lnd}{P}, \; \frac{2N + b \cdot \text{KV}(n)}{B} \right),
$$

where $\text{KV}(n)$ is the cache size of one sequence at context length $n$ (Section 4), and the $2N$ in the memory term counts 16-bit weight bytes. The step produces $b$ tokens, one per sequence. So:

- **Throughput**, the total tokens per second across all users, is $b / t_{\text{step}}(b)$.
- **Per-user speed**, the tokens per second each user sees, is $1 / t_{\text{step}}(b)$.

When the batch is small and the context is short, the weight read dominates $t_{\text{step}}$, and it does not grow with $b$. Adding sequences to the batch is then almost free: throughput grows nearly linearly with $b$ while each user's speed barely changes. This continues until either the compute term catches up with the memory term (the step becomes compute-bound, at roughly $b \approx I^\star$ for short contexts) or the KV cache term takes over (at longer contexts), or memory runs out.

The following code evaluates this model for Llama 3 8B on our hypothetical accelerator:

```python
def decode_step(batch, context, n_params=8e9, bytes_per_param=2, L=32, d=4096,
                kv_bytes_per_token=131_072,                  # Llama 3 8B, 16-bit
                peak_flops=1e15, bandwidth=3e12, capacity=80e9):
    weight_bytes = n_params * bytes_per_param
    kv_bytes = batch * context * kv_bytes_per_token
    if weight_bytes + kv_bytes > capacity:
        return None                                          # does not fit in memory
    flops = batch * (2 * n_params + 4 * L * context * d)
    t_mem, t_compute = (weight_bytes + kv_bytes) / bandwidth, flops / peak_flops
    return max(t_mem, t_compute), ("memory" if t_mem > t_compute else "compute")

for context in [100, 2000]:
    for batch in [1, 8, 32, 128, 512]:
        r = decode_step(batch, context)
        if r is None:
            print(f"context {context:>5}, batch {batch:>3}: does not fit")
            continue
        t, bound = r
        print(f"context {context:>5}, batch {batch:>3}: {t*1e3:5.1f} ms/step, "
              f"{batch/t:7.0f} tok/s total, {1/t:4.0f} tok/s per user ({bound}-bound)")
```

It prints:

```
context   100, batch   1:   5.3 ms/step,     187 tok/s total,  187 tok/s per user (memory-bound)
context   100, batch   8:   5.4 ms/step,    1490 tok/s total,  186 tok/s per user (memory-bound)
context   100, batch  32:   5.5 ms/step,    5847 tok/s total,  183 tok/s per user (memory-bound)
context   100, batch 128:   5.9 ms/step,   21722 tok/s total,  170 tok/s per user (memory-bound)
context   100, batch 512:   8.2 ms/step,   62296 tok/s total,  122 tok/s per user (compute-bound)
context  2000, batch   1:   5.4 ms/step,     184 tok/s total,  184 tok/s per user (memory-bound)
context  2000, batch   8:   6.0 ms/step,    1326 tok/s total,  166 tok/s per user (memory-bound)
context  2000, batch  32:   8.1 ms/step,    3936 tok/s total,  123 tok/s per user (memory-bound)
context  2000, batch 128:  16.5 ms/step,    7749 tok/s total,   61 tok/s per user (memory-bound)
context  2000, batch 512: does not fit
```

The results illustrate three regimes:

1. **Short context, small to moderate batch.** At context 100, going from batch 1 to batch 128 multiplies throughput by more than 100 while each user's speed drops by less than 10 percent. The weights are read once per step regardless, and more sequences simply use FLOPs that would otherwise be wasted.
2. **Compute-bound.** At batch 512 and short context, the step becomes compute-bound. Beyond this point, throughput stops growing and per-user latency grows linearly with batch size.
3. **KV-bound.** At context 2,000, the KV cache reads grow with the batch and soon dominate. Throughput still improves with batch size, but much less, per-user speed falls noticeably, and at batch 512 the caches no longer fit in memory at all.

These are idealized lower bounds, and real systems reach only a fraction of peak bandwidth and compute. But the shape is right, and it explains the main strategies of the rest of the chapter:

- **Batching** (Section 6) is how a server moves from the left of the roofline toward the ridge point. Its value is enormous at low batch sizes, which is why continuous batching was such an important innovation.
- **Shrinking the KV cache** (Section 4) raises the batch size at which memory runs out and reduces the KV term that limits throughput at long context.
- **Quantizing weights** (Section 6) reduces the weight term, which dominates at small batch, so it helps latency-sensitive, low-batch serving the most.
- **The latency-throughput tradeoff** (Section 8) is visible directly: larger batches give more total tokens per second and a lower cost per token, but each user's tokens arrive more slowly. A service must choose where on this curve to operate.

There is one more subtlety. For a single user, batching does not help at all: that user's decode is stuck at about 1 FLOP per byte. Speculative decoding (Section 6) is a way to create a "batch" of several tokens from a single sequence, raising intensity for one user.

## Beyond the roofline

The roofline model ignores several effects that matter in practice:

- **Kernel launch and synchronization overheads.** Each layer launches many small GPU kernels. At batch 1, a small model can spend a noticeable share of each step on launch overhead rather than on memory traffic or arithmetic. Techniques such as CUDA Graphs, which record a sequence of kernel launches once and replay it, and kernel fusion (Section 6) reduce this.
- **Non-matmul operations.** Normalizations, activations, and rotary embeddings have very low intensity. Executed as separate kernels, each reads and writes whole activation tensors, adding memory traffic out of proportion to their FLOPs. Fusing them into neighboring kernels removes most of this traffic.
- **Achievable vs. peak.** Well-tuned large matrix multiplies can reach a large fraction of peak FLOP/s, and good decode kernels a large fraction of peak bandwidth, but less-optimized code can fall far short of both. The roofline tells you the ceiling, not what your code achieves.
- **Communication.** When a model spans several devices, the interconnect becomes a fourth roof. Section 7 treats this.

## Key takeaways

- Three hardware numbers dominate inference: peak compute (FLOP/s), memory bandwidth (bytes/s), and memory capacity (bytes); compute has grown faster than bandwidth for years.
- Arithmetic intensity is FLOPs per byte moved; a computation is memory-bound below the ridge point $P/B$ and compute-bound above it, and the roofline gives attainable performance as $\min(P, I \cdot B)$.
- A matrix multiply over a batch of $b$ tokens has an intensity of about $b$ FLOPs per byte (16-bit), so single-sequence decode is deeply memory-bound, while long-prompt prefill is compute-bound.
- Attention over the KV cache during decode has an intensity of about $h / h_{kv}$ regardless of batch size, so it is always memory-bound.
- Batching amortizes weight reads across sequences, raising throughput almost for free until the system becomes compute-bound, KV-bound, or runs out of memory; per-user speed falls as the batch grows.

## Further reading

Pope, Reiner, et al. "Efficiently Scaling Transformer Inference." In *Proceedings of Machine Learning and Systems 5*, 2023. https://arxiv.org/abs/2211.05102.

Shazeer, Noam. "Fast Transformer Decoding: One Write-Head Is All You Need." arXiv preprint arXiv:1911.02150, 2019. https://arxiv.org/abs/1911.02150.

Williams, Samuel, et al. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM* 52, no. 4 (2009): 65–76.
