# 13.4 The KV Cache

Section 1 noted that the decoding loop only needs the logits for the last position, yet a naive implementation recomputes the whole sequence at every step. The **key-value (KV) cache** removes that waste. It is the single most important data structure in LLM inference: it turns generation from quadratic to linear work per token, it is the reason decode can process one token at a time (Section 3), and it is very often the thing that limits how many users a server can handle and how long their contexts can be. This section explains what the cache stores, how to compute its size, why it creates memory pressure, and the architectural and systems techniques used to shrink and manage it.

## What is stored and why

### Recomputation without a cache

Recall the causal self-attention of Chapter 6. In one layer and one head, each position $t$ has a query $\mathbf{q}_t = W_Q \mathbf{x}_t$, a key $\mathbf{k}_t = W_K \mathbf{x}_t$, and a value $\mathbf{v}_t = W_V \mathbf{x}_t$, where $\mathbf{x}_t$ is the layer input at that position. The attention output at position $t$ is

```math
\mathbf{o}_t = \sum_{\tau=1}^{t} \alpha_{t\tau} \mathbf{v}_\tau, \qquad \alpha_{t\tau} = \frac{\exp\big(\mathbf{q}_t^\top \mathbf{k}_\tau / \sqrt{d_h}\big)}{\sum_{\tau'=1}^{t} \exp\big(\mathbf{q}_t^\top \mathbf{k}_{\tau'} / \sqrt{d_h}\big)}.
```

Two facts follow from the causal mask:

1. The output at position $t$ depends only on inputs at positions $\le t$. Adding a new token at position $t+1$ does not change anything computed for positions $1, \dots, t$, in this layer or any other.
2. To compute the output for the new position, we need its query, plus the keys and values of *all* earlier positions.

Without a cache, generating token $t+1$ means running the full network on all $t$ tokens again, recomputing every earlier key and value just to throw them away except for the last position's output. The total work to generate $g$ tokens grows like $g^2$ times the per-token cost.

### Caching keys and values

Fact 1 says the earlier keys and values never change. So we compute them once and store them. At each decode step, for each layer:

1. Compute $`\mathbf{q}_{t}, \mathbf{k}_{t}, \mathbf{v}_{t}`$ for the *new* token only.
2. Append $`\mathbf{k}_{t}`$ and $`\mathbf{v}_{t}`$ to the cache: $`K_{1:t} = [K_{1:t-1}; \mathbf{k}_t^\top]`$, $`V_{1:t} = [V_{1:t-1}; \mathbf{v}_t^\top]`$.
3. Compute attention for the new query against all cached keys and values.

Everything else in the layer (the output projection, the MLP, the normalizations) operates on one position at a time and needs no history. So keys and values are *all* that must be kept. Queries are not cached because each query is used only once, at its own position.

Why cache keys and values rather than, say, the layer inputs $\mathbf{x}_\tau$? We could store the inputs and recompute $K$ and $V$ from them, which would take the same memory for multi-head attention but cost extra matrix multiplies on every step. Caching $K$ and $V$ directly trades memory for compute, which is the right trade when, as in decode, compute is cheap relative to everything else. (With grouped-query attention, below, caching $K$ and $V$ also takes *less* memory than caching inputs.)

With the cache, a decode step does $O(1)$ weight work per token plus $O(t)$ attention work against the cache, instead of redoing all $t$ tokens. Prefill (Section 3) is simply the step that fills the cache for the prompt in one parallel pass.

### A tiny KV cache

The following NumPy code implements a single attention head with and without a cache, checks that the outputs are identical, and times both. (This is the second suggested code lab.)

```python
import time
import numpy as np

rng = np.random.default_rng(0)
d, n_steps = 256, 512
Wq, Wk, Wv = (rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(3))

def attend(q, K, V):
    s = K @ q / np.sqrt(d)
    a = np.exp(s - s.max()); a /= a.sum()
    return V.T @ a

def step_no_cache(xs):
    """Recompute keys and values for every past token at every step."""
    X = np.stack(xs)
    K, V = X @ Wk, X @ Wv
    return attend(xs[-1] @ Wq, K, V)

class KVCache:
    def __init__(self, max_len, d):
        self.K = np.empty((max_len, d)); self.V = np.empty((max_len, d)); self.n = 0
    def append(self, k, v):
        self.K[self.n], self.V[self.n] = k, v; self.n += 1
    def view(self):
        return self.K[:self.n], self.V[:self.n]

def step_with_cache(x, cache):
    """Compute k, v only for the new token; reuse everything else."""
    cache.append(x @ Wk, x @ Wv)
    K, V = cache.view()
    return attend(x @ Wq, K, V)

xs = [rng.standard_normal(d) for _ in range(n_steps)]

t0 = time.perf_counter()
out_a = [step_no_cache(xs[:t + 1]) for t in range(n_steps)]
t1 = time.perf_counter()
cache = KVCache(n_steps, d)
out_b = [step_with_cache(x, cache) for x in xs]
t2 = time.perf_counter()

assert np.allclose(out_a, out_b)          # identical outputs
print(f"no cache: {t1 - t0:.3f} s, with cache: {t2 - t1:.3f} s")
```

On an ordinary laptop the cached version is faster by well over an order of magnitude at this length, and the gap widens as `n_steps` grows, because the uncached version's total work grows quadratically. Note that the cache is preallocated to a maximum length. Real systems must decide how much memory to reserve for sequences whose final length is unknown, which is the problem PagedAttention solves later in this section.

## Computing its size

For every layer, every key-value head, and every token, the cache holds one key vector and one value vector of dimension $d_h$. The total size in bytes is

```math
\text{KV bytes} = 2 \times L \times h_{kv} \times d_h \times n \times b \times s,
```

where the leading 2 counts keys and values, $L$ is the number of layers, $h_{kv}$ the number of key-value heads (equal to the number of attention heads $h$ in standard multi-head attention), $d_h$ the head dimension, $n$ the sequence length, $b$ the batch size (number of sequences), and $s$ the bytes per element (2 for FP16 or BF16, 1 for FP8 or INT8).

It is convenient to separate out the size *per token*:

```math
\text{KV bytes per token} = 2 \, L \, h_{kv} \, d_h \, s.
```

Some examples, all with 16-bit entries ($s = 2$):

| Model | $L$ | $h_{kv}$ | $d_h$ | KV per token | One 4,096-token sequence |
|---|---|---|---|---|---|
| Llama 2 7B (multi-head) | 32 | 32 | 128 | 512 KiB | 2 GiB |
| Llama 3 8B (grouped-query) | 32 | 8 | 128 | 128 KiB | 512 MiB |
| Llama 2 70B (grouped-query) | 80 | 8 | 128 | 320 KiB | 1.25 GiB |

For Llama 2 7B, the calculation is $2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$ bytes, exactly half a mebibyte per token. A single 4,096-token conversation needs 2 GiB of cache, and a batch of 16 such conversations needs 32 GiB, more than twice the roughly 13.5 GB of the model's own BF16 weights.

```python
def kv_cache_bytes(L, n_kv_heads, d_head, seq_len, batch=1, bytes_per_elem=2):
    return 2 * L * n_kv_heads * d_head * seq_len * batch * bytes_per_elem

GiB = 2 ** 30
print(kv_cache_bytes(32, 32, 128, 4096) / GiB)             # Llama 2 7B:  2.0
print(kv_cache_bytes(32, 8, 128, 4096) / GiB)              # Llama 3 8B:  0.5
print(kv_cache_bytes(32, 8, 128, 128 * 1024) / GiB)        # Llama 3 8B at 128K context: 16.0
```

## Memory pressure at long context and large batch

The formula is linear in both $n$ and $b$, and this is exactly the problem. A server's accelerator memory must hold the model weights, a working area for activations, and the KV caches of every active request. Whatever is left after the weights determines how many tokens of cache fit, and that number limits the product of batch size and context length.

**Capacity.** Consider serving Llama 3 8B in BF16 on our hypothetical 80 GB accelerator from Section 3. The weights take about 16 GB. If about 60 GB remain for cache after activations and overheads, then at 128 KiB per token the device can hold roughly

```math
\frac{60 \times 10^9}{131{,}072} \approx 458{,}000 \text{ tokens of cache}.
```

That could be about 55 concurrent requests at 8,000 tokens each, or only three requests at the full 128,000-token context. Long context and high concurrency compete for the same memory. When a server runs out of cache space, it must reject, queue, or preempt requests.

**Bandwidth.** The cache must not only be stored but also *read*. On every decode step, each sequence's attention reads its entire cache in every layer. Section 3 estimated decode time as bytes read divided by bandwidth. For a batch of $b$ sequences, the bytes read per step are roughly

```math
\underbrace{2N}_{\text{weights (16-bit)}} + \underbrace{b \times 2 L h_{kv} d_h n s}_{\text{KV caches}}.
```

The weights are read once per step no matter how large the batch, but each sequence's cache is its own. For Llama 3 8B at 128,000 tokens, one sequence's cache (16 GiB) is about as large as the weights, so a single long-context request decodes at roughly half the speed of a short one. With many long requests in a batch, cache reads dominate. Unlike weight reads, cache reads cannot be amortized by batching, a point Section 5 makes precise with the roofline model.

**Why capacity limits throughput.** Section 5 shows that larger batches use the hardware more efficiently. The KV cache is usually what stops a server from growing the batch further. Every technique that shrinks the cache therefore also raises throughput, not just maximum context length.

## Reducing it

There are two broad strategies. **Architectural** changes reduce the number of bytes per token by changing the model. **Systems** techniques reduce waste in how the cache is stored and shared, without changing the model.

### Multi-query and grouped-query attention

In standard multi-head attention (MHA), each of the $h$ query heads has its own key head and value head, so $h_{kv} = h$. Shazeer observed that decode was dominated by loading keys and values and proposed **multi-query attention (MQA)**: all query heads share a *single* key head and value head, so $h_{kv} = 1$. The KV cache shrinks by a factor of $h$ (for example, 32 times), and so does the memory traffic for reading it. The cost is some loss of model quality and, in some reports, less stable training.

**Grouped-query attention (GQA)**, introduced by Ainslie et al., interpolates between the two. The $h$ query heads are divided into $h_{kv}$ groups, and each group shares one key head and one value head:

```math
\text{head } j \text{ uses } K^{(g(j))}, V^{(g(j))}, \qquad g(j) = \left\lfloor \frac{j}{h / h_{kv}} \right\rfloor \quad (\text{heads numbered from } 0).
```

With $h_{kv} = h$, GQA is MHA; with $h_{kv} = 1$, it is MQA. Ainslie et al. found that intermediate values achieve quality close to MHA with speed close to MQA, and showed that an existing MHA checkpoint can be converted to GQA by mean-pooling its key and value heads within each group and then continuing training ("uptraining") for a small fraction of the original compute. GQA has become the default in many open models: the Llama 3 8B entry in the table above uses 8 key-value heads for 32 query heads, a 4 times reduction relative to Llama 2 7B.

GQA also changes the arithmetic of decode. Each key and value loaded from memory is now used by $h / h_{kv}$ query heads instead of one, so attention does more FLOPs per byte read (Section 5).

### Other ways to shrink the cache

- **Multi-head latent attention (MLA)**, introduced in DeepSeek-V2, stores a single compressed low-rank *latent* vector per token and layer, from which keys and values for all heads are reconstructed. The cache holds only the latent, which is much smaller than full keys and values.
- **KV cache quantization** stores keys and values in 8-bit (FP8 or INT8) or even lower precision, halving or more the memory and bandwidth relative to 16-bit (Section 6 covers quantization in general).
- **Sliding-window attention** (Section 2) caches only the last $w$ tokens in the affected layers, capping the cache size for those layers.
- **Eviction and compression** methods drop or merge cached entries judged less important. Xiao et al. observed that many models put large attention weight on the first few tokens ("attention sinks") and showed that keeping those sink tokens plus a recent window allows streaming generation well beyond the training length, though the model loses access to the evicted middle. Eviction methods are approximations, and their effect on quality must be measured for each task.

### PagedAttention

Even with a compact cache format, *how* the cache is laid out in memory matters. Early serving systems allocated one contiguous buffer per request, sized for the maximum possible length, because the final output length is not known in advance. This wastes memory in three ways: space reserved for tokens that are never generated, *internal fragmentation* when a request finishes far short of its reservation, and *external fragmentation* when free memory is split into gaps too small for a new request's contiguous buffer. Kwon et al. measured that such systems left a large fraction of KV memory unused, which directly limited batch size.

**PagedAttention**, introduced by Kwon et al. with the vLLM serving system, borrows the idea of virtual memory and paging from operating systems:

- The cache is divided into fixed-size **blocks**, each holding keys and values for a fixed number of tokens (for example 16).
- Each request has a **block table** mapping its *logical* blocks (tokens 0 to 15, 16 to 31, and so on) to *physical* blocks anywhere in memory. Physical blocks need not be contiguous.
- Blocks are allocated on demand as the sequence grows. Waste is limited to the unfilled part of each request's last block.
- The attention kernel is modified to follow the block table, gathering keys and values from scattered blocks.

Paging also makes **sharing** easy. Several sequences can point to the same physical blocks, with a reference count on each block. This helps whenever sequences share a prefix:

- **Parallel sampling.** Generating several samples for the same prompt (as in the self-consistency method of Section 9) stores the prompt's cache once.
- **Beam search.** Beams share the blocks of their common history.
- **Shared system prompts and multi-turn chat.** The cache for a common prefix can be reused across requests, the basis of prefix caching (Section 3).

When a sequence needs to write into a shared block, the block is copied first (**copy-on-write**), just as an operating system handles a forked process. When memory runs out, a paged system can also *preempt* a request, either swapping its blocks to CPU memory or discarding them and recomputing them later, and resume it when space frees up.

By nearly eliminating fragmentation, paging lets a server fit many more concurrent sequences into the same memory, which raises throughput for the reasons developed in Section 5. Paged KV memory management is now standard in major serving engines (Section 8).

## Key takeaways

- The KV cache stores the keys and values of every past token in every layer, so each decode step computes only the new token's projections instead of reprocessing the whole sequence.
- Its size is $2 \times L \times h_{kv} \times d_h \times n \times b \times s$ bytes; for a 7B multi-head model in 16-bit precision, that is 0.5 MiB per token.
- The cache grows linearly with context length and batch size, often exceeding the weights; it limits concurrency, and reading it adds to every decode step's memory traffic in a way batching cannot amortize.
- MQA and GQA share key-value heads across query heads; MLA, KV quantization, sliding windows, and eviction shrink the cache further.
- PagedAttention stores the cache in fixed-size blocks mapped through a block table, eliminating most fragmentation and enabling prefix sharing with copy-on-write.

## Further reading

Ainslie, Joshua, et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 2023. https://arxiv.org/abs/2305.13245.

DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." arXiv preprint arXiv:2405.04434, 2024. https://arxiv.org/abs/2405.04434.

Kwon, Woosuk, et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." In *Proceedings of the 29th Symposium on Operating Systems Principles*, 2023. https://arxiv.org/abs/2309.06180.

Pope, Reiner, et al. "Efficiently Scaling Transformer Inference." In *Proceedings of Machine Learning and Systems 5*, 2023. https://arxiv.org/abs/2211.05102.

Shazeer, Noam. "Fast Transformer Decoding: One Write-Head Is All You Need." arXiv preprint arXiv:1911.02150, 2019. https://arxiv.org/abs/1911.02150.

Xiao, Guangxuan, et al. "Efficient Streaming Language Models with Attention Sinks." In *International Conference on Learning Representations*, 2024. https://arxiv.org/abs/2309.17453.
