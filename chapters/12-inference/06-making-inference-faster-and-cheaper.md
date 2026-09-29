# 12.6 Making Inference Faster and Cheaper

The previous sections built a cost model for inference. Prefill is compute-bound; decode is memory-bandwidth-bound; the KV cache competes with the weights for memory and bandwidth; and larger batches amortize the cost of reading weights. This section uses that model to explain the main techniques practitioners use to make inference faster and cheaper. Each technique attacks one term of the cost model:

| Technique | What it reduces | Helps most |
|---|---|---|
| Quantization | Bytes per weight (and per KV entry, per activation) | Memory-bound decode; fitting models on fewer devices |
| Batching | Weight reads per generated token | Throughput and cost per token |
| Speculative decoding | Sequential target-model passes per token | Latency of memory-bound, low-batch decode |
| Kernel optimizations | Wasted memory traffic and overheads | Attention at long context; all small operations |
| Model-level options | Parameters, or active parameters, per token | Everything, but requires changing the model |

These techniques combine. A typical production deployment might serve a mixture-of-experts model with 8-bit weights, continuous batching, FlashAttention-style kernels, a paged KV cache, and speculative decoding.

## Quantization

*Quantization* stores numbers in fewer bits. Since decode time is roughly proportional to the bytes read (Section 3), halving the bytes per weight nearly halves the time per token at small batch sizes, and also halves the memory the weights occupy, leaving more room for the KV cache or allowing a model to fit on fewer devices.

### Number formats

| Format | Bits | Layout | Typical use |
|---|---|---|---|
| FP32 | 32 | 1 sign, 8 exponent, 23 mantissa | Rarely used for LLM inference weights |
| FP16 | 16 | 1 sign, 5 exponent, 10 mantissa | Inference; narrow range can overflow |
| BF16 | 16 | 1 sign, 8 exponent, 7 mantissa | Default training and inference format |
| FP8 (E4M3 / E5M2) | 8 | 4 or 5 exponent bits, 3 or 2 mantissa | Weights, activations, KV cache on newer hardware |
| INT8 | 8 | Integer in $[-128, 127]$ plus a scale | Weights and activations |
| INT4 | 4 | Integer in $[-8, 7]$ plus a scale | Weights |

**BF16** keeps the 8-bit exponent of FP32, so it covers the same dynamic range with less precision. This makes it robust for models trained in BF16, and it is the usual "unquantized" baseline for inference. **FP16** has more precision but a much narrower range, so activations that are large in some models can overflow. The 8-bit floating-point formats, standardized for deep learning by Micikevicius et al., come in two variants: E4M3 (more precision) and E5M2 (more range). Hardware that supports FP8 matrix multiplies can run them at roughly twice the 16-bit rate.

Integer formats represent real numbers through a **scale** (and optionally a zero point). Their value comes from how that scale is chosen.

### Uniform quantization

The most common scheme is *symmetric* ("absmax") quantization. For a group of real values $\mathbf{w}$ and a $k$-bit signed integer format with $q_{\max} = 2^{k-1} - 1$:

$$
s = \frac{\max_i |w_i|}{q_{\max}}, \qquad q_i = \text{clip}\Big(\text{round}\big(w_i / s\big), -q_{\max}, q_{\max}\Big), \qquad \hat{w}_i = s \, q_i.
$$

The integers $q_i$ are stored, together with the scale $s$ in higher precision. *Asymmetric* quantization adds a zero point $z$ so that the integer range covers $[\min w, \max w]$ rather than a symmetric interval: $q_i = \text{round}(w_i / s) + z$ and $\hat{w}_i = s(q_i - z)$.

The rounding error for each value is at most $s/2$. The scale is set by the largest magnitude in the group, so a single outlier makes the scale large and the resolution for every other value in the group coarse. This is why the **granularity** of scales matters:

- **Per-tensor:** one scale for the whole matrix. Cheapest, but most sensitive to outliers.
- **Per-channel:** one scale per output row. Standard for INT8 weights.
- **Per-group:** one scale per group of, say, 64 or 128 consecutive weights. Standard for 4-bit weights. The scales add a little storage (for groups of 128 with 16-bit scales, 16/128 = 0.125 extra bits per weight).

The following code quantizes a random matrix and measures the error in a matrix-vector product:

```python
import numpy as np

def quantize_per_channel(W, bits=8):
    """Symmetric per-output-channel quantization of a weight matrix W [out, in]."""
    qmax = 2 ** (bits - 1) - 1                      # 127 for INT8, 7 for INT4
    scale = np.abs(W).max(axis=1, keepdims=True) / qmax
    Q = np.clip(np.round(W / scale), -qmax, qmax).astype(np.int8)
    return Q, scale

def dequantize(Q, scale):
    return Q.astype(np.float32) * scale

rng = np.random.default_rng(0)
W = rng.standard_normal((4096, 4096)).astype(np.float32) * 0.02
x = rng.standard_normal(4096).astype(np.float32)

for bits in [8, 4]:
    Q, s = quantize_per_channel(W, bits)
    y, y_hat = W @ x, dequantize(Q, s) @ x
    rel_err = np.linalg.norm(y - y_hat) / np.linalg.norm(y)
    print(f"INT{bits}: relative output error {rel_err:.4f}")

def quantize_per_group(W, bits=4, group=128):
    """Symmetric quantization with one scale per group of `group` input weights."""
    out_dim, in_dim = W.shape
    Wg = W.reshape(out_dim, in_dim // group, group)
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(Wg).max(axis=2, keepdims=True) / qmax
    Q = np.clip(np.round(Wg / scale), -qmax, qmax).astype(np.int8)
    return Q, scale

Q, s = quantize_per_group(W, bits=4, group=128)
y_hat = (Q.astype(np.float32) * s).reshape(W.shape) @ x
print(f"INT4, groups of 128: relative output error {np.linalg.norm(W @ x - y_hat) / np.linalg.norm(W @ x):.4f}")
```

It prints relative output errors of about 0.9 percent for INT8, 16 percent for per-channel INT4, and 12 percent for INT4 with groups of 128. Two lessons follow. Each bit removed doubles the rounding step, so INT4 is far noisier than INT8. And finer groups help; they help much more on real LLM weights, which, unlike this Gaussian toy matrix, contain outliers. Naive 4-bit rounding is therefore not good enough for real models, and better 4-bit methods matter (see below). The fourth suggested code lab asks you to quantize a small real model to INT8 and compare speed, memory, and output quality.

### Weight-only vs. weight-and-activation quantization

There are two different goals, and they call for different kinds of quantization.

**Weight-only quantization** (often written W8A16 or W4A16: 8- or 4-bit weights, 16-bit activations) stores weights in low precision but performs the arithmetic in 16-bit. A kernel loads the compact weights from memory, dequantizes them in fast on-chip memory, and multiplies. The FLOPs are unchanged, but the bytes read shrink by a factor of 2 (INT8) or about 4 (INT4). Since small-batch decode is memory-bound, this translates almost directly into faster tokens and lower memory. Weight-only quantization is the standard approach for local inference (for example the 4-bit and 5-bit formats in llama.cpp) and for latency-sensitive serving.

**Weight-and-activation quantization** (for example W8A8 in INT8 or FP8) quantizes both operands, so the matrix multiply itself runs on low-precision tensor cores at higher FLOP/s. This helps the *compute-bound* regime: prefill and large-batch decode. It is harder, because activations are computed on the fly (so their scales must be estimated or computed dynamically), and because LLM activations contain **outliers**: a few feature dimensions with magnitudes far larger than the rest. Dettmers et al. found that such outlier features emerge systematically in large transformers and proposed LLM.int8(), which performs the few outlier dimensions in 16-bit and the rest in INT8. Xiao et al. proposed SmoothQuant, which notes that weights are easy to quantize and activations are hard, and migrates the difficulty with a mathematically equivalent per-channel rescaling: activations are divided by a factor $\mathbf{s}$ and the corresponding weight rows multiplied by it, $Y = (X \, \text{diag}(\mathbf{s})^{-1}) (\text{diag}(\mathbf{s}) \, W)$.

The **KV cache** can be quantized too (Section 4), typically to 8 bits, which reduces both its memory and the bandwidth spent reading it at long context.

### Post-training quantization methods

Most LLM quantization is *post-training quantization* (PTQ): it takes a trained model and a small calibration set and produces a quantized model without retraining.

- **Round-to-nearest (RTN)** simply applies the formula above. It works well for 8-bit weights and poorly for 4-bit.
- **GPTQ** (Frantar et al.) quantizes the weights of each layer one column at a time and, after each column, updates the not-yet-quantized columns to compensate for the error, using second-order (Hessian) information computed from calibration activations. The goal is to minimize the error in the layer's *output*, $\lVert WX - \hat{W}X \rVert^2$, not in the weights themselves.
- **AWQ** (Lin et al.) observes that a small fraction of weight channels matter disproportionately, namely those multiplying large activations. It scales those channels up before quantization (and the corresponding activations down), protecting them from rounding error without mixed-precision storage.

With methods like these, 8-bit weight quantization is usually close to lossless, and 4-bit weight quantization usually costs a small amount of quality, more for smaller models and for demanding tasks such as math and code. The only reliable way to know for a given model and use case is to evaluate the quantized model on the tasks you care about (Chapter 11), not just perplexity.

*Quantization-aware training* (QAT) simulates quantization during training or fine-tuning so the model learns to be robust to it. It costs training compute but can recover quality at very low bit widths. QLoRA (Dettmers et al., 2023) combined a 4-bit "NormalFloat" (NF4) weight format with low-rank adapters for memory-efficient fine-tuning, and its NF4 format is also used for inference.

## Batching

Section 5 showed that batching many sequences into one forward pass amortizes the cost of reading weights and can multiply throughput with little effect on per-user latency. The question is how to form batches when requests arrive at random times and generate outputs of unpredictable length.

### Static batching

The simplest approach collects a fixed set of requests, pads their prompts to the same length, and runs them together until *every* sequence has finished. This works for offline jobs, but it wastes a lot of compute for interactive serving:

- A sequence that finishes early (after 20 tokens) occupies its slot until the longest sequence in the batch (perhaps 2,000 tokens) finishes. Its slot does no useful work, and it cannot be given to a waiting request.
- New requests must wait for the entire batch to finish before starting, which increases their time to first token.
- Padding wastes computation and memory on tokens that carry no information.

*Dynamic batching* improves on this by forming batches from whatever requests have arrived within a short time window, but the batch still runs as a unit until its longest member finishes.

### Continuous batching

**Continuous batching**, also called *iteration-level scheduling* or *in-flight batching*, was introduced by Yu et al. in the Orca serving system. The key idea is to make scheduling decisions at every decode iteration rather than once per batch:

1. At each iteration, the scheduler looks at the set of running sequences.
2. Sequences that emitted a stop token or hit their length limit are removed immediately, and their KV cache memory is freed.
3. Waiting requests are admitted into the freed slots, as long as there is memory for their KV cache.
4. The engine runs one forward pass over all running sequences, producing one new token for each (and running the prefill for newly admitted requests).

```python
def serve_loop(engine, waiting_queue, max_batch_tokens):
    running = []
    while True:
        running = [seq for seq in running if not seq.finished]     # free finished slots
        while waiting_queue and engine.can_admit(waiting_queue[0], running, max_batch_tokens):
            running.append(waiting_queue.pop(0))                    # admit new requests
        if not running:
            engine.wait_for_requests(); continue
        next_tokens = engine.step(running)      # prefill for new sequences, decode for others
        for seq, tok in zip(running, next_tokens):
            seq.append(tok)                     # also streams the token to the client
```

The GPU is never held hostage by a long sequence, new requests start almost immediately, and batch sizes stay high. Different sequences in the same iteration have different lengths, which is awkward for attention; Orca's *selective batching* batches the token-wise operations (the linear layers and MLP, which dominate FLOPs) across all sequences while computing attention separately per sequence. Combined with paged KV cache management (Section 4), which lets sequences grow and shrink without fragmentation, continuous batching is the foundation of modern serving engines (Section 8).

Two refinements are common. **Chunked prefill** (Section 3) splits long prompts into pieces so that admitting a new request does not stall the decode steps of everyone else. **Preemption** handles the case where running sequences grow until the KV cache is full: the scheduler pauses some sequences, either swapping their cache to CPU memory or discarding it to be recomputed later, and resumes them when memory frees up.

## Speculative decoding

Batching helps throughput, but it does nothing for a single user's sequence, which is still limited to one token per pass through the full model. **Speculative decoding** reduces the number of sequential passes of the large model needed per token, without changing the output distribution.

### The idea

Decode of a large *target* model $p$ is memory-bound: verifying several tokens in one pass costs little more than generating one, because the weights are read once either way (Section 5). Suppose a much smaller, faster *draft* model $q$ proposes $\gamma$ tokens. The target model can score all $\gamma$ proposals in a single forward pass, exactly as it would process a short prompt, computing its own distributions $p(\cdot \mid x_{\lt t+i})$ at every proposed position. If the draft guessed well, several tokens are accepted for the price of one target pass.

This idea was developed independently by Leviathan et al. and by Chen et al. (who called it *speculative sampling*), building on earlier blockwise parallel decoding by Stern et al.

### The algorithm

One step of speculative decoding with draft length $\gamma$:

1. **Draft.** Run the draft model autoregressively for $\gamma$ steps, sampling $\tilde{x}_1, \dots, \tilde{x}_\gamma$ with $\tilde{x}_i \sim q(\cdot \mid x, \tilde{x}_{\lt i})$, and record the draft probabilities.
2. **Verify.** Run the target model once on the context plus all $\gamma$ draft tokens, obtaining target distributions $p_i = p(\cdot \mid x, \tilde{x}_{\lt i})$ for $i = 1, \dots, \gamma + 1$.
3. **Accept or reject.** For $i = 1, 2, \dots$, accept $\tilde{x}_i$ with probability

$$
\min\left(1, \; \frac{p_i(\tilde{x}_i)}{q_i(\tilde{x}_i)}\right).
$$

   At the first rejection, discard the remaining draft tokens and instead sample a replacement from the *residual* distribution

$$
p'_i(x) = \frac{\max\big(0, \, p_i(x) - q_i(x)\big)}{\sum_{x'} \max\big(0, \, p_i(x') - q_i(x')\big)}.
$$

4. **Bonus token.** If all $\gamma$ drafts are accepted, sample one extra token from $p_{\gamma+1}$, which the verification pass already computed.

Every step therefore produces between 1 and $\gamma + 1$ tokens for one target pass.

### Why the output distribution is exact

The acceptance rule is a form of rejection sampling. Consider one position with draft distribution $q$ and target distribution $p$. The probability that the procedure emits token $x$ is the probability that the draft proposes $x$ and it is accepted, plus the probability that some proposal is rejected and the residual produces $x$:

$$
\Pr[\text{emit } x] = q(x) \min\left(1, \frac{p(x)}{q(x)}\right) + \Big(1 - \sum_{x'} \min\big(q(x'), p(x')\big)\Big) \, p'(x).
$$

The first term equals $\min(q(x), p(x))$. The total rejection probability is $1 - \sum_{x'} \min(q(x'), p(x')) = \sum_{x'} \max(0, p(x') - q(x'))$, which is exactly the normalizer of $p'$, so the second term equals $\max(0, p(x) - q(x))$. Adding them gives $\min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x)$. The emitted token has exactly the target distribution, however poor the draft is. A bad draft only costs speed, not quality. (With greedy decoding, the rule reduces to accepting draft tokens as long as they match the target's argmax.)

Here is an implementation of the verification step, with a simulation confirming that the first emitted token follows the target distribution:

```python
import numpy as np

def speculative_step(p_target, q_draft, draft_tokens, rng):
    """One verification step of speculative sampling.
    q_draft[i]:  draft distribution used to sample draft_tokens[i]
    p_target[i]: target distribution at the same position (len(draft_tokens) + 1 rows,
                 the last row is for the position after all draft tokens)."""
    out = []
    for i, x in enumerate(draft_tokens):
        p, q = p_target[i], q_draft[i]
        if rng.random() < min(1.0, p[x] / q[x]):     # accept with prob min(1, p/q)
            out.append(x)
            continue
        residual = np.maximum(p - q, 0.0)            # reject: resample from (p - q)+
        out.append(int(rng.choice(len(p), p=residual / residual.sum())))
        return out
    out.append(int(rng.choice(len(p_target[-1]), p=p_target[-1])))   # all accepted: bonus token
    return out

# Check that the first emitted token follows the target distribution exactly.
rng = np.random.default_rng(0)
p = np.array([0.5, 0.3, 0.15, 0.05])
q = np.array([0.25, 0.25, 0.25, 0.25])
counts = np.zeros(4)
for _ in range(200_000):
    x = int(rng.choice(4, p=q))
    first = speculative_step([p, p], [q], [x], rng)[0]
    counts[first] += 1
print(np.round(counts / counts.sum(), 3))           # close to [0.5, 0.3, 0.15, 0.05]
```

### How much it helps

Let $\alpha$ be the probability that a draft token is accepted, and assume for simplicity that acceptances are independent. Leviathan et al. showed that the expected number of tokens produced per target pass is

$$
\mathbb{E}[\text{tokens per step}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}.
$$

With $\alpha = 0.8$ and $\gamma = 4$, this is $(1 - 0.8^5)/0.2 \approx 3.4$ tokens per target pass. The wall-clock speedup is lower than this, because each step also pays for $\gamma$ draft passes and a slightly more expensive verification pass. The best $\gamma$ balances these costs against the acceptance rate.

Acceptance rates depend on how well the draft predicts the target, which depends on the task: code and structured or repetitive text are often easy to predict, creative writing less so. Speculative decoding helps most when decode is memory-bound, that is, at small batch sizes. At large batch sizes, the target model is already closer to compute-bound, the extra verification FLOPs are no longer free, and the gains shrink or disappear.

### Variants

- **Choosing the draft.** A smaller model from the same family (sharing the tokenizer) is the classic choice.
- **Self-drafting heads.** Medusa (Cai et al.) adds extra prediction heads to the target model that guess several future tokens at once, verified with a tree of candidates. EAGLE (Li et al.) trains a lightweight draft head that predicts from the target model's own hidden features.
- **Model-free drafts.** *Prompt lookup* or n-gram drafting proposes tokens by copying spans from the prompt or earlier output, which works surprisingly well for tasks such as editing code or summarizing with quotations.
- **Tree verification.** Instead of one draft sequence, verify a tree of alternatives in one pass using a specially masked attention, increasing the chance that some branch is accepted.

## Kernel optimizations

A *kernel* is a function that runs on the accelerator, such as one matrix multiply or one normalization. Many of the biggest practical speedups in inference come not from new algorithms but from kernels that move less data.

### FlashAttention

The standard implementation of attention computes the score matrix $S = QK^\top / \sqrt{d_h}$ (an $n \times n$ matrix per head), writes it to HBM, reads it back to compute the softmax $A$, writes $A$ to HBM, and reads it back to compute $AV$. For long sequences, the $n \times n$ matrices are enormous, and all this traffic makes attention memory-bound even though its FLOPs are high.

**FlashAttention** (Dao et al.) computes exactly the same result without ever writing $S$ or $A$ to HBM. It splits $Q$, $K$, and $V$ into blocks small enough to fit in on-chip memory, and for each block of queries it streams through the blocks of keys and values, accumulating the output. The obstacle is the softmax, which normally needs the maximum and the sum over *all* keys before any output can be computed. FlashAttention uses the **online softmax** trick: keep a running maximum $m$, a running normalizer $\ell$, and a running unnormalized output $\mathbf{o}$, and rescale them when a new block changes the maximum. For a new block of scores $\mathbf{s}^{(j)}$ with values $V^{(j)}$:

$$
m' = \max\big(m, \max_k s^{(j)}_k\big), \qquad \ell' = e^{m - m'} \ell + \sum_k e^{s^{(j)}_k - m'}, \qquad \mathbf{o}' = e^{m - m'} \mathbf{o} + \sum_k e^{s^{(j)}_k - m'} \mathbf{v}^{(j)}_k.
$$

After the last block, the output is $\mathbf{o} / \ell$. This is exact, not an approximation. Because the $n \times n$ matrices are never materialized, attention's memory use becomes linear in $n$ instead of quadratic, and HBM traffic drops sharply. Dao et al. described this as making attention *IO-aware*. Follow-up versions (FlashAttention-2 and FlashAttention-3) improved how work is partitioned across the GPU and exploited features of newer hardware.

During decode, there is only one query per sequence, so parallelizing over query blocks leaves most of the GPU idle at small batch sizes. Decode-specific attention kernels therefore split the *keys and values* across many parallel workers and combine their partial results with the same rescaling rule. Paged attention kernels (Section 4) additionally gather keys and values through a block table.

### Fused operations

Every separate kernel reads its inputs from HBM and writes its outputs back. For low-intensity operations (Section 5), that traffic, not the arithmetic, is the whole cost. **Kernel fusion** combines several operations into one kernel so intermediate results stay in on-chip memory:

- Fusing RMSNorm, rotary position embeddings, activation functions (such as SiLU and the gating multiply in SwiGLU), and residual additions into adjacent kernels.
- Fusing dequantization into matrix multiplies, so compact weights are expanded only in on-chip memory (essential for weight-only quantization to pay off).
- Fusing the final softmax and sampling steps.

A related overhead is **kernel launch** cost. A forward pass may launch hundreds of kernels, and at small batch sizes, the time the CPU spends launching them can rival the time the GPU spends running them. *CUDA Graphs* capture the whole sequence of launches for a decode step once and replay it with a single call. Compilers such as `torch.compile`, and engines such as TensorRT-LLM, perform fusion and graph capture automatically.

## Model-level options

The techniques above make a given model cheaper to run. Often the biggest savings come from changing the model itself.

**Distillation.** Knowledge distillation (Hinton et al.) trains a smaller *student* model to imitate a larger *teacher*, typically by matching the teacher's output distributions rather than only the hard labels:

$$
\mathcal{L}_{\text{KD}} = \sum_t \mathrm{KL}\big(p_{\text{teacher}}(\cdot \mid x_{\lt t}) \,\|\, p_{\text{student}}(\cdot \mid x_{\lt t})\big).
$$

The soft targets carry more information per example than the one-hot next token. For LLMs, distillation also takes the simpler form of fine-tuning a small model on text generated by a large one (sequence-level distillation), which is how many small instruction-following and reasoning models are produced (Chapter 8). A distilled student costs a fraction of the teacher to serve.

**Mixture-of-experts (MoE).** An MoE layer replaces the single MLP with $E$ expert MLPs and a small router that sends each token to the top $k$ of them (Shazeer et al., 2017; Fedus et al., 2022). A model can then have a very large total parameter count, and the capacity that goes with it, while the FLOPs per token scale with the *active* parameters only (Section 2). Mixtral 8x7B, for example, routes each token to 2 of 8 experts. For inference, MoE changes the cost model in an important way: compute per token is low, but *memory* must hold all experts, and at small batch sizes different tokens touch different experts, so a decode step may read many experts' weights for few tokens. MoE models are therefore most economical at large batch sizes and are usually served across many devices with expert parallelism (Section 7).

**Smaller models, trained longer.** The Chinchilla analysis (Hoffmann et al.) found the model size that minimizes loss for a fixed *training* budget. But a model is trained once and served many times. If inference dominates lifetime cost, it pays to train a smaller model on more tokens than is "compute-optimal" for training, accepting a higher training cost for a model that is cheaper per token forever after. The LLaMA models (Touvron et al., 2023) made this argument explicitly, and many recent small models are trained on far more tokens per parameter than the Chinchilla ratio.

**Pruning and sparsity.** Pruning removes weights (unstructured sparsity) or entire neurons, heads, or layers (structured pruning). SparseGPT (Frantar and Alistarh) showed that large GPT-family models can be pruned to substantial sparsity in one shot with limited loss. Unstructured sparsity is hard to turn into real speedups on GPUs; structured pruning and hardware-supported patterns (such as 2:4 sparsity, where two of every four weights are zero) are more practical.

## Key takeaways

- Quantization shrinks bytes per weight: weight-only formats (W8A16, W4A16) speed up memory-bound decode, while weight-and-activation formats (W8A8, FP8) speed up compute-bound prefill; outliers and scale granularity determine quality, and methods like GPTQ and AWQ make 4-bit weights practical.
- Continuous batching schedules at every iteration, removing finished sequences and admitting new ones immediately, which keeps batches large and GPUs busy.
- Speculative decoding lets a cheap draft propose tokens that the target model verifies in one pass; its rejection-sampling rule preserves the target distribution exactly, and it helps most at small batch sizes.
- FlashAttention computes exact attention in on-chip tiles with an online softmax, never materializing the $n \times n$ matrix; fusion and graph capture remove the overhead of many small kernels.
- Distillation, mixture-of-experts, smaller over-trained models, and pruning reduce cost by changing the model; MoE cuts FLOPs per token but not memory.

## Further reading

Cai, Tianle, et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." arXiv preprint arXiv:2401.10774, 2024. https://arxiv.org/abs/2401.10774.

Chen, Charlie, et al. "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv preprint arXiv:2302.01318, 2023. https://arxiv.org/abs/2302.01318.

Dao, Tri. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv preprint arXiv:2307.08691, 2023. https://arxiv.org/abs/2307.08691.

Dao, Tri, et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2205.14135.

Dettmers, Tim, et al. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2208.07339.

Dettmers, Tim, et al. "QLoRA: Efficient Finetuning of Quantized LLMs." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2305.14314.

Fedus, William, et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." *Journal of Machine Learning Research* 23, no. 120 (2022): 1–39. https://arxiv.org/abs/2101.03961.

Frantar, Elias, et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." In *International Conference on Learning Representations*, 2023. https://arxiv.org/abs/2210.17323.

Frantar, Elias, et al. "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." In *Proceedings of the 40th International Conference on Machine Learning*, 2023. https://arxiv.org/abs/2301.00774.

Hinton, Geoffrey, et al. "Distilling the Knowledge in a Neural Network." arXiv preprint arXiv:1503.02531, 2015. https://arxiv.org/abs/1503.02531.

Hoffmann, Jordan, et al. "Training Compute-Optimal Large Language Models." arXiv preprint arXiv:2203.15556, 2022. https://arxiv.org/abs/2203.15556.

Leviathan, Yaniv, et al. "Fast Inference from Transformers via Speculative Decoding." In *Proceedings of the 40th International Conference on Machine Learning*, 2023. https://arxiv.org/abs/2211.17192.

Li, Yuhui, et al. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." In *Proceedings of the 41st International Conference on Machine Learning*, 2024. https://arxiv.org/abs/2401.15077.

Lin, Ji, et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." In *Proceedings of Machine Learning and Systems 6*, 2024. https://arxiv.org/abs/2306.00978.

Micikevicius, Paulius, et al. "FP8 Formats for Deep Learning." arXiv preprint arXiv:2209.05433, 2022. https://arxiv.org/abs/2209.05433.

Shah, Jay, et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv preprint arXiv:2407.08608, 2024. https://arxiv.org/abs/2407.08608.

Shazeer, Noam, et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." In *International Conference on Learning Representations*, 2017. https://arxiv.org/abs/1701.06538.

Stern, Mitchell, et al. "Blockwise Parallel Decoding for Deep Autoregressive Models." In *Advances in Neural Information Processing Systems 31*, 2018. https://arxiv.org/abs/1811.03115.

Touvron, Hugo, et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971, 2023. https://arxiv.org/abs/2302.13971.

Xiao, Guangxuan, et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." In *Proceedings of the 40th International Conference on Machine Learning*, 2023. https://arxiv.org/abs/2211.10438.

Yu, Gyeong-In, et al. "Orca: A Distributed Serving System for Transformer-Based Generative Models." In *16th USENIX Symposium on Operating Systems Design and Implementation*, 2022.
