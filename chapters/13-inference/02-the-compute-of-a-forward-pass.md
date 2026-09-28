# 13.2 The Compute of a Forward Pass

Section 1 showed that generating text means running the model once per new token. Before we can reason about speed, memory, or cost, we need to know how much arithmetic one of those runs takes. This section derives the most useful rule of thumb in LLM inference: a forward pass costs roughly **two floating-point operations per parameter per token**. We then look at where those operations go inside a transformer layer, and at the one part of the computation that does not scale with the parameter count at all: attention over the context, whose cost grows with the length of the sequence.

Throughout, we use the notation of Chapter 6: $L$ layers, model (residual stream) width $d$, $h$ attention heads of dimension $d_h$ (so usually $h \, d_h = d$), feed-forward width $d_{\text{ff}}$, vocabulary size $V$, and context length $n$. $N$ denotes the number of parameters.

## Counting FLOPs

A *FLOP* here means one floating-point operation: one addition or one multiplication. (Hardware specifications report FLOP/s, operations per second; we write FLOPs for counts and FLOP/s for rates.)

### The cost of a matrix multiply

Almost all the arithmetic in a transformer is matrix multiplication. Multiplying an $m \times k$ matrix by a $k \times n$ matrix produces $mn$ outputs, each a dot product of length $k$: $k$ multiplications and $k - 1$ additions, which we round to $2k$ operations. The total is

$$
\text{FLOPs}(m \times k \cdot k \times n) \approx 2mkn.
$$

Now consider a linear layer with weight matrix $W \in \mathbb{R}^{k \times n}$ applied to a batch of $m$ token vectors. The layer has $kn$ parameters and costs $2mkn$ FLOPs. Per token, that is $2kn$: **two FLOPs for every parameter**, one multiply and one add. Every weight is touched exactly once per token.

### Two FLOPs per parameter per token

Because nearly all of a transformer's parameters live in linear layers (the attention projections and the MLP), the same rule holds for the whole network. For a model with $N$ parameters, the forward pass costs approximately

$$
C_{\text{forward}} \approx 2N \quad \text{FLOPs per token}.
$$

This is the approximation used by Kaplan et al. in their scaling-law analysis, together with its training counterpart: the backward pass costs about twice the forward pass, so training costs about $6N$ FLOPs per token. For inference, only the forward pass matters.

The rule is remarkably accurate for standard dense models at moderate context lengths. Its main omissions are:

- **Attention scores.** Computing $QK^\top$ and multiplying the attention weights by $V$ involves no parameters, so it is not counted in $2N$. We treat it separately below; it becomes significant at long context.
- **Embedding lookup.** The input embedding matrix ($V \times d$ parameters) is used as a lookup table, not a matrix multiply, so it costs essentially nothing even though it contributes to $N$. The *output* unembedding $W_U$ is a real matrix multiply and does cost $2Vd$ FLOPs per token. (When the input and output embeddings are tied, the same parameters serve both roles.)
- **Everything else.** Normalization layers, activation functions, residual additions, rotary position embeddings, and the softmax cost a handful of operations per activation, which is $O(d)$ or $O(nh)$ per token per layer rather than $O(d^2)$. These are negligible in FLOPs, though, as Section 5 explains, they are not negligible in memory traffic.

A consequence is worth stating plainly: **the arithmetic cost of a token depends on the model's size, not on what the token is.** A 7-billion-parameter model spends about 14 billion FLOPs on every token, whether it is a comma or the answer to a hard question. This is one motivation for the test-time compute methods of Section 9, which spend more tokens, and so more FLOPs, on harder problems.

### Mixture-of-experts models

In a mixture-of-experts (MoE) model (Section 6), each token is routed to only a few of many expert MLPs. The FLOPs per token then depend on the *active* parameters, not the total:

$$
C_{\text{forward}} \approx 2N_{\text{active}}.
$$

For example, Mixtral 8x7B has about 47 billion total parameters but routes each token through 2 of its 8 experts per layer, so only about 13 billion parameters are active per token. It costs roughly as many FLOPs per token as a 13B dense model, while needing memory for all 47B parameters. Keep this distinction between *compute* (active parameters) and *memory* (total parameters) in mind; it matters throughout the chapter.

## Where the compute goes: attention vs. feed-forward layers

To see where the $2N$ FLOPs are spent, count the parameters in one transformer block.

**Attention projections.** Standard multi-head attention has four $d \times d$ matrices: $W_Q$, $W_K$, $W_V$ project the input into queries, keys, and values, and $W_O$ projects the concatenated head outputs back to the residual stream. That is $4d^2$ parameters. With grouped-query attention (Section 4), where only $h_{kv} < h$ key-value heads are used, $W_K$ and $W_V$ shrink to $d \times h_{kv} d_h$ each, and the total becomes $2d^2 + 2 d \, h_{kv} d_h$.

**Feed-forward (MLP).** The original transformer MLP has two matrices, $d \times d_{\text{ff}}$ and $d_{\text{ff}} \times d$, with $d_{\text{ff}} = 4d$, for $8d^2$ parameters. Many modern models use a gated MLP such as SwiGLU,

$$
\text{MLP}(\mathbf{x}) = W_{\text{down}}\big(\text{SiLU}(W_{\text{gate}} \mathbf{x}) \odot W_{\text{up}} \mathbf{x}\big),
$$

which has three matrices. To keep the parameter count comparable, $d_{\text{ff}}$ is usually reduced to about $\tfrac{8}{3}d$, giving $3 \cdot \tfrac{8}{3} d \cdot d = 8d^2$ again.

So a standard block has about $4d^2 + 8d^2 = 12d^2$ parameters, and the whole model has

$$
N \approx 12 L d^2 + (\text{embedding parameters}).
$$

The split is roughly **one third attention projections, two thirds MLP**. Since FLOPs follow parameters, the same split holds for the parameter-dependent compute. The MLP, not attention, is where most of the arithmetic in a short-context forward pass goes.

### A worked example: Llama 2 7B and Llama 3 8B

Llama 2 7B has $L = 32$, $d = 4096$, 32 heads of dimension 128, a SwiGLU MLP with $d_{\text{ff}} = 11008$, and a vocabulary of 32,000 tokens with separate input and output embeddings. The formula gives

$$
12 \cdot 32 \cdot 4096^2 \approx 6.44\text{B} \quad \text{(blocks)}, \qquad 2 \cdot 32000 \cdot 4096 \approx 0.26\text{B} \quad \text{(embeddings)},
$$

for a total of about 6.7 billion, in line with the published size. (The MLP has $3 \cdot 4096 \cdot 11008 \approx 135$M parameters per layer, very close to $8d^2 \approx 134$M.)

Llama 3 8B has the same $L$ and $d$ but uses grouped-query attention with 8 key-value heads, a wider MLP ($d_{\text{ff}} = 14336$), and a much larger vocabulary (128,256 tokens). Counting exactly is easy in code:

```python
def param_count(L, d, n_heads, n_kv_heads, d_ff, vocab, tied_embeddings=False, gated_mlp=True):
    d_head = d // n_heads
    attn = 2 * d * d + 2 * d * (n_kv_heads * d_head)       # W_Q, W_O, W_K, W_V
    mlp = (3 if gated_mlp else 2) * d * d_ff
    norms = 2 * d                                            # two RMSNorm weight vectors
    blocks = L * (attn + mlp + norms)
    emb = vocab * d * (1 if tied_embeddings else 2)
    return blocks, emb, blocks + emb

blocks, emb, total = param_count(L=32, d=4096, n_heads=32, n_kv_heads=8, d_ff=14336, vocab=128256)
print(f"blocks {blocks/1e9:.2f}B, embeddings {emb/1e9:.2f}B, total {total/1e9:.2f}B")
# blocks 6.98B, embeddings 1.05B, total 8.03B
```

Per layer, Llama 3 8B has about 42M attention parameters and 176M MLP parameters, so its MLP share is even larger than two thirds. Note also that its embeddings account for more than 1 billion parameters. The input embedding costs no FLOPs, but the output projection to 128,256 logits costs $2 \cdot 128256 \cdot 4096 \approx 1.05$ GFLOPs per token, more than the $\approx 0.44$ GFLOPs of a single block's attention projections.

A reasonable estimate of the FLOPs per token, excluding attention scores, is therefore $2 \times (N - N_{\text{input embedding}}) \approx 2 \times 7.5\text{B} = 15$ GFLOPs.

## Attention cost grows with context length

The parameter count tells us nothing about the length of the context. Attention is the one place where tokens interact, and its cost depends on how many tokens there are.

### Cost for one new token

Suppose we are generating the token at position $n$, and the keys and values for all $n$ positions are available (Section 4 explains how they are cached). For each head, the new query $\mathbf{q} \in \mathbb{R}^{d_h}$ is compared with $n$ keys, and the resulting weights combine $n$ values:

$$
\mathbf{s} = \frac{K \mathbf{q}}{\sqrt{d_h}} \in \mathbb{R}^{n}, \qquad \mathbf{a} = \text{softmax}(\mathbf{s}), \qquad \mathbf{o} = V^\top \mathbf{a} \in \mathbb{R}^{d_h}.
$$

Computing $\mathbf{s}$ costs $2 n d_h$ FLOPs and computing $\mathbf{o}$ another $2 n d_h$. Summed over $h$ query heads and $L$ layers, and using $h d_h = d$:

$$
C_{\text{attn}}(n) \approx 4 L n d \quad \text{FLOPs per new token at context length } n.
$$

(Grouped-query attention shares keys and values across query heads but does not change this count: every query head still attends over all $n$ positions.)

Compare this with the parameter-dependent cost of about $2 \cdot 12 L d^2 = 24 L d^2$:

$$
\frac{C_{\text{attn}}}{C_{\text{params}}} \approx \frac{4 L n d}{24 L d^2} = \frac{n}{6d}.
$$

Attention arithmetic equals the rest of the forward pass when $n \approx 6d$. For a model with $d = 4096$, that is a context of about 24,000 tokens. At a context of 2,000 tokens, attention adds under 10 percent to the FLOPs; at 128,000 tokens, it costs several times more than all the weight multiplications combined. Larger models have larger $d$, so the crossover point moves out: attention FLOPs matter relatively less for big models at a fixed context length.

### Cost for a whole sequence

Processing an entire sequence of length $n$ from scratch, as in the prefill phase of Section 3, sums this cost over positions. With a causal mask, position $t$ attends to $t$ positions, so the total is

$$
\sum_{t=1}^{n} 4 L t d \approx 2 L d n^2.
$$

This is the famous quadratic cost of attention. Averaged over the sequence, it is $2 L n d$ per token, which matches the context-dependent term in the per-token estimate of Kaplan et al., $C_{\text{forward}} \approx 2N + 2 L n d$. Dense kernels that compute the full $n \times n$ score matrix and then apply the mask do up to twice this arithmetic; efficient kernels such as FlashAttention (Section 6) skip the fully masked blocks.

The quadratic term is why long-context inference is expensive. Doubling the prompt from 64,000 to 128,000 tokens roughly doubles the weight FLOPs of prefill but quadruples its attention FLOPs.

### Memory grows with context too

FLOPs are only half the story. To attend over $n$ earlier positions, the model needs their keys and values. During generation these are stored in the KV cache, whose size grows linearly with $n$ and which must be read from memory on every decoding step. Section 4 shows that at long context, reading the KV cache can take more time than the attention arithmetic itself, and can even exceed the cost of reading the weights. The naive attention implementation also materializes the $n \times n$ score matrix, which is quadratic in memory; FlashAttention avoids that.

### Architectural responses

Because the attention term grows without bound, many architectures modify it:

- **Sliding-window (local) attention** limits each token to the last $w$ positions, making per-token cost $O(w)$ instead of $O(n)$. Some models interleave local and global attention layers.
- **Sparse attention patterns** let each token attend to a structured subset of positions.
- **Linear attention and state-space models** replace softmax attention with a recurrence whose state has a fixed size, so per-token cost and memory are constant in $n$. Hybrid models combine a few full attention layers with many such layers.

Each of these trades some modeling flexibility for cost. Standard full attention remains the default in most large models, which is why the KV cache and attention kernels get so much engineering attention.

## Putting numbers together

The following function estimates the forward FLOPs to generate one token at a given context length, and the FLOPs of a full prefill:

```python
def flops_per_token(n_params_nonembed, L, d, vocab, context):
    weights = 2 * n_params_nonembed           # attention projections + MLP
    lm_head = 2 * vocab * d                   # output projection to logits
    attention = 4 * L * context * d           # q.K and a.V over the context
    return weights + lm_head + attention

def prefill_flops(n_params_nonembed, L, d, vocab, prompt_len):
    weights = 2 * n_params_nonembed * prompt_len
    lm_head = 2 * vocab * d                   # logits are needed only for the last position
    attention = 2 * L * d * prompt_len ** 2   # causal: sum over t of 4*L*t*d
    return weights + lm_head + attention

# Llama 3 8B: about 6.98e9 parameters in the blocks
for ctx in [1_000, 8_000, 32_000, 128_000]:
    f = flops_per_token(6.98e9, L=32, d=4096, vocab=128256, context=ctx)
    print(f"context {ctx:>7,}: {f/1e9:6.1f} GFLOPs per generated token")
```

For Llama 3 8B, this gives about 15 GFLOPs per token at short context and roughly 82 GFLOPs at 128,000 tokens, where attention dominates. The third suggested code lab asks you to extend this function with the KV cache size from Section 4 for a model configuration of your choice.

Knowing the FLOPs does not yet tell us the time. A modern accelerator can perform hundreds of trillions of operations per second, which would suggest thousands of tokens per second for an 8B model. Real single-user generation is much slower than that. The reason is the subject of the next two sections: during generation, the bottleneck is usually not arithmetic but moving data.

## Key takeaways

- A matrix multiply of $m \times k$ by $k \times n$ costs about $2mkn$ FLOPs, so each weight costs two FLOPs per token.
- A dense transformer's forward pass costs about $2N$ FLOPs per token; an MoE model's costs about $2N_{\text{active}}$, while its memory still scales with total parameters.
- A standard block has about $12d^2$ parameters: roughly one third in attention projections and two thirds in the MLP, so the MLP dominates short-context compute.
- Attention over the context adds about $4Lnd$ FLOPs per new token, overtaking the weight FLOPs near $n \approx 6d$; processing a whole sequence costs about $2Ldn^2$, quadratic in length.
- FLOP counts are the starting point, not the answer: at inference, memory traffic often dominates.

## Further reading

Hoffmann, Jordan, et al. "Training Compute-Optimal Large Language Models." arXiv preprint arXiv:2203.15556, 2022. https://arxiv.org/abs/2203.15556.

Jiang, Albert Q., et al. "Mixtral of Experts." arXiv preprint arXiv:2401.04088, 2024. https://arxiv.org/abs/2401.04088.

Kaplan, Jared, et al. "Scaling Laws for Neural Language Models." arXiv preprint arXiv:2001.08361, 2020. https://arxiv.org/abs/2001.08361.

Shazeer, Noam. "GLU Variants Improve Transformer." arXiv preprint arXiv:2002.05202, 2020. https://arxiv.org/abs/2002.05202.

Touvron, Hugo, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv preprint arXiv:2307.09288, 2023. https://arxiv.org/abs/2307.09288.

Vaswani, Ashish, et al. "Attention Is All You Need." In *Advances in Neural Information Processing Systems 30*, 2017. https://arxiv.org/abs/1706.03762.
