# 12.7 Scaling Across Devices

So far we have mostly imagined a model running on a single accelerator. Many models do not fit. A 70-billion-parameter model needs about 140 GB for its BF16 weights alone, before any KV cache, and frontier models, especially large mixture-of-experts models, are far bigger. Even when a model fits, splitting it across devices can make each token faster, because the devices read their shares of the weights in parallel and their memory bandwidths add up.

This section covers the three main ways to split a model for inference, tensor parallelism, pipeline parallelism, and expert parallelism, and then the cost they all share: communication between devices.

## Why split a model?

There are three reasons, and they lead to different choices.

1. **Capacity.** The weights plus the KV cache plus activations must fit in the combined memory of the devices. This is a hard constraint.
2. **Latency.** Decode time per token is roughly bytes read divided by bandwidth (Section 3). If $p$ devices each hold $1/p$ of the weights and read them simultaneously, the effective bandwidth is $p$ times larger and, ideally, the time per token falls by a factor of $p$.
3. **Throughput.** More memory for KV cache means larger batches (Section 5), and more devices mean more total compute.

The simplest form of scaling, **data parallelism** (running independent copies, or *replicas*, of the model on separate devices and spreading requests among them), addresses throughput only. It needs no communication between replicas, but each replica must fit the whole model and serve each request at single-replica speed. The techniques below split a *single* copy of the model.

## Tensor parallelism

**Tensor parallelism** (TP) splits the individual weight matrices of each layer across devices, so every device participates in every layer for every token. The standard scheme for transformers was introduced in Megatron-LM by Shoeybi et al.

### Splitting the MLP

Consider a two-layer MLP, $Y = \phi(X A)$ followed by $Z = Y B$, where $X \in \mathbb{R}^{b \times d}$, $A \in \mathbb{R}^{d \times d_{\text{ff}}}$, $B \in \mathbb{R}^{d_{\text{ff}} \times d}$, and $\phi$ is an elementwise activation. Split $A$ by **columns** across $p$ devices, $A = [A_1, A_2, \dots, A_p]$. Every device holds the full input $X$ and computes

$$
Y_i = \phi(X A_i),
$$

a slice of the hidden activations. Because $\phi$ is elementwise, no communication is needed yet. Now split $B$ by **rows**, $B = [B_1; B_2; \dots; B_p]$, to match. Device $i$ computes a partial output $Z_i = Y_i B_i$, and the full output is the sum:

$$
Z = Y B = \sum_{i=1}^{p} Y_i B_i.
$$

Summing the partial results across devices, so that every device ends up with the full $Z$, is an **all-reduce** operation. The pattern (column-parallel first matrix, row-parallel second matrix) needs exactly one all-reduce per MLP. Gated MLPs such as SwiGLU work the same way, with the gate and up projections both split by columns.

### Splitting attention

Attention splits naturally by **heads**. Each device takes $h/p$ of the query heads, with the corresponding columns of $W_Q$, $W_K$, $W_V$, computes attention for its heads entirely locally, and applies its row slice of the output projection $W_O$. Another all-reduce sums the partial outputs. So a transformer block under tensor parallelism needs **two all-reduces per layer** in the forward pass: one after attention and one after the MLP.

A pleasant side effect is that the **KV cache is split too**: each device stores keys and values only for its own heads. With grouped-query attention, if there are fewer key-value heads than devices ($h_{kv} \lt p$), some key-value heads must be replicated on several devices, and the cache savings are smaller.

### Properties

- **Memory and bandwidth scale.** Each device holds and reads about $1/p$ of the weights and KV cache, so per-token decode time can drop nearly by a factor of $p$, if communication is cheap.
- **Communication is frequent.** Two all-reduces per layer, on every forward pass, and the next layer cannot start until they finish. For a model with 80 layers, that is 160 synchronization points per generated token.
- **Therefore TP is kept within a node.** Tensor parallelism is almost always used across devices connected by a fast, low-latency interconnect, such as the 4 or 8 GPUs in one server linked by NVLink, and rarely across the slower network between servers.

## Pipeline parallelism

**Pipeline parallelism** (PP) splits the model by **layers**. With $p$ stages, device 1 holds layers $1$ to $L/p$, device 2 the next $L/p$, and so on. A token's activations flow through the stages in order; each boundary requires sending one activation tensor of size $b \times d$ from one device to the next, a cheap point-to-point transfer rather than a collective operation.

Pipelining has a well-known problem: while device 2 works on a batch, device 1 is idle unless it has other work. In training, GPipe (Huang et al.) addressed this by splitting a batch into *micro-batches* that follow each other through the pipeline, and the remaining idle time at the start and end is called the *pipeline bubble*. For inference, the natural micro-batches are different groups of requests: while stage 2 processes group A's current token, stage 1 can process group B's. With enough concurrent groups, all stages stay busy.

The tradeoffs are the mirror image of tensor parallelism:

- **Communication is light and tolerant of slow links.** Only activations cross stage boundaries, once per stage per forward pass. Pipeline parallelism works across servers connected by ordinary data-center networking.
- **It does not reduce latency for a single request.** A token must still pass through all $L$ layers in sequence, now with added transfer time between stages. Each stage reads only $1/p$ of the weights, but the stages run one after another for any single token. Pipeline parallelism adds *capacity* and *throughput*, not per-token speed.
- **Load balance matters.** The slowest stage sets the pace. Layers must be divided evenly, and the first and last stages also hold the embedding and output layers.

In practice, large models are often served with tensor parallelism inside each server and pipeline parallelism across servers.

## Expert parallelism

Mixture-of-experts models (Section 6) add a third option. Each MoE layer has many expert MLPs, and each token uses only a few of them. **Expert parallelism** (EP) places different experts on different devices, an approach used at scale in GShard (Lepikhin et al.) and Switch Transformers (Fedus et al.). A forward pass through an MoE layer then works as follows:

1. Each device computes the router scores for its tokens and decides which experts each token goes to.
2. **Dispatch:** tokens are sent to the devices holding their chosen experts. Because every device may send tokens to every other device, this is an **all-to-all** exchange.
3. Each device runs its experts on the tokens it received.
4. **Combine:** results are sent back to the tokens' original devices (a second all-to-all) and combined using the router weights.

The attention layers and other dense parts are typically handled with data or tensor parallelism alongside.

Expert parallelism lets a model with hundreds of billions of total parameters spread its expert weights across many devices, with each device holding and reading only its own experts. Its challenges are specific:

- **All-to-all communication** in every MoE layer, in both directions, with volume proportional to the number of tokens times $k$ (the number of experts per token) times $d$.
- **Load imbalance.** If the router sends many tokens to the same expert, the device holding it becomes a straggler while others wait. Training uses auxiliary load-balancing objectives to encourage even routing, and serving systems may replicate popular experts.
- **Small-batch inefficiency.** At small batch sizes, each expert receives only a few tokens, so its weights are read for little work. MoE serving is most efficient with large batches spread across many devices.

## Other forms of parallelism

For very long prompts, even the activations and attention of a single sequence can be too much for one device. **Sequence** or **context parallelism** splits the sequence dimension across devices. In Ring Attention (Liu et al.), each device holds a block of queries and passes blocks of keys and values around a ring of devices, computing blockwise attention with the same online-softmax rescaling used in FlashAttention (Section 6), while overlapping the transfers with computation. This is used mainly to speed up the prefill of very long contexts.

## Communication costs between GPUs

Every form of model parallelism replaces some memory traffic with network traffic. Whether the trade pays off depends on the interconnect.

### A simple cost model

A standard way to estimate the time to send a message of $M$ bytes over a link is

$$
T(M) = \alpha + \frac{M}{\beta},
$$

where $\alpha$ is the fixed **latency** per message (software overhead, synchronization, link latency) and $\beta$ is the link **bandwidth**. Small messages are dominated by $\alpha$, large ones by $M / \beta$.

Collective operations are built from many such messages. The classic **ring all-reduce** of $M$ bytes over $p$ devices proceeds in $2(p-1)$ steps (a reduce-scatter phase and an all-gather phase), and each device sends and receives $\frac{2(p-1)}{p} M$ bytes in total:

$$
T_{\text{all-reduce}} \approx 2(p-1)\,\alpha + \frac{2(p-1)}{p} \cdot \frac{M}{\beta}.
$$

The bandwidth term barely grows with $p$ (it approaches $2M/\beta$), but the latency term grows linearly. Tree-based and hardware-accelerated algorithms reduce the latency term for small messages.

### Interconnect hierarchy

Links between devices vary by orders of magnitude:

- **Within a server**, GPUs are typically connected by a dedicated high-bandwidth fabric (NVLink and NVSwitch on NVIDIA systems, similar fabrics on other accelerators), with bandwidths in the hundreds of gigabytes per second per GPU and low latency.
- **Between a GPU and its host**, PCIe offers considerably less bandwidth.
- **Between servers**, networks such as InfiniBand or high-speed Ethernet (often with RDMA) connect the machines, again with lower bandwidth per GPU and higher latency than in-server links.

This hierarchy is why the parallelism strategy follows the topology: the most communication-hungry scheme (tensor parallelism) runs on the fastest links, and the most tolerant scheme (pipeline parallelism) crosses the slowest.

### A worked estimate

How much does tensor-parallel communication cost relative to the useful work? Consider a 70B model with $L = 80$ and $d = 8192$ (the shape of Llama 2 70B) split across $p = 4$ of our hypothetical accelerators (1,000 TFLOP/s, 3 TB/s), with illustrative link parameters of $\beta = 400$ GB/s and $\alpha = 2\ \mu\text{s}$ per ring step. Each all-reduce moves the activations for all tokens in the step, $M = b \cdot d \cdot 2$ bytes in 16-bit:

```python
def tp_step_estimate(tokens, n_params=70e9, L=80, d=8192, p=4, bytes_per=2,
                     peak_flops=1e15, hbm_bw=3e12, link_bw=400e9, link_latency=2e-6):
    """Very rough per-step time for tensor parallelism over p devices (one forward pass)."""
    compute = 2 * n_params * tokens / (p * peak_flops)
    weights = n_params * bytes_per / (p * hbm_bw)              # each device reads its shard
    msg = tokens * d * bytes_per                               # activations per all-reduce
    per_allreduce = 2 * (p - 1) * link_latency + 2 * (p - 1) / p * msg / link_bw
    comm = 2 * L * per_allreduce                               # two all-reduces per layer
    return max(compute, weights), comm

for tokens, label in [(1, "decode, batch 1"), (64, "decode, batch 64"), (2000, "prefill 2,000")]:
    local, comm = tp_step_estimate(tokens)
    print(f"{label:>17}: compute/memory {local*1e3:6.2f} ms, all-reduce {comm*1e3:6.2f} ms")
```

The output:

```
  decode, batch 1: compute/memory  11.67 ms, all-reduce   1.93 ms
 decode, batch 64: compute/memory  11.67 ms, all-reduce   2.55 ms
    prefill 2,000: compute/memory  70.00 ms, all-reduce  21.58 ms
```

Several lessons follow, even from such a rough model:

- **Decode communication is latency-bound.** At batch 1, each all-reduce moves only 16 KB, and the cost is dominated by the 160 per-token synchronizations, not by bytes. It stays nearly flat as the batch grows to 64. This is why low-latency links and optimized small-message all-reduce implementations matter so much for tensor-parallel decode.
- **Prefill communication is bandwidth-bound**, and it can be a significant fraction of compute time. Implementations overlap communication with computation where possible, for example by starting to communicate one chunk while computing the next.
- **Scaling TP further has diminishing returns.** Doubling $p$ halves the per-device memory time, but the latency term of each all-reduce grows with $p$. Beyond some degree of tensor parallelism (commonly the size of one server), adding devices speeds up each token very little.

The link parameters here are assumptions for illustration; real values depend on the hardware generation, topology, and communication library, and should be measured.

### Choosing a configuration

Putting the pieces together, a serving deployment typically chooses:

- **The minimum number of devices that fits the model and enough KV cache** for the target batch size and context length.
- **Tensor parallelism within a server** to reduce per-token latency, up to the point where communication overhead eats the gains.
- **Pipeline parallelism across servers** when a model does not fit in one server.
- **Expert parallelism** for MoE layers, often across many devices, with the dense layers handled separately.
- **Data parallelism (replicas)** on top, to scale throughput with demand.

Because splitting always adds communication, the most cost-efficient configuration for throughput is often the *smallest* degree of model parallelism that fits, with many replicas, while the configuration for lowest latency uses more tensor parallelism per replica. The serving systems of Section 8 expose these choices as configuration options.

## Key takeaways

- Models are split across devices for capacity, for lower latency (aggregating memory bandwidth), and for throughput; data-parallel replicas scale throughput only.
- Tensor parallelism splits weight matrices (column-parallel then row-parallel), needs two all-reduces per layer, splits the KV cache by heads, and is kept within a server on fast links.
- Pipeline parallelism splits layers into stages with cheap point-to-point transfers; it adds capacity and throughput but does not speed up a single token.
- Expert parallelism places MoE experts on different devices and needs two all-to-all exchanges per MoE layer; load balance and batch size determine its efficiency.
- Communication costs follow $T = \alpha + M/\beta$: tensor-parallel decode is latency-bound, prefill is bandwidth-bound, and the parallelism strategy should match the interconnect hierarchy.

## Further reading

Fedus, William, et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." *Journal of Machine Learning Research* 23, no. 120 (2022): 1–39. https://arxiv.org/abs/2101.03961.

Huang, Yanping, et al. "GPipe: Efficient Training of Giant Neural Networks Using Pipeline Parallelism." In *Advances in Neural Information Processing Systems 32*, 2019. https://arxiv.org/abs/1811.06965.

Lepikhin, Dmitry, et al. "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." In *International Conference on Learning Representations*, 2021. https://arxiv.org/abs/2006.16668.

Liu, Hao, et al. "Ring Attention with Blockwise Transformers for Near-Infinite Context." In *International Conference on Learning Representations*, 2024. https://arxiv.org/abs/2310.01889.

Pope, Reiner, et al. "Efficiently Scaling Transformer Inference." In *Proceedings of Machine Learning and Systems 5*, 2023. https://arxiv.org/abs/2211.05102.

Shoeybi, Mohammad, et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv preprint arXiv:1909.08053, 2019. https://arxiv.org/abs/1909.08053.
