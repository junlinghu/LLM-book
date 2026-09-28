# 8.6 Parameter-Efficient Fine-Tuning

Section 5 ended with a memory bill: full fine-tuning with Adam needs about 16 bytes per parameter for model states, most of it for the optimizer's view of every weight that is trained. It also produces a complete new copy of the model for every fine-tuned variant, which is expensive to store and to serve when a team wants one assistant for customer support, another for code review, and a third for a particular language. **Parameter-efficient fine-tuning** (PEFT) attacks both problems with one idea: freeze the pretrained weights and train a small number of new parameters. This section surveys the early methods, derives LoRA, counts its parameters for a real model, implements it from scratch, and repeats the experiment of Section 5 with it. It ends with QLoRA and with the evidence on what training so few parameters costs.

## The idea: freeze and add

Fine-tuning a model with parameters $`\theta_0`$ means learning a change $`\Delta\theta`$. Full fine-tuning lets $`\Delta\theta`$ be anything. Parameter-efficient methods restrict it to a family described by far fewer numbers $`\phi`$, and train only those:

```math
\theta = \theta_0 + \Delta\theta(\phi), \qquad |\phi| \ll |\theta_0|
```

The frozen weights still take part in every forward and backward pass, since the gradient with respect to $`\phi`$ flows through the whole network, and the activations must still be stored. But the frozen weights receive no updates, so they need no gradients and no optimizer state. Of the 16 bytes per parameter in Section 5, only the 2 bytes of the BF16 weight remain for $`\theta_0`$. The methods differ in where they put $`\phi`$: in new layers, in new inputs, or in a low-rank change to existing weight matrices.

## Adapters, prefixes, and soft prompts

The first widely used approach inserted small new layers into a frozen Transformer. An **adapter** is a bottleneck: a down-projection from the model width $`h`$ to a small width $`m`$, a nonlinearity $`f`$, and an up-projection back to $`h`$, wrapped in a residual connection:

```math
\text{adapter}(\mathbf{z}) = \mathbf{z} + W_{\text{up}}\, f(W_{\text{down}} \mathbf{z}), \qquad W_{\text{down}} \in \mathbb{R}^{m \times h}, \quad W_{\text{up}} \in \mathbb{R}^{h \times m}
```

Houlsby et al. (2019) placed adapters inside every block of BERT, initialized close to the identity so that the network started out computing what the pretrained model computed. Each adapter adds about $`2hm`$ parameters. On the GLUE benchmark, adapters came within 0.4% of the performance of full fine-tuning while adding only 3.6% of the parameters per task. Their drawback is that they are extra layers applied one after another with the rest of the block, so they add latency at inference.

A second family leaves the network alone and learns new *inputs*. A handwritten prompt is restricted to embeddings of real tokens, but nothing stops us from prepending vectors that are not the embedding of any token and training them by gradient descent. **Prompt tuning** (Lester et al. 2021) prepends such a "soft prompt" to the input embeddings. **Prefix tuning** (Li and Liang 2021) prepends trained vectors to the keys and values of the attention in every layer (Section 6.5), so that every token can attend to them as if they were "virtual tokens". Li and Liang found that, learning only 0.1% of the parameters, prefix tuning matched fine-tuning on table-to-text generation and summarization with the full data and beat it in low-data settings. Lester et al. found that prompt tuning becomes more competitive with scale, matching the tuning of all weights once T5 models exceed billions of parameters. The costs of both are that the prefix occupies part of the context and that, for smaller models, a few input vectors are a weak lever.

## LoRA

**Low-Rank Adaptation**, or LoRA (Hu et al. 2022), changes the weights, but in a restricted way. Take a weight matrix $`W_0 \in \mathbb{R}^{d \times k}`$ of the pretrained model, for example the query projection of an attention layer. Instead of an arbitrary update with $`dk`$ entries, LoRA learns an update of rank at most $`r`$, written as the product of two thin matrices:

```math
W = W_0 + \frac{\alpha}{r} BA, \qquad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}
```

with $`r`$ much smaller than $`d`$ and $`k`$. $`W_0`$ is frozen; only $`A`$ and $`B`$ are trained. For an input $`\mathbf{x}`$, the layer computes

```math
\mathbf{h} = W_0 \mathbf{x} + \frac{\alpha}{r} B (A \mathbf{x})
```

$`A`$ projects the input down to $`r`$ dimensions and $`B`$ projects it back up. This is a bottleneck like an adapter's, but it runs *in parallel* with the frozen weight, and it is linear, which is what makes merging possible. The bet behind it is that the change fine-tuning needs is much simpler than the weights themselves; Hu et al. studied this rank deficiency and found small ranks were often enough in their experiments.

### Initialization and the scale

LoRA initializes $`A`$ randomly and $`B`$ to zero, so $`BA = 0`$ and training starts exactly at the pretrained model. The asymmetry matters. With $`\mathbf{g}`$ the gradient of the loss with respect to $`\mathbf{h}`$, the gradients of the two factors are

```math
\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r}\, \mathbf{g}\, (A\mathbf{x})^{\top}, \qquad \frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r}\, B^{\top} \mathbf{g}\, \mathbf{x}^{\top}
```

With $`B = 0`$, the gradient of $`A`$ vanishes but that of $`B`$ does not, so the first update moves $`B`$, and from then on both factors train. If both started at zero, neither would ever receive a gradient; if both were random, training would start away from the pretrained function.

The factor $`\alpha / r`$ scales the update. Hu et al. fixed $`\alpha`$ rather than tuning it, so that changing the rank does not require retuning everything else. In practice $`\alpha`$ acts as a second learning-rate knob for the adapter; common choices are $`\alpha = r`$ or $`\alpha = 2r`$, and the experiment below uses $`\alpha = 2r`$.

### Counting parameters

A LoRA update of a $`d \times k`$ matrix has $`r(d + k)`$ parameters instead of $`dk`$. For a $`4096 \times 4096`$ projection, full fine-tuning trains 16,777,216 weights, while LoRA with $`r = 8`$ trains

```math
8 \cdot (4096 + 4096) = 65{,}536
```

about 0.4% as many. For GPT-3 175B, Hu et al. reported 10,000 times fewer trainable parameters and 3 times less GPU memory than full fine-tuning with Adam, with model quality on par with or better than full fine-tuning on the models and tasks they tested.

Real attention layers are often not square. SmolLM2-135M has width 576, 30 layers, and 9 query heads of dimension 64, but its keys and values have only 3 heads, shared among the query heads to save memory. Its query and output projections are $`576 \times 576`$, while its key and value projections map 576 dimensions to $`3 \times 64 = 192`$. With rank 8 on all four:

| Projection | Shape ($`d \times k`$) | LoRA parameters $`r(d + k)`$ |
|---|---|---|
| Query | $`576 \times 576`$ | 9,216 |
| Key | $`192 \times 576`$ | 6,144 |
| Value | $`192 \times 576`$ | 6,144 |
| Output | $`576 \times 576`$ | 9,216 |
| **Per layer** | | **30,720** |

Over 30 layers this is 921,600 parameters, exactly what the implementation counts (Appendix A.1): 0.68% of the 135,436,608 parameters of the wrapped model. The count is linear in $`r`$, so rank 2 gives 230,400 and rank 32 gives 3,686,400.

LoRA can wrap any weight matrix. Hu et al. concentrated on the attention projections (Section 6.5) and found that, for a fixed budget, adapting several of them at a low rank worked better than adapting one at a higher rank. Many later recipes also wrap the FFN matrices, which hold most of a block's parameters. The embedding and output matrices usually stay frozen, with the exception noted in Section 3: rows for newly added role tokens must be trained.

## LoRA from scratch

LoRA is short enough to write by hand (Appendix A.1). A wrapper module holds the original `nn.Linear` layer with its parameters frozen, a trainable $`A`$ drawn from a Gaussian with standard deviation $`1/\sqrt{k}`$, and a trainable $`B`$ of zeros; its forward pass adds the scaled low-rank term to the frozen layer's output. A helper replaces the query, key, value, and output projections in all 30 layers of SmolLM2-135M. Before training, the largest difference between the logits of the wrapped model and the base model is exactly 0.0, as it must be when $`B = 0`$.

Appendix A.2 then repeats the full fine-tuning experiment of Section 5 with LoRA at ranks 2, 8, and 32: the same 600 training examples, the same masked loss, batches of 8, 120 steps with 10 steps of warmup and linear decay, and the same evaluation. Because the adapters start from zero, LoRA is commonly run with a much higher learning rate than full fine-tuning; these runs use $`10^{-3}`$, 20 times the rate of Section 5. The three runs together took 3 minutes on 4 CPU threads of an idle machine and 6 minutes in the busier appendix run, against 9 and 12 minutes for the two full fine-tuning runs of Section 5.

| Model | Trainable parameters | Seen phrasings | Held-out phrasings | Shakespeare perplexity |
|---|---|---|---|---|
| Base model, no fine-tuning | 0 | 0.00 | 0.00 | 41.7 |
| Full fine-tuning (Section 5) | 134.5 million | 0.63 | 0.60 | 47.0 |
| LoRA, $`r = 2`$ | 230,400 | 0.00 | 0.00 | 56.4 |
| LoRA, $`r = 8`$ | 921,600 | 0.57 | 0.43 | 74.1 |
| LoRA, $`r = 32`$ | 3,686,400 | 0.73 | 0.50 | 322.8 |

The results are more mixed than the usual summary of LoRA suggests.

**Rank matters.** At rank 2, 120 steps did not get a single test example right, although perplexity rose to 56.4, so the adapter did change the model. At rank 8, with 0.68% of the parameters, LoRA reached 0.57 on the seen phrasings, close to full fine-tuning's 0.63. At rank 32 it reached 0.73, the best score on seen phrasings of any run in this chapter.

**It generalized less well to new phrasings.** Both LoRA runs that learned the tasks did worse on held-out phrasings than on seen ones, by much more than full fine-tuning did: 0.57 against 0.43 at rank 8, and 0.73 against 0.50 at rank 32 (seven of the 30 examples), compared with 0.63 against 0.60. The adapters fit the training templates but carried less of the task over to unfamiliar wording.

**Here, LoRA forgot more, not less.** Perplexity on held-out Shakespeare rose to 74.1 at rank 8 and 322.8 at rank 32, against 47.0 for full fine-tuning. The likely cause is the setup rather than LoRA itself. The learning rate of $`10^{-3}`$ was not tuned, and a low-rank update is not necessarily a small one: $`BA`$ can have a large norm even when its rank is small. Fewer trainable parameters restrict the *directions* in which the weights can move, not how far they move. A lower learning rate, fewer steps, or a smaller $`\alpha`$ would very likely have reduced the damage, at some cost in accuracy. Each configuration was run once, on a toy task, so these numbers illustrate trade-offs rather than measure them, but the lesson holds generally: LoRA's rank, scale, and learning rate matter, and its reputation for preserving the base model is not a guarantee.

## Merging the adapter

Because the update is linear, it can be folded into the frozen weight after training, $`W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA`$. The merged matrix has the shape of $`W_0`$, so the merged model has the base model's architecture and speed; unlike adapters and prefixes, LoRA costs nothing extra at inference. Appendix A.3 merges the rank-8 adapter. The merged model contains no LoRA modules, and its logits differ from the unmerged model's by at most $`1.7 \times 10^{-4}`$, the size of floating-point rounding when the same numbers are summed in a different order.

Keeping adapters separate has its own uses. The rank-8 adapter is 921,600 numbers, under 4 MB in FP32, against about 540 MB for the whole model. A server can keep one copy of the base model and switch among many adapters, by merging and unmerging (subtracting $`\frac{\alpha}{r}BA`$ again) or by computing each request's low-rank term separately.

## QLoRA: a frozen model in 4 bits

Once only adapters are trained, the frozen weights dominate the memory, and since they never change they need not be stored in high precision. **QLoRA** (Dettmers et al. 2023) stores the frozen base model in 4 bits and trains LoRA adapters on top in 16-bit precision, backpropagating through the quantized weights, which are dequantized on the fly for each matrix multiplication. It rests on three ideas:

- **4-bit NormalFloat (NF4)**, a data type designed for normally distributed values, as trained weights approximately are;
- **double quantization**, which also quantizes the scaling constants of the 4-bit format, reducing their overhead;
- **paged optimizers**, which move optimizer state between GPU and CPU memory to survive memory spikes.

At about half a byte per parameter plus the scaling constants, the weights of a 7-billion-parameter model need roughly 3.5 GB instead of 14 GB. Dettmers et al. fine-tuned a 65-billion-parameter model on a single 48 GB GPU while preserving the task performance of 16-bit fine-tuning. Their Guanaco models reached 99.3% of ChatGPT's performance level on the Vicuna benchmark after 24 hours of fine-tuning on one GPU, though the same paper warned that chatbot benchmarks of the time were not trustworthy, a problem Section 7 returns to. QLoRA is a large part of why LoRA-style fine-tuning became the default for individuals and small teams.

## LoRA learns less and forgets less

Is a low-rank update enough? Biderman et al. (2024) compared LoRA and full fine-tuning on programming and mathematics, with instruction fine-tuning on about 100,000 prompt-response pairs and with continued pretraining on 20 billion tokens. In standard low-rank settings, LoRA substantially underperformed full fine-tuning. But it better maintained the base model's performance outside the target domain, mitigated forgetting more than weight decay or dropout, and kept generations more diverse. Full fine-tuning, they found, learned weight perturbations with a rank 10 to 100 times greater than typical LoRA configurations, which may explain part of the gap.

This fits the superficial alignment hypothesis of Section 1. If SFT mainly teaches format and style, a low-rank change is often enough, and LoRA is an excellent default. If the data is large and genuinely new, such as a new domain of code or mathematics, full fine-tuning or a much higher rank may be needed. The experiment above adds a caution: with an untuned learning rate, even a low-rank adapter can move the model far, so "forgets less" should be checked with a regression measure such as held-out perplexity, not assumed.

## Key takeaways

- Parameter-efficient fine-tuning freezes the pretrained weights and trains a few new parameters, removing most optimizer memory, making each variant a small file, and letting one base model serve many variants.
- Adapters insert bottleneck layers into each block (within 0.4% of full fine-tuning on GLUE with 3.6% of the parameters); prefix and prompt tuning train vectors prepended to the keys and values or to the input embeddings.
- LoRA learns a low-rank update $`\frac{\alpha}{r}BA`$ of a frozen matrix, with $`A`$ random and $`B = 0`$ so training starts at the pretrained model; it trains $`r(d + k)`$ parameters per matrix, 921,600 for rank 8 on SmolLM2-135M's attention projections.
- The update can be merged into the weights, leaving the original architecture and no extra inference cost; the merged SmolLM2 model matched the unmerged one to within $`1.7 \times 10^{-4}`$ in its logits.
- QLoRA stores the frozen model in 4-bit NF4 and trains LoRA adapters on top, which made it possible to fine-tune a 65-billion-parameter model on a single 48 GB GPU.
- LoRA usually learns less than full fine-tuning from large or unfamiliar data and forgets less, but its hyperparameters matter: on the toy tasks, rank-32 LoRA at an untuned learning rate beat full fine-tuning on seen phrasings but did worse on new phrasings and forgot far more.

## Further reading

Biderman, Dan, et al. "LoRA Learns Less and Forgets Less." *Transactions on Machine Learning Research*, 2024. https://arxiv.org/abs/2405.09673.

Dettmers, Tim, et al. "QLoRA: Efficient Finetuning of Quantized LLMs." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2305.14314.

Houlsby, Neil, et al. "Parameter-Efficient Transfer Learning for NLP." In *Proceedings of the 36th International Conference on Machine Learning*, 2019. https://arxiv.org/abs/1902.00751.

Hu, Edward J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." In *International Conference on Learning Representations*, 2022. https://arxiv.org/abs/2106.09685.

Lester, Brian, Rami Al-Rfou, and Noah Constant. "The Power of Scale for Parameter-Efficient Prompt Tuning." In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, 2021. https://arxiv.org/abs/2104.08691.

Li, Xiang Lisa, and Percy Liang. "Prefix-Tuning: Optimizing Continuous Prompts for Generation." In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics*, 2021. https://arxiv.org/abs/2101.00190.

## Appendix: Code for Section 8.6

These listings continue the Python session of the appendix to Section 8.5: run A.1 to A.4 there first, which define the synthetic data, the `encode` and `collate` functions, the base `model`, and the `report` function used here. The three LoRA runs in A.2 take a few minutes together on 4 CPU threads.

### A.1 A LoRA layer from scratch

Wrap a frozen linear layer with a trainable low-rank update whose B matrix starts at zero.

```python
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    def merged(self):
        """A plain nn.Linear with the update folded into the weight."""
        out = nn.Linear(self.base.in_features, self.base.out_features,
                        bias=self.base.bias is not None)
        with torch.no_grad():
            out.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
            if self.base.bias is not None:
                out.bias.copy_(self.base.bias)
        return out

def add_lora(model, targets=("q_proj", "k_proj", "v_proj", "o_proj"), r=8, alpha=16):
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in targets:
            setattr(attn, name, LoRALinear(getattr(attn, name), r, alpha))
    return model

def merge_lora(model):
    for layer in model.model.layers:
        attn = layer.self_attn
        for name, child in list(attn.named_children()):
            if isinstance(child, LoRALinear):
                setattr(attn, name, child.merged())
    return model
```

```python
torch.manual_seed(0)
lora_model = add_lora(AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32))
x = torch.tensor([encode(*test_seen[0])[0]])
with torch.no_grad():
    diff = (lora_model(x).logits - model(x).logits).abs().max().item()
trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in lora_model.parameters())
print(f"max logit difference before training: {diff}")
print(f"trainable {trainable:,} of {total:,} parameters ({100 * trainable / total:.2f}%)")
cfg = lora_model.config
print("width", cfg.hidden_size, "| layers", cfg.num_hidden_layers, "| heads", cfg.num_attention_heads,
      "| key-value heads", cfg.num_key_value_heads)
```

Output:

```
max logit difference before training: 0.0
trainable 921,600 of 135,436,608 parameters (0.68%)
width 576 | layers 30 | heads 9 | key-value heads 3
```

### A.2 Training with LoRA

Train only the adapters on the synthetic tasks, with the same data and steps as full fine-tuning, for several ranks.

```python
def finetune_lora(r, steps=120, batch_size=8, lr=1e-3, warmup=10, seed=0):
    torch.manual_seed(seed)
    m = add_lora(AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", dtype=torch.float32), r=r, alpha=2 * r)
    data = [encode(q, a) for q, a in train]
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, (steps - s) / (steps - warmup)))
    order = random.Random(seed)
    m.train()
    for step in range(steps):
        ids, labels, attn = collate(order.sample(data, batch_size))
        loss = m(input_ids=ids, attention_mask=attn, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad()
    return m, sum(p.numel() for p in params)

start = time.time()
adapters = {}
for r in [2, 8, 32]:
    m, n = finetune_lora(r)
    adapters[r] = m
    report(f"LoRA r={r} ({n:,} trainable)", m)
print(f"(three runs: {(time.time() - start) / 60:.0f} minutes on 4 CPU threads)")
```

Output:

```
LoRA r=2 (230,400 trainable)       seen 0.00  held-out phrasing 0.00  Shakespeare ppl 56.4
LoRA r=8 (921,600 trainable)       seen 0.57  held-out phrasing 0.43  Shakespeare ppl 74.1
LoRA r=32 (3,686,400 trainable)    seen 0.73  held-out phrasing 0.50  Shakespeare ppl 322.8
(three runs: 6 minutes on 4 CPU threads)
```

### A.3 Merging the adapter

Fold the trained update into the weights and check that the merged model computes the same function.

```python
m = adapters[8]
x = torch.tensor([encode(*test_new[0])[0]])
with torch.no_grad():
    before = m(x).logits
    merged = merge_lora(m)
    after = merged(x).logits
print("max logit difference after merging:", f"{(before - after).abs().max().item():.1e}")
print("LoRA modules left:", sum(isinstance(c, LoRALinear) for c in merged.modules()))
```

Output:

```
max logit difference after merging: 1.7e-04
LoRA modules left: 0
```
