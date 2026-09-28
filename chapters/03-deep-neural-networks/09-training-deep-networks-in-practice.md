# 3.9 Training Deep Networks in Practice

The previous sections introduced the ingredients one at a time. This section assembles them into a working recipe and adds the practical knowledge that separates a training run that works from one that silently doesn't: mixed-precision arithmetic, a debugging checklist, reproducibility habits, and what changes when the recipe is scaled up. The goal is a default you can reach for when training any deep network, together with a systematic way to find out what is wrong when it misbehaves.

## A baseline recipe

For a new deep network, the following choices are a strong starting point. Each line refers back to the section that explains it.

| Ingredient | Default choice | Section |
|---|---|---|
| Initialization | He for ReLU-family, Xavier for tanh; small init for residual branch outputs | 3, 4 |
| Architecture | Residual blocks | 4 |
| Normalization | Pre-norm LayerNorm or RMSNorm (BatchNorm for conv nets with large batches) | 5 |
| Optimizer | AdamW, $`\beta_1 = 0.9`$, $`\beta_2 \in [0.95, 0.999]`$, weight decay about 0.01–0.1 on weight matrices | 6, 8 |
| Learning rate | Chosen by a range test or a short sweep | 7 |
| Schedule | Linear warmup, then cosine or linear decay | 7 |
| Gradient clipping | Global norm, threshold around 1.0 | 2 |
| Regularization | Dropout and augmentation as needed, judged from the train/validation gap | 8 |
| Precision | BF16 mixed precision where supported | This section |

Here is the recipe as a compact PyTorch training loop for a residual MLP classifier. It is deliberately minimal but contains every ingredient:

```python
import math, torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, d, p_drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):                       # pre-norm residual block
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.norm(x)))))

class ResMLP(nn.Module):
    def __init__(self, d_in, d, n_blocks, n_classes):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        for b in self.blocks:                   # small init for residual outputs
            nn.init.normal_(b.fc2.weight, std=0.02 / math.sqrt(2 * n_blocks))
            nn.init.zeros_(b.fc2.bias)
    def forward(self, x):
        x = self.inp(x.flatten(1))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))

def train(model, train_loader, total_steps, peak_lr=1e-3, warmup=500, device="cuda"):
    model.to(device).train()
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                             {"params": no_decay, "weight_decay": 0.0}], lr=peak_lr)
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        s = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    step = 0
    while step < total_steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print(f"step {step} loss {loss.item():.4f} gnorm {gnorm:.2f} "
                      f"lr {sched.get_last_lr()[0]:.2e}")
            if step >= total_steps:
                break
```

The chapter's final suggested code lab trains a model like this on Fashion-MNIST, then runs an **ablation**: remove one ingredient at a time (no residuals, no normalization, plain SGD instead of AdamW, no warmup, no clipping) and compare the loss curves. Ablations are the most reliable way to learn which ingredients matter for a given problem; the answer varies with depth, data, and scale.

## Mixed-precision training

Neural network training is dominated by matrix multiplications, and modern accelerators perform them much faster, and store them in half the memory, in 16-bit floating point than in 32-bit. **Mixed-precision training** uses 16-bit arithmetic for most of the computation while keeping the parts that need precision in 32-bit.

### The formats

| Format | Bits (sign / exponent / mantissa) | Range | Precision |
|---|---|---|---|
| FP32 | 1 / 8 / 23 | About $`10^{-38}`$ to $`10^{38}`$ | About 7 decimal digits |
| FP16 | 1 / 5 / 10 | About $`6 \times 10^{-5}`$ (normal) to $`65{,}504`$ | About 3 decimal digits |
| BF16 | 1 / 8 / 7 | Same as FP32 | About 2 decimal digits |

FP16 has more precision than BF16 but a much narrower range: numbers above 65,504 overflow to infinity, and small numbers (such as many gradients) underflow to zero. BF16 ("brain floating point") keeps FP32's 8-bit exponent and therefore its range, at the cost of precision.

### The recipe

Micikevicius et al. (2018) described the standard approach:

1. **FP32 master weights.** Keep an FP32 copy of the weights, which the optimizer updates. Each step's update $`\eta \Delta`$ is often many orders of magnitude smaller than the weight itself; in 16-bit, adding it to the weight would round away entirely, so updates must be applied in FP32.
2. **16-bit forward and backward passes.** Cast weights to 16-bit for the matrix multiplications of the forward and backward passes. Activations and gradients are mostly in 16-bit, which also halves activation memory.
3. **FP32 for sensitive operations.** Reductions such as sums in softmax, normalization statistics, and the loss are computed or accumulated in FP32.
4. **Loss scaling (for FP16).** Many gradient values are so small that they underflow to zero in FP16. Multiplying the loss by a large factor $`S`$ before the backward pass scales all gradients up by $`S`$ (by the chain rule), shifting them into FP16's representable range. The gradients are divided by $`S`$ before the optimizer step. **Dynamic loss scaling** starts with a large $`S`$, halves it and skips the step whenever gradients overflow to Inf or NaN, and periodically tries doubling it again.

With **BF16**, the range matches FP32's, so gradient underflow is rarely a problem and loss scaling is usually unnecessary. This simplicity is why BF16 is the common choice for training large models on hardware that supports it. In PyTorch, `torch.autocast` handles the casting (keeping sensitive operations in FP32 automatically), and for FP16 a `torch.amp.GradScaler` handles dynamic loss scaling:

```python
scaler = torch.amp.GradScaler("cuda")         # only needed for FP16
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = loss_fn(model(x), y)
scaler.scale(loss).backward()
scaler.unscale_(opt)                           # so clipping sees true gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(opt)                               # skips the step if grads overflowed
scaler.update()
```

Note the `unscale_` call before clipping: clipping must operate on the true gradients, not the scaled ones.

## A debugging checklist

Deep networks fail silently. A bug that would crash an ordinary program often just makes a network train somewhat worse. The following checklist, applied in order, catches most problems.

**1. Check the initial loss against its expected value.** Before any training, compute the loss on a batch. For a $`C`$-class classifier with cross-entropy, a freshly initialized network should predict roughly uniform probabilities, so the loss should be close to

```math
-\ln\frac{1}{C} = \ln C.
```

For 10 classes, that is about 2.30; for a language model with a vocabulary of 50,000 tokens, about 10.8. An initial loss far above this means the network starts out confidently wrong, often from an output layer initialized too large. An initial loss far below it suggests a bug such as label leakage.

**2. Overfit a single batch.** Take one small batch (a handful of examples) and train on it repeatedly with regularization off. A working model and training loop should drive the loss on that batch to nearly zero within a few hundred steps. If it cannot, the problem is in the model or the loop, not the data or the hyperparameters: look for a bug in the loss, detached tensors, a missing `optimizer.step()`, labels misaligned with inputs, or an absurd learning rate. This is the single most useful debugging step, because it separates "can the model learn at all" from "does it generalize."

**3. Monitor gradient norms.** Log the global gradient norm (before clipping) every few steps, and per-layer norms occasionally (Section 2). A steadily growing norm, sudden spikes, or norms of exactly zero in some layers all point to specific problems: instability, bad batches, or disconnected parameters.

**4. Monitor the update-to-weight ratio.** For each weight matrix, compute the ratio of the size of the update to the size of the weights:

```math
\frac{\|\Delta W\|}{\|W\|} = \frac{\|W_{t} - W_{t-1}\|}{\|W_{t-1}\|}.
```

This ratio says how much each step changes the layer in relative terms. A commonly cited rule of thumb is that it should be somewhere around $`10^{-3}`$ per step: much larger suggests the learning rate is too high for that layer, and much smaller suggests the layer is barely learning. It is a rough heuristic, not a law, but plotting it per layer quickly reveals layers that are frozen or thrashing.

```python
@torch.no_grad()
def update_ratios(model, prev_params):
    out = {}
    for name, p in model.named_parameters():
        if p.ndim >= 2:
            out[name] = ((p - prev_params[name]).norm() / (prev_params[name].norm() + 1e-12)).item()
    return out
# usage: prev = {n: p.detach().clone() for n, p in model.named_parameters()}; opt.step(); update_ratios(model, prev)
```

**5. Watch for loss spikes and NaN values.** A loss spike is a sudden jump in the training loss. Small spikes that recover quickly are common; large ones that do not recover indicate instability. NaN losses are fatal. Common causes and fixes:

| Symptom | Common causes | First fixes |
|---|---|---|
| NaN from the first steps | Bug (log of zero, division by zero), bad init, learning rate far too high | Check the loss computation, lower the learning rate |
| Loss spikes early | Warmup too short, learning rate too high | Longer warmup, lower peak learning rate |
| Loss spikes mid-training | Bad data batch, instability at high learning rate, Adam $`\hat{v}`$ adapting too slowly | Inspect the batch, lower learning rate, lower $`\beta_2`$, check clipping |
| FP16 NaN or Inf | Overflow in activations or gradients | Use dynamic loss scaling, or BF16 |

When a NaN appears, find the first operation that produced it; PyTorch's `torch.autograd.set_detect_anomaly(True)` can help, at a large speed cost.

**6. Compare train and validation curves.** Once training works, the gap between training and validation loss tells you whether to add regularization (large gap) or capacity (both losses high, small gap), as in Section 8.

## Reproducibility

Deep learning experiments are noisy: two runs that differ only in their random seed can end up with noticeably different results. Reproducibility habits make it possible to tell a real improvement from noise and to recover from failures.

**Seeds.** Set the random seeds for Python, NumPy, and PyTorch at the start of each run, and record them:

```python
import random, numpy as np, torch
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Seeds control initialization, data shuffling, dropout masks, and augmentation. Even with fixed seeds, results on GPUs may not be bit-for-bit identical, because some operations are nondeterministic for speed (such as some atomic additions in parallel reductions). PyTorch's `torch.use_deterministic_algorithms(True)` forces deterministic implementations where available, at some cost in speed. For comparing methods, the more important practice is to run **several seeds** and report the spread, rather than one run each.

**Logging.** Record everything needed to rerun and interpret an experiment: the code version (a commit hash), the full configuration of hyperparameters, the data version, and training curves (loss, validation metrics, learning rate, gradient norm) over time. Experiment-tracking tools make this easy, but even a structured log file is far better than nothing.

**Checkpointing.** Save checkpoints periodically, including not only the model weights but also the **optimizer state** (Adam's moments), the **learning-rate scheduler state**, the step count, and the random number generator states. Resuming from weights alone restarts Adam's moment estimates from zero and resets the schedule, which can cause a loss spike and changes the run. For long runs, checkpoints are also insurance against hardware failures.

```python
torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "step": step,
            "rng": torch.get_rng_state()}, "ckpt.pt")
```

## Scaling up the recipe

The same recipe carries over to larger models and larger batches, but instabilities that were rare at small scale become common, and each failure costs far more compute. Experience from large-scale training, including LLM pretraining, points to a few consistent adjustments:

- **Lower peak learning rates for larger models.** The largest stable learning rate tends to decrease as models get wider and deeper, so a learning rate tuned on a small model usually needs to be reduced for a large one.
- **Larger batches, with learning rates adjusted accordingly** (Section 7), to use more hardware in parallel, up to the point of diminishing returns.
- **Gradient clipping is always on.** At scale, occasional very large gradients are expected; clipping keeps them from wrecking the weights.
- **Warmup is always used,** often over a few thousand steps.
- **Monitoring for loss spikes** becomes an operational task. Large runs log gradient norms, loss, and other statistics continuously. When a spike does not recover, a common response is to roll back to a checkpoint from before the spike and resume, sometimes skipping the data batches around the spike or lowering the learning rate.
- **Lower $`\beta_2`$** (such as 0.95) is common in large-scale training, so Adam's second-moment estimate adapts faster to sudden changes in gradient scale.
- **Normalization and initialization details matter more.** Pre-norm placement, RMSNorm or LayerNorm choices, and the scaling of residual branches at initialization (Sections 4 and 5) all affect whether very deep networks train stably.

None of these are new ideas; they are the ingredients of this chapter, applied with more care because the cost of a failed run is higher.

## Key takeaways

- A strong default: He/Xavier initialization, residual blocks, pre-norm LayerNorm or RMSNorm, AdamW, warmup plus cosine or linear decay, gradient clipping, and regularization tuned to the train/validation gap.
- Mixed precision computes in FP16 or BF16 while keeping FP32 master weights; FP16 needs loss scaling to prevent gradient underflow, and BF16's wider range usually does not.
- Debug in order: check the initial loss (about $`\ln C`$ for $`C`$ classes), overfit a single batch, monitor gradient norms and update-to-weight ratios, and investigate loss spikes and NaNs.
- Record seeds, configurations, and curves; checkpoint the optimizer and scheduler state along with the weights; compare methods over several seeds.
- Scaling up keeps the recipe but lowers learning rates, keeps clipping and warmup on, and treats loss-spike monitoring as routine.

## Further reading

Goodfellow, Ian, et al. *Deep Learning*. Cambridge, MA: MIT Press, 2016. Chapter 11. https://www.deeplearningbook.org/.

Kalamkar, Dhiraj, et al. "A Study of BFLOAT16 for Deep Learning Training." arXiv preprint arXiv:1905.12322, 2019. https://arxiv.org/abs/1905.12322.

Micikevicius, Paulius, et al. "Mixed Precision Training." In *International Conference on Learning Representations*, 2018. https://arxiv.org/abs/1710.03740.
