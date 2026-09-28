# 8.5 The Training Recipe

The previous sections defined what SFT optimizes, how conversations are laid out, and where the data comes from. This section puts the pieces together into a training run. The recipe is short, because SFT reuses almost everything from pretraining: the model, the loss, the optimizer, and the training loop of Chapter 7. What changes is the scale and the starting point. The model starts from pretrained weights rather than from random ones, the dataset is tiny compared with the pretraining corpus, and the main risks are no longer divergence and slow progress but overfitting to the demonstrations and forgetting what pretraining taught. To make the recipe concrete, this section fine-tunes SmolLM2-135M on a small synthetic instruction dataset, on a CPU, and uses the result to look at batching, overfitting, forgetting, and the memory cost of full fine-tuning.

## Starting from pretrained weights

Fine-tuning continues the optimization that pretraining began, so the settings are those of Chapter 7 with a few adjustments.

- **A smaller learning rate.** The pretrained weights already sit in a good region of the loss surface. Large steps would move them far from it and destroy useful structure, so peak learning rates for full fine-tuning are typically an order of magnitude or more below those used in pretraining.
- **A short warmup and a decaying schedule.** A few warmup steps protect the pretrained weights from large, poorly calibrated early updates (Chapter 3), and a linear or cosine decay to a small value brings training to a gentle stop.
- **Few epochs.** An SFT dataset may contain thousands or hundreds of thousands of examples, which the model sees a few times, compared with the trillions of tokens it saw once during pretraining.

Published recipes illustrate the range. The Llama 2 authors fine-tuned with a cosine learning-rate schedule, an initial learning rate of $`2 \times 10^{-5}`$, weight decay of 0.1, a batch size of 64, and sequences of 4,096 tokens, for 2 epochs; they concatenated prompts and answers into one sequence and zeroed out the loss on the prompt tokens, exactly as in Section 2 (Touvron et al. 2023). InstructGPT's SFT models were trained for 16 epochs with a cosine learning-rate decay and residual dropout of 0.2 (Ouyang et al. 2022). The number of epochs differs by a factor of eight, which says something about how much the right setting depends on the data and on what the model is later used for, a point the discussion of overfitting below returns to.

## A small experiment

The experiment in Appendix A.1 to A.5 follows the recipe on a scale that runs on a laptop. The base model is SmolLM2-135M. The data consists of six tasks whose answers can be checked exactly: writing a word in capital letters, giving its first letter, repeating a word three times, counting the words in a short phrase, giving the last word of a phrase, and adding a two-digit number and a one-digit number. Each task has four phrasings, for example "Convert to uppercase: {x}", "Write {x} in capital letters.", "Uppercase this word: {x}", and "Rewrite the word {x} using only capital letters.". Three phrasings of each task are used for training, and the fourth is held out.

From these templates, the code draws 600 training examples (100 per task) and two test sets of 30 examples each. The first test set uses the training phrasings with newly drawn inputs; because the tasks draw words from a list of only 66, 6 of these 30 examples happen to coincide with training examples. The second test set uses only the held-out phrasings, so every prompt in it is worded in a way the model never saw during fine-tuning. It measures whether the model learned the task or just the templates.

Each example is formatted with the ChatML template of Section 3, with a user turn containing the instruction and an assistant turn containing the answer. For "Write pepper in capital letters.", the formatted sequence is 19 tokens, of which 4 are trained: `P`, `EP`, `PER`, and the end-of-turn token `<|im_end|>`. Everything else, including the user turn and the assistant header, is masked with the ignore label as in Section 2.

The training loop is the one from Chapter 7, restricted to 120 optimizer steps: AdamW with a peak learning rate of $`5 \times 10^{-5}`$ and no weight decay, 10 steps of linear warmup followed by linear decay to zero, batches of 8 examples, and gradient clipping at a norm of 1.0. That is 960 examples in total, or 1.6 passes over the training set. Two such runs, one with the prompt masked and one with the loss on all tokens, took about 9 minutes on 4 CPU threads of an otherwise idle machine, and 12 minutes in the appendix run on a busier one (Appendix A.5).

## Batching: padding and packing

SFT examples vary in length much more than pretraining blocks, which are cut to a fixed size from a continuous stream (Section 7.3). There are two ways to batch them.

**Padding** extends every example in a batch to the length of the longest with a padding token, sets the labels of the padded positions to the ignore value, and passes an attention mask that keeps real tokens from attending to padding (Section 6.4). It is simple, and it is what the experiment uses. Its cost is wasted computation: a batch containing one long example and seven short ones spends most of its work on padding. Sorting or bucketing examples by length reduces the waste.

**Packing** concatenates several examples into one sequence of a fixed length, as in pretraining, so that almost no position is wasted. Plain concatenation has a problem, however. Under an ordinary causal mask, tokens of the second example can attend to the first, so the model sees an unrelated conversation as context, a form of cross-contamination between examples. The fix is a **block-diagonal** attention mask, in which each example attends only to itself, together with position IDs that restart at zero for each example so that every example is encoded as if it stood alone (Section 7.3; Krell et al. 2021). The loss masks travel with the tokens: prompt positions of every packed example keep their ignore labels.

Appendix A.3 checks this with the first two training examples. Packed into one sequence with a block-diagonal mask and restarted positions, they produce the same logits as when each is run separately, to within $`10^{-3}`$. With a plain causal mask and consecutive positions, the logits for the second example differ from the separate run by up to 24.18, which is far from a rounding error: the model is predicting a different distribution because it is reading a different context. Packing combined with loss averaging also interacts with the normalization question of Section 2, because a packed sequence contains a variable number of response tokens from a variable number of examples.

## What the experiment shows

The fine-tuned models were compared with two references: the base model before fine-tuning, and SmolLM2-135M-Instruct, which its authors built from the same base model with SFT on a large instruction mixture followed by preference optimization (Allal et al. 2025). Accuracy is exact match between the greedy answer and the target string. Forgetting is measured by perplexity on 20 blocks of 128 tokens from the end of the tiny Shakespeare text used in earlier chapters, text that none of these fine-tuning runs saw (Appendix A.4 and A.5).

| Model | Seen phrasings | Held-out phrasings | Shakespeare perplexity |
|---|---|---|---|
| Base model, no fine-tuning | 0.00 | 0.00 | 41.7 |
| SmolLM2-135M-Instruct (reference) | 0.00 | 0.00 | 56.5 |
| Fine-tuned, prompt masked | 0.63 | 0.60 | 47.0 |
| Fine-tuned, loss on all tokens | 0.67 | 0.57 | 53.0 |

Several things stand out.

**Before fine-tuning, both reference models score zero.** The base model does not answer in the expected form; it continues the text. The instruction-tuned model does answer, but in its own style, with full sentences rather than the bare string the check requires. Exact match is a strict metric, and both models fail it for reasons of format, not necessarily of knowledge. Teaching a format is exactly what SFT does well, and 120 steps on 600 examples take the base model from zero to about 60 percent.

**The model generalizes to new phrasings, imperfectly.** Accuracy on the held-out phrasings (0.60 for the masked run) is close to accuracy on the training phrasings (0.63), so most of what the model learned is tied to the task rather than to the exact template. The failures are instructive. Asked "Rewrite the word river using only capital letters.", a phrasing it never saw, the masked model answered "RULE OF FORCE". Asked "Output the word ticket thrice.", it wrote "ticket ticket", apparently not connecting "thrice" with the three repetitions it learned from "three times" and "3 times". Other held-out prompts, such as "What do you get when you add 47 to 4?" and "Tell me the initial letter of planet.", were answered correctly.

**Masking made no measurable difference to accuracy, but a visible one to forgetting.** With 30 test examples, one example is worth 0.033, so the differences in accuracy between the two runs (0.63 against 0.67, and 0.60 against 0.57) amount to a single example each, well within the noise of such a small test. The difference in perplexity is larger: 47.0 with the prompt masked against 53.0 with the loss on all tokens, starting from 41.7. A likely reason is that the unmasked run also spends its updates on learning to predict the repetitive template text of the prompts, which pulls the model further from general text. One run per setting is not enough to be sure of a difference of this size, and this is a toy task, but the direction agrees with the reasons for masking given in Section 2.

The training losses in the appendix cannot be compared between the two runs. The masked loss fell from 1.738 at step 40 to 0.222 at step 120, but the unmasked loss averages over prompt tokens as well, which are easier to predict in some places and impossible in others (no model can predict which word a user will ask about), so its value of 0.658 at step 120 measures something else.

## Watching for overfitting

Because an SFT dataset is small, a model can fit it closely within a few epochs, and the usual signal of overfitting appears: the loss on held-out demonstrations stops falling and starts to rise. In SFT, however, that signal can be misleading. The InstructGPT authors observed that their SFT models overfit on validation loss after one epoch, yet training for more epochs still improved both the score of their reward model and the ratings of human labelers (Ouyang et al. 2022). Held-out loss measures how well the model predicts the particular wording of the reference responses, and a model can become worse at predicting that wording while becoming better at producing good responses of its own.

The practical lesson is to select checkpoints by the quantities that matter: task metrics such as the exact-match accuracy above, checks on held-out phrasings and held-out tasks, and samples read by people. Validation loss remains useful as a warning of gross overfitting, but it is not the objective. Section 7 returns to evaluation.

## Catastrophic forgetting

Training a network on new data alone can erode what it learned before, a phenomenon known as **catastrophic forgetting**. Kirkpatrick et al. (2017) studied it in networks trained on a sequence of tasks and proposed elastic weight consolidation, which adds a penalty for moving the weights that mattered most for earlier tasks, measured by the Fisher information. In LLM fine-tuning, the "earlier task" is pretraining, and forgetting shows up as lost knowledge or skills that were not represented in the SFT data.

The experiment shows a mild version. Perplexity on held-out Shakespeare rose from 41.7 to 47.0 after 120 steps on the synthetic tasks, even with the prompt masked. SmolLM2-135M-Instruct, trained on far more data and with a further stage of preference optimization, is at 56.5. Higher perplexity on old plays is not necessarily a problem for an assistant, and it partly reflects a shift in what kind of text the model expects. But it is a signal that the model has moved away from its pretraining distribution, and the same movement can cost abilities that do matter.

Three remedies are common. A lower learning rate and fewer steps limit how far the weights move. **Mixing** some pretraining-style text, or data from earlier tasks, into the SFT data keeps the old objective partly in play; this is the language-model counterpart of rehearsal in continual learning. And training fewer parameters, the subject of Section 6, constrains the update directly.

## The memory bill for full fine-tuning

Full fine-tuning updates every parameter, so it needs the same memory as pretraining per parameter, even if the dataset is small. With the Adam optimizer and mixed-precision training (Chapter 3), each parameter carries:

- 2 bytes for the BF16 weight used in the forward and backward passes,
- 2 bytes for its BF16 gradient,
- 4 bytes for an FP32 master copy of the weight, to which updates are applied,
- 8 bytes for Adam's first and second moments, 4 bytes each in FP32.

This accounting follows Rajbhandari et al. (2020). For a model with $`P`$ parameters, the memory for these **model states** is

```math
M_{\text{states}} \approx (2 + 2 + 4 + 4 + 4)\,P = 16P \ \text{bytes}
```

For SmolLM2-135M, whose parameter count is 135 million, that is about 2.2 GB, easily within the memory of a laptop. For a 7-billion-parameter model it is 112 GB, more than the memory of any single common GPU, before counting **activations**, the intermediate values saved during the forward pass for use in the backward pass. Activation memory grows with the batch size, the sequence length, the width, and the number of layers, and for long sequences it can exceed the model states. The experiment above trained entirely in FP32 on a CPU, with 4 bytes each for weights and gradients and 8 for Adam's moments, which also comes to 16 bytes per parameter.

## Reducing memory

Several techniques bring the bill down without changing what is learned.

- **Gradient checkpointing** stores only some of the activations during the forward pass and recomputes the rest during the backward pass, trading extra computation for less activation memory.
- **Gradient accumulation** processes a large batch as several small micro-batches and sums their gradients before each optimizer step, so the batch size no longer determines the activation memory. As Section 2 explained, the loss normalization must be handled carefully across micro-batches.
- **Sharding** splits the model states across devices instead of replicating them on each one. ZeRO partitions first the optimizer states, then the gradients, and finally the parameters themselves, so that memory for model states per device falls roughly in proportion to the number of devices (Rajbhandari et al. 2020).

The most direct saving, though, is to train fewer parameters. The 12 bytes per parameter for the master copy and Adam's moments are needed only for parameters that are trained. If the pretrained weights are frozen and only a small number of new parameters are trained, most of the bill disappears, and the frozen weights can even be stored in reduced precision. That is the idea of parameter-efficient fine-tuning.

## Key takeaways

- SFT reuses the pretraining loop with a smaller learning rate, a short warmup, a decaying schedule, and a few epochs over a small dataset; published recipes range from 2 epochs (Llama 2) to 16 (InstructGPT).
- Examples can be padded, or packed into fixed-length sequences with a block-diagonal attention mask and restarted position IDs; without the mask, packed examples contaminate each other, and the loss masks must travel with the tokens.
- On a toy task, 120 steps of full fine-tuning took SmolLM2-135M from 0 to about 60 percent exact-match accuracy, with similar accuracy on held-out phrasings, at the cost of a rise in perplexity on held-out text from 41.7 to 47.0.
- Validation loss on demonstrations can rise while response quality still improves, so checkpoints should be chosen with task metrics and samples.
- Catastrophic forgetting is limited by lower learning rates, fewer steps, mixing in pretraining-style data, and training fewer parameters.
- Full fine-tuning with Adam in mixed precision needs about 16 bytes per parameter for model states, 112 GB for a 7-billion-parameter model before activations; checkpointing, accumulation, and sharding reduce the burden.

## Further reading

Allal, Loubna Ben, et al. "SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model." arXiv preprint arXiv:2502.02737, 2025. https://arxiv.org/abs/2502.02737.

Kirkpatrick, James, et al. "Overcoming Catastrophic Forgetting in Neural Networks." *Proceedings of the National Academy of Sciences* 114, no. 13 (2017): 3521–3526. https://arxiv.org/abs/1612.00796.

Krell, Mario Michael, et al. "Efficient Sequence Packing without Cross-Contamination: Accelerating Large Language Models without Impacting Performance." arXiv preprint arXiv:2107.02027, 2021. https://arxiv.org/abs/2107.02027.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Rajbhandari, Samyam, et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC20)*, 2020. https://arxiv.org/abs/1910.02054.

Touvron, Hugo, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv preprint arXiv:2307.09288, 2023. https://arxiv.org/abs/2307.09288.

## Appendix: Code for Section 8.5

These listings reproduce the examples in this section. Run them in order in one Python session; they need the `torch` and `transformers` packages, download the models from the Hugging Face Hub on first use, and run on a CPU. The fine-tuning runs in A.5 take several minutes each on 4 CPU threads; the evaluation in A.4 downloads the tiny Shakespeare text if it is not already present.

### A.1 A synthetic instruction dataset

Six tasks with checkable answers, each with three training phrasings and one held-out phrasing.

```python
import random, math, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(4)
WORDS = ("apple river stone cloud garden window candle forest pencil mirror ocean silver "
         "basket rocket island winter summer planet dragon castle butter meadow violin "
         "tiger lemon piano wallet bridge jacket carpet rabbit button harbor lantern "
         "orange parrot saddle ticket valley wizard anchor bottle circle desert engine "
         "feather guitar hammer jungle kettle ladder marble needle pepper puzzle ribbon "
         "shadow thunder tunnel velvet walnut yellow zipper blanket cactus dolphin").split()

def sentence(rng):
    return " ".join(rng.sample(WORDS, rng.randint(3, 6)))

TASKS = {  # name: (phrasings; the last one is held out, answer function, input maker)
    "upper": (["Convert to uppercase: {x}", "Write {x} in capital letters.",
               "Uppercase this word: {x}", "Rewrite the word {x} using only capital letters."],
              lambda x: x.upper(), lambda r: r.choice(WORDS)),
    "first": (["What is the first letter of '{x}'?", "Give the first letter of the word {x}.",
               "Which letter does '{x}' start with?", "Tell me the initial letter of {x}."],
              lambda x: x[0], lambda r: r.choice(WORDS)),
    "repeat": (["Repeat the word {x} three times.", "Say {x} three times, separated by spaces.",
                "Write '{x}' 3 times.", "Output the word {x} thrice."],
               lambda x: " ".join([x] * 3), lambda r: r.choice(WORDS)),
    "count": (["How many words are in this text: {x}", "Count the words: {x}",
               "Number of words in '{x}'?", "Tell me how many words this has: {x}"],
              lambda x: str(len(x.split())), sentence),
    "last": (["What is the last word of: {x}", "Give the final word of '{x}'.",
              "Which word comes last in: {x}", "Name the word at the end of: {x}"],
             lambda x: x.split()[-1], sentence),
    "add": (["What is {a} + {b}?", "Add {a} and {b}.", "Compute {a} plus {b}.",
             "What do you get when you add {a} to {b}?"],
            None, lambda r: (r.randint(10, 49), r.randint(1, 9))),
}

def make_example(task, rng, held_out=False):
    phrasings, answer, make = TASKS[task]
    p = phrasings[-1] if held_out else rng.choice(phrasings[:-1])
    x = make(rng)
    if task == "add":
        return p.format(a=x[0], b=x[1]), str(x[0] + x[1])
    return p.format(x=x), answer(x)

rng = random.Random(0)
train = [make_example(t, rng) for _ in range(100) for t in TASKS]
test_seen = [make_example(t, rng) for _ in range(5) for t in TASKS]
test_new = [make_example(t, rng, held_out=True) for _ in range(5) for t in TASKS]
print(len(train), "training examples,", len(test_seen), "test (seen phrasings),",
      len(test_new), "test (held-out phrasings)")
print("seen-phrasing test examples that also occur in the training set:",
      sum(ex in set(train) for ex in test_seen))
for q, a in train[:6]:
    print(f"  {q!r} -> {a!r}")
```

Output:

```
600 training examples, 30 test (seen phrasings), 30 test (held-out phrasings)
seen-phrasing test examples that also occur in the training set: 6
  'Write pepper in capital letters.' -> 'PEPPER'
  "What is the first letter of 'lantern'?" -> 'l'
  "Write 'zipper' 3 times." -> 'zipper zipper zipper'
  'Count the words: yellow feather bridge cactus planet' -> '5'
  "Give the final word of 'basket harbor dragon wizard'." -> 'wizard'
  'What is 14 + 6?' -> '20'
```

### A.2 Tokenizing with loss masks and padding

Format every example with the ChatML template and mask everything except the answer and its end-of-turn token.

```python
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
IM_START, IM_END = "<|im_start|>", "<|im_end|>"

def prompt_text(q):
    return f"{IM_START}user\n{q}{IM_END}\n{IM_START}assistant\n"

def encode(q, a, mask_prompt=True):
    p = tok(prompt_text(q), add_special_tokens=False).input_ids
    r = tok(a + IM_END, add_special_tokens=False).input_ids
    labels = ([-100] * len(p) if mask_prompt else p) + r
    return p + r, labels

def collate(batch):
    n = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), n), tok.pad_token_id or 0)
    labels = torch.full((len(batch), n), -100)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)], labels[i, :len(y)], attn[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    return ids, labels, attn

ids, labels = encode(*train[0])
print("prompt + answer tokens:", len(ids), "| trained:", sum(l != -100 for l in labels))
print("trained:", [tok.decode([t]) for t, l in zip(ids, labels) if l != -100])
```

Output:

```
prompt + answer tokens: 19 | trained: 4
trained: ['P', 'EP', 'PER', '<|im_end|>']
```

### A.3 Packing with a block-diagonal mask

Check that two packed examples give exactly the same logits as the examples run separately.

```python
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32).eval()
a, _ = encode(*train[0]); b, _ = encode(*train[1])
with torch.no_grad():
    la = model(torch.tensor([a])).logits[0]
    lb = model(torch.tensor([b])).logits[0]
    n, L = len(a) + len(b), len(a)
    allowed = torch.zeros(n, n, dtype=torch.bool)
    allowed[:L, :L] = torch.tril(torch.ones(L, L, dtype=torch.bool))
    allowed[L:, L:] = torch.tril(torch.ones(n - L, n - L, dtype=torch.bool))
    mask = torch.zeros(1, 1, n, n).masked_fill(~allowed, float("-inf"))
    pos = torch.tensor([list(range(L)) + list(range(n - L))])
    packed = model(torch.tensor([a + b]), attention_mask=mask, position_ids=pos).logits[0]
    naive = model(torch.tensor([a + b])).logits[0]
same = torch.allclose(packed[:L], la, atol=1e-3) and torch.allclose(packed[L:], lb, atol=1e-3)
print("block-diagonal mask, same logits as separate runs (within 1e-3):", same)
print("plain causal mask, max logit difference on the second example:",
      round((naive[L:] - lb).abs().max().item(), 2))
```

Output:

```
block-diagonal mask, same logits as separate runs (within 1e-3): True
plain causal mask, max logit difference on the second example: 24.18
```

### A.4 Evaluation: exact match and forgetting

Greedy answers scored by exact match, and perplexity on held-out Shakespeare as a measure of forgetting.

```python
import os, urllib.request
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("shakespeare.txt"):
    urllib.request.urlretrieve(URL, "shakespeare.txt")
text = open("shakespeare.txt", encoding="utf-8").read()
held = tok(text[-20000:], add_special_tokens=False).input_ids
blocks = [held[i:i + 128] for i in range(0, 20 * 128, 128)]

@torch.no_grad()
def accuracy(model, data):
    model.eval()
    correct = 0
    for q, a in data:
        ids = torch.tensor([tok(prompt_text(q), add_special_tokens=False).input_ids])
        out = model.generate(ids, max_new_tokens=12, do_sample=False,
                             eos_token_id=tok.convert_tokens_to_ids(IM_END),
                             pad_token_id=tok.convert_tokens_to_ids(IM_END))
        pred = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        correct += pred == a
    return correct / len(data)

@torch.no_grad()
def perplexity(model):
    model.eval()
    x = torch.tensor(blocks)
    return math.exp(model(x, labels=x).loss.item())

def report(name, model):
    print(f"{name:34s} seen {accuracy(model, test_seen):.2f}  held-out phrasing "
          f"{accuracy(model, test_new):.2f}  Shakespeare ppl {perplexity(model):.1f}")

report("base, no fine-tuning", model)
instruct = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct",
                                                dtype=torch.float32)
report("SmolLM2-135M-Instruct (reference)", instruct)
del instruct
```

Output:

```
base, no fine-tuning               seen 0.00  held-out phrasing 0.00  Shakespeare ppl 41.7
SmolLM2-135M-Instruct (reference)  seen 0.00  held-out phrasing 0.00  Shakespeare ppl 56.5
```

### A.5 Full fine-tuning

Train the base model on the synthetic data with AdamW, warmup, and linear decay, with and without the prompt mask.

```python
def finetune(mask_prompt, steps=120, batch_size=8, lr=5e-5, warmup=10, seed=0):
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
    data = [encode(q, a, mask_prompt) for q, a in train]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, (steps - s) / (steps - warmup)))
    order = random.Random(seed)
    model.train()
    for step in range(steps):
        batch = order.sample(data, batch_size)
        ids, labels, attn = collate(batch)
        loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        if (step + 1) % 40 == 0:
            print(f"  step {step + 1}: training loss {loss.item():.3f}")
    return model

start = time.time()
masked = finetune(mask_prompt=True)
report("fine-tuned, prompt masked", masked)
unmasked = finetune(mask_prompt=False)
report("fine-tuned, loss on all tokens", unmasked)
print(f"(both runs: {(time.time() - start) / 60:.0f} minutes on 4 CPU threads)")
```

Output:

```
  step 40: training loss 1.738
  step 80: training loss 0.312
  step 120: training loss 0.222
fine-tuned, prompt masked          seen 0.63  held-out phrasing 0.60  Shakespeare ppl 47.0
  step 40: training loss 1.343
  step 80: training loss 0.744
  step 120: training loss 0.658
fine-tuned, loss on all tokens     seen 0.67  held-out phrasing 0.57  Shakespeare ppl 53.0
(both runs: 12 minutes on 4 CPU threads)
```

```python
for q, a in test_new[::5]:
    ids = torch.tensor([tok(prompt_text(q), add_special_tokens=False).input_ids])
    with torch.no_grad():
        out = masked.generate(ids, max_new_tokens=12, do_sample=False,
                              eos_token_id=tok.convert_tokens_to_ids(IM_END),
                              pad_token_id=tok.convert_tokens_to_ids(IM_END))
    print(f"{q!r}: {tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)!r} (target {a!r})")
```

Output:

```
'Rewrite the word river using only capital letters.': 'RULE OF FORCE' (target 'RIVER')
'What do you get when you add 47 to 4?': '51' (target '51')
'Name the word at the end of: wizard circle lemon feather cloud meadow': 'meadow' (target 'meadow')
'Tell me how many words this has: orange mirror tunnel butter rocket': '5' (target '5')
'Output the word ticket thrice.': 'ticket ticket' (target 'ticket ticket ticket')
'Tell me the initial letter of planet.': 'p' (target 'p')
```

### A.6 The memory bill for full fine-tuning

Count the bytes for weights, gradients, and Adam state under mixed precision.

```python
P = sum(p.numel() for p in masked.parameters())
for name, n in [("SmolLM2-135M", P), ("a 7-billion-parameter model", 7e9)]:
    print(f"{name}: {n / 1e6:,.0f}M parameters x 16 bytes = {16 * n / 1e9:.1f} GB "
          f"(weights 2, gradients 2, FP32 master weights 4, Adam moments 8)")
```

Output:

```
SmolLM2-135M: 135M parameters x 16 bytes = 2.2 GB (weights 2, gradients 2, FP32 master weights 4, Adam moments 8)
a 7-billion-parameter model: 7,000M parameters x 16 bytes = 112.0 GB (weights 2, gradients 2, FP32 master weights 4, Adam moments 8)
```
