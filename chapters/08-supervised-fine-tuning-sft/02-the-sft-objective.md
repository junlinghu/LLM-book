# 8.2 The SFT Objective

Supervised fine-tuning uses the same loss as pretraining, next-token cross-entropy, applied to a different kind of data and with one important restriction: only the tokens of the response are scored. This section writes the objective down, shows how the restriction is implemented with a **loss mask**, extends it to conversations with several turns, and then looks at two details that are easy to get wrong: how the loss is averaged over a batch, and what it means that the model is always trained on reference text.

## The objective

A training example for SFT is a pair: a **prompt** $`\mathbf{x}`$, which may include a system message and earlier turns of a conversation, and a **response** $`\mathbf{y} = (y_1, \dots, y_m)`$, the text we want the model to produce. A language model defines the probability of the response given the prompt by the chain rule, exactly as in Chapter 7, except that every factor also conditions on the prompt:

```math
p_\theta(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{m} p_\theta(y_t \mid \mathbf{x}, y_{\lt t})
```

SFT maximizes the log of this probability, averaged over the demonstrations in the dataset $`\mathcal{D}`$. Written as a loss to minimize:

```math
\mathcal{L}_{\text{SFT}}(\theta) = -\,\mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \mathcal{D}} \left[ \sum_{t=1}^{m} \log p_\theta(y_t \mid \mathbf{x}, y_{\lt t}) \right]
```

This is maximum likelihood estimation, the same principle behind the cross-entropy loss of Chapter 2. It has a useful interpretation. If the demonstrations are samples from some distribution $`p_{\text{data}}(\mathbf{y} \mid \mathbf{x})`$, the expected loss for a given prompt is the entropy of that distribution plus the KL divergence from it to the model:

```math
-\,\mathbb{E}_{\mathbf{y} \sim p_{\text{data}}} \left[ \log p_\theta(\mathbf{y} \mid \mathbf{x}) \right] = H\big(p_{\text{data}}\big) + \mathrm{KL}\big(p_{\text{data}} \,\|\, p_\theta\big)
```

The entropy does not depend on $`\theta`$, so minimizing the SFT loss drives the model's distribution toward the distribution of the demonstrations. This form of the KL divergence penalizes the model heavily for assigning low probability to any response that appears in the data, so the model is pushed to cover everything the demonstrators did. It says nothing directly about responses that do not appear in the data. If the demonstrations are good, the model learns to produce responses like them; if they contain mistakes, the model learns those too, with the same weight.

## Implementing it: concatenate, then mask

In practice, the prompt and the response are not handled by separate machinery. They are concatenated into one token sequence, formatted with a chat template (Section 3), and fed to the model exactly as in pretraining. The model predicts every next token in parallel, using the causal mask (Section 6.4), and the cross-entropy is computed at every position.

The restriction to response tokens is applied through the labels. For each position, the label is the next token of the sequence, as in pretraining, except that positions whose next token belongs to the prompt get a special **ignore value**, conventionally $`-100`$ (Section 5.9). The cross-entropy functions in PyTorch skip positions with this label, so they contribute nothing to the loss or the gradient. If $`m_t = 1`$ when token $`t`$ is part of a response and $`m_t = 0`$ otherwise, and $`\ell_t = -\log p_\theta(s_t \mid s_{\lt t})`$ is the per-token loss for the concatenated sequence $`\mathbf{s}`$, the loss for one example is

```math
\mathcal{L} = \frac{\sum_{t} m_t \, \ell_t}{\sum_{t} m_t}
```

Here the sum is divided by the number of response tokens, the default in most libraries. The next part of this section shows that this choice matters once examples are batched.

To see what gets trained in a real conversation, take a system message ("You are a concise assistant."), a user question ("What is the capital of France?"), the answer "Paris.", a follow-up question ("And of Japan?"), and the answer "Tokyo.". Rendered with the chat template of SmolLM2-135M-Instruct, this conversation is 49 tokens. Only 7 of them are trained: `Paris`, `.`, `<|im_end|>` for the first answer, and `To`, `kyo`, `.`, `<|im_end|>` for the second (Appendix A.1). Everything else, including the system message, both user turns, and the `<|im_start|>assistant` header that opens each answer, is masked.

Building these masks correctly is fiddly, and it depends on the template. The approach in Appendix A.1 renders the conversation one turn at a time, records which new tokens each turn adds, and marks as trainable only the tokens of assistant turns after their header. It then checks that the concatenated tokens decode to exactly the text the template produces for the whole conversation. That check catches a common bug: tokenizing turns separately and concatenating them can produce different tokens from tokenizing the whole string, because BPE merges can cross the boundary between two pieces of text (Section 5.3).

## Why mask the prompt

Training on prompt tokens would not be harmful in an obvious way; the model would simply also learn to predict user messages. There are several reasons not to:

- **The goal is to produce responses.** At inference, the model never generates the prompt; it is given. Capacity and gradient spent on predicting prompts are spent on a skill the assistant does not need.
- **Prompts are often repetitive.** Many datasets reuse the same system prompt or instruction template across thousands of examples. Unmasked, these repeated tokens make up a large share of the loss and are learned almost perfectly, which can drown out the signal from the responses, especially when responses are short.
- **Prompts may be low quality by design.** User messages in a dataset can contain typos, jailbreak attempts, or unsafe requests that the assistant is supposed to handle well. We want the model to learn the good response to a bad request, not the bad request.

The per-token losses of the two SmolLM2 models on the 49-token conversation illustrate what fine-tuning has done to them (Appendix A.2):

| Model | Loss on response tokens | Loss on all tokens | Loss on prompt tokens |
|---|---|---|---|
| SmolLM2-135M (base) | 5.764 | 5.784 | 5.787 |
| SmolLM2-135M-Instruct | 1.591 | 1.694 | 1.711 |

The base model's loss is high everywhere, because it was never trained on this format: special tokens such as `<|im_start|>` and `<|im_end|>` were in its vocabulary but almost never in its training text. The instruction-tuned model's loss is much lower, and lower on responses than on prompts, which is what training on responses would produce. Computing the loss with the ignore labels through the Hugging Face model gives 1.591, the same as the response-only column, which confirms that the labels mask what we intended.

Whether to train on prompt tokens at all is an empirical question, and Section 5 compares the two choices on a small task. The masked version is the standard default.

## Multi-turn conversations

A conversation with several exchanges could be turned into several training examples, one per assistant turn, each with the earlier turns as prompt. That wastes compute, because the early turns would be processed again for every later turn. The standard approach is to process the whole conversation once and put the loss on *every* assistant turn. Because of the causal mask, the prediction at each position depends only on earlier tokens, so each assistant turn is automatically conditioned on the entire conversation before it: the second answer sees the first question, the first answer, and the second question. One forward pass trains all turns.

The masking rule is therefore about roles, not position: tokens produced by the assistant are trained, and tokens from the system, the user, or tools are masked. Some pipelines deliberately mask certain assistant turns too, for example early turns that were written badly and are kept only as context, or turns generated by a different model.

## Learning to stop

The end-of-turn token is part of the response and must be included in the loss. It is how the model learns to stop. At inference, generation ends when the model produces this token (Section 5.6), so a model that never learned to produce it will keep going until it hits a length limit, as the base model did in Section 1. Conversely, if the responses in the training data are consistently long, the model learns that the end-of-turn token rarely comes early, and it becomes verbose. Response length is one of the most visible properties an SFT dataset controls.

In the conversation above, the template also puts a newline after each `<|im_end|>`, between turns. That newline is part of the formatting, not of the response, so it is masked. Details like this are specific to each template, and they are one more reason to build masks from the template itself rather than by hand.

## Averaging the loss over a batch

When several examples are batched, the per-token losses can be combined in two ways. The **token average** sums the losses over all response tokens in the batch and divides by the total number of response tokens. The **example average** first averages each example's losses over its own response tokens, then averages the per-example results. With $`N`$ examples, where example $`i`$ has $`n_i`$ response tokens and per-token losses $`\ell_{i,t}`$:

```math
\mathcal{L}_{\text{token}} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{n_i} \ell_{i,t}}{\sum_{i=1}^{N} n_i}, \qquad \mathcal{L}_{\text{example}} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{n_i} \sum_{t=1}^{n_i} \ell_{i,t}
```

They agree when all responses have the same length, and differ otherwise. In the token average, every token carries the same weight, so long responses dominate the gradient. In the example average, every example carries the same weight, so each token of a short response counts for more.

A small batch makes the difference concrete. Two examples ask "Is 7 a prime number?". One answer is "Yes." (3 trained tokens, counting the period and the end-of-turn token), the other a 38-token explanation. Under SmolLM2-135M-Instruct (Appendix A.3):

| | Trained tokens | Mean loss |
|---|---|---|
| Short response | 3 | 3.753 |
| Long response | 38 | 1.374 |
| Token average | 41 | 1.548 |
| Example average | | 2.563 |

The token average is close to the long response's loss, because 38 of the 41 tokens come from it; each token of the short response has weight 1/41. In the example average, each of the three short-response tokens has weight 1/6, almost seven times as much. Neither choice is wrong, but they train different models. A dataset that mixes one-word answers with long essays will, under the token average, be dominated by the essays.

The same issue appears in less obvious places. With **gradient accumulation**, a large batch is split into micro-batches whose gradients are summed before the optimizer step. If each micro-batch computes a token average and the results are averaged, the effective weighting is neither of the two rules above, because micro-batches with few response tokens get as much weight as those with many. Getting the token average right requires dividing by the total number of response tokens across all micro-batches. The same care is needed when examples are **packed** into one sequence (Section 5) and when training is spread across devices. Hugging Face's Transformers library, one of the most widely used, fixed exactly this bug in its gradient accumulation in October 2024, after users noticed that training losses changed when accumulation was turned on (Hugging Face 2024).

## SFT is teacher forcing

Like the Transformer trained on translation in Section 6.9, an SFT model is trained with **teacher forcing**: at every position, it conditions on the reference tokens of the response, never on tokens it generated itself. This is what makes training efficient, since all positions are computed in one parallel pass, but it creates a mismatch with how the model is used. At inference, the model conditions on its own outputs. If it makes an early mistake, it finds itself in a situation that never occurred in training, a sequence that no demonstrator would have written, and nothing in the loss taught it how to recover. This mismatch is the **exposure bias** already met in Section 6.9 (Ranzato et al. 2016). Scheduled sampling, which gradually replaces reference tokens with the model's own predictions during training, was an early attempt to reduce it for recurrent models (Bengio et al. 2015).

For short responses the effect is usually small. For long responses, such as multi-step reasoning, it matters more: an early wrong step can derail everything after it. SFT also never shows the model examples of what *not* to do. The loss rewards the reference response and is silent about every alternative, good or bad. Section 7 returns to these limits, which are the main motivation for training on the model's own outputs with preferences or rewards after SFT.

## Key takeaways

- The SFT loss is next-token cross-entropy on the response given the prompt; minimizing it moves the model's distribution toward the distribution of the demonstrations, mistakes included.
- In practice, prompt and response are concatenated and formatted with a chat template, and prompt positions get the ignore label $`-100`$ so that only response tokens contribute to the loss.
- Masking the prompt focuses training on producing responses, and keeps repetitive or low-quality prompt text out of the objective.
- In a multi-turn conversation, all assistant turns are trained in one pass, each conditioned on everything before it; the end-of-turn token is trained so that the model learns to stop.
- Token averaging and example averaging weight short and long responses differently, and gradient accumulation, packing, and multi-device training must preserve whichever normalization is intended.
- SFT is teacher forcing: the model is trained only on reference text, never on its own outputs or on examples of what to avoid.

## Further reading

Bengio, Samy, et al. "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks." In *Advances in Neural Information Processing Systems 28*, 2015. https://arxiv.org/abs/1506.03099.

Hugging Face. "Fixing Gradient Accumulation." Blog post, October 16, 2024. https://huggingface.co/blog/gradient_accumulation.

Ranzato, Marc'Aurelio, et al. "Sequence Level Training with Recurrent Neural Networks." In *International Conference on Learning Representations*, 2016. https://arxiv.org/abs/1511.06732.

## Appendix: Code for Section 8.2

These listings reproduce the examples in this section. Run them in order in one Python session; they need the `torch` and `transformers` packages, download the models from the Hugging Face Hub on first use, and run on a CPU.

### A.1 Building labels with a loss mask

Render a two-turn conversation with SmolLM2's chat template and mark which tokens are trained.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
IGNORE = -100

def build_example(messages, tok):
    """Token IDs and labels for a conversation; only assistant turns (and their
    end-of-turn token) are trained, everything else gets the label IGNORE."""
    ids, labels = [], []
    for k in range(len(messages)):
        # Render the conversation up to and including turn k, and keep the new tokens.
        text = tok.apply_chat_template(messages[:k + 1], tokenize=False)
        new = tok(text, add_special_tokens=False).input_ids[len(ids):]
        if messages[k]["role"] == "assistant":
            header = tok.apply_chat_template(messages[:k], tokenize=False,
                                             add_generation_prompt=True)
            n_header = len(tok(header, add_special_tokens=False).input_ids) - len(ids)
            # The assistant header ("<|im_start|>assistant\n") is prompt, the rest is trained,
            # except the final newline after <|im_end|>, which the template adds between turns.
            body = new[n_header:]
            lab = [IGNORE] * n_header + body[:-1] + [IGNORE]
        else:
            lab = [IGNORE] * len(new)
        ids += new
        labels += lab
    return ids, labels

conversation = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And of Japan?"},
    {"role": "assistant", "content": "Tokyo."},
]
ids, labels = build_example(conversation, tok)
assert tok.decode(ids) == tok.apply_chat_template(conversation, tokenize=False)
trained = [i for i, l in zip(ids, labels) if l != IGNORE]
print("tokens:", len(ids), "| trained:", len(trained))
print("trained tokens:", [tok.decode([i]) for i in trained])
```

Output:

```
tokens: 49 | trained: 7
trained tokens: ['Paris', '.', '<|im_end|>', 'To', 'kyo', '.', '<|im_end|>']
```

### A.2 Masked and unmasked loss

Compute the loss on response tokens only and on all tokens, for the base and the instruction-tuned model.

```python
def token_losses(model, ids):
    """Cross-entropy of each token given the ones before it (position t predicts t+1)."""
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x).logits[0, :-1]
    return F.cross_entropy(logits, x[0, 1:], reduction="none")

mask = torch.tensor([l != IGNORE for l in labels[1:]])
for name in ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct"]:
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    losses = token_losses(model, ids)
    print(f"{name.split('/')[1]:24s} response-only {losses[mask].mean():.3f}"
          f"  all tokens {losses.mean():.3f}  prompt-only {losses[~mask].mean():.3f}")
```

Output:

```
SmolLM2-135M             response-only 5.764  all tokens 5.784  prompt-only 5.787
SmolLM2-135M-Instruct    response-only 1.591  all tokens 1.694  prompt-only 1.711
```

```python
x, y = torch.tensor([ids]), torch.tensor([labels])
with torch.no_grad():
    hf_loss = model(input_ids=x, labels=y).loss
print(f"Hugging Face loss with IGNORE labels: {hf_loss:.3f}")
```

Output:

```
Hugging Face loss with IGNORE labels: 1.591
```

### A.3 Averaging over tokens or over examples

Two examples with a short and a long response give different batch losses under the two averaging rules.

```python
short = [{"role": "user", "content": "Is 7 a prime number?"},
         {"role": "assistant", "content": "Yes."}]
long = [{"role": "user", "content": "Is 7 a prime number?"},
        {"role": "assistant", "content": "Yes. Seven is prime because its only divisors are 1 and "
                                         "itself; it is not divisible by 2, 3, 4, 5, or 6."}]
per_example = []
for conv in [short, long]:
    ids_c, lab_c = build_example(conv, tok)
    m = torch.tensor([l != IGNORE for l in lab_c[1:]])
    per_example.append(token_losses(model, ids_c)[m])
for name, l in zip(["short", "long"], per_example):
    print(f"{name}: {len(l)} trained tokens, mean loss {l.mean():.3f}")
all_tokens = torch.cat(per_example)
print(f"token average:   {all_tokens.mean():.3f}")
print(f"example average: {torch.stack([l.mean() for l in per_example]).mean():.3f}")
n_short, n_long = len(per_example[0]), len(per_example[1])
print(f"weight of one short-response token: token average 1/{n_short + n_long}, "
      f"example average 1/{2 * n_short}")
```

Output:

```
short: 3 trained tokens, mean loss 3.753
long: 38 trained tokens, mean loss 1.374
token average:   1.548
example average: 2.563
weight of one short-response token: token average 1/41, example average 1/6
```
