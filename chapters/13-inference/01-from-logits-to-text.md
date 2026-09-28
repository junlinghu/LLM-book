# 13.1 From Logits to Text

A trained language model does not produce text. It produces a probability distribution over the next token. Everything between that distribution and the words a user reads is the job of the *decoding* procedure: a loop that repeatedly runs the model, turns its output into a choice of token, appends the token to the input, and runs the model again. The decoding procedure has a surprisingly large effect on quality. The same model can sound creative or robotic, stay on topic or ramble, produce valid JSON or broken JSON, depending only on how its logits are turned into tokens.

This section walks through that loop and the main decoding strategies used in practice: greedy decoding, sampling with temperature, top-k and top-p truncation, repetition penalties and stop conditions, beam search, and constrained decoding for structured outputs. The rest of the chapter then asks what each iteration of this loop costs and how to make it cheaper.

## Autoregressive generation, one token at a time

Recall from Chapter 7 that a GPT-style model defines the probability of a sequence as a product of next-token conditionals:

$$
p_\theta(x_1, \dots, x_N) = \prod_{t=1}^{N} p_\theta(x_t \mid x_{\lt t}).
$$

At each position the transformer produces a final hidden state $\mathbf{h}_t \in \mathbb{R}^{d}$. The *unembedding* (or "LM head") matrix $W_U \in \mathbb{R}^{V \times d}$ maps it to a vector of *logits*, one real number per vocabulary entry:

$$
\mathbf{z}_t = W_U \mathbf{h}_t \in \mathbb{R}^{V}.
$$

The softmax turns logits into probabilities:

$$
p_\theta(x_{t+1} = i \mid x_{\le t}) = \frac{\exp(z_{t,i})}{\sum_{j=1}^{V} \exp(z_{t,j})}.
$$

Generation runs this model in a loop. Starting from a prompt $x_1, \dots, x_m$, we compute the distribution for position $m+1$, choose a token from it, append that token, and repeat until a stop condition is met:

```python
def generate(model, prompt_ids, choose_token, max_new_tokens=256, eos_id=None):
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        logits = model(ids)[-1]          # logits for the next position only
        next_id = choose_token(logits)   # greedy, sampling, ... (this section)
        ids.append(next_id)
        if next_id == eos_id:
            break
    return ids[len(prompt_ids):]
```

Two properties of this loop drive almost everything else in the chapter.

**It is sequential.** Token $t+1$ cannot be computed until token $t$ has been chosen, because it is part of the input. During training, teacher forcing let us process all positions of a sequence in parallel (Chapter 7); during generation, we cannot. Generating 1,000 tokens requires 1,000 dependent passes through the network.

**Each step only needs one new row of logits.** The model computes a hidden state for every position, but we only use the last one. Naively rerunning the whole sequence at every step wastes almost all of the work. Section 4 shows how the *KV cache* avoids that waste by saving intermediate results from earlier steps.

The function `choose_token` is the decoding strategy. The simplest strategies look only at the current logits; more elaborate ones (beam search, constrained decoding) keep extra state across steps.

## Greedy decoding vs. sampling

**Greedy decoding** always picks the most likely token:

$$
x_{t+1} = \arg\max_{i} \; z_{t,i}.
$$

Greedy decoding is deterministic, cheap, and often a good choice for short, factual outputs where there is one right answer: extracting a date, classifying a sentence, answering a multiple-choice question. It is a poor choice for long open-ended text. Choosing the locally most likely token at every step does not produce the most likely sequence overall, and in practice greedy continuations of open-ended prompts often fall into loops, repeating the same phrase or sentence. Holtzman et al. studied this *degeneration* in detail and showed that text maximizing likelihood looks very different from human text: human writing regularly contains tokens that the model considers fairly unlikely, whereas greedy and beam search outputs stay in the high-probability region and become bland and repetitive.

**Sampling** instead draws the next token at random from the model's distribution:

$$
x_{t+1} \sim \text{Categorical}\big(\text{softmax}(\mathbf{z}_t)\big).
$$

Pure ("ancestral") sampling produces text with the same statistics the model learned, which gives it variety. But the model's distribution has a long tail. With a vocabulary of 100,000 or more tokens, even if each individual unlikely token has tiny probability, their total can be substantial, and sampling from the tail now and then produces an incoherent token that derails the rest of the generation. Nearly every practical sampler therefore *reshapes* or *truncates* the distribution before drawing from it. The next subsections describe the common knobs.

A useful mental model is that decoding trades off two failure modes: *too little randomness* (repetition, blandness, identical answers to every user) and *too much randomness* (incoherence, factual drift, format errors). Different tasks sit at different points on this spectrum.

## Temperature, top-k, and top-p (nucleus) sampling

### Temperature

Temperature $T \gt 0$ divides the logits before the softmax:

$$
p_T(i) = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}.
$$

- $T = 1$ leaves the model's distribution unchanged.
- $T \lt 1$ sharpens it: differences between logits are magnified, so probable tokens become more probable. As $T \to 0$, the distribution collapses onto the argmax and sampling becomes greedy decoding. Many APIs treat "temperature 0" as a special case meaning greedy.
- $T \gt 1$ flattens it toward uniform, making rare tokens more likely.

Temperature does not change the *ranking* of tokens, only how peaked the distribution is. For two tokens with logits $z_a \gt z_b$, the probability ratio is $p_T(a)/p_T(b) = \exp\big((z_a - z_b)/T\big)$: halving the temperature squares this ratio.

In practice, low temperatures (roughly 0 to 0.3) are common for code, math, extraction, and other tasks with a single right answer, while values around 0.7 to 1.0 are common for chat and creative writing. These are conventions, not laws; the right value depends on the model and on how it was post-trained, because RLHF and similar methods already make a model's distribution much sharper than a base model's.

### Top-k sampling

Top-k sampling, popularized for story generation by Fan et al., keeps only the $k$ tokens with the highest logits, sets the probability of all others to zero, and renormalizes. With $k = 1$ it is greedy decoding. It removes the long tail, but a fixed $k$ fits the shape of the distribution poorly. After "The capital of France is", almost all the mass is on one token and $k = 50$ admits 49 bad options; after "My favorite food is", there may be hundreds of reasonable continuations and $k = 50$ cuts many of them off.

### Top-p (nucleus) sampling

Top-p sampling, proposed by Holtzman et al. under the name *nucleus sampling*, adapts the cutoff to the distribution. Sort the tokens by probability, $p_{(1)} \ge p_{(2)} \ge \dots$, and keep the smallest prefix whose cumulative probability reaches $p$:

$$
k^\star = \min \Big\{ k : \sum_{r=1}^{k} p_{(r)} \ge p \Big\}, \qquad \text{nucleus} = \{(1), \dots, (k^\star)\}.
$$

Then renormalize over the nucleus and sample. When the model is confident, the nucleus is small (perhaps a single token); when it is uncertain, the nucleus is large. Typical values of $p$ are between 0.9 and 0.95.

Other truncation rules follow the same pattern. *Min-p* sampling, for example, keeps every token whose probability is at least a fixed fraction of the top token's probability, so the threshold also scales with the model's confidence.

### Putting the knobs together

Implementations apply these transformations in a fixed order, usually: penalties on the raw logits, then temperature, then top-k, then top-p, then sampling. The order matters. Applying temperature before top-p means a high temperature enlarges the nucleus; applying it after does not. Here is a compact NumPy implementation:

```python
import numpy as np

def softmax(z):
    z = z - z.max()                      # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum()

def sample_next(logits, temperature=1.0, top_k=None, top_p=None, rng=np.random.default_rng()):
    z = np.asarray(logits, dtype=np.float64)
    if temperature == 0:                 # convention: temperature 0 means greedy
        return int(np.argmax(z))
    z = z / temperature
    if top_k is not None:                # keep the k largest logits
        kth = np.sort(z)[-top_k]
        z = np.where(z < kth, -np.inf, z)
    probs = softmax(z)
    if top_p is not None:                # keep the smallest set with mass >= top_p
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        cutoff = np.searchsorted(cum, top_p) + 1
        keep = order[:cutoff]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))

logits = np.array([4.0, 3.5, 2.0, 0.5, -1.0])
print(sample_next(logits, temperature=0))                  # always 0
print(sample_next(logits, temperature=0.7, top_p=0.9))     # 0 or 1, occasionally 2
```

The first suggested code lab for this chapter asks you to wire a function like this to a real model's logits and compare the outputs of greedy decoding, several temperatures, and several values of $p$ on the same prompt.

## Repetition penalties and stop sequences

### Repetition penalties

Even with sampling, models sometimes repeat themselves. Several heuristics push against this by modifying the logits of tokens that have already appeared.

The *repetition penalty* introduced with the CTRL model by Keskar et al. discounts the logits of tokens that already occur in the context by a factor $\theta \gt 1$. Because dividing a negative logit would *raise* it, common implementations divide positive logits and multiply negative ones:

$$
z_i' = \begin{cases} z_i / \theta & \text{if } i \text{ appeared before and } z_i \gt 0, \\ z_i \cdot \theta & \text{if } i \text{ appeared before and } z_i \le 0, \\ z_i & \text{otherwise.} \end{cases}
$$

*Frequency and presence penalties*, exposed by many chat APIs, are additive. If token $i$ has appeared $c_i$ times so far,

$$
z_i' = z_i - \alpha_{\text{freq}} \, c_i - \alpha_{\text{pres}} \, \mathbb{1}[c_i \gt 0].
$$

The frequency penalty grows with each repetition; the presence penalty is a one-time charge for having used a token at all, which nudges the model toward new topics. A related option, the *no-repeat n-gram* rule, forbids any token that would complete an n-gram already present in the output.

All of these are blunt instruments. They cannot tell a harmful repetition ("the the the") from a necessary one (a variable name that must appear many times in code, or a person's name in a biography). Strong penalties degrade code and structured output. Modern instruction-tuned models repeat themselves much less than early base models did, and many deployments use little or no repetition penalty, relying instead on sensible temperature and top-p settings.

### Stop sequences

Generation must end somewhere. Three stop conditions are standard:

1. **End-of-sequence token.** Models are trained to emit a special token (such as `<|endoftext|>` or an end-of-turn marker in a chat template, Chapter 8) when a document or turn is complete. Emitting it ends generation.
2. **Maximum length.** A hard cap on the number of new tokens bounds cost and latency. When a generation stops because it hit the cap, APIs typically report a finish reason such as "length" so the caller knows the output was truncated.
3. **Stop strings.** The caller can supply strings, such as `"\nUser:"` or `"</answer>"`, that end generation when they appear.

Stop strings are subtler than they look, because the model generates *tokens* and the stop condition is defined on *text*. A stop string can be split across several tokens, and a single token can contain the end of the stop string plus extra characters. A correct implementation therefore checks the detokenized text after every step and truncates at the first match. When streaming output to a user (Section 8), the server must also hold back any trailing text that could be the beginning of a stop string until it knows whether the full string will appear; otherwise it may stream half of a stop sequence and then have to take it back.

## Beam search

Greedy decoding keeps one partial hypothesis. *Beam search* keeps $B$ of them. At each step, every hypothesis in the beam is extended by every possible next token, each extension is scored by its total log-probability,

$$
s(x_{1:t}) = \sum_{\tau=1}^{t} \log p_\theta(x_\tau \mid x_{\lt \tau}),
$$

and the $B$ highest-scoring extensions become the new beam. Finished hypotheses (those that emitted the end token) are set aside, and the search ends when enough hypotheses have finished or a length limit is reached. With $B = 1$, beam search is greedy decoding.

Because every additional token multiplies in a probability less than one, raw log-probability favors short outputs. Implementations therefore usually apply *length normalization*, for example dividing the score by $t^\alpha$ for some $\alpha$ between 0 and 1, before comparing finished hypotheses.

```python
import numpy as np

def beam_search(step_logprobs, bos, eos, beam_size=4, max_len=20, alpha=0.7):
    """step_logprobs(prefix) -> array of log-probabilities over the vocabulary."""
    beams = [([bos], 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for prefix, score in beams:
            lp = step_logprobs(prefix)
            for tok in np.argsort(-lp)[:beam_size]:        # only the best few per beam
                candidates.append((prefix + [int(tok)], score + float(lp[tok])))
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for prefix, score in candidates:
            (finished if prefix[-1] == eos else beams).append((prefix, score))
            if len(beams) == beam_size:
                break
        if not beams:
            break
    finished.extend(beams)
    return max(finished, key=lambda c: c[1] / (len(c[0]) ** alpha))
```

Beam search was the workhorse of neural machine translation and summarization, where the output is tightly determined by the input and "the most probable output" is a reasonable target. For open-ended generation it works poorly. It amplifies exactly the problems of greedy decoding: high-likelihood text is generic and repetitive, and larger beams can make this worse rather than better. Meister et al. argued that beam search works well in translation *because* of the specific biases it introduces, not because it finds the most probable sequence. Beam search also costs roughly $B$ times the compute and $B$ times the KV-cache memory of greedy decoding (though hypotheses share a common prefix, which systems like PagedAttention in Section 4 exploit).

As a result, chat assistants almost always use sampling, not beam search. Beam-style search reappears in Section 9, but over reasoning *steps* scored by a verifier rather than over individual tokens scored by likelihood.

## Constrained decoding: JSON, grammars, and tool calls

Many applications need output in a strict format: a JSON object matching a schema, a SQL query, a call to a function with typed arguments, a choice from a fixed list of labels. Prompting the model to follow the format usually works, but "usually" is not good enough for a program that will parse the output. *Constrained decoding* guarantees the format by changing which tokens the model is allowed to choose.

### Logit masking

The core mechanism is simple. At each step, compute the set $\mathcal{A}_t \subseteq \{1, \dots, V\}$ of tokens that keep the output a valid prefix of *some* string in the target language, and mask the rest:

$$
z_i' = \begin{cases} z_i & i \in \mathcal{A}_t, \\ -\infty & i \notin \mathcal{A}_t. \end{cases}
$$

Then apply any decoding strategy (greedy, sampling) to the masked logits. The model still chooses *which* valid continuation to produce; the mask only removes invalid ones. For a classification task with labels `positive`, `negative`, and `neutral`, the mask at the first step allows only the tokens that begin one of those labels.

### From schemas to automata

The hard part is computing $\mathcal{A}_t$ quickly for rich formats. Two classes of formats cover most needs:

- **Regular languages.** Many formats, including JSON objects with a fixed schema (with bounded nesting), dates, phone numbers, and enumerations, can be described by a regular expression. A regular expression compiles to a finite-state machine (FSM). The decoder tracks the FSM state as tokens are generated. Willard and Louf showed how to make this efficient for LLMs: because the vocabulary is fixed, one can precompute, for every FSM state, which vocabulary tokens lead to a valid next state. At generation time, finding $\mathcal{A}_t$ is then a lookup rather than a scan over the whole vocabulary. This is the approach behind the Outlines library.
- **Context-free grammars.** Arbitrary nested JSON, programming languages, and SQL need a context-free grammar (CFG), which requires a stack (a pushdown automaton) rather than a finite set of states. Grammar-constrained decoding tracks the parser state and allows tokens that the parser can accept. llama.cpp, for example, supports grammars written in its GBNF notation, and several serving engines accept JSON Schema and compile it into a grammar or automaton internally.

A complication runs through all of this: the automaton is defined over *characters*, but the model emits *tokens*, and a single token may span several grammar symbols (for example `"},{"`). The mask must therefore check, for each token, whether its entire character string can be consumed from the current state. Precomputing these token-level transitions, and handling the case where a token's string straddles several states, is where most of the engineering effort in constrained-decoding libraries goes.

### Tool calls

Tool or function calling (discussed as a post-training behavior in Chapter 8) is usually implemented as structured output. The model is trained to emit a special marker followed by a function name and a JSON argument object. Many serving systems combine this training with constrained decoding: once the model emits the tool-call marker, the decoder switches to a grammar derived from the declared tool signatures, so that the function name is one of the declared tools and the arguments match their schemas.

### Costs and caveats

Constrained decoding guarantees *syntactic* validity, not correctness: the model can still fill a well-formed JSON field with a wrong value. It also changes the distribution the model samples from. If the model's preferred continuation is invalid, masking forces it onto a path it considered less likely, and a model that "wanted" to write a sentence of explanation before the JSON may produce worse JSON when forced to start with `{`. A practical pattern is to describe the format in the prompt as well, so that the constraint rarely has to override the model, and to let the model reason in free text before a constrained final answer when the task needs reasoning. Computing masks adds some per-token CPU work; good implementations overlap it with the GPU forward pass.

## A note on determinism

Setting temperature to 0 is often assumed to make outputs reproducible. In practice, the same prompt can still produce different outputs across runs. Floating-point addition is not associative, and high-performance kernels may sum numbers in different orders depending on the batch size, the other requests batched alongside yours, or the hardware. A tiny change in a logit can flip an argmax when two tokens are nearly tied, and from that point the generations diverge. Fully deterministic inference is possible but requires deliberately choosing batch-invariant kernels, usually at some cost in speed. When you evaluate a model (Chapter 12), report the decoding settings and do not assume that greedy decoding makes results exactly repeatable.

## Key takeaways

- A language model outputs logits; a decoding loop turns them into text one token at a time, and each token requires another forward pass.
- Greedy decoding is deterministic and good for short factual answers but tends to be repetitive on open-ended text; sampling adds variety but needs its tail controlled.
- Temperature rescales logits; top-k keeps a fixed number of tokens; top-p keeps the smallest set covering probability $p$ and adapts to the model's confidence.
- Repetition, frequency, and presence penalties are blunt heuristics; stop tokens, stop strings, and length caps end generation, and stop strings must be matched on text, not tokens.
- Beam search maximizes approximate sequence likelihood and suits translation-like tasks, but not open-ended chat.
- Constrained decoding masks invalid tokens using finite-state machines or grammars, guaranteeing valid JSON, grammar-conforming output, and well-formed tool calls, though not correct content.

## Further reading

Fan, Angela, et al. "Hierarchical Neural Story Generation." In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics*, 2018. https://arxiv.org/abs/1805.04833.

Holtzman, Ari, et al. "The Curious Case of Neural Text Degeneration." In *International Conference on Learning Representations*, 2020. https://arxiv.org/abs/1904.09751.

Keskar, Nitish Shirish, et al. "CTRL: A Conditional Transformer Language Model for Controllable Generation." arXiv preprint arXiv:1909.05858, 2019. https://arxiv.org/abs/1909.05858.

Meister, Clara, et al. "If Beam Search Is the Answer, What Was the Question?" In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*, 2020. https://arxiv.org/abs/2010.02650.

Nguyen, Minh Nhat, et al. "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs." arXiv preprint arXiv:2407.01082, 2024. https://arxiv.org/abs/2407.01082.

Willard, Brandon T., et al. "Efficient Guided Generation for Large Language Models." arXiv preprint arXiv:2307.09702, 2023. https://arxiv.org/abs/2307.09702.
