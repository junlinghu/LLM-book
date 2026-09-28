# 12.2 Intrinsic Metrics

An *intrinsic* metric measures the language model as a language model: how well it predicts text, independent of any particular downstream task. An *extrinsic* metric measures how well the model performs a task that people care about, such as answering questions or writing code. This section covers the most important intrinsic metric, perplexity, and then the problem that undermines intrinsic and extrinsic metrics alike: making sure the test data really is held out.

## Cross-entropy and next-token loss

Recall from Chapter 7 that a GPT-style model defines a probability distribution over the next token given the previous ones, $p_\theta(x_t \mid x_{<t})$. For a sequence of $N$ tokens $x_1, \dots, x_N$, the model assigns the whole sequence the probability

$$
p_\theta(x_1, \dots, x_N) = \prod_{t=1}^{N} p_\theta(x_t \mid x_{<t}).
$$

The average negative log-likelihood per token is

$$
\mathcal{L} = -\frac{1}{N} \sum_{t=1}^{N} \log p_\theta(x_t \mid x_{<t}).
$$

This is exactly the pretraining loss. It is also the *cross-entropy* between the empirical distribution of the text and the model's distribution. If we use natural logarithms, $\mathcal{L}$ is measured in *nats* per token; with base-2 logarithms, it is measured in *bits* per token.

The information-theoretic reading is useful. By Shannon's source coding theorem, a model that assigns probability $p$ to an event can be turned (with arithmetic coding) into a compressor that spends about $-\log_2 p$ bits encoding it. So a model with a cross-entropy of 3 bits per token could compress text to about 3 bits per token on average. Better language models are better compressors. This connection is why some researchers report results as *bits per byte* (see below) and why "compression is intelligence" is a recurring slogan in the field.

## Perplexity

Perplexity is simply the exponentiated cross-entropy:

$$
\text{PPL} = \exp(\mathcal{L}) = \exp\!\left(-\frac{1}{N} \sum_{t=1}^{N} \log p_\theta(x_t \mid x_{<t})\right) = \left(\prod_{t=1}^N p_\theta(x_t \mid x_{<t})\right)^{-1/N}.
$$

The last form shows that perplexity is the inverse of the geometric mean of the per-token probabilities. It has an intuitive interpretation: a perplexity of $k$ means the model is, on average, as uncertain as if it were choosing uniformly among $k$ equally likely tokens at every step. A model that predicts every token with certainty has perplexity 1. A model that guesses uniformly over a vocabulary of $V$ tokens has perplexity $V$.

### A worked example

Suppose a model reads the four-token sequence "the cat sat down" and assigns these conditional probabilities:

| Position | Token | $p_\theta(x_t \mid x_{<t})$ | $-\ln p$ |
|---|---|---|---|
| 1 | the | 0.20 | 1.609 |
| 2 | cat | 0.05 | 2.996 |
| 3 | sat | 0.30 | 1.204 |
| 4 | down | 0.40 | 0.916 |

The average negative log-likelihood is $(1.609 + 2.996 + 1.204 + 0.916)/4 = 6.725 / 4 \approx 1.681$ nats per token, so the perplexity is $e^{1.681} \approx 5.37$. The model is about as uncertain as if it were picking among five or six equally likely words at each step. Notice how the single hard token, "cat," contributes almost half of the total loss. Perplexity is dominated by surprising tokens.

### Computing perplexity in practice

With the Hugging Face `transformers` library, computing perplexity takes a few lines. The model returns the mean cross-entropy loss when you pass the input IDs as labels; the library shifts the labels internally so that each position predicts the next token.

```python
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def perplexity(text: str, model_name: str = "gpt2", window: int = 512, stride: int = 256) -> float:
    """Perplexity of `text` using a sliding window so long texts fit in context."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    ids = tok(text, return_tensors="pt").input_ids[0]

    total_nll, total_tokens, prev_end = 0.0, 0, 0
    for start in range(0, len(ids), stride):
        end = min(start + window, len(ids))
        chunk = ids[start:end].unsqueeze(0)
        labels = chunk.clone()
        # Only score tokens not already scored by the previous window.
        n_new = end - prev_end
        labels[:, :-n_new] = -100          # -100 = ignore in the loss
        with torch.no_grad():
            loss = model(chunk, labels=labels).loss
        scored = (labels[:, 1:] != -100).sum().item()   # first position is never scored
        total_nll += loss.item() * scored
        total_tokens += scored
        prev_end = end
        if end == len(ids):
            break
    return math.exp(total_nll / total_tokens)
```

Two details matter. First, models have a finite context window, so long documents must be split. Splitting into disjoint chunks makes the first tokens of each chunk artificially hard (they have no context), which inflates perplexity. A sliding window with overlap, where each window only scores the tokens not already scored, gives every token some left context. Second, you should report exactly which tokens were scored and how, because these choices change the number.

## What perplexity measures and what it misses

Perplexity is valuable because it is cheap, deterministic, needs no labels beyond raw text, and correlates with many downstream abilities when comparing models trained in similar ways. It is the standard metric for:

- Comparing pretraining runs, architectures, or data mixtures that share a tokenizer.
- Monitoring training for divergence or loss spikes.
- Fitting scaling laws that predict how loss falls with compute, data, and parameters.
- Measuring domain fit, for example whether a model continued on medical text now predicts medical text better.

But it has sharp limitations.

**Perplexity depends on the tokenizer.** Perplexity is per *token*, and different tokenizers split the same text into different numbers of tokens. A tokenizer with a larger vocabulary produces fewer, "bigger" tokens, each of which is harder to predict. Comparing perplexities across models with different tokenizers is therefore meaningless. The standard fix is to normalize by a tokenizer-independent unit. *Bits per byte* (BPB) divides the total negative log-likelihood in bits by the number of UTF-8 bytes in the text:

$$
\text{BPB} = \frac{-\sum_{t=1}^{N} \log_2 p_\theta(x_t \mid x_{<t})}{\text{number of bytes in the text}}.
$$

Because the numerator is the total code length of the whole text, and the denominator depends only on the text, BPB can be compared across tokenizers. Bits per character and per-word perplexity are similar ideas.

**Perplexity depends on the evaluation text.** A model trained mostly on English web text will have low perplexity on news articles and high perplexity on Python code or on Swahili. A single perplexity number without a named evaluation corpus is almost meaningless. Benchmark suites such as Paloma report perplexity separately across many domains for this reason.

**Perplexity does not measure truth.** A model can assign high probability to a fluent falsehood that appears often on the web. TruthfulQA (Section 4) was built around this observation: its authors found that larger models were often *less* truthful on questions designed to elicit common misconceptions, because they imitated human falsehoods better.

**Perplexity does not measure instruction following, helpfulness, or safety.** These are properties of how a model responds to a user, not of how well it predicts web text. As Section 1 noted, post-training can increase perplexity on generic text while making the model far more useful.

**Perplexity is not well defined for everything we evaluate.** For reasoning models that generate long hidden chains of thought, or for API models that do not expose token probabilities, perplexity may be impossible to compute.

The practical rule: use perplexity (or BPB) to compare *base* models on the *same* evaluation corpus, ideally with the *same* tokenizer. Use task metrics for everything else.

### Likelihood-based scoring of multiple-choice tasks

A related use of token probabilities is scoring multiple-choice benchmarks without generating text. For each answer option $a_i$ to a question $q$, compute the log-likelihood $\log p_\theta(a_i \mid q)$ and pick the highest. Because longer options accumulate more negative log-probability, harnesses often normalize by the number of tokens or bytes in the option. This "cloze-style" scoring is how many benchmarks were originally evaluated, and it is still useful for small base models that cannot yet follow the instruction "Answer with A, B, C, or D." The choice between length-normalized likelihood, unnormalized likelihood, and generating a letter can change scores by several points, which is one reason the same model can have different reported scores on the same benchmark (Section 3).

## Held-out data: contamination, leakage, and train/test overlap

Everything above assumes the evaluation text is *held out*: the model never saw it during training. For classic ML datasets that assumption was easy to enforce. For LLMs trained on trillions of tokens scraped from the web, it is very hard.

### What contamination is

*Data contamination* (also called test-set leakage or train/test overlap) occurs when evaluation examples, or information that makes them easier, appear in the training data. It comes in several forms:

- **Verbatim contamination.** The exact test questions, often with answers, appear in the pretraining corpus. This happens easily: benchmark datasets are posted on GitHub, discussed in blog posts, and quoted in papers.
- **Near-duplicate contamination.** Paraphrased, translated, or reformatted versions of test items appear in training data.
- **Answer or solution leakage.** The question may not appear, but its solution does, such as the fix for a GitHub issue used in a coding benchmark.
- **Fine-tuning contamination.** Benchmark-like data, sometimes generated by other LLMs that had themselves seen the benchmark, is included in SFT or RL data.
- **Indirect (decision) leakage.** Nothing from the test set enters the training data, but developers repeatedly choose data mixes, hyperparameters, or prompts by looking at test scores. This is the adversarial and regressional Goodhart effect from Section 1.

Contamination matters because it turns a test of generalization into a test of memory. A model that has memorized benchmark answers can score far above its real ability.

### Evidence that contamination is real

Several careful studies show measurable effects. Zhang et al. commissioned GSM1k, a new set of grade-school math problems written to match the style and difficulty of GSM8K but guaranteed never to have been published. Evaluating many models on both, they observed accuracy drops of up to 8 percent on GSM1k for some model families, with a positive relationship between how likely a model was to generate GSM8K examples and how much worse it did on the fresh problems. They also found that many frontier models showed minimal signs of overfitting, which is an important nuance: contamination is common but its impact varies widely.

In coding, OpenAI reported in February 2026 that every frontier model it tested could reproduce the original human-written fix (the "gold patch") or verbatim problem-statement details for some tasks in SWE-bench Verified, and it stopped reporting that benchmark as a result. Section 3 tells that story in more detail. Because SWE-bench tasks come from public open-source repositories, their solutions were almost certain to be in training data sooner or later.

### Detecting contamination: n-gram overlap

The oldest and most common detection method checks for overlapping n-grams between test examples and the training corpus. The GPT-3 paper, for example, analyzed contamination by looking for 13-gram overlaps between benchmark examples and its training data, then compared performance on "clean" and "dirty" subsets.

Formally, let $G_n(x)$ be the set of $n$-grams (sequences of $n$ consecutive tokens or words) in a test example $x$, and let $G_n(D)$ be the set of $n$-grams in the training corpus $D$. One common overlap score is the fraction of the example's $n$-grams that also occur in the corpus:

$$
\text{overlap}_n(x, D) = \frac{|G_n(x) \cap G_n(D)|}{|G_n(x)|}.
$$

An example is flagged as contaminated if the overlap exceeds a threshold (for instance, any single long n-gram match, or more than half of its n-grams). The choice of $n$ is a trade-off. Small $n$ (say 5) produces false positives, because common phrases like "which of the following is" appear everywhere. Large $n$ (say 50) produces false negatives, because a paraphrase or a change in formatting breaks every match.

Here is a minimal implementation. Real systems use hashing, Bloom filters, or suffix arrays to handle trillion-token corpora, but the logic is the same.

```python
import re

def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

def build_index(corpus_docs, n: int) -> set[tuple[str, ...]]:
    index = set()
    for doc in corpus_docs:
        index |= ngrams(doc, n)
    return index

def overlap_score(example: str, index: set, n: int) -> float:
    grams = ngrams(example, n)
    if not grams:
        return 0.0
    return len(grams & index) / len(grams)

corpus = ["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."]
index = build_index(corpus, n=8)
test_item = "Natalia sold clips to 48 of her friends in April, and then she sold half as many in May. How many clips did she sell?"
print(round(overlap_score(test_item, index, n=8), 2))   # a high score flags likely contamination
```

The normalization (lowercasing, stripping punctuation) matters as much as $n$: without it, a trivial change in capitalization or whitespace defeats the check.

### Detecting contamination without the training data

Often you want to test a model whose training data you cannot see. Several methods probe the model itself:

- **Membership inference from token probabilities.** A model tends to assign unusually high probability to text it was trained on. Shi et al. proposed *Min-K% Prob*: compute the log-probability of each token in a candidate text, take the $k$ percent of tokens with the lowest probability, and average them. Unseen text usually contains a few very surprising tokens; memorized text does not. A high Min-K% score suggests the text was in the training data.
- **Guided completion.** Golchin and Surdeanu prompt the model with the dataset name, split, and the first part of a test instance, and check whether it completes the rest verbatim or nearly so. If a model can reproduce the second half of a benchmark question from its first half, it has probably seen it.
- **Exchangeability tests.** Oren et al. observed that benchmark datasets are usually published in a fixed order. If a model has memorized the dataset, it will assign higher likelihood to the canonical ordering of examples than to shuffled orderings. Because this is a statistical test against a clear null hypothesis, it can provide provable guarantees of contamination without access to training data.
- **Performance on perturbed or fresh versions.** If a model does much worse on rephrased questions, changed numbers, or newly written items of the same kind (as in GSM1k), memorization is a likely explanation.

None of these methods is conclusive on its own. Membership inference on individual examples is noisy, and a clean result does not prove absence of contamination. They are best used together.

### Preventing contamination

Because detection is imperfect, benchmark designers and model developers also try to prevent contamination:

- **Canary strings.** BIG-bench and many later benchmarks embed a unique random string (a "canary GUID") in their files and ask developers to filter any document containing it from training data. This only works if developers comply and if copies of the data retain the string.
- **Decontamination filters.** Responsible training pipelines remove documents that have high n-gram overlap with known benchmarks before training. Model reports increasingly describe such filters.
- **Private or held-out test sets.** Some benchmarks keep part of their data secret and evaluate submitted models themselves. Humanity's Last Exam, for instance, maintains a private held-out set in addition to its public questions, and SWE-bench Pro keeps some of its evaluation data confidential.
- **Time-based (live) benchmarks.** LiveCodeBench collects new competitive programming problems continuously and tags each with its release date. You can evaluate a model only on problems published after its training cutoff, which makes contamination much less likely.
- **Dynamic or procedurally generated benchmarks.** Generating fresh instances from templates or programs for every evaluation run makes memorization useless, though it can make items less natural.

### A note on the development set

Contamination has a subtler cousin inside your own project. Chapter 2 introduced the split between training, validation (development), and test data. For LLM work the same discipline applies to *evaluation decisions*. Use a development set for choosing prompts, checkpoints, and data mixtures; touch the test set as rarely as possible; and when you report results, say how many times you looked. If you tuned the prompt on the test set, the test set has become training data.

## Key takeaways

- Cross-entropy (next-token loss) and its exponential, perplexity, measure how well a model predicts text; perplexity is the inverse geometric mean of per-token probabilities.
- Perplexity depends on the tokenizer and on the evaluation corpus; use bits per byte to compare models with different tokenizers, and always name the corpus.
- Perplexity is good for comparing base models and monitoring training, but it does not measure truth, helpfulness, instruction following, or safety.
- Contamination turns a test of generalization into a test of memory. It is common on the web-scale data LLMs train on and has measurably inflated some benchmark scores.
- Detect contamination with n-gram overlap, membership-inference probes (such as Min-K% Prob), guided completion, exchangeability tests, and fresh test sets; prevent it with canaries, decontamination filters, private sets, and live benchmarks.

## Further reading

Brown, Tom B., et al. "Language Models Are Few-Shot Learners." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2005.14165.

Golchin, Shahriar, et al. "Time Travel in LLMs: Tracing Data Contamination in Large Language Models." arXiv preprint arXiv:2308.08493, 2023. https://arxiv.org/abs/2308.08493.

Jain, Naman, et al. "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code." arXiv preprint arXiv:2403.07974, 2024. https://arxiv.org/abs/2403.07974.

Magnusson, Ian, et al. "Paloma: A Benchmark for Evaluating Language Model Fit." arXiv preprint arXiv:2312.10523, 2023. https://arxiv.org/abs/2312.10523.

Oren, Yonatan, et al. "Proving Test Set Contamination in Black Box Language Models." arXiv preprint arXiv:2310.17623, 2023. https://arxiv.org/abs/2310.17623.

Sainz, Oscar, et al. "NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for Each Benchmark." arXiv preprint arXiv:2310.18018, 2023. https://arxiv.org/abs/2310.18018.

Shi, Weijia, et al. "Detecting Pretraining Data from Large Language Models." arXiv preprint arXiv:2310.16789, 2023. https://arxiv.org/abs/2310.16789.

Zhang, Hugh, et al. "A Careful Examination of Large Language Model Performance on Grade School Arithmetic." arXiv preprint arXiv:2405.00332, 2024. https://arxiv.org/abs/2405.00332.
