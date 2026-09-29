# 5.7 Choosing the Vocabulary Size

Every subword algorithm takes one number as input: the target vocabulary size $`V`$. It is one of the few tokenizer hyperparameters that is chosen deliberately for each model, and it affects the model in several directions at once. A larger vocabulary makes sequences shorter, which saves compute and stretches the context window, but it also adds parameters, makes the output softmax more expensive, and leaves more tokens with too little training data. This section works through each effect, measures the compression side of the tradeoff on a real corpus, surveys the sizes used in practice, and explains how to evaluate a tokenizer before committing to it.

## What $`V`$ controls in the model

The vocabulary size appears in exactly two places in the model, the same two places where token IDs enter and leave (Section 1).

**The embedding table** has one row of $`d`$ numbers per token, where $`d`$ is the model width, so it holds $`V \times d`$ parameters. **The output layer** maps a $`d`$-dimensional vector to $`V`$ logits, another $`V \times d`$ weight matrix. Some models **tie** the two matrices, using the same weights for both, which halves the cost; GPT-2 does this.

For small models these matrices are a large share of the total. GPT-2's smallest model has width 768 and a vocabulary of 50,257, so its embedding table alone has

```math
50{,}257 \times 768 = 38{,}597{,}376
```

parameters, nearly a third of the model's roughly 124 million. For large models the share is smaller, because the rest of the network grows faster than the embeddings, but the absolute numbers are large: a vocabulary of 128,000 at width 4,096 is about 524 million parameters per matrix.

**Compute** at the output grows with $`V`$ too. Producing the logits costs about $`2 V d`$ floating-point operations per token (a multiply and an add for each weight), and the softmax and the cross-entropy loss touch all $`V`$ logits. The embedding lookup, by contrast, is nearly free: it copies one row. When the vocabulary is large relative to the model, the output layer can become a noticeable fraction of the cost of each training step and each generated token.

## The benefit: shorter sequences

Against those costs, a larger vocabulary gives shorter sequences. Longer tokens mean fewer positions per document, and fewer positions means:

- **More text per context window.** A model with a fixed context length of $`T`$ tokens can read more words if each token covers more text.
- **Less compute per document.** The cost of the rest of the model grows at least linearly with the number of positions, and attention grows faster than linearly. Shorter sequences are cheaper to train on and to run.
- **Faster generation.** A model generates one token per step, so text that needs fewer tokens is produced in fewer steps.

To see how compression changes with vocabulary size, we can train byte-level BPE tokenizers of increasing size on the first 90 percent of tiny Shakespeare and measure bytes per token on the held-out 10 percent. The Hugging Face `tokenizers` library trains these in seconds (Appendix A.1):

| Vocabulary size | Held-out tokens | Bytes per token | Entries seen fewer than 10 times |
|---|---|---|---|
| 512 | 59,401 | 1.88 | 193 |
| 1,000 | 49,650 | 2.25 | 223 |
| 2,000 | 43,755 | 2.55 | 373 |
| 4,000 | 38,542 | 2.89 | 850 |
| 8,000 | 35,070 | 3.18 | 5,291 |
| 16,000 | 33,720 | 3.31 | 13,917 |

The last column counts the vocabulary entries that occur fewer than ten times when the tokenizer encodes its own training data.

Two patterns stand out. First, **compression has diminishing returns**. Going from 512 to 1,000 entries cuts the held-out token count by 16 percent; going from 8,000 to 16,000, which doubles the vocabulary again, cuts it by less than 4 percent. Each additional merge covers a rarer string than the one before, so it saves fewer tokens. The same curve appears at any scale, which is why vocabulary sizes grow slowly compared with model and data sizes.

Second, **rare tokens multiply**. With 16,000 entries trained on only about a megabyte of text, most of the vocabulary appears fewer than ten times in the tokenizer's own training data. (At 512, most of the 193 rare entries are byte values that never occur in this ASCII-only text.) Many rare entries are intermediate merges. In the 16,000-entry tokenizer, " Glou" never occurs on its own when the training text is encoded: it was needed only as a step toward " Gloucester", which occurs 37 times and absorbs every use of it. The same is true of " notwith" on the way to " notwithstanding". The corpus here is tiny; real tokenizers are trained on far more data, but the same effect appears at their scale.

## The cost of rare tokens

A token that is rare in the training data is a problem for the model, not just the tokenizer. Its embedding row receives a gradient only at positions where the token appears, and its output logit is pushed up only there. A token seen a handful of times ends up with a poorly trained embedding, and the model has little idea when to predict it. Section 8 describes an extreme case, **glitch tokens**: tokens that were frequent in the tokenizer's training data but nearly absent from the model's, with embeddings so poorly trained that they cause bizarre behavior.

So the choice of $`V`$ balances three things: compression (favoring large $`V`$), parameter and output-layer cost (favoring small $`V`$, relative to model size), and enough training examples per token (favoring small $`V`$ relative to the amount of training data).

## Sizes used in practice

| Tokenizer | Algorithm | Vocabulary size |
|---|---|---|
| BERT (`bert-base-uncased`) | WordPiece | 30,522 |
| GPT-2 | Byte-level BPE | 50,257 |
| Original LLaMA | SentencePiece BPE with byte fallback | 32,000 |
| `cl100k_base` (tiktoken) | Byte-level BPE | 100,277 |
| Llama 3 | Byte-level BPE (tiktoken-based) | about 128,000 |
| Qwen2.5 | Byte-level BPE | 151,665 |
| `o200k_base` (tiktoken) | Byte-level BPE | 200,019 |

The `tiktoken` and Hugging Face sizes are the numbers the libraries report, including special tokens. The Llama 3 report describes its vocabulary as 128K tokens, combining 100K tokens from `tiktoken` with 28K additional tokens to better support non-English languages, and it reports that the new tokenizer improved compression on a sample of English data from 3.17 to 3.94 characters per token compared with the Llama 2 tokenizer (Grattafiori et al. 2024).

The trend is clear: vocabularies have grown from tens of thousands to one or two hundred thousand. Two forces drive it. Models are larger, so the embedding matrices are a smaller fraction of the total and the savings from shorter sequences dominate. And models are expected to serve many languages and code well, which requires tokens for many scripts and programming constructs, not just English words.

## Hardware-friendly sizes

GPU matrix-multiplication kernels run fastest when matrix dimensions are multiples of a suitable power of two, such as 64 or 128. A vocabulary of 50,257 is not; 50,257 divided by 64 is about 785.3. It is common to **pad** the embedding and output matrices up to the next convenient size, for example 50,304 (786 × 64), and simply never use the extra rows. The padded rows cost a little memory, while the faster kernels can save noticeably more time than that. The tokenizer itself does not change; only the model's matrices get extra rows.

## Measuring a tokenizer

Before training a model with a tokenizer, measure it on text like the text the model will see. Three measurements are standard:

- **Compression**: bytes (or characters) per token on held-out text, reported separately for each language and domain that matters. A single overall number hides the differences that Section 8 discusses.
- **Fertility**: the average number of tokens per word, for languages where words are well defined. A fertility close to 1 means most words are single tokens.
- **Vocabulary utilization**: how many vocabulary entries actually occur, and how often, in a large sample of the model's training data. Entries that almost never occur are wasted parameters and potential glitch tokens.

Measured on one English sentence and our translations of it into nine other languages, three OpenAI tokenizers give these token counts, with the ratio to the English count for the same tokenizer in parentheses (Appendix A.2):

| Language | UTF-8 bytes | GPT-2 | `cl100k_base` | `o200k_base` |
|---|---|---|---|---|
| English | 66 | 18 (1.0x) | 18 (1.0x) | 18 (1.0x) |
| French | 71 | 29 (1.6x) | 18 (1.0x) | 17 (0.9x) |
| Spanish | 64 | 25 (1.4x) | 19 (1.1x) | 16 (0.9x) |
| German | 71 | 28 (1.6x) | 19 (1.1x) | 14 (0.8x) |
| Russian | 102 | 63 (3.5x) | 30 (1.7x) | 16 (0.9x) |
| Chinese | 51 | 32 (1.8x) | 22 (1.2x) | 13 (0.7x) |
| Japanese | 66 | 29 (1.6x) | 23 (1.3x) | 15 (0.8x) |
| Korean | 70 | 64 (3.6x) | 33 (1.8x) | 19 (1.1x) |
| Hindi | 153 | 89 (4.9x) | 62 (3.4x) | 16 (0.9x) |
| Arabic | 87 | 52 (2.9x) | 34 (1.9x) | 17 (0.9x) |
 One sentence is only an illustration, and the ratios would shift with different text, but the pattern is large enough to trust. GPT-2's tokenizer, trained mostly on English web text, needs nearly five times as many tokens for the Hindi sentence as for the English one. `cl100k_base` narrows the gap for European languages but still needs 3.4 times as many for Hindi. `o200k_base`, with twice the vocabulary, brings every one of these languages close to parity with English. Petrov et al. (2023) measured such disparities systematically on parallel text in many languages and found differences in tokenized length of up to 15 times between languages for some tokenizers, with consequences for cost, latency, and how much content fits in the context window.

## The tokenizer's training data matters as much as its size

The multilingual table shows that vocabulary size alone does not determine compression. What the vocabulary is spent on depends on the text the tokenizer was trained on. A tokenizer trained mostly on English prose spends its merges on English words and compresses other languages, code, and mathematical notation poorly.

Code makes the point sharply. Measured on held-out Shakespeare and on the Python source of the tokenizer we build in Section 10, saved as `bpe.py` (Appendix A.3):

| Text | Bytes | GPT-2 | `cl100k_base` | `o200k_base` |
|---|---|---|---|---|
| Shakespeare (held out) | 111,540 | 3.09 | 3.54 | 3.60 |
| Python (`bpe.py`) | 3,411 | 2.02 | 3.98 | 4.00 |

The numbers are bytes per token. On Shakespeare the three tokenizers differ by less than 17 percent. On Python they differ by a factor of two. Much of the difference is indentation. GPT-2's tokenizer has no tokens for runs of spaces, so a line indented by eight spaces costs it seven single-space tokens before the eighth space attaches to the first word, while `cl100k_base` encodes the first seven spaces as a single token. Section 8 returns to whitespace. The lesson for building a tokenizer is to train it on a sample that matches the mixture of languages, code, and other content the model will be trained on, and then to check compression on each part of that mixture separately.

## Key takeaways

- The vocabulary size $`V`$ sets the size of the embedding table and the output layer ($`V \times d`$ each, unless tied) and the cost of the output softmax.
- Larger vocabularies give shorter sequences, which means more text per context window, less compute per document, and faster generation, but with sharply diminishing returns.
- Large vocabularies contain many rare tokens whose embeddings are poorly trained.
- Typical sizes have grown from about 30,000 to 50,000 in early models to 100,000 to 200,000 in recent ones, driven by larger models and multilingual and code coverage.
- Matrices are often padded to a multiple of 64 or 128 for hardware efficiency.
- Evaluate a tokenizer by compression per language and domain, fertility, and vocabulary utilization, and train it on data that matches the model's training mixture.

## Further reading

Grattafiori, Aaron, et al. "The Llama 3 Herd of Models." arXiv preprint arXiv:2407.21783, 2024. https://arxiv.org/abs/2407.21783.

Petrov, Aleksandar, et al. "Language Model Tokenizers Introduce Unfairness Between Languages." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2305.15425.

Radford, Alec, et al. "Language Models Are Unsupervised Multitask Learners." OpenAI technical report, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

Touvron, Hugo, et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971, 2023. https://arxiv.org/abs/2302.13971.

## Appendix: Code for Section 5.7

These listings reproduce the measurements in this section. Run them in order in one Python session; they need the `tokenizers` and `tiktoken` packages, download tiny Shakespeare, and, for A.3, expect the Section 10 tokenizer saved as `bpe.py` in the working directory.

### A.1 Compression as a function of vocabulary size

Train byte-level BPE tokenizers of six sizes on 90 percent of tiny Shakespeare and measure them on the rest.

Notebook: [5.7-A.1-compression-as-a-function-of-vocabulary-size.ipynb](../../code/05-tokenizer/5.7-A.1-compression-as-a-function-of-vocabulary-size.ipynb)

### A.2 Token counts across languages

Count tokens for one sentence in ten languages with three OpenAI tokenizers.

Notebook: [5.7-A.2-token-counts-across-languages.ipynb](../../code/05-tokenizer/5.7-A.2-token-counts-across-languages.ipynb)

### A.3 Prose versus code

Compare compression on Shakespeare and on Python source.

Notebook: [5.7-A.3-prose-versus-code.ipynb](../../code/05-tokenizer/5.7-A.3-prose-versus-code.ipynb)

