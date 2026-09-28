# 5.5 The Unigram Language Model and SentencePiece

BPE and WordPiece build a vocabulary from the bottom up, one merge at a time. The **unigram language model** tokenizer, introduced by Kudo (2018), works in the opposite direction. It starts with a large set of candidate pieces, assigns each a probability, and prunes the candidates that contribute least until the vocabulary is the desired size. Because it is a probabilistic model, it does not just give one segmentation of a word; it gives a probability to every possible segmentation, which makes it natural to choose the most likely one or to sample.

The unigram tokenizer is usually used through **SentencePiece** (Kudo and Richardson 2018), a library that also implements BPE and that changed how tokenizers handle whitespace. This section derives the unigram model, implements its encoder with the Viterbi algorithm, explains how it is trained, and then covers SentencePiece, including the byte fallback that lets it handle any input.

## The model

A unigram language model treats a tokenized text as a sequence of tokens drawn independently from a fixed distribution over the vocabulary. Each vocabulary entry $`x`$ has a probability $`p(x)`$, with the probabilities summing to 1. The probability of a particular segmentation $`\mathbf{x} = (x_1, \dots, x_M)`$ is the product of its tokens' probabilities:

```math
P(\mathbf{x}) = \prod_{i=1}^{M} p(x_i).
```

A given string $`X`$ can usually be segmented in many ways. Writing $`S(X)`$ for the set of all segmentations of $`X`$ into vocabulary pieces, the tokenizer encodes $`X`$ as the most probable one:

```math
\mathbf{x}^* = \arg\max_{\mathbf{x} \in S(X)} P(\mathbf{x}).
```

Because every token multiplies in a probability less than 1, segmentations with fewer tokens tend to be more probable, but a split into two common pieces can beat a single rare piece.

The model is called "unigram" because it ignores context: the probability of a token does not depend on its neighbors. That is a crude model of language, but it is only used to choose segmentations, not to generate text.

## Encoding with the Viterbi algorithm

The number of segmentations grows exponentially with the length of the string, so we cannot enumerate them all. But the best segmentation has a structure that dynamic programming can exploit: the best segmentation of the first $`j`$ characters must end with some piece $`X[i{:}j]`$, preceded by the best segmentation of the first $`i`$ characters. Filling in a table of the best score for each prefix, from left to right, and then following back-pointers gives the **Viterbi algorithm**. Working in log probabilities turns the product into a sum, and the whole algorithm is about fifteen lines of Python (Appendix A.1). The running time is proportional to the length of the text times the maximum piece length.

A toy example makes this concrete. Take a vocabulary of twelve pieces whose probabilities sum to 1: "h", "u", "g", and "gs", "hu", "ugs", and "b" with probability 0.05 each; "s", "ug", and "un" with 0.10 each; "hug" with 0.15; and "bun" with 0.20. The word "hugs" can be segmented in seven ways. Viterbi picks "hug" + "s", with probability $`0.15 \times 0.10 = 0.015`$. Normalizing over all seven segmentations shows what it chose among:

| Segmentation | Probability | Share of the seven |
|---|---|---|
| hug · s | 0.015000 | 0.718 |
| h · ugs | 0.002500 | 0.120 |
| hu · gs | 0.002500 | 0.120 |
| h · ug · s | 0.000500 | 0.024 |
| hu · g · s | 0.000250 | 0.012 |
| h · u · gs | 0.000125 | 0.006 |
| h · u · g · s | 0.000013 | 0.001 |

Viterbi's answer "hug" + "s" has 71.8 percent of the probability mass among the seven segmentations. The other segmentations are less likely but not impossible, and that is what subword regularization takes advantage of.

## Training with expectation-maximization

Training has to choose both the vocabulary and the probabilities. Kudo (2018) proposed the following procedure:

1. **Seed a large vocabulary.** Start with all characters plus a large set of frequent substrings of the corpus, much larger than the target size. SentencePiece finds frequent substrings efficiently with a suffix array.
2. **Estimate probabilities with EM.** Given the current vocabulary, the true segmentation of each word is unknown, so it is treated as a hidden variable. The expectation-maximization (EM) algorithm alternates between computing, for every word, how likely each segmentation is under the current probabilities (the E-step, done efficiently with a forward-backward pass over the same lattice that Viterbi uses), and re-estimating each piece's probability from its expected count (the M-step).
3. **Score each piece by its loss.** For each piece, estimate how much the total log-likelihood of the corpus would decrease if that piece were removed from the vocabulary and the words that used it had to be segmented with other pieces.
4. **Prune.** Keep the pieces with the largest losses, discarding a fixed fraction of the rest (for example the bottom 20 percent). Single characters are always kept, so that every word remains segmentable.
5. **Repeat** steps 2 to 4 until the vocabulary reaches the target size.

The contrast with BPE is worth spelling out. BPE decides greedily, one merge at a time, and never revisits a decision. The unigram trainer starts from many candidates and removes the ones that the whole corpus needs least, re-estimating everything after each round. The result is a vocabulary chosen with a global objective, the likelihood of the corpus, rather than a sequence of local frequency decisions.

## Subword regularization

Because the unigram model defines a distribution over segmentations, we can **sample** a segmentation instead of always taking the best one. Kudo (2018) called this **subword regularization**: during training of the downstream model, each time a sentence is used, it is segmented by sampling from

```math
P(\mathbf{x} \mid X) \propto P(\mathbf{x})^{\alpha},
```

where the smoothing parameter $`\alpha`$ controls how peaked the distribution is. Large $`\alpha`$ concentrates almost all the mass on the Viterbi segmentation; small $`\alpha`$ makes the distribution flatter, so unusual segmentations appear more often. Sampling can be restricted to the $`\ell`$ best segmentations, or done over all segmentations using a forward-filtering, backward-sampling pass over the lattice. As with BPE-dropout (Section 3), the goal is to make the model robust to segmentation variation, and Kudo reported consistent improvements in machine translation, especially in low-resource and out-of-domain settings. At test time the Viterbi segmentation is used.

## SentencePiece

**SentencePiece** (Kudo and Richardson 2018) is a library that implements both the unigram model and BPE. Its main design idea is to treat the input as a raw stream of Unicode characters, with no separate, language-specific pre-tokenization step. Spaces are not boundaries that the tokenizer removes; they are ordinary characters that the tokenizer keeps, displayed as the symbol `▁` (U+2581, "lower one eighth block"). "Hello world" is treated as `▁Hello▁world`, and a piece such as `▁world` means "world at the start of a word". Because the space is part of the text being segmented, decoding is simple: concatenate the pieces and replace `▁` with a space.

This matters for languages that do not use spaces between words, such as Chinese, Japanese, and Thai. SentencePiece needs no word segmenter for them: it simply learns pieces from the character stream.

Training a SentencePiece model takes a single function call (Appendix A.2). We trained both a unigram and a BPE model with 2,000 pieces each on tiny Shakespeare, with byte fallback enabled, and encoded "Wherefore art thou Romeo? 你好":

| Model | Pieces |
|---|---|
| Unigram | `▁Where` `for` `e` `▁art` `▁thou` `▁Romeo` `?` `▁` `<0xE4>` `<0xBD>` `<0xA0>` `<0xE5>` `<0xA5>` `<0xBD>` |
| BPE | `▁Where` `fore` `▁art` `▁thou` `▁Romeo` `?` `▁` `<0xE4>` `<0xBD>` `<0xA0>` `<0xE5>` `<0xA5>` `<0xBD>` |

The two algorithms agree on most of the sentence and differ on "Wherefore": the unigram model prefers "▁Where" + "for" + "e", while BPE learned "fore" as a piece. Both decode back to the input exactly.

## Byte fallback

The Chinese characters "你好" never appear in Shakespeare, so they are not in either vocabulary. Without further help, SentencePiece would map each to the unknown piece `<unk>`, losing the text. With `byte_fallback=True`, the vocabulary also includes 256 byte pieces, written `<0x00>` to `<0xFF>`, and any character not in the vocabulary is encoded as its UTF-8 bytes. "你" is the three bytes E4 BD A0, and those are exactly the pieces in the table above. The vocabulary begins with the three special pieces `<unk>`, `<s>`, and `</s>`, followed by the byte pieces `<0x00>`, `<0x01>`, and so on.

Byte fallback gives SentencePiece models the same guarantee as byte-level BPE: nothing is ever unknown. The original LLaMA models, for example, used SentencePiece's BPE with byte fallback and additionally split all numbers into individual digits (Touvron et al. 2023). The two designs differ in their base units. Byte-level BPE works on bytes everywhere, so a common character that has not been merged may be split across several tokens. SentencePiece with byte fallback works on characters and uses bytes only for characters outside its vocabulary.

## Normalization and reversibility

SentencePiece is often described as fully reversible, and it is, but only with respect to the *normalized* text. By default it applies a normalization rule called `nmt_nfkc`, based on Unicode NFKC normalization (Section 6), and it removes leading, trailing, and repeated whitespace. Encoding and decoding with the default settings therefore changes some inputs. For example, the string `"  two  spaces\tand a tab ﬁ"`, with leading and doubled spaces, a tab, and the "ﬁ" ligature, comes back as `"two spaces and a tab fi"` (Appendix A.3).

The doubled spaces are collapsed, the tab has become a space, and the "ﬁ" ligature (a single character) has been replaced by the two letters "fi". For natural-language text these changes are often harmless or even helpful. For code, where whitespace carries meaning, they are not. Training with `normalization_rule_name="identity"` and `remove_extra_whitespaces=False` turns them off. A model trained that way keeps every space as its own `▁` piece where needed, encodes the tab (which never occurs in Shakespeare) as its byte piece `<0x09>`, encodes the ligature as its three bytes `<0xEF>` `<0xAC>` `<0x81>`, and reproduces the test string exactly.

## Sampling segmentations in SentencePiece

SentencePiece exposes subword regularization directly. With `enable_sampling=True`, each call to `encode` samples a segmentation; `alpha` is the smoothing parameter from the formula above, and `nbest_size=-1` samples from all segmentations. Four samples for "Wherefore art thou Romeo?" with $`\alpha = 0.1`$ looked like this in one run (Appendix A.4; the samples change from run to run):

- `▁` `W` `he` `re` `for` `e` `▁` `art` `▁thou` `▁Romeo` `?`
- `▁Where` `f` `o` `re` `▁art` `▁thou` `▁` `R` `om` `e` `o` `?`
- `▁W` `he` `re` `f` `or` `e` `▁a` `rt` `▁thou` `▁Romeo` `?`
- `▁W` `h` `er` `e` `f` `or` `e` `▁` `art` `▁t` `h` `o` `u` `▁Romeo` `?`

At this setting the samples are far from the Viterbi segmentation, which is useful to see but more variation than one would typically train with. In our runs, $`\alpha = 0.5`$ returned the Viterbi segmentation in most samples.

## Comparing the three algorithms

| | BPE | WordPiece | Unigram LM |
|---|---|---|---|
| Training direction | Bottom up: merge pairs | Bottom up: merge pairs | Top down: prune a large candidate set |
| Selection criterion | Most frequent pair | Largest likelihood gain (pair cohesion) | Smallest loss in corpus likelihood when removed |
| What is stored | Vocabulary and ordered merges | Vocabulary | Vocabulary with a probability per piece |
| Encoding | Apply merges in rank order | Greedy longest match first | Viterbi (most probable segmentation), or sampling |
| Word boundaries | Space attached to the following token (GPT-2 style) or `▁` (SentencePiece) | `##` on continuation pieces; spaces discarded | `▁` for spaces (SentencePiece) |
| Unknown input | None with byte-level BPE or byte fallback | `[UNK]` | None with byte fallback |
| Typical users | GPT-2 and later GPT models, LLaMA, many open LLMs | BERT and derived encoders | T5 and other SentencePiece-based models |

In practice the differences in model quality between the algorithms are usually small compared with the effect of the tokenizer's training data, vocabulary size, and pre-tokenization rules. Bostrom and Durrett (2020) compared BPE and unigram tokenization for pretraining and found that unigram segmentations aligned more closely with morphology and that models pretrained with them matched or outperformed BPE on downstream tasks in English and Japanese, but byte-level BPE remains the most common choice for large generative models.

## Key takeaways

- The unigram LM tokenizer assigns a probability to every vocabulary piece and scores a segmentation by the product of its pieces' probabilities.
- Encoding picks the most probable segmentation with the Viterbi algorithm, in time linear in the text length for a bounded piece length.
- Training starts from a large candidate vocabulary and alternates EM probability estimation with pruning of the pieces whose removal costs the corpus the least likelihood.
- Subword regularization samples segmentations during model training, with a smoothing parameter $`\alpha`$ controlling how far samples stray from the best one.
- SentencePiece treats text as a raw character stream with spaces as the visible symbol `▁`, implements both unigram and BPE, and uses byte fallback so nothing is unknown.
- SentencePiece's default normalization collapses whitespace and applies NFKC, so exact round-tripping requires the identity normalization rule.

## Further reading

Bostrom, Kaj, et al. "Byte Pair Encoding Is Suboptimal for Language Model Pretraining." In *Findings of the Association for Computational Linguistics: EMNLP 2020*, 2020. https://arxiv.org/abs/2004.03720.

Kudo, Taku. "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates." In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics*, 66–75, 2018. https://arxiv.org/abs/1804.10959.

Kudo, Taku, et al. "SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing." In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 66–71, 2018. https://arxiv.org/abs/1808.06226.

Touvron, Hugo, et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971, 2023. https://arxiv.org/abs/2302.13971.

## Appendix: Code for Section 5.5

These listings reproduce the examples in this section. Run them in order in one Python session; they need the `sentencepiece` package, and the second listing downloads tiny Shakespeare.

### A.1 Viterbi segmentation under a unigram model

Find the most probable segmentation with dynamic programming, then enumerate all segmentations of a short word to check it.

```python
import math

def viterbi(text, logp, max_len=10):
    """Most probable segmentation of `text` under a unigram model."""
    n = len(text)
    best = [0.0] + [-math.inf] * n     # best[j]: best log-prob of text[:j]
    back = [0] * (n + 1)
    for end in range(1, n + 1):
        for start in range(max(0, end - max_len), end):
            piece = text[start:end]
            if piece in logp and best[start] + logp[piece] > best[end]:
                best[end] = best[start] + logp[piece]
                back[end] = start
    pieces, end = [], n
    while end > 0:
        pieces.append(text[back[end]:end])
        end = back[end]
    return pieces[::-1], best[n]
```

```python
probs = {"h": .05, "u": .05, "g": .05, "s": .10, "hu": .05, "ug": .10,
         "gs": .05, "hug": .15, "ugs": .05, "b": .05, "un": .10, "bun": .20}
logp = {k: math.log(v) for k, v in probs.items()}
pieces, lp = viterbi("hugs", logp)
print(pieces, f"{math.exp(lp):.6f}")
```

Output:

```
['hug', 's'] 0.015000
```

```python
def all_segmentations(text, logp):
    if not text:
        yield [], 0.0
        return
    for i in range(1, len(text) + 1):
        if text[:i] in logp:
            for rest, lp in all_segmentations(text[i:], logp):
                yield [text[:i]] + rest, logp[text[:i]] + lp

segs = sorted(all_segmentations("hugs", logp), key=lambda s: -s[1])
Z = sum(math.exp(lp) for _, lp in segs)
for s, lp in segs:
    print(s, f"{math.exp(lp):.6f}", f"{math.exp(lp) / Z:.3f}")
```

Output:

```
['hug', 's'] 0.015000 0.718
['h', 'ugs'] 0.002500 0.120
['hu', 'gs'] 0.002500 0.120
['h', 'ug', 's'] 0.000500 0.024
['hu', 'g', 's'] 0.000250 0.012
['h', 'u', 'gs'] 0.000125 0.006
['h', 'u', 'g', 's'] 0.000013 0.001
```

### A.2 Training SentencePiece models

Train a unigram and a BPE model on tiny Shakespeare and encode a line that includes characters outside the vocabulary; then list the first vocabulary entries.

```python
import os, urllib.request
import sentencepiece as spm

if not os.path.exists("shakespeare.txt"):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "shakespeare.txt")

for model_type in ["unigram", "bpe"]:
    spm.SentencePieceTrainer.train(
        input="shakespeare.txt", model_prefix=f"shakes_{model_type}",
        vocab_size=2000, model_type=model_type, byte_fallback=True,
        character_coverage=1.0, minloglevel=2)
    sp = spm.SentencePieceProcessor(model_file=f"shakes_{model_type}.model")
    s = "Wherefore art thou Romeo? 你好"
    ids = sp.encode(s)
    print(model_type, sp.encode(s, out_type=str), sp.decode(ids) == s)
```

Output:

```
unigram ['▁Where', 'for', 'e', '▁art', '▁thou', '▁Romeo', '?', '▁', '<0xE4>', '<0xBD>', '<0xA0>', '<0xE5>', '<0xA5>', '<0xBD>'] True
bpe ['▁Where', 'fore', '▁art', '▁thou', '▁Romeo', '?', '▁', '<0xE4>', '<0xBD>', '<0xA0>', '<0xE5>', '<0xA5>', '<0xBD>'] True
```

```python
sp = spm.SentencePieceProcessor(model_file="shakes_unigram.model")
print([sp.id_to_piece(i) for i in range(8)])
```

Output:

```
['<unk>', '<s>', '</s>', '<0x00>', '<0x01>', '<0x02>', '<0x03>', '<0x04>']
```

### A.3 Normalization and exact round-tripping

Default normalization changes whitespace and ligatures; the identity rule keeps the text exactly.

```python
s = "  two  spaces\tand a tab ﬁ"
print(repr(sp.decode(sp.encode(s))))
```

Output:

```
'two spaces and a tab fi'
```

```python
spm.SentencePieceTrainer.train(
    input="shakespeare.txt", model_prefix="shakes_exact",
    vocab_size=2000, model_type="unigram", byte_fallback=True,
    character_coverage=1.0, normalization_rule_name="identity",
    remove_extra_whitespaces=False, minloglevel=2)
sp_exact = spm.SentencePieceProcessor(model_file="shakes_exact.model")
print(sp_exact.encode(s, out_type=str))
print(sp_exact.decode(sp_exact.encode(s)) == s)
```

Output:

```
['▁', '▁', '▁two', '▁', '▁sp', 'ace', 's', '<0x09>', 'and', '▁a', '▁ta', 'b', '▁', '<0xEF>', '<0xAC>', '<0x81>']
True
```

### A.4 Sampling segmentations

Sample segmentations for subword regularization.

```python
# output varies from run to run
for _ in range(4):
    print(sp.encode("Wherefore art thou Romeo?", out_type=str,
                    enable_sampling=True, alpha=0.1, nbest_size=-1))
```

One run printed (the samples change from run to run):

```
['▁', 'W', 'he', 're', 'for', 'e', '▁', 'art', '▁thou', '▁Romeo', '?']
['▁Where', 'f', 'o', 're', '▁art', '▁thou', '▁', 'R', 'om', 'e', 'o', '?']
['▁W', 'he', 're', 'f', 'or', 'e', '▁a', 'rt', '▁thou', '▁Romeo', '?']
['▁W', 'h', 'er', 'e', 'f', 'or', 'e', '▁', 'art', '▁t', 'h', 'o', 'u', '▁Romeo', '?']
```
