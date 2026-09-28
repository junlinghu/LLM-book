# 5.3 Byte Pair Encoding (BPE)

**Byte pair encoding (BPE)** is the most widely used tokenization algorithm in LLMs. GPT-2 and its successors, the original LLaMA models, and many others use a variant of it. The idea is simple enough to run by hand: start from single characters (or bytes), and repeatedly glue together the pair of adjacent symbols that occurs most often. Each glued pair becomes a new vocabulary entry. After enough merges, frequent words are single symbols and rare words are sequences of frequent pieces.

This section derives BPE training and encoding, works a complete example by hand, and then covers the two refinements that GPT-2 introduced and nearly every later tokenizer kept: running BPE over bytes instead of characters, and splitting text into chunks with a regular expression before BPE sees it. It closes with BPE-dropout, a training-time variation.

## From compression to tokenization

BPE began as a data compression algorithm. Gage (1994) described a scheme that repeatedly finds the most frequent pair of adjacent bytes in a file and replaces every occurrence with a single unused byte value, recording each replacement in a table so the process can be reversed. Repeating this shrinks the file.

Sennrich et al. (2016) noticed that the same procedure, applied to words in a text corpus, produces a good subword vocabulary. Frequent character sequences, such as common words and common suffixes, get merged early and become single symbols. Rare sequences are never merged and stay split into smaller pieces. The number of merges controls the vocabulary size directly: each merge adds exactly one new symbol.

## Training

BPE training takes a corpus and a target vocabulary size and produces two things: the vocabulary itself and an ordered list of **merge rules**.

1. Split the corpus into words (the **pre-tokenization** step, discussed below and in Section 6), and count how often each distinct word occurs.
2. Write each word as a sequence of base symbols: characters in the original algorithm, bytes in byte-level BPE. The base symbols form the initial vocabulary.
3. Count every pair of adjacent symbols across all words, weighting each word by its frequency.
4. Pick the most frequent pair, say $`(a, b)`$. Add the merged symbol $`ab`$ to the vocabulary and append the rule $`(a, b) \to ab`$ to the merge list.
5. Replace every occurrence of $`a`$ followed by $`b`$, in every word, with $`ab`$.
6. Repeat from step 3 until the vocabulary reaches the target size.

Merges never cross word boundaries, because each word is processed separately. And the algorithm works on a table of distinct words with counts, not on the raw text, which makes it fast: a corpus with billions of words may contain only a few million distinct ones. The whole algorithm fits in about twenty lines of Python; the appendix at the end of this section contains it, along with code for every other example here.

## A worked example

Sennrich et al. illustrated BPE with a tiny corpus of four words and their frequencies: "low" 5 times, "lower" 2 times, "newest" 6 times, and "widest" 3 times. Every word starts as a sequence of characters. Counting adjacent pairs, weighted by word frequency, gives:

| Pair | Count | Comes from |
|---|---|---|
| e s | 9 | newest (6) + widest (3) |
| s t | 9 | newest (6) + widest (3) |
| w e | 8 | lower (2) + newest (6) |
| l o | 7 | low (5) + lower (2) |
| o w | 7 | low (5) + lower (2) |
| n e | 6 | newest (6) |
| e w | 6 | newest (6) |
| w i, i d, d e | 3 each | widest (3) |
| e r | 2 | lower (2) |

Two pairs tie at 9. Real implementations need a deterministic tie-breaking rule so that training is reproducible; ours takes the pair encountered first, which is "e s". Running six merges gives:

| Merge | Pair merged | Count | Words afterward |
|---|---|---|---|
| 1 | e + s | 9 | l o w · l o w e r · n e w es t · w i d es t |
| 2 | es + t | 9 | l o w · l o w e r · n e w est · w i d est |
| 3 | l + o | 7 | lo w · lo w e r · n e w est · w i d est |
| 4 | lo + w | 7 | low · low e r · n e w est · w i d est |
| 5 | n + e | 6 | low · low e r · ne w est · w i d est |
| 6 | ne + w | 6 | low · low e r · new est · w i d est |

Notice how the counts change as merges happen. Before any merge, "w e" occurred 8 times. After the first merge turned "newest" into `n e w es t`, the "w" in "newest" is followed by "es", not "e", so "w e" survives only in "lower" and its count drops to 2. Every merge changes the pair statistics, which is why the counts must be updated after each step.

After six merges the vocabulary contains the ten starting characters plus the six new symbols "es", "est", "lo", "low", "ne", and "new". The suffix "est" and the word "low" emerged from pure frequency counting, with no knowledge of English.

## Encoding new text

To encode a new word with a trained BPE model, split it into base symbols and apply the learned merges **in the order they were learned**. The position of a merge in the list is its **rank**; earlier merges have lower rank and higher priority. With the six merges above, four words that never appeared in the corpus are segmented as follows:

| Word | Segmentation |
|---|---|
| lowest | low · est |
| newer | new · e · r |
| wider | w · i · d · e · r |
| slow | s · low |

None of these words appeared in the training corpus, yet each gets a sensible segmentation from the pieces that did. "lowest" becomes "low" + "est". "wider" stays in characters because no merge involving those letters was learned (the "wid" in "widest" was never merged in six steps).

It is important that encoding replays the merges by rank rather than, say, greedily taking the longest vocabulary entry that matches at each position. The two can give different results, and only rank order reproduces the segmentations the model saw during training. Looping over the entire merge list for every word, as the simple encoder in the appendix does, is correct but slow for a vocabulary with tens of thousands of merges. Practical encoders instead look at the pairs actually present in the word, apply whichever of them has the lowest rank, and repeat until no present pair has a rule. Section 10 implements that version.

## Decoding

Decoding maps each token to its string and concatenates. For character-level BPE, the only subtlety is spacing: since merges never cross word boundaries, the tokenizer must record where the spaces were. Sennrich et al. appended an end-of-word marker to each word; SentencePiece (Section 5) instead turns spaces into a visible symbol. Byte-level BPE, as used by GPT-2, keeps the space as part of the following token, so decoding is just concatenation of bytes. Because every merge is a concatenation of two byte strings, decoding a byte-level BPE sequence always gives back exactly the original bytes: the scheme is **lossless**.

Encoding must also be **deterministic**: the same text must always produce the same token IDs. A model is trained on one specific segmentation of each string and has little experience with others, so a tokenizer that sometimes produced different splits for the same input would degrade the model. That is why tie-breaking rules matter and why encoding follows a fixed rank order.

## The cost of training

The simple training loop recounts every pair in every distinct word after each merge. With $`W`$ distinct words and $`M`$ merges, that is roughly $`M`$ passes over all the words. For tens of thousands of merges over millions of distinct words, the naive approach is too slow. Production trainers keep the pair counts in a data structure and update them incrementally: when a merge is applied, only the words that contain the merged pair change, so only the counts of pairs that overlap those occurrences need adjusting, and a priority queue finds the next most frequent pair quickly. The algorithm is the same; only the bookkeeping differs.

## Byte-level BPE

Character-level BPE still has an unknown-symbol problem: a character that did not appear in the tokenizer's training corpus has no base symbol. GPT-2 (Radford et al. 2019) solved this by running BPE over the **bytes** of the UTF-8 encoding instead of over Unicode characters. The base vocabulary is then exactly the 256 possible byte values, every string in every language can be represented, and nothing is ever unknown. Frequent multi-byte characters, such as common accented letters or common Chinese characters, simply become merged tokens during training.

GPT-2's vocabulary of 50,257 entries is exactly 256 byte tokens, plus 50,000 learned merges, plus one special token, `<|endoftext|>`, that marks document boundaries.

One implementation detail causes confusion when you open GPT-2's vocabulary files. Many byte values, such as the space (byte 32), the newline (byte 10), and control characters, are awkward to store in a text file of merge rules. GPT-2's code therefore maps each of the 256 byte values to a printable Unicode character, and stores the vocabulary and merges in terms of those characters. Printable ASCII characters (and most printable Latin-1 characters) map to themselves, while the remaining byte values are shifted to characters starting at U+0100. The space becomes `Ġ`, the newline `Ċ`, and the tab `ĉ`. After a version line, GPT-2's `merges.txt` file begins with the merges `Ġ t`, `Ġ a`, `h e`, `i n`, `r e`, `o n`, and `Ġt he`, one per line.

The very first merge GPT-2 learned was a space followed by "t", and by the seventh it had built " the". When you see `Ġ` in GPT-2 tokens, read it as a leading space. The mapping is purely cosmetic: it is undone during decoding, and the underlying algorithm operates on bytes. Section 10 includes the mapping function in order to load GPT-2's files.

## Pre-tokenization with a regular expression

If BPE were run directly on raw text, it would happily learn tokens that span word boundaries and punctuation, such as "dog." and "dog!" and "dog?" as separate tokens, wasting vocabulary on combinations. GPT-2 prevents this by first splitting the text into chunks with a regular expression and running BPE within each chunk separately. The pattern's alternatives, tried in order at each position, are:

- common English contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`);
- an optional space followed by a run of letters (any Unicode letter, written `\p{L}` in the pattern);
- an optional space followed by a run of digits (`\p{N}`);
- an optional space followed by a run of other symbols such as punctuation;
- a run of whitespace that is not followed by a non-space character, and then any remaining whitespace.

Applied to the text "Hello world! It's 2024, isn't it?   Yes." followed by two newlines, two spaces, and "dog. dog!", the pattern produces these chunks (Appendix A.3 has the exact pattern):

`Hello` · `␣world` · `!` · `␣It` · `'s` · `␣2024` · `,` · `␣isn` · `'t` · `␣it` · `?` · `␣␣` · `␣Yes` · `.` · `⏎⏎␣` · `␣dog` · `.` · `␣dog` · `!`

where `⏎` marks a newline. The whitespace rules are subtle. In "it?   Yes", the three spaces are split into two spaces as their own chunk and one space attached to " Yes". That keeps " Yes" identical to the token that appears in ordinary single-spaced text.

Because merges never cross chunk boundaries, " dog" is learned once and reused before any punctuation. GPT-2 encodes " dog.", " dog!", and " dog?" as the same token 3290 for " dog" followed by the punctuation mark's own token (13, 0, and 30), rather than spending three vocabulary entries on the combinations.

The pre-tokenizer is also a limit on what BPE can learn: a token can never contain a chunk boundary. GPT-2's pattern allows " 2024" (a space and digits) to become a token but never "2024," with its comma. Later tokenizers changed the pattern, for example to split numbers into groups of at most three digits, and Section 6 compares these choices. The contraction rules are specific to English, and they only match lowercase forms: the pattern splits "it's" into "it" and "'s", but "IT'S" into "IT", "'", and "S". That is a small example of how English-centric design decisions end up baked into a tokenizer.

## BPE-dropout

Standard BPE encoding is deterministic, so the model sees only one segmentation of each word, and it never sees how a frequent word decomposes into smaller pieces. Provilkov et al. (2020) proposed **BPE-dropout**: during training (of the model, not the tokenizer), each time a merge could be applied, skip it with some probability $`p`$. The same word then gets different segmentations each time it appears, from the usual one down to nearly character level. At test time, encoding is deterministic again ($`p = 0`$). This acts as data augmentation and makes the model more robust to rare words and misspellings; the authors reported gains in machine translation.

The change to the encoder is small: at each step, randomly discard candidate merges before choosing the lowest-rank one among those that remain (Appendix A.5). Encoding "lowest" four times with the toy merges and $`p = 0.5`$ gave "l o w es t", "low e s t", "lo w est", and "l o w est".

With $`p = 0`$ this procedure gives the standard segmentation, "low" + "est". With $`p = 0.5`$, which is deliberately high for this demonstration, every one of the four samples is a finer segmentation than the standard one. Provilkov et al. usually used $`p = 0.1`$. BPE-dropout is a training-time technique for the model; most LLM pretraining pipelines tokenize the corpus once, deterministically, and do not use it.

## Key takeaways

- BPE training starts from characters or bytes and repeatedly merges the most frequent adjacent pair, counting within words weighted by word frequency, until the vocabulary reaches its target size.
- The output of training is a vocabulary and an ordered list of merges; encoding applies the merges in rank order, not greedy longest match, and must be deterministic.
- Byte-level BPE uses the 256 byte values as its base vocabulary, so no input is ever unknown and decoding is lossless; GPT-2's vocabulary is 256 bytes plus 50,000 merges plus `<|endoftext|>`.
- GPT-2's `Ġ` and `Ċ` are printable stand-ins for the space and newline bytes, not part of the algorithm.
- A regular-expression pre-tokenizer splits text into chunks that merges cannot cross, which keeps tokens like " dog" reusable but also limits what the tokenizer can learn.
- BPE-dropout randomly skips merges during model training to expose the model to multiple segmentations.

## Further reading

Gage, Philip. "A New Algorithm for Data Compression." *The C Users Journal* 12, no. 2 (1994): 23–38.

Provilkov, Ivan, et al. "BPE-Dropout: Simple and Effective Subword Regularization." In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 1882–1892, 2020. https://arxiv.org/abs/1910.13267.

Radford, Alec, et al. "Language Models Are Unsupervised Multitask Learners." OpenAI technical report, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

Sennrich, Rico, et al. "Neural Machine Translation of Rare Words with Subword Units." In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715–1725, 2016. https://arxiv.org/abs/1508.07909.

## Appendix: Code for Section 5.3

These listings reproduce the examples in this section. Run them in order in one Python session; they need the `regex` and `tiktoken` packages.

### A.1 Character-level BPE training

Train BPE on a table of word frequencies and run it on the toy corpus from the worked example.

```python
from collections import Counter

def merge_word(word, pair):
    """Replace each occurrence of `pair` in the tuple `word` with the merged symbol."""
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(word[i] + word[i + 1])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)

def train_bpe(word_freqs, num_merges):
    words = {tuple(w): n for w, n in word_freqs.items()}
    merges = []
    for step in range(num_merges):
        stats = Counter()
        for w, n in words.items():
            for pair in zip(w, w[1:]):
                stats[pair] += n
        pair = max(stats, key=stats.get)          # ties: the pair seen first
        merges.append(pair)
        words = {merge_word(w, pair): n for w, n in words.items()}
        print(step + 1, pair, stats[pair], list(words))
    return merges
```

```python
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
merges = train_bpe(corpus, 6)
```

Output:

```
1 ('e', 's') 9 [('l', 'o', 'w'), ('l', 'o', 'w', 'e', 'r'), ('n', 'e', 'w', 'es', 't'), ('w', 'i', 'd', 'es', 't')]
2 ('es', 't') 9 [('l', 'o', 'w'), ('l', 'o', 'w', 'e', 'r'), ('n', 'e', 'w', 'est'), ('w', 'i', 'd', 'est')]
3 ('l', 'o') 7 [('lo', 'w'), ('lo', 'w', 'e', 'r'), ('n', 'e', 'w', 'est'), ('w', 'i', 'd', 'est')]
4 ('lo', 'w') 7 [('low',), ('low', 'e', 'r'), ('n', 'e', 'w', 'est'), ('w', 'i', 'd', 'est')]
5 ('n', 'e') 6 [('low',), ('low', 'e', 'r'), ('ne', 'w', 'est'), ('w', 'i', 'd', 'est')]
6 ('ne', 'w') 6 [('low',), ('low', 'e', 'r'), ('new', 'est'), ('w', 'i', 'd', 'est')]
```

### A.2 Encoding by replaying merges

Encode new words by applying the learned merges in rank order.

```python
def encode_word(word, merges):
    symbols = tuple(word)
    for pair in merges:                 # apply merges in learned order
        symbols = merge_word(symbols, pair)
    return list(symbols)

for w in ["lowest", "newer", "wider", "slow"]:
    print(w, encode_word(w, merges))
```

Output:

```
lowest ['low', 'est']
newer ['new', 'e', 'r']
wider ['w', 'i', 'd', 'e', 'r']
slow ['s', 'low']
```

### A.3 GPT-2's pre-tokenization pattern

Split text into chunks with GPT-2's regular expression. The third-party `regex` module is needed because Python's built-in `re` does not support `\p{L}` and `\p{N}`.

```python
import regex as re    # the third-party `regex` module supports \p{L} and \p{N}

GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
print(re.findall(GPT2_PATTERN, "Hello world! It's 2024, isn't it?   Yes.\n\n  dog. dog!"))
```

Output:

```
['Hello', ' world', '!', ' It', "'s", ' 2024', ',', ' isn', "'t", ' it', '?', '  ', ' Yes', '.', '\n\n ', ' dog', '.', ' dog', '!']
```

### A.4 Punctuation after the same word

Show that " dog" is the same token before different punctuation marks.

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")
for s in [" dog.", " dog!", " dog?"]:
    ids = enc.encode(s)
    print(repr(s), ids, [enc.decode([i]) for i in ids])
```

Output:

```
' dog.' [3290, 13] [' dog', '.']
' dog!' [3290, 0] [' dog', '!']
' dog?' [3290, 30] [' dog', '?']
```

### A.5 BPE-dropout

Encode a word several times while skipping each candidate merge with probability `p`.

```python
import random

def encode_word_dropout(word, merges, p, rng):
    ranks = {pair: r for r, pair in enumerate(merges)}
    symbols = tuple(word)
    while len(symbols) >= 2:
        candidates = [pr for pr in zip(symbols, symbols[1:])
                      if pr in ranks and rng.random() >= p]
        if not candidates:
            break
        best = min(candidates, key=ranks.get)
        symbols = merge_word(symbols, best)
    return list(symbols)

rng = random.Random(0)
print([encode_word_dropout("lowest", merges, 0.5, rng) for _ in range(4)])
```

Output:

```
[['l', 'o', 'w', 'es', 't'], ['low', 'e', 's', 't'], ['lo', 'w', 'est'], ['l', 'o', 'w', 'est']]
```
