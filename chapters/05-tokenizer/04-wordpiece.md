# 5.4 WordPiece

**WordPiece** is the subword algorithm behind BERT and the many encoder models derived from it. It builds its vocabulary bottom-up by merging symbols, just like BPE, and at a glance its vocabularies look similar. The two algorithms differ in how they choose each merge, in how they mark word boundaries, and in how they encode new text. This section explains those three differences, runs WordPiece's merge criterion on the same toy corpus as Section 3, implements WordPiece encoding, and checks it against BERT's tokenizer.

## Origins

WordPiece was introduced by Schuster and Nakajima (2012) for Japanese and Korean voice search at Google. Neither language separates words with spaces, and Korean in particular has an enormous number of possible syllable combinations, so a fixed word list was not an option. They learned a vocabulary of word pieces from data by repeatedly adding the unit that most improved a language model of the training text. Wu et al. (2016) later used a WordPiece model in Google's neural machine translation system, where it played the same role as BPE in the work of Sennrich et al.: rare words are split into pieces that the model can translate or copy.

Devlin et al. (2019) used WordPiece for BERT, with a vocabulary of about 30,000 tokens (the released `bert-base-uncased` tokenizer has 30,522 entries). BERT's popularity made WordPiece the default for encoder models for several years.

## The merge criterion: likelihood, not frequency

BPE merges the most frequent pair. WordPiece instead merges the pair whose merger most increases the likelihood of the training data under a simple model in which each token is drawn independently (a **unigram** model).

To see what that means, let $`c_a`$ be the number of times symbol $`a`$ occurs in the segmented training data, $`c_{ab}`$ the number of times $`a`$ is immediately followed by $`b`$, and $`N`$ the total number of symbols. Under a unigram model, symbol $`a`$ has probability $`p_a = c_a / N`$. Merging $`a`$ and $`b`$ replaces each of the $`c_{ab}`$ adjacent pairs with one symbol $`ab`$. Ignoring the small change in $`N`$ and in the other counts, the log-likelihood of the data changes by approximately

```math
\Delta \log L \approx c_{ab} \log \frac{p_{ab}}{p_a \, p_b}.
```

The fraction inside the logarithm compares how often $`a`$ and $`b`$ actually appear together with how often they would appear together by chance. Its logarithm is the **pointwise mutual information** of the pair. A pair of common symbols that happen to be adjacent often, such as "e" followed by "s", has a high count but a modest ratio. A pair of rarer symbols that almost always appear together has a high ratio.

The original papers describe the criterion in terms of likelihood and give few implementation details. A widely used simplified form drops the leading count and scores each pair by the ratio alone:

```math
\text{score}(a, b) = \frac{c_{ab}}{c_a \, c_b}.
```

This prefers merging pieces that rarely occur apart. The Python below runs that simplified criterion on the toy corpus from Section 3. Following WordPiece's convention (explained in the next section), every symbol that does not start a word carries the prefix `##`:

```python
from collections import Counter

def wordpiece_merges(word_freqs, num_merges):
    # non-initial symbols carry the "##" prefix
    words = {tuple([w[0]] + ["##" + c for c in w[1:]]): n for w, n in word_freqs.items()}
    for step in range(num_merges):
        pair_counts, sym_counts = Counter(), Counter()
        for w, n in words.items():
            for s in w:
                sym_counts[s] += n
            for pair in zip(w, w[1:]):
                pair_counts[pair] += n
        score = {p: c / (sym_counts[p[0]] * sym_counts[p[1]])
                 for p, c in pair_counts.items()}
        best = max(score, key=score.get)
        new = best[0] + best[1].removeprefix("##")
        print(step + 1, best, pair_counts[best], round(score[best], 4))
        merged = {}
        for w, n in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(new); i += 2
                else:
                    out.append(w[i]); i += 1
            merged[tuple(out)] = n
        words = merged
    print(list(words))

wordpiece_merges({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 6)
```

```
1 ('w', '##i') 3 0.3333
2 ('wi', '##d') 3 0.3333
3 ('l', '##o') 7 0.1429
4 ('##s', '##t') 9 0.1111
5 ('lo', '##w') 7 0.0769
6 ('##e', '##r') 2 0.0588
[('low',), ('low', '##er'), ('n', '##e', '##w', '##e', '##st'), ('wid', '##e', '##st')]
```

The order is very different from BPE's. BPE's first merge was "e s", the most frequent pair (9 occurrences). WordPiece's first merge is "w" + "##i", which occurs only 3 times, because the word-initial "w" and the "##i" never appear anywhere except together in "widest". Its sixth merge, "##e" + "##r", has a count of only 2. Meanwhile the frequent "e s" pair is never merged in six steps: "##e" appears in three different words and "##s" in two, so the pair's ratio stays low. After the same number of merges, BPE had built "est", "low", and "new"; WordPiece built "wid", "low", "##er", and "##st".

On a real corpus the two criteria produce vocabularies that overlap heavily, since frequent words are frequent under either criterion. The difference shows up at the margins: WordPiece is more willing to add pieces that are rare but very cohesive.

## Marking word boundaries with ##

BPE in the GPT-2 style marks the *start* of a word by attaching the preceding space to it. WordPiece takes the opposite approach: it removes whitespace entirely during pre-tokenization and marks the pieces that *continue* a word with the prefix `##`. So "tokenization" is encoded as "token" followed by "##ization", and a word-initial "token" and a word-internal "##token" are two different vocabulary entries.

This means the vocabulary contains many `##` entries. In `bert-base-uncased`, 5,828 of the 30,522 entries start with `##`.

Because the spaces themselves are discarded, a WordPiece tokenizer cannot always reproduce the original text exactly: it knows where word boundaries were, but not whether there was one space, two spaces, or a newline. BERT's tokenizer also does more than split. Its pre-tokenization step, which BERT's code calls basic tokenization, splits off every punctuation character, puts spaces around Chinese, Japanese, and Korean ideographs so that each one is treated as a separate word, and, in the uncased models, lowercases the text and strips accents. Section 6 discusses these normalization steps; for now, note that decoding a BERT token sequence gives back a normalized version of the text, not the original.

## Encoding: greedy longest match first

The biggest practical difference from BPE is encoding. A WordPiece tokenizer does not store a merge list at all, only the vocabulary. To encode a word, it takes the **longest prefix of the word that is in the vocabulary**, emits it, and repeats on the remainder (with the `##` prefix added), until the word is used up. If at some point no piece matches, the entire word becomes the unknown token `[UNK]`.

```python
def wordpiece_encode(word, vocab, unk="[UNK]"):
    """Greedy longest-match-first segmentation of a single word."""
    pieces, start = [], 0
    while start < len(word):
        end = len(word)
        while end > start:
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                break
            end -= 1
        if end == start:              # no piece matches: the whole word is unknown
            return [unk]
        pieces.append(piece)
        start = end
    return pieces
```

We can check this against the real BERT tokenizer, using its vocabulary:

```python
from transformers import AutoTokenizer

bert = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab = bert.get_vocab()
for w in ["tokenization", "unaffable", "playing", "xyzzy"]:
    print(w, wordpiece_encode(w, vocab), bert.tokenize(w))
```

```
tokenization ['token', '##ization'] ['token', '##ization']
unaffable ['una', '##ffa', '##ble'] ['una', '##ffa', '##ble']
playing ['playing'] ['playing']
xyzzy ['x', '##y', '##zzy'] ['x', '##y', '##zzy']
```

Our short function agrees with BERT on every word. "playing" is common enough to be a single entry. "unaffable" shows a weakness of greedy matching: the longest prefix in the vocabulary is "una", which commits the rest of the word to "##ffa" + "##ble", rather than the more meaningful "un" + "##aff" + "##able". Greedy longest match finds a segmentation quickly but not necessarily the best one by any measure.

The `[UNK]` fallback matters more than it might seem. BERT's vocabulary contains a limited set of characters, and any word containing a character outside it becomes `[UNK]` as a whole:

```python
print(bert.tokenize("I love 🙂 emoji"))
print(bert.tokenize("Ωmega ∮ integral"))
```

```
['i', 'love', '[UNK]', 'em', '##oj', '##i']
['ω', '##me', '##ga', '[UNK]', 'integral']
```

Both the emoji and the contour-integral sign are unknown, and their meaning is lost. (The uncased model also lowercased "Ω" to "ω".) A byte-level BPE tokenizer can never produce an unknown token; a WordPiece tokenizer can. BERT's implementation also turns any single word longer than 100 characters into `[UNK]`, a safeguard against pathological inputs.

The simple implementation above is quadratic in the length of the word in the worst case, because it may try every prefix at every position. Song et al. (2021) showed that WordPiece encoding can be done in time linear in the input length, using a trie with precomputed failure links in the style of the Aho-Corasick string-matching algorithm, and reported substantial speedups over existing implementations.

## Special tokens in BERT

BERT's vocabulary also defines special tokens that its training setup relies on: `[CLS]` at the start of every input (its final vector is used for classification), `[SEP]` between and after segments, `[PAD]` for padding, `[MASK]` for the masked-language-modeling objective, and `[UNK]`. The tokenizer adds `[CLS]` and `[SEP]` automatically:

```python
print(bert.convert_ids_to_tokens(bert("Hello world")["input_ids"]))
```

```
['[CLS]', 'hello', 'world', '[SEP]']
```

Section 6 discusses special tokens in general, and Section 9 shows a common bug that this automatic insertion causes.

## Where WordPiece is used today

WordPiece remains the tokenizer of BERT and of many BERT-derived encoders used for classification, retrieval, and embedding models, because those models are still widely deployed and a model cannot switch tokenizers without retraining. Decoder-only LLMs, however, have mostly moved to byte-level BPE (Section 3) or to SentencePiece models with byte fallback (Section 5). The reasons follow from this section: WordPiece's `[UNK]` token loses information on unseen characters, its whitespace handling is not reversible, and generative models need to produce exactly the text they mean, including spacing, code indentation, emoji, and every script.

## Key takeaways

- WordPiece builds its vocabulary by merging pairs, like BPE, but chooses the merge that most increases the likelihood of the training data under a unigram model, which favors cohesive pairs over merely frequent ones.
- A common simplified score is $`c_{ab} / (c_a c_b)`$; on the toy corpus it merges rare, cohesive pairs such as "w" + "##i" before the frequent "e" + "s".
- WordPiece marks word-continuation pieces with `##` and discards the whitespace itself, so decoding does not always reproduce the original spacing.
- Encoding is greedy longest-match-first over the vocabulary, with no merge list; a word that cannot be segmented becomes `[UNK]`.
- WordPiece is still used in BERT-family encoders, while generative LLMs mostly use byte-level BPE or SentencePiece with byte fallback.

## Further reading

Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics*, 4171–4186, 2019. https://arxiv.org/abs/1810.04805.

Schuster, Mike, et al. "Japanese and Korean Voice Search." In *2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 5149–5152, 2012. https://doi.org/10.1109/ICASSP.2012.6289079.

Song, Xinying, et al. "Fast WordPiece Tokenization." In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, 2021. https://arxiv.org/abs/2012.15524.

Wu, Yonghui, et al. "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation." arXiv preprint arXiv:1609.08144, 2016. https://arxiv.org/abs/1609.08144.
