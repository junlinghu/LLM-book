# 5.10 Building a Byte-Level BPE Tokenizer from Scratch

This section builds a complete byte-level BPE tokenizer in about a hundred lines of Python, trains it on tiny Shakespeare, and then proves that it is the same algorithm GPT-2 uses: loaded with GPT-2's published vocabulary and merges, it produces exactly the same token IDs as `tiktoken` on every test string and on the entire Shakespeare corpus. The code was written for this book. It favors clarity over speed, so it is far slower than production libraries, but every step of Sections 3 and 6 appears in it explicitly: pre-tokenization with GPT-2's regular expression, BPE training over bytes, rank-ordered encoding, lossless decoding, and special tokens that are recognized only when the application allows it.

The plan follows the seven steps of the outline:

1. Convert text to UTF-8 bytes and split it into chunks with a regex pre-tokenizer.
2. Train: count pairs, merge the most frequent pair, record the merge, and repeat.
3. Encode by repeatedly applying the lowest-rank merge available in each chunk.
4. Decode by concatenating each token's bytes and decoding UTF-8.
5. Add special tokens, matched only when the application allows it.
6. Test round-tripping and compare compression with GPT-2's tokenizer.
7. Load GPT-2's merges and verify that the IDs match `tiktoken` exactly.

The only dependency beyond the standard library is the `regex` module, which supports the Unicode property classes `\p{L}` and `\p{N}` that GPT-2's pattern uses (Python's built-in `re` does not).

## The complete tokenizer

Here is the whole tokenizer. Save it as `bpe.py`; the rest of the section walks through it and then puts it to work.

```python
import regex as re
from collections import Counter

# GPT-2's pre-tokenization pattern (Radford et al. 2019).
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in the tuple `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class ByteBPE:
    def __init__(self, pattern=GPT2_PATTERN):
        self.pattern = re.compile(pattern)
        self.merges = {}                               # (id, id) -> new id
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self.special = {}                              # str -> id
        self.byte_ids = list(range(256))               # byte value -> id

    # ---- training: Steps 1 and 2 ---------------------------------------
    def train(self, text, vocab_size):
        chunks = Counter(self.pattern.findall(text))
        words = {tuple(c.encode("utf-8")): n for c, n in chunks.items()}
        for new_id in range(256, vocab_size):
            stats = Counter()
            for w, n in words.items():
                for pair in zip(w, w[1:]):
                    stats[pair] += n
            if not stats:
                break
            pair = max(stats, key=stats.get)   # ties: first pair seen
            words = {merge(w, pair, new_id): n for w, n in words.items()}
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    # ---- encoding: Step 3 ----------------------------------------------
    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            # Merge the pair that was learned earliest. Merge IDs grow with
            # rank, so the pair with the smallest merge ID has the lowest rank.
            pair = min(zip(ids, ids[1:]),
                       key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return list(ids)

    def encode_ordinary(self, text):
        ids = []
        for chunk in self.pattern.findall(text):
            ids.extend(self._encode_chunk(
                tuple(self.byte_ids[b] for b in chunk.encode("utf-8"))))
        return ids

    def encode(self, text, allow_special=False):
        if not (allow_special and self.special):
            return self.encode_ordinary(text)
        split = "(" + "|".join(re.escape(s) for s in self.special) + ")"
        ids = []
        for part in re.split(split, text):
            if part in self.special:
                ids.append(self.special[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    # ---- special tokens and decoding: Steps 4 and 5 --------------------
    def add_special_tokens(self, tokens):
        for t in tokens:
            if t not in self.special:
                new_id = len(self.vocab)
                self.special[t] = new_id
                self.vocab[new_id] = t.encode("utf-8")

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
```

The tokenizer's state is four tables. `vocab` maps each token ID to the bytes it stands for; it starts with the 256 single bytes. `merges` maps a pair of IDs to the ID of the token that replaces them, in the order the merges were learned. `special` maps special-token strings to their IDs. `byte_ids` maps each byte value to the ID of its single-byte token; for a tokenizer we train ourselves this is the identity, but GPT-2 numbers its byte tokens differently, and Step 7 needs the table.

### Steps 1 and 2: pre-tokenization and training

`train` first splits the text into chunks with the pre-tokenizer pattern from Section 3, counts how often each distinct chunk occurs, and converts each chunk to a tuple of UTF-8 byte values. From then on it never looks at the raw text again. It works on the table of distinct chunks with their counts, which for the training portion of tiny Shakespeare is 14,134 entries instead of about a million characters.

Each iteration of the loop is one merge. It counts every adjacent pair of IDs across all chunks, weighted by the chunk's frequency, picks the most frequent pair (on ties, the pair counted first), and rewrites every chunk with that pair replaced by the new ID. It records the merge and the new token's bytes, which are simply the two parts' bytes concatenated. New IDs are assigned in order, 256, 257, 258, and so on, so a token's ID also tells us when it was learned.

This is the naive algorithm from Section 3: it recounts all pairs after every merge. That makes training slow for large vocabularies, but the result is identical to what an optimized trainer with incremental counts would produce.

### Step 3: encoding

`encode_ordinary` splits the input into chunks with the same pattern and encodes each chunk independently, so merges never cross chunk boundaries. Within a chunk, `_encode_chunk` repeatedly looks at the adjacent pairs actually present, finds the one with the lowest rank among those that have a merge rule, and applies it everywhere in the chunk. When no present pair has a rule, the chunk is fully encoded. Because merge IDs increase with rank, "the pair whose merge ID is smallest" is the same as "the pair that was learned earliest", which is exactly the rank-order rule from Section 3.

### Steps 4 and 5: decoding and special tokens

`decode` concatenates the bytes of every token and decodes the result as UTF-8. Since every token's bytes are the concatenation of its parts' bytes, the concatenation is exactly the original input, and decoding is lossless. The `errors="replace"` argument only matters if someone decodes an arbitrary ID sequence that ends in the middle of a character, such as a single token from a split emoji; for streaming, use the incremental decoder from Section 6.

`add_special_tokens` gives each special string the next free ID. `encode` treats special strings as special **only** when called with `allow_special=True`. In that case it splits the text around every occurrence of a special string, emits the special ID for each occurrence, and encodes the pieces between them normally. By default, and always in `encode_ordinary`, a special string in the input is encoded as ordinary text. That is the safe default from Section 6: the application inserts special tokens deliberately, and text from users or documents can never produce them.

## Training on tiny Shakespeare

We train on the first 90 percent of tiny Shakespeare and keep the last 10 percent to measure compression on unseen text. A vocabulary of 512 means 256 merges:

```python
import os, urllib.request

if not os.path.exists("shakespeare.txt"):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "shakespeare.txt")
text = open("shakespeare.txt", encoding="utf-8").read()
split = int(0.9 * len(text))
train, held_out = text[:split], text[split:]

tok = ByteBPE()
tok.train(train, vocab_size=512)
print([tok.vocab[i] for i in range(256, 276)])
print([tok.vocab[i] for i in range(500, 512)])
```

```
[b' t', b'he', b' a', b'ou', b' s', b' m', b'in', b' w', b're', b'ha', b' the', b'nd', b' b', b'is', b'or', b' f', b'er', b'll', b'it', b'on']
[b'ond', b' their', b'self', b' there', b' know', b' king', b' if', b'ep', b'ind', b' tr', b' br', b' O']
```

On the machine used for this book, training took about 6 seconds for 512 entries and about 16 seconds for 1,024; the naive loop gets slower with every merge, since it rescans all chunks each time. The first merges are the most frequent pairs in English: a space followed by "t", then "he", then a space followed by "a". The eleventh merge already produces " the". By the last merges, the tokenizer is adding whole words that are frequent in Shakespeare, such as " king" and " know". GPT-2's own first merges, visible in the first lines of its `merges.txt` (Section 3), were similar: " t", " a", "he", "in", "re".

Encoding a famous line shows the tokenizer at work:

```python
line = "To be, or not to be: that is the question."
ids = tok.encode(line)
print(len(ids), ids)
print([tok.vocab[i].decode("utf-8") for i in ids])
```

```
18 [396, 304, 44, 32, 270, 321, 287, 304, 58, 322, 326, 266, 32, 113, 117, 377, 395, 46]
['To', ' be', ',', ' ', 'or', ' not', ' to', ' be', ':', ' that', ' is', ' the', ' ', 'q', 'u', 'est', 'ion', '.']
```

Frequent short words, with their leading spaces, are single tokens. "question" is still built from pieces, "q", "u", "est", and "ion", and with only 256 merges the tokenizer has learned "or" but not " or", so the space before it is a token of its own.

## Step 6: testing and comparing

A tokenizer must round-trip every input, not just the text it was trained on. Our tokenizer has never seen a character outside the 65 that occur in Shakespeare, but because its base vocabulary is all 256 bytes, it can still encode anything:

```python
tests = ["Hello, world!", "Emoji 🙂🚀 and accents: café, naïve, Zürich",
         "中文分词很难。日本語のテキスト。한국어 텍스트.",
         "def f(x):\n    return x ** 2  # square\n", "   spaces\t\ttabs\n\n\nnewlines   "]
print(all(tok.decode(tok.encode(t)) == t for t in tests))
print(len("中文分词很难。".encode("utf-8")), len(tok.encode("中文分词很难。")))
```

```
True
21 21
```

Every test string round-trips exactly. The Chinese text, though, gets no compression at all: its 21 bytes become 21 tokens, one per byte, because none of its byte pairs occurred in the training data. That is the multilingual cost of Section 8 in its most extreme form.

Next, special tokens. We add an end-of-text token and check that it is recognized only when allowed:

```python
tok.add_special_tokens(["<|endoftext|>"])
eot = tok.special["<|endoftext|>"]
s = "the end<|endoftext|>A new document"
print(eot, tok.encode(s, allow_special=True))
print(tok.encode(s).count(eot), tok.decode(tok.encode(s, allow_special=True)) == s)
```

```
512 [116, 257, 334, 267, 512, 65, 428, 119, 382, 99, 117, 109, 340]
0 True
```

With `allow_special=True`, the special string becomes the single ID 512 in the middle of the sequence, and decoding restores the original text. Without it, the ID never appears (the count is 0): the characters "<", "|", and so on are encoded as ordinary text.

Finally, compression on the held-out text, compared with GPT-2's tokenizer. We also train a second tokenizer with 1,024 entries:

```python
import tiktoken

n_bytes = len(held_out.encode("utf-8"))
tok_1024 = ByteBPE()
tok_1024.train(train, vocab_size=1024)
gpt2 = tiktoken.get_encoding("gpt2")
for name, n in [("ours, 512", len(tok.encode(held_out))),
                ("ours, 1024", len(tok_1024.encode(held_out))),
                ("GPT-2, 50257", len(gpt2.encode_ordinary(held_out)))]:
    print(name, n, round(n_bytes / n, 2))
```

```
ours, 512 59401 1.88
ours, 1024 49416 2.26
GPT-2, 50257 36059 3.09
```

With 512 entries our tokenizer reaches 1.88 bytes per token on unseen Shakespeare, and with 1,024 entries 2.26. GPT-2's tokenizer, with a vocabulary about fifty times larger and trained on web text rather than Shakespeare, reaches 3.09. These are the numbers quoted in Section 2. The 512-entry figure also matches, token for token, the Hugging Face BPE trainer with the same settings in Section 7, which is a useful independent check that the training loop is right.

## Step 7: matching GPT-2 exactly

If our implementation is really the same algorithm as GPT-2's, then loading GPT-2's vocabulary and merges into it should reproduce GPT-2's tokenization exactly. OpenAI published those files with GPT-2, and they are mirrored on the Hugging Face Hub as `vocab.json` (a map from token strings to IDs) and `merges.txt` (the 50,000 merges in rank order).

Two details need care. First, both files write bytes using GPT-2's printable-character map from Section 3 (`Ġ` for a space and so on), so we need the inverse of that map. Second, GPT-2 does not number its single-byte tokens by byte value: ID 0 is "!", not byte 0. That is what `byte_ids` is for. The loader below builds all four tables from the two files:

```python
import json


def bytes_to_unicode():
    """GPT-2's reversible map from the 256 byte values to printable characters."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def load_gpt2(tok, vocab_path, merges_path):
    """Load GPT-2's published vocab.json and merges.txt into a ByteBPE."""
    byte_decoder = {c: b for b, c in bytes_to_unicode().items()}
    to_bytes = lambda s: bytes(byte_decoder[c] for c in s)
    with open(vocab_path, encoding="utf-8") as f:
        encoder = json.load(f)                        # token string -> id
    tok.vocab = {i: to_bytes(s) for s, i in encoder.items()}
    token_to_id = {b: i for i, b in tok.vocab.items()}
    tok.byte_ids = [token_to_id[bytes([b])] for b in range(256)]
    tok.merges = {}
    with open(merges_path, encoding="utf-8") as f:
        lines = f.read().split("\n")[1:]              # skip the version line
    for line in lines:
        if not line:
            continue
        a, b = line.split(" ")
        a, b = to_bytes(a), to_bytes(b)
        tok.merges[(token_to_id[a], token_to_id[b])] = token_to_id[a + b]
    # The merge dict maps pairs to ids; GPT-2's ids grow with merge rank,
    # so "lowest id" and "earliest merge" pick the same pair.
    tok.special = {"<|endoftext|>": encoder["<|endoftext|>"]}
    return tok
```

Before trusting the loader, we check the property that `_encode_chunk` relies on: in GPT-2's files, the token created by the merge of rank $`r`$ has ID $`256 + r`$, so merge IDs grow with rank.

```python
from huggingface_hub import hf_hub_download

vocab_path = hf_hub_download("openai-community/gpt2", "vocab.json")
merges_path = hf_hub_download("openai-community/gpt2", "merges.txt")
gpt2_tok = load_gpt2(ByteBPE(), vocab_path, merges_path)

with open(merges_path, encoding="utf-8") as f:
    ranked = [line for line in f.read().split("\n")[1:] if line]
ok = all(gpt2_tok.merges[pair] == 256 + r
         for r, pair in enumerate(gpt2_tok.merges))
print(len(ranked), len(gpt2_tok.merges), len(gpt2_tok.vocab), ok)
```

```
50000 50000 50257 True
```

All 50,000 merges satisfy it (Python dictionaries preserve insertion order, so enumerating `merges` visits them in rank order). Now the real test: compare our IDs with `tiktoken`'s on a set of deliberately awkward strings, then on the entire tiny Shakespeare corpus:

```python
tests = [
    "Hello, world!",
    "Tokenization isn't magic: it's just bytes and merges.",
    "   leading spaces\tand\ttabs\n\n\nnewlines   ",
    "Numbers: 1234567 + 89 = 1234656",
    "Emoji 🙂🚀 and accents: café, naïve, Zürich",
    "中文分词很难。日本語のテキスト。한국어 텍스트.",
    "def f(x):\n    return x ** 2  # square\n",
    "<|endoftext|> is special only when allowed",
]
for t in tests:
    ours, ref = gpt2_tok.encode_ordinary(t), gpt2.encode_ordinary(t)
    assert ours == ref, (t, ours, ref)
    assert gpt2_tok.decode(ours) == t
print("all", len(tests), "test strings match")
print(gpt2_tok.encode("<|endoftext|> hi", allow_special=True),
      gpt2.encode("<|endoftext|> hi", allowed_special={"<|endoftext|>"}))

ours = gpt2_tok.encode_ordinary(text)
ref = gpt2.encode_ordinary(text)
print(len(text), len(ours), ours == ref)
```

```
all 8 test strings match
[50256, 23105] [50256, 23105]
1115394 338025 True
```

Every test string produces exactly the same IDs as `tiktoken`, including emoji, three scripts, tabs and runs of spaces, and code. The special token is handled identically. And all 338,025 tokens of the full 1,115,394-character corpus match. Our hundred lines of Python implement GPT-2's tokenizer.

They are also much slower. On the same machine, our encoder took about 7 seconds for the whole corpus and `tiktoken` about 0.4 seconds, and the gap grows for larger inputs. Production encoders are written in compiled languages, cache the encodings of frequent chunks, and use better data structures than rescanning each chunk for the minimum-rank pair. None of that changes the output.

## What we built and what production tokenizers add

The tokenizer in this section contains every essential idea of byte-level BPE: a 256-byte base vocabulary that makes every input encodable, a regex pre-tokenizer that keeps merges inside chunks, merges learned by frequency and applied in rank order, lossless decoding, and special tokens that are inserted deliberately rather than parsed from text. Production libraries add speed (compiled code, caching, incremental training), serialization formats such as `tokenizer.json`, offset tracking for mapping tokens back to characters, and conveniences such as padding, truncation, and chat templates, all covered in Sections 6 and 9.

The suggested code labs for this chapter build on exactly this code: train the tokenizer on text of your choice, match GPT-2 on a larger sample, extend it with chat role markers and a streaming decoder, and use its IDs to build the packed training blocks, padded batches, and embedding lookups of Section 9.

## Key takeaways

- A complete byte-level BPE tokenizer needs only four tables (vocabulary, merges, special tokens, and the byte-to-ID map) and a few short functions.
- Training counts pairs over a table of distinct pre-tokenized chunks, merges the most frequent pair, and assigns new IDs in order, so IDs double as merge ranks.
- Encoding repeatedly applies the lowest-rank merge present in each chunk; decoding concatenates bytes, which makes the round trip lossless for any input.
- Special tokens should be recognized only when the caller explicitly allows it.
- Loaded with GPT-2's published vocabulary and merges, the from-scratch tokenizer reproduces `tiktoken`'s GPT-2 token IDs exactly, on awkward test strings and on all 338,025 tokens of tiny Shakespeare; production libraries differ in speed, not in output.

## Further reading

Radford, Alec, et al. "Language Models Are Unsupervised Multitask Learners." OpenAI technical report, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

Sennrich, Rico, et al. "Neural Machine Translation of Rare Words with Subword Units." In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715–1725, 2016. https://arxiv.org/abs/1508.07909.
