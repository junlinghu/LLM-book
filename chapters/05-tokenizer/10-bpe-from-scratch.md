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

## The design

The whole tokenizer is a single class, `ByteBPE`, plus one helper function, about ninety lines in all. The full listing is in Appendix A.1; saved as `bpe.py`, it is the file the experiments in this section, and the code labs, build on.

The tokenizer's state is four tables. `vocab` maps each token ID to the bytes it stands for; it starts with the 256 single bytes. `merges` maps a pair of IDs to the ID of the token that replaces them, in the order the merges were learned. `special` maps special-token strings to their IDs. `byte_ids` maps each byte value to the ID of its single-byte token; for a tokenizer we train ourselves this is the identity, but GPT-2 numbers its byte tokens differently, and Step 7 needs the table.

The class has five public methods, one or two per step of the plan:

| Method | Step | What it does |
|---|---|---|
| `train(text, vocab_size)` | 1, 2 | Pre-tokenizes the text and learns merges until the vocabulary reaches `vocab_size` |
| `encode_ordinary(text)` | 1, 3 | Encodes text, treating every character as ordinary text |
| `encode(text, allow_special=False)` | 3, 5 | Encodes text, recognizing special-token strings only if `allow_special` is true |
| `decode(ids)` | 4 | Turns IDs back into a string |
| `add_special_tokens(tokens)` | 5 | Registers special-token strings with new IDs |

A single helper, `merge(ids, pair, new_id)`, does the one operation that both training and encoding need: it scans a sequence of IDs left to right and replaces every occurrence of a pair with the new ID. Everything else is bookkeeping around it.

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

We train on the first 90 percent of tiny Shakespeare and keep the last 10 percent to measure compression on unseen text. A vocabulary of 512 means 256 merges (Appendix A.2 has the code for this and the rest of the experiments).

On the machine used for this book, training took about 6 seconds for 512 entries and about 16 seconds for 1,024; the naive loop gets slower with every merge, since it rescans all chunks each time. The first merges are the most frequent pairs in English:

| Merge | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| New token | ␣t | he | ␣a | ou | ␣s | ␣m | in | ␣w | re | ha | ␣the |

The eleventh merge already produces " the". By the last merges, the tokenizer is adding whole words that are frequent in Shakespeare, such as " their", " there", " know", and " king". GPT-2's own first merges, visible in the first lines of its `merges.txt` (Section 3), were similar: " t", " a", "he", "in", "re".

Encoding a famous line shows the tokenizer at work. "To be, or not to be: that is the question." becomes 18 tokens:

`To` · `␣be` · `,` · `␣` · `or` · `␣not` · `␣to` · `␣be` · `:` · `␣that` · `␣is` · `␣the` · `␣` · `q` · `u` · `est` · `ion` · `.`

Frequent short words, with their leading spaces, are single tokens. "question" is still built from pieces, "q", "u", "est", and "ion", and with only 256 merges the tokenizer has learned "or" but not " or", so the space before it is a token of its own.

## Step 6: testing and comparing

A tokenizer must round-trip every input, not just the text it was trained on. Our tokenizer has never seen a character outside the 65 that occur in Shakespeare, but because its base vocabulary is all 256 bytes, it can still encode anything. We tested it on five strings chosen to be awkward: plain English, emoji and accented letters, Chinese, Japanese, and Korean, a snippet of Python, and runs of spaces, tabs, and newlines (Appendix A.3). Every test string round-trips exactly. The Chinese text, though, gets no compression at all: the seven characters "中文分词很难。" are 21 bytes, and they become 21 tokens, one per byte, because none of its byte pairs occurred in the training data. That is the multilingual cost of Section 8 in its most extreme form.

Next, special tokens. We add an end-of-text token, which gets the next free ID, 512, and encode the string "the end<|endoftext|>A new document" both ways. With `allow_special=True`, the special string becomes the single ID 512 in the middle of a 13-token sequence, and decoding restores the original text. Without it, ID 512 never appears: the characters "<", "|", and so on are encoded as ordinary text.

Finally, compression on the held-out text, compared with GPT-2's tokenizer. We also train a second tokenizer with 1,024 entries:

| Tokenizer | Vocabulary | Held-out tokens | Bytes per token |
|---|---|---|---|
| Ours | 512 | 59,401 | 1.88 |
| Ours | 1,024 | 49,416 | 2.26 |
| GPT-2 | 50,257 | 36,059 | 3.09 |

GPT-2's tokenizer, with a vocabulary about fifty times larger and trained on web text rather than Shakespeare, compresses unseen Shakespeare best. These are the numbers quoted in Section 2. The 512-entry figure also matches, token for token, the Hugging Face BPE trainer with the same settings in Section 7, which is a useful independent check that the training loop is right.

## Step 7: matching GPT-2 exactly

If our implementation is really the same algorithm as GPT-2's, then loading GPT-2's vocabulary and merges into it should reproduce GPT-2's tokenization exactly. OpenAI published those files with GPT-2, and they are mirrored on the Hugging Face Hub as `vocab.json` (a map from token strings to IDs) and `merges.txt` (the 50,000 merges in rank order).

Two details need care. First, both files write bytes using GPT-2's printable-character map from Section 3 (`Ġ` for a space and so on), so we need the inverse of that map. Second, GPT-2 does not number its single-byte tokens by byte value: ID 0 is "!", not byte 0. That is what `byte_ids` is for.

The loader, `load_gpt2` (Appendix A.4), fills in all four tables of a fresh `ByteBPE`. It rebuilds GPT-2's byte-to-character map and inverts it, so that every token string in the files can be turned back into bytes. It reads `vocab.json` into `vocab`, looks up the ID of each single-byte token to fill `byte_ids`, and then reads `merges.txt` line by line: each line names two tokens, and the merge maps their pair of IDs to the ID of their concatenation. Finally, it registers `<|endoftext|>` as the only special token.

Before trusting the loader, we check the property that `_encode_chunk` relies on: in GPT-2's files, the token created by the merge of rank $`r`$ has ID $`256 + r`$, so merge IDs grow with rank. The loaded tokenizer has 50,000 merges and 50,257 vocabulary entries, and all 50,000 merges satisfy the property. (Python dictionaries preserve insertion order, so enumerating `merges` visits them in rank order.)

Now the real test: compare our IDs with `tiktoken`'s on a set of eight deliberately awkward strings, then on the entire tiny Shakespeare corpus (Appendix A.5). Every test string produces exactly the same IDs as `tiktoken`, including emoji, three scripts, tabs and runs of spaces, numbers, and code. The special token is handled identically: "<|endoftext|> hi" becomes [50256, 23105] in both when special tokens are allowed. And all 338,025 tokens of the full 1,115,394-character corpus match. Our hundred lines of Python implement GPT-2's tokenizer.

They are also much slower. On the same machine, our encoder took about 7 seconds for the whole corpus and `tiktoken` about 0.4 seconds, and the gap grows for larger inputs. Production encoders are written in compiled languages, cache the encodings of frequent chunks, and use better data structures than rescanning each chunk for the minimum-rank pair. None of that changes the output.

## What we built and what production tokenizers add

The tokenizer in this section contains every essential idea of byte-level BPE: a 256-byte base vocabulary that makes every input encodable, a regex pre-tokenizer that keeps merges inside chunks, merges learned by frequency and applied in rank order, lossless decoding, and special tokens that are inserted deliberately rather than parsed from text. Production libraries add speed (compiled code, caching, incremental training), serialization formats such as `tokenizer.json`, offset tracking for mapping tokens back to characters, and conveniences such as padding, truncation, and chat templates, all covered in Sections 6 and 9.

The suggested code labs for this chapter build on exactly this code, the listing in Appendix A.1: train the tokenizer on text of your choice, match GPT-2 on a larger sample, extend it with chat role markers and a streaming decoder, and use its IDs to build the packed training blocks, padded batches, and embedding lookups of Section 9.

## Key takeaways

- A complete byte-level BPE tokenizer needs only four tables (vocabulary, merges, special tokens, and the byte-to-ID map) and a few short functions.
- Training counts pairs over a table of distinct pre-tokenized chunks, merges the most frequent pair, and assigns new IDs in order, so IDs double as merge ranks.
- Encoding repeatedly applies the lowest-rank merge present in each chunk; decoding concatenates bytes, which makes the round trip lossless for any input.
- Special tokens should be recognized only when the caller explicitly allows it.
- Loaded with GPT-2's published vocabulary and merges, the from-scratch tokenizer reproduces `tiktoken`'s GPT-2 token IDs exactly, on awkward test strings and on all 338,025 tokens of tiny Shakespeare; production libraries differ in speed, not in output.

## Further reading

Radford, Alec, et al. "Language Models Are Unsupervised Multitask Learners." OpenAI technical report, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

Sennrich, Rico, et al. "Neural Machine Translation of Rare Words with Subword Units." In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715–1725, 2016. https://arxiv.org/abs/1508.07909.

## Appendix: Code for Section 5.10

The first listing is the complete tokenizer; save it as `bpe.py`. The remaining listings reproduce the experiments in this section. Run them in order in one Python session after running A.1 (or after `from bpe import ByteBPE`). They need the `regex`, `tiktoken`, and `huggingface_hub` packages and download tiny Shakespeare and GPT-2's vocabulary files.

### A.1 The complete tokenizer (`bpe.py`)

The whole byte-level BPE tokenizer: pre-tokenization, training, encoding, decoding, and special tokens.

Notebook: [5.10-A.1-the-complete-tokenizer-bpe-py.ipynb](../../code/05-tokenizer/5.10-A.1-the-complete-tokenizer-bpe-py.ipynb)

### A.2 Training on tiny Shakespeare

Train a 512-entry tokenizer on 90 percent of the corpus, show the first and last merges, and encode a line.

Notebook: [5.10-A.2-training-on-tiny-shakespeare.ipynb](../../code/05-tokenizer/5.10-A.2-training-on-tiny-shakespeare.ipynb)

### A.3 Testing and comparing

Round-trip tests, special tokens, and compression on held-out text compared with GPT-2.

Notebook: [5.10-A.3-testing-and-comparing.ipynb](../../code/05-tokenizer/5.10-A.3-testing-and-comparing.ipynb)

### A.4 Loading GPT-2's vocabulary and merges

Convert GPT-2's published files into the four tables and check that merge IDs grow with rank.

Notebook: [5.10-A.4-loading-gpt-2-vocabulary-and-merges.ipynb](../../code/05-tokenizer/5.10-A.4-loading-gpt-2-vocabulary-and-merges.ipynb)

### A.5 Matching tiktoken

Compare IDs with `tiktoken` on awkward strings and on the whole corpus.

Notebook: [5.10-A.5-matching-tiktoken.ipynb](../../code/05-tokenizer/5.10-A.5-matching-tiktoken.ipynb)

