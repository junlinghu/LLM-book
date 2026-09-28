# 5.9 Tokenizers in Practice

The previous sections built up tokenizers from their algorithms. In practice you will rarely implement one for production use; you will load an existing tokenizer, occasionally train a new one, and spend most of your effort making sure that the right tokenizer is used in the right way. This section covers the three libraries you are most likely to meet, `tiktoken`, Hugging Face `tokenizers` and `transformers`, and SentencePiece; how to inspect a tokenizer; the bugs that come up most often; the step from token IDs to the tensors a model consumes; and how to add tokens to an existing vocabulary.

## tiktoken

`tiktoken` is OpenAI's BPE library, written in Rust with a Python interface. It is fast, it only encodes and decodes (it does not train tokenizers), and it ships the encodings used by OpenAI's models: `gpt2`, `cl100k_base`, `o200k_base`, and a few others. `encoding_for_model` maps a model name to its encoding: `gpt2` to `gpt2`, `gpt-4` to `cl100k_base`, and `gpt-4o` to `o200k_base`.

The most common use is counting tokens before sending a request, to stay within a context window or to estimate cost. The prompt "Summarize the following report in three sentences." is 10 tokens in `o200k_base`, whose vocabulary has 200,019 entries.

Chat APIs add a few tokens per message for the chat template (Section 6), so a count of the message text alone is a close lower bound rather than an exact figure. `tiktoken` also enforces the special-token rule from Section 6 by default: `encode` raises an error on text that contains a special token's string unless you say whether it should be allowed, and `encode_ordinary` always treats such text as ordinary.

## Hugging Face tokenizers and transformers

The Hugging Face `tokenizers` library implements BPE, WordPiece, and unigram models in Rust, including training, and the `transformers` library wraps it so that every model on the Hugging Face Hub comes with its matching tokenizer. `AutoTokenizer.from_pretrained` downloads and loads the right one. Loaded this way, GPT-2's tokenizer encodes "Hello world" as [15496, 995], decodes it back, and reports `is_fast` as `True`. The `is_fast` flag says whether the tokenizer is backed by the Rust library; fast tokenizers support features such as offset mappings (Section 6). A tokenizer is saved as a small set of files, most importantly `tokenizer.json`, which describes the whole pipeline (normalizer, pre-tokenizer, model with its vocabulary and merges, post-processor, and decoder) in one JSON document.

**Training a new tokenizer** with the same pipeline as an existing one takes one call. Here we train a GPT-2-style tokenizer with 4,000 entries on tiny Shakespeare, reusing GPT-2's pre-tokenizer, byte-level alphabet, and special tokens (Appendix A.2). Comparing it with GPT-2's own tokenizer on two lines:

| Text | GPT-2 (50,257 entries) | Shakespeare-trained (4,000 entries) |
|---|---|---|
| GLOUCESTER: Now is the winter of our discontent | 13 tokens: `GL` `OU` `C` `ES` `TER` `:` `ĠNow` … `Ġdiscontent` | 10 tokens: `GLOUCESTER` `:` `ĠNow` … `Ġdisc` `ontent` |
| def f(x): return x | 7 tokens: `def` `Ġf` `(` `x` `):` `Ġreturn` `Ġx` | 10 tokens: `de` `f` `Ġf` `(` `x` `)` `:` `Ġreturn` `Ġ` `x` |

Saving the new tokenizer writes two files, `tokenizer.json` and `tokenizer_config.json`. The new tokenizer, with less than a tenth of GPT-2's vocabulary, encodes the Shakespeare line in fewer tokens, because "GLOUCESTER" (a speaker label that appears constantly in the corpus) earned a token of its own. On a line of Python, which never appears in Shakespeare, it does worse than GPT-2. Section 7's lesson in miniature: a tokenizer is good at the text it was trained on. For full control over each pipeline component, the `tokenizers` library can also build a tokenizer from parts, as in the vocabulary-size experiment of Section 7.

## SentencePiece

SentencePiece (Section 5) trains and runs unigram and BPE models. It can be used from Python, as in Section 5, or with its command-line tools, which are built with the C++ library. A single `spm_train` command, with options for the input file, the model prefix, the vocabulary size (2,000), the model type (unigram), byte fallback, and character coverage, trains a model on tiny Shakespeare. The Python package accepts the same options as a single string passed to `SentencePieceTrainer.train`.

Training produces `shakes.model`, a single binary file containing the vocabulary, the piece scores, and the normalization rules, and `shakes.vocab`, a readable list of pieces and scores. The `.model` file is all that is needed to encode and decode. Two options deserve attention when training: `character_coverage`, the fraction of characters in the training data that must be covered by the vocabulary (lower values, such as 0.9995, drop very rare characters, which is useful for languages with large character sets), and the normalization settings discussed in Section 5.

## Inspecting a tokenizer

Before using a tokenizer, and especially before training a model with one, look at it:

- **List the vocabulary and merges.** Skim the longest tokens, the last merges learned, and any tokens that look like fragments of usernames, URLs, or boilerplate; they hint at the tokenizer's training data and at possible glitch tokens (Section 8).
- **Visualize token boundaries** on representative text: prose, code, numbers, and every language the model should handle. Showing each token's decoded string, as the tables in this chapter do, is usually enough.
- **Check round-tripping.** For a set of unusual inputs (emoji, mixed scripts, combining characters, long runs of whitespace, very long numbers, control characters), confirm that decoding the encoding gives back the input, or understand exactly why it does not.

As an example, take six test strings: emoji with an accented word ("🙂🚀 and café"), three East Asian scripts ("中文 日本語 한국어"), a letter with two combining marks, tabs and newlines around a word, a long decimal expansion of π, and a string containing a NUL character. Both byte-level BPE tokenizers, GPT-2's and `o200k_base`, round-trip every test string. BERT's does not:

| Input | BERT encode, then decode |
|---|---|
| 🙂🚀 and café | [UNK] and cafe |
| 中文 日本語 한국어 | 中 文 日 本 語 한국어 |
| x with two combining marks | x |

BERT's tokenizer replaces the emoji with `[UNK]`, strips accents (turning "café" into "cafe" and dropping the combining marks entirely), and inserts spaces between Chinese and Japanese characters. The Korean word looks unchanged, but it is not: accent stripping works by decomposing characters (Unicode NFD) and removing the combining marks, and the decomposition also split each of the three Hangul syllables into its component letters (jamo), so the decoded word has eight code points where the input had three and no longer compares equal to it. None of that is a bug in BERT's tokenizer, which was designed for classification, but it would be a serious problem for a model that must generate text.

## Common bugs

A handful of mistakes account for most tokenization bugs in practice.

**Adding special tokens twice.** Many tokenizers add special tokens automatically. If you also add them to the text yourself, you get duplicates. BERT's tokenizer turns "[CLS] hello world [SEP]" into `[CLS]` `[CLS]` `hello` `world` `[SEP]` `[SEP]`, while plain "hello world" gives the intended `[CLS]` `hello` `world` `[SEP]`.

The same happens with beginning-of-sequence tokens when a chat template already includes one and the tokenizer adds another. Either let the tokenizer add special tokens or pass `add_special_tokens=False` and add them yourself, but not both.

**Mismatched tokenizer and model.** A model must be used with exactly the tokenizer it was trained with. Loading a different checkpoint's tokenizer, even one from the same family with the same vocabulary size, can silently produce IDs that mean different things to the model. The output is not an error message but degraded text. Always load the tokenizer from the same checkpoint as the model.

**The wrong padding side.** When a batch of prompts of different lengths is padded for generation, a decoder-only model generates after the *last* position of each row. If padding is added on the right, the last position of a short prompt is a padding token, and the model continues from padding. For batched generation, pad on the left; for training, where every position is scored and masked individually, either side works as long as the attention mask and loss mask are correct. The next section shows both.

**Silent truncation.** Tokenizers can be configured to truncate inputs at a maximum length, which is often what you want, but a document that is cut off without anyone noticing is a subtle data bug. GPT-2's tokenizer, for example, records a maximum length of 1,024 tokens, and with `truncation=True` it drops everything after that: a 2,000-word input comes back as exactly 1,024 IDs, with no warning.

Count tokens and log how many inputs were truncated, rather than discovering it later.

## From token IDs to model inputs

A model does not consume strings or even lists of IDs; it consumes batches of equal-length integer tensors. Getting from a tokenized corpus to those tensors takes a few steps, and the details matter for training.

**Packing a corpus for pretraining.** A language model is trained to predict each token from the ones before it. The standard recipe is to tokenize every document, join them into one long stream with an end-of-text token between documents, and cut training examples out of the stream as fixed-length blocks. For each block of $`T`$ input tokens, the targets are the same tokens shifted one position to the left: the target at position $`t`$ is the input at position $`t+1`$. Because the stream is one long array, it is usually saved to disk once, in a compact integer type, and loaded directly for training (Appendix A.5).

A tiny example shows the layout. Three short documents, "First document.", "Second, slightly longer document.", and "Third.", become a stream of 14 GPT-2 token IDs, with the end-of-text token (ID 50256) after each document:

`[5962, 3188, 13, 50256, 12211, 11, 4622, 2392, 3188, 13, 50256, 22747, 13, 50256]`

Drawing two random blocks of eight tokens gives an input batch `x` and a target batch `y`, each of shape 2 × 8. The first input block decodes to ".<|endoftext|>Second, slightly longer document." and its targets to "<|endoftext|>Second, slightly longer document.<|endoftext|>": every row of `y` is the matching row of `x` shifted by one. A block can span a document boundary, as both of these do; the end-of-text token tells the model that what follows is unrelated to what came before. GPT-2's IDs are below 65,536, so they fit in 16-bit unsigned integers, halving the storage compared with 32 bits. A vocabulary larger than 65,536, such as `o200k_base`, needs 32-bit storage.

**Padding a batch of prompts.** For inputs that must stay separate, such as prompts in a batch or fine-tuning examples, sequences of different lengths are padded to a common length, and an **attention mask** records which positions are real (1) and which are padding (0), so the model can ignore the padding. GPT-2 has no padding token, and a common convention is to reuse the end-of-text token for padding, relying on the mask. Padding "Hello there" (two tokens) to the length of a seven-token prompt gives:

| Padding side | IDs of the short prompt | Attention mask |
|---|---|---|
| Right | 15496, 612, 50256, 50256, 50256, 50256, 50256 | 1, 1, 0, 0, 0, 0, 0 |
| Left | 50256, 50256, 50256, 50256, 50256, 15496, 612 | 0, 0, 0, 0, 0, 1, 1 |

With left padding, the last position of every row is a real token, which is what batched generation needs. When training on padded batches, the loss must also skip padding positions; a common convention is to set their target to a value such as $`-100`$ that the loss function ignores.

**Turning IDs into vectors.** The final step is the embedding lookup from Section 1: each ID selects a row of a learned table of shape $`V \times d`$. In PyTorch that table is `nn.Embedding`. With GPT-2's vocabulary and a width of 64, the table has shape 50,257 × 64, and looking up the 2 × 8 batch from the packing example gives a tensor of shape 2 × 8 × 64. In general, a batch of shape (batch size, sequence length) of IDs becomes a tensor of shape (batch size, sequence length, model width), and from here on the model works only with vectors. The number of rows in the table must equal the tokenizer's vocabulary size (or exceed it, if padded for hardware as in Section 7); an ID outside that range is an indexing error.

## Extending a tokenizer

Sometimes an existing model needs tokens it does not have: domain terms that it splits into many pieces, a new language, or new special tokens for a chat template or tool calls. A vocabulary can be extended after training, within limits. New tokens are added to the tokenizer, the embedding table (and output layer) gets new rows, and the model is fine-tuned so that it learns to use them.

For example, GPT-2 splits " tokenizer" into `Ġtoken` and `izer` (IDs 11241 and 7509). After `tok.add_tokens([" tokenizer"])`, the vocabulary has 50,258 entries, and " my tokenizer works" tokenizes as `Ġmy`, ` tokenizer`, `Ġworks` (Appendix A.7). The new token gets ID 50257, the first free ID. Tokens added this way are matched as whole strings before the BPE model runs, which is why it is shown with a real space rather than `Ġ`.

How the new embedding rows are initialized matters. A random row is a vector the model has never seen, which can disrupt its outputs until fine-tuning catches up. A better default is to initialize each new token's embedding as the average of the embeddings of the pieces it replaces, so the model starts with a reasonable guess of its meaning. For " tokenizer", that is the mean of the rows for `Ġtoken` and `izer`, copied into row 50257 of a new table of shape 50,258 × 64.

With a Hugging Face model, `model.resize_token_embeddings(len(tok))` adds the rows, which can then be initialized the same way. Extending a vocabulary is a patch, not a substitute for a good tokenizer: the new tokens have far less training than the original ones, and existing merges are not revisited, so the tokenizer still splits the rest of the text as before.

## Key takeaways

- `tiktoken` encodes and decodes OpenAI's encodings quickly and is strict about special tokens; Hugging Face `tokenizers` and `transformers` load the matching tokenizer for any Hub model and can train new ones; SentencePiece trains and runs unigram and BPE models from a single `.model` file.
- Inspect a tokenizer before using it: read the vocabulary, visualize token boundaries on representative text, and test round-tripping on unusual inputs.
- The most common bugs are duplicated special tokens, a tokenizer that does not match the model, right-padding for batched generation, and silent truncation.
- For pretraining, join tokenized documents with end-of-text tokens, save the stream compactly, and cut fixed-length blocks whose targets are the inputs shifted by one.
- For separate sequences, pad to a common length and pass an attention mask; the embedding table then turns IDs of shape (batch, length) into vectors of shape (batch, length, width).
- New tokens can be added to an existing vocabulary, with their embedding rows best initialized from the pieces they replace, followed by fine-tuning.

## Further reading

Kudo, Taku, et al. "SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing." In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 66–71, 2018. https://arxiv.org/abs/1808.06226.

Radford, Alec, et al. "Language Models Are Unsupervised Multitask Learners." OpenAI technical report, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

## Appendix: Code for Section 5.9

These listings reproduce the examples in this section. Run the Python listings in order in one session; they need `tiktoken`, `transformers`, `torch`, and `numpy`, and A.2 downloads tiny Shakespeare. A.3 is a shell command that needs the SentencePiece command-line tools.

### A.1 tiktoken basics

Map model names to encodings and count the tokens in a prompt.

```python
import tiktoken

for model in ["gpt2", "gpt-4", "gpt-4o"]:
    print(model, tiktoken.encoding_for_model(model).name)
```

Output:

```
gpt2 gpt2
gpt-4 cl100k_base
gpt-4o o200k_base
```

```python
enc = tiktoken.get_encoding("o200k_base")
prompt = "Summarize the following report in three sentences."
print(len(enc.encode(prompt)), enc.n_vocab)
```

Output:

```
10 200019
```

### A.2 Loading and training Hugging Face tokenizers

Load GPT-2's tokenizer, then train a 4,000-entry tokenizer on tiny Shakespeare with the same pipeline.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
enc_out = tok("Hello world")
print(enc_out["input_ids"], tok.decode(enc_out["input_ids"]), tok.is_fast)
```

Output:

```
[15496, 995] Hello world True
```

```python
import os, urllib.request

if not os.path.exists("shakespeare.txt"):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "shakespeare.txt")
corpus = open("shakespeare.txt", encoding="utf-8").read().split("\n\n")

new_tok = tok.train_new_from_iterator(corpus, vocab_size=4000)
for s in ["GLOUCESTER: Now is the winter of our discontent", "def f(x): return x"]:
    print(len(tok.tokenize(s)), tok.tokenize(s))
    print(len(new_tok.tokenize(s)), new_tok.tokenize(s))
new_tok.save_pretrained("shakes-tokenizer")
print(sorted(os.listdir("shakes-tokenizer")))
```

Output:

```
13 ['GL', 'OU', 'C', 'ES', 'TER', ':', 'ĠNow', 'Ġis', 'Ġthe', 'Ġwinter', 'Ġof', 'Ġour', 'Ġdiscontent']
10 ['GLOUCESTER', ':', 'ĠNow', 'Ġis', 'Ġthe', 'Ġwinter', 'Ġof', 'Ġour', 'Ġdisc', 'ontent']
7 ['def', 'Ġf', '(', 'x', '):', 'Ġreturn', 'Ġx']
10 ['de', 'f', 'Ġf', '(', 'x', ')', ':', 'Ġreturn', 'Ġ', 'x']
['tokenizer.json', 'tokenizer_config.json']
```

### A.3 Training with the SentencePiece command line

Train a unigram model with byte fallback.

```bash
spm_train --input=shakespeare.txt --model_prefix=shakes --vocab_size=2000 \
          --model_type=unigram --byte_fallback=true --character_coverage=1.0
```

### A.4 Round-trip tests and common bugs

Round-trip unusual strings, add special tokens twice, and truncate a long input.

```python
tests = ["🙂🚀 and café", "中文 日本語 한국어", "x\u0301\u0302", "\t\t  indented\n\n\n",
         "3.14159265358979323846", "a\x00b"]
for name in ["gpt2", "o200k_base"]:
    e = tiktoken.get_encoding(name)
    print(name, all(e.decode(e.encode(t)) == t for t in tests))
bert = AutoTokenizer.from_pretrained("bert-base-uncased")
for t in tests[:3]:
    print(repr(t), "->", repr(bert.decode(bert(t, add_special_tokens=False)["input_ids"])))
```

Output:

```
gpt2 True
o200k_base True
'🙂🚀 and café' -> '[UNK] and cafe'
'中文 日本語 한국어' -> '中 文 日 本 語 한국어'
'x́̂' -> 'x'
```

```python
print(bert.convert_ids_to_tokens(bert("[CLS] hello world [SEP]")["input_ids"]))
print(bert.convert_ids_to_tokens(bert("hello world")["input_ids"]))
```

Output:

```
['[CLS]', '[CLS]', 'hello', 'world', '[SEP]', '[SEP]']
['[CLS]', 'hello', 'world', '[SEP]']
```

```python
print(tok.model_max_length, len(tok("word " * 2000, truncation=True)["input_ids"]))
```

Output:

```
1024 1024
```

### A.5 Packing a corpus into training blocks

Join documents with end-of-text tokens, save the stream as 16-bit integers, and draw input and target blocks.

```python
import numpy as np
import torch

enc = tiktoken.get_encoding("gpt2")
docs = ["First document.", "Second, slightly longer document.", "Third."]
stream = []
for d in docs:
    stream.extend(enc.encode_ordinary(d))
    stream.append(enc.eot_token)                 # <|endoftext|>, ID 50256
print(len(stream), stream)

np.array(stream, dtype=np.uint16).tofile("corpus.bin")   # GPT-2 IDs fit in 16 bits
data = torch.from_numpy(np.fromfile("corpus.bin", dtype=np.uint16).astype(np.int64))

def get_batch(data, batch_size, block_size, generator=None):
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x, y

g = torch.Generator().manual_seed(0)
x, y = get_batch(data, batch_size=2, block_size=8, generator=g)
print(x)
print(y)
print(repr(enc.decode(x[0].tolist())), "->", repr(enc.decode(y[0].tolist())))
```

Output:

```
14 [5962, 3188, 13, 50256, 12211, 11, 4622, 2392, 3188, 13, 50256, 22747, 13, 50256]
tensor([[   13, 50256, 12211,    11,  4622,  2392,  3188,    13],
        [50256, 12211,    11,  4622,  2392,  3188,    13, 50256]])
tensor([[50256, 12211,    11,  4622,  2392,  3188,    13, 50256],
        [12211,    11,  4622,  2392,  3188,    13, 50256, 22747]])
'.<|endoftext|>Second, slightly longer document.' -> '<|endoftext|>Second, slightly longer document.<|endoftext|>'
```

### A.6 Padding and embedding

Pad a batch on the right and on the left, then look up embeddings.

```python
tok.pad_token = tok.eos_token
prompts = ["Hello there", "A much longer prompt than the first"]
for side in ["right", "left"]:
    tok.padding_side = side
    batch = tok(prompts, padding=True, return_tensors="pt")
    print(side, batch["input_ids"].tolist(), batch["attention_mask"].tolist())
```

Output:

```
right [[15496, 612, 50256, 50256, 50256, 50256, 50256], [32, 881, 2392, 6152, 621, 262, 717]] [[1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]]
left [[50256, 50256, 50256, 50256, 50256, 15496, 612], [32, 881, 2392, 6152, 621, 262, 717]] [[0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1]]
```

```python
torch.manual_seed(0)
emb = torch.nn.Embedding(num_embeddings=enc.n_vocab, embedding_dim=64)
h = emb(x)
print(emb.weight.shape, h.shape)
```

Output:

```
torch.Size([50257, 64]) torch.Size([2, 8, 64])
```

### A.7 Adding a token

Add " tokenizer" to GPT-2's vocabulary and initialize its embedding from the pieces it replaces.

```python
tok = AutoTokenizer.from_pretrained("gpt2")
old_vocab = len(tok)
pieces = tok(" tokenizer")["input_ids"]
print(pieces, tok.convert_ids_to_tokens(pieces))
tok.add_tokens([" tokenizer"])
print(len(tok), tok.convert_ids_to_tokens(tok(" my tokenizer works")["input_ids"]))
```

Output:

```
[11241, 7509] ['Ġtoken', 'izer']
50258 ['Ġmy', ' tokenizer', 'Ġworks']
```

```python
torch.manual_seed(0)
old_emb = torch.nn.Embedding(old_vocab, 64)      # stands in for a trained embedding table
new_emb = torch.nn.Embedding(len(tok), 64)
with torch.no_grad():
    new_emb.weight[:old_vocab] = old_emb.weight
    new_emb.weight[old_vocab] = old_emb.weight[pieces].mean(dim=0)
print(new_emb.weight.shape)
```

Output:

```
torch.Size([50258, 64])
```
