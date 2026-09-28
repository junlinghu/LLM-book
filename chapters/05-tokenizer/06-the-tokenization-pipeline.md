# 5.6 The Tokenization Pipeline

Sections 3 to 5 described the subword algorithms, the part of a tokenizer that decides how to cut a chunk of text into vocabulary pieces. A production tokenizer wraps that algorithm in a pipeline of other steps, and those steps cause at least as many surprises as the algorithm itself. This section walks through the pipeline in order: normalization, pre-tokenization, the subword model, post-processing with special tokens, and decoding. Along the way it covers chat templates, the security problem of special tokens in user text, streaming decoding, and mapping tokens back to character positions.

## The stages

Most tokenizers, whatever their subword algorithm, can be described as five stages:

1. **Normalization** cleans up the raw string: Unicode normalization, and optionally lowercasing, accent removal, or whitespace cleanup.
2. **Pre-tokenization** splits the normalized string into chunks, such as words, numbers, and punctuation. The subword model never merges across chunk boundaries.
3. **The subword model** (BPE, WordPiece, or unigram) segments each chunk into vocabulary pieces and maps them to IDs.
4. **Post-processing** adds special tokens, such as a beginning-of-sequence token or BERT's `[CLS]` and `[SEP]`.
5. **Decoding** turns a sequence of IDs back into a string, undoing whatever conventions the earlier stages introduced.

The Hugging Face `tokenizers` library represents a tokenizer literally as these components, which makes it easy to see how two tokenizers differ (Appendix A.1):

| Stage | `bert-base-uncased` | `gpt2` |
|---|---|---|
| Normalizer | `BertNormalizer` | none |
| Pre-tokenizer | `BertPreTokenizer` | `ByteLevel` |
| Model | `WordPiece` | `BPE` |
| Post-processor | `TemplateProcessing` | `ByteLevel` |
| Decoder | `WordPiece` | `ByteLevel` |

GPT-2 has no normalizer at all, a byte-level pre-tokenizer, and a BPE model. BERT normalizes, splits on whitespace and punctuation, runs WordPiece, and adds its special tokens through a template. We take the stages in order.

## Normalization

Unicode often offers more than one way to write what a reader sees as the same text. The letter "é" can be stored as a single code point (U+00E9) or as "e" followed by a combining acute accent (U+0301). The two strings look identical but are different sequences of code points, and therefore different bytes and different tokens. In `cl100k_base`, the composed "café" is two tokens, [936, 59958], while the decomposed one is three, [936, 1897, 54939]. After NFC normalization, described next, the two strings are equal.

**Unicode normalization** maps such equivalent strings to a canonical form. The standard defines four forms, of which two matter most for tokenizers:

- **NFC** (canonical composition) combines base characters and combining marks into single code points where possible. It only merges sequences that Unicode defines as *canonically equivalent*, meaning they represent exactly the same character, so it never changes what the text means.
- **NFKC** (compatibility composition) goes further and also replaces *compatibility* characters with their plain equivalents: ligatures, full-width forms, superscripts, circled numbers, and similar variants.

A few examples show the difference:

| Input | After NFKC |
|---|---|
| café | café (unchanged) |
| ﬁne (with the "ﬁ" ligature) | fine |
| ①② | 12 |
| ＡＢＣ１２３ (full-width) | ABC123 |
| x² | x2 |

NFKC makes text more uniform, which helps a tokenizer's statistics, but it also destroys information. "x²" and "x2" mean different things, and a model trained on NFKC-normalized text can never output a superscript. That is why NFKC-style normalization is common in tokenizers built for natural-language understanding, such as SentencePiece's default rule (Section 5), but rare in tokenizers for generative models.

Other normalizations are more aggressive still. BERT's uncased models **lowercase** everything and **strip accents**, and BERT's normalizer also puts spaces around Chinese, Japanese, and Korean ideographs. It turns "Héllo WÖRLD 你好" into `"hello world  你  好 "`.

For a classifier, treating "Apple" and "apple" as the same word may be acceptable. For a generative model, it is not: the model must be able to write capital letters, accents, and exact spacing. Most LLM tokenizers therefore do little or no normalization. GPT-2's byte-level tokenizer does none; whatever bytes come in are what the model sees. The cost is that equivalent strings, like the two spellings of "café" above, get different tokens, so a pipeline that feeds such a model may want to apply NFC itself before tokenizing, since NFC never changes meaning.

**Whitespace cleanup** is another normalization choice. Collapsing runs of spaces or converting tabs to spaces makes natural text more uniform, but for source code, where indentation carries meaning, it is destructive. Section 5 showed that SentencePiece's default settings collapse whitespace.

## Pre-tokenization

Pre-tokenization splits the text into chunks before the subword model runs, and merges never cross chunk boundaries. It is the stage where most of a tokenizer's hand-designed rules live.

The simplest pre-tokenizers split on whitespace and punctuation. BERT's splits off every punctuation character and discards the whitespace. GPT-2's uses the regular expression from Section 3 and keeps the spaces, attached to the following word and displayed as `Ġ`. On the string "Hello, world!  It's", with two spaces before "It's", the two produce these chunks:

| Pre-tokenizer | Chunks |
|---|---|
| BERT | `Hello` `,` `world` `!` `It` `'` `s` |
| GPT-2 | `Hello` `,` `Ġworld` `!` `Ġ` `ĠIt` `'s` |

Each chunk also comes with its character offsets in the original string. BERT split "It's" into three pieces, while GPT-2 kept the contraction "'s" together. Of the two spaces before "It's", GPT-2 made one its own chunk and attached the other to " It".

**Numbers** get special treatment in many newer tokenizers, because the way digits are grouped affects how easily a model learns arithmetic (Section 8). GPT-2's pattern allows any run of digits to be a chunk, so BPE learned tokens for whatever digit strings were frequent in its training data. Later tokenizers restrict this. The original LLaMA models split every number into individual digits (Touvron et al. 2023). The `cl100k_base` and `o200k_base` patterns allow chunks of at most three digits and never attach a space to a number. Here is how the three tokenizers split " 1234567 and 2024-09-28":

| Tokenizer | Tokens |
|---|---|
| `gpt2` | ` 123` `45` `67` ` and` ` 2024` `-` `09` `-` `28` |
| `cl100k_base`, `o200k_base` | ` ` `123` `456` `7` ` and` ` ` `202` `4` `-` `09` `-` `28` |

GPT-2 split 1234567 into "123", "45", "67", because those happened to be learned tokens, and it has a single token for " 2024". The newer tokenizers split every number left to right into groups of three, with the space as a separate token, so the year becomes " ", "202", "4".

The pre-tokenizer is a hard limit on what the vocabulary can contain. If the pattern separates letters from digits, no token can contain both. If it attaches spaces to the following word, no token can end with a space. These rules are chosen by hand and are rarely revisited once a model is trained.

## Special tokens

**Special tokens** are vocabulary entries that do not correspond to any text the tokenizer would produce from ordinary input. They carry structural signals:

- **Beginning and end of sequence.** Many models use a token such as `<s>` or `<|begin_of_text|>` at the start of every input, and a token such as `</s>` or `<|endoftext|>` at the end of a document. GPT-2 uses a single token, `<|endoftext|>` (ID 50256), placed between documents during training. When a model generates the end token, the application stops generation.
- **Padding.** A `[PAD]` or `<pad>` token fills out shorter sequences in a batch to a common length. Padding positions are masked out (Section 9).
- **Unknown.** `[UNK]` or `<unk>`, for tokenizers that can meet unencodable input (Section 4).
- **Separator and classification.** BERT uses `[CLS]` at the start and `[SEP]` between and after segments.
- **Chat role markers.** Chat models add tokens that mark the start and end of each turn and who is speaking, as described below.

Special tokens are added by the application or by the tokenizer's post-processing step, not produced from the text. That raises a security question: what should happen if the *user's text* contains the literal string `<|endoftext|>`?

The answer must be that it is encoded as ordinary text. If user input could produce real special tokens, a user could end a document early, forge a system message, or impersonate the assistant in a chat, a form of **prompt injection** at the token level. `tiktoken` is strict about this by default: it refuses to encode text that contains a special token's string unless told what to do. Encoding "hi <|endoftext|>" with GPT-2's encoding raises a `ValueError` ("Encountered text corresponding to disallowed special token"). Passing `allowed_special={"<|endoftext|>"}` gives [5303, 220, 50256]: "hi", a space, and the real special token, ID 50256. Passing `disallowed_special=()` gives [5303, 1279, 91, 437, 1659, 5239, 91, 29], where the string is spelled with seven ordinary tokens for "<", "|", "end", "of", "text", "|", ">", and the model sees harmless text (Appendix A.4). `enc.encode_ordinary` does the same. The application should use the second behavior for any text that comes from a user or a document, and insert real special tokens itself.

Not every library is this careful. Hugging Face tokenizers, by default, *do* turn special-token strings in the input into special tokens. Using the tokenizer of Qwen2.5-0.5B-Instruct, a chat model whose turns end with `<|im_end|>`, the input string "<|im_end|>" becomes the single token [151645], the real end-of-turn token. Passing `split_special_tokens=True` encodes the string as ordinary text. When you build prompts from untrusted text, check which behavior your library uses.

## Chat templates

A chat model is still a model of a single token sequence. A conversation with a system message and several user and assistant turns has to be flattened into one sequence, with special tokens marking where each turn begins and ends and who is speaking. The exact format is the model's **chat template**, and it is fixed when the model is fine-tuned for chat: a model works best with exactly the template it was trained on. Hugging Face tokenizers store the template and apply it (Appendix A.5). For Qwen2.5, a conversation with a system message, a user question ("What is a token?"), an assistant answer ("A unit of text."), and a second user message is flattened into one string in which every turn has the same shape:

- `<|im_start|>system` ⏎ `You are a helpful assistant.<|im_end|>`
- `<|im_start|>user` ⏎ `What is a token?<|im_end|>`
- `<|im_start|>assistant` ⏎ `A unit of text.<|im_end|>`
- `<|im_start|>user` ⏎ `Give an example.<|im_end|>`
- `<|im_start|>assistant` ⏎ (left open for the reply)

Here ⏎ marks a newline, and each turn is followed by a newline as well. The markers `<|im_start|>` and `<|im_end|>` are the single special tokens 151644 and 151645.

With `add_generation_prompt=True`, the template ends with an open assistant turn, so the model's next tokens are its reply; the application stops generating when the model emits `<|im_end|>`. Other model families use different markers and layouts, which is why using the wrong template, or building the prompt string by hand with a small mistake such as a missing newline, quietly degrades a chat model.

Chat templates are also where the special-token rule above matters most. The template's markers must be real special tokens, and the message contents, which come from users and documents, must never be able to produce them.

## Decoding

Decoding maps IDs back to text and undoes the conventions of the earlier stages: byte-level tokenizers concatenate bytes and decode UTF-8, SentencePiece replaces `▁` with spaces, and WordPiece joins `##` pieces to the preceding piece and puts spaces between the rest. Special tokens are either rendered as their strings or skipped, depending on a flag.

One trap concerns **streaming**. Applications usually display a model's output as it is generated, decoding one token at a time. With a byte-level tokenizer, a single token can end in the middle of a multi-byte UTF-8 character. GPT-2's tokenizer encodes "I ❤️ 東京" as nine tokens:

| Token ID | 40 | 43074 | 97 | 37929 | 10545 | 251 | 109 | 12859 | 105 |
|---|---|---|---|---|---|---|---|---|---|
| Bytes | `I` | `␣ E2 9D` | `A4` | `EF B8 8F` | `␣ E6` | `9D` | `B1` | `E4 BA` | `AC` |
| Decoded alone | I | ␣� | � | (U+FE0F) | ␣� | � | � | � | � |

The heart "❤" is the three bytes E2 9D A4, and GPT-2's tokenizer split them across two tokens. The emoji's second code point, the invisible variation selector U+FE0F, is a token of its own. The character "東" is split across three tokens. Decoding each token separately turns every incomplete byte sequence into the replacement character `�`, and the display fills with garbage, even though decoding the whole sequence at once gives the right text.

The fix is to buffer bytes until they form complete characters. Python's incremental UTF-8 decoder does exactly that: feed it each token's bytes as they arrive, and it returns only the characters that are complete, holding back any trailing partial character. A streaming decoder built on it takes about ten lines (Appendix A.6). On the same nine tokens it yields "I", " ", "❤", the variation selector, " ", "東", and "京". Every piece is now valid text. The decoder emitted the space as soon as it arrived and held back the partial heart until its last byte came in. Any streaming interface built on a byte-level tokenizer needs this buffering.

## Offsets and alignment

Many applications need to know which characters of the original text each token came from: highlighting the tokens that influenced a prediction, labeling spans such as names in the input, or pointing to the exact passage a model quoted. Because normalization can change the text and pre-tokenization can drop characters, the correspondence is not always obvious. The fast (Rust) tokenizers in Hugging Face track it through every stage and can return an **offset mapping**, the character span of each token in the original string. For "Tokenizers map text to IDs." with GPT-2's tokenizer:

| Token | Token | izers | ␣map | ␣text | ␣to | ␣IDs | . |
|---|---|---|---|---|---|---|---|
| Span | (0, 5) | (5, 10) | (10, 14) | (14, 19) | (19, 22) | (22, 26) | (26, 27) |

Each pair is a start and end character index. Note that the span of " map" is (10, 14), which includes its leading space. When a token holds only part of a character, as in the byte-split examples above, its span covers the whole character, so neighboring tokens can report overlapping spans. In the same tokenizer, the three tokens that together spell " 東" in "I ❤️ 東京" report the spans (4, 6), (5, 6), and (5, 6). Span-based applications must be prepared for tokens that share a character.

## Key takeaways

- A tokenizer is a pipeline: normalization, pre-tokenization, the subword model, post-processing with special tokens, and decoding.
- Unicode normalization maps equivalent strings to one form; NFC preserves meaning, while NFKC, lowercasing, and accent stripping destroy information, which is why generative LLM tokenizers normalize little or not at all.
- Pre-tokenization rules, such as regex splitting and grouping numbers into at most three digits, set hard limits on what tokens can exist.
- Special tokens mark structure and must be inserted by the application; text from users or documents must never be converted into special tokens, and libraries differ in their defaults.
- Chat templates flatten a conversation into one token sequence with role markers, and a model should be prompted with exactly its own template.
- Decoding one token at a time with a byte-level tokenizer requires buffering incomplete UTF-8 bytes, and fast tokenizers can map tokens back to character offsets.

## Further reading

Kudo, Taku, et al. "SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing." In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 66–71, 2018. https://arxiv.org/abs/1808.06226.

Radford, Alec, et al. "Language Models Are Unsupervised Multitask Learners." OpenAI technical report, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

Touvron, Hugo, et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971, 2023. https://arxiv.org/abs/2302.13971.

## Appendix: Code for Section 5.6

These listings reproduce the examples in this section. Run them in order in one Python session; they need the `transformers` and `tiktoken` packages and download the tokenizers on first use.

### A.1 Inspecting pipeline components

List the five components of the BERT and GPT-2 tokenizers.

```python
from transformers import AutoTokenizer

for name in ["bert-base-uncased", "gpt2"]:
    t = AutoTokenizer.from_pretrained(name).backend_tokenizer
    parts = [t.normalizer, t.pre_tokenizer, t.model, t.post_processor, t.decoder]
    print(name, [type(p).__name__ if p else None for p in parts])
```

Output:

```
bert-base-uncased ['BertNormalizer', 'BertPreTokenizer', 'WordPiece', 'TemplateProcessing', 'WordPiece']
gpt2 [None, 'ByteLevel', 'BPE', 'ByteLevel', 'ByteLevel']
```

### A.2 Normalization

Composed and decomposed "café", NFKC examples, and BERT's normalizer.

```python
import unicodedata, tiktoken

enc = tiktoken.get_encoding("cl100k_base")
composed, decomposed = "caf\u00e9", "cafe\u0301"
print(composed == decomposed, enc.encode(composed), enc.encode(decomposed))
print(unicodedata.normalize("NFC", decomposed) == composed)
```

Output:

```
False [936, 59958] [936, 1897, 54939]
True
```

```python
for s in ["café", "ﬁne", "①②", "ＡＢＣ１２３", "x²"]:
    print(repr(s), "->", repr(unicodedata.normalize("NFKC", s)))
```

Output:

```
'café' -> 'café'
'ﬁne' -> 'fine'
'①②' -> '12'
'ＡＢＣ１２３' -> 'ABC123'
'x²' -> 'x2'
```

```python
bert = AutoTokenizer.from_pretrained("bert-base-uncased").backend_tokenizer
print(repr(bert.normalizer.normalize_str("Héllo WÖRLD 你好")))
```

Output:

```
'hello world  你  好 '
```

### A.3 Pre-tokenization

BERT's and GPT-2's pre-tokenizers, and digit grouping in three OpenAI encodings.

```python
bert_pre = AutoTokenizer.from_pretrained("bert-base-uncased").backend_tokenizer.pre_tokenizer
gpt2_pre = AutoTokenizer.from_pretrained("gpt2").backend_tokenizer.pre_tokenizer
print(bert_pre.pre_tokenize_str("Hello, world!  It's"))
print(gpt2_pre.pre_tokenize_str("Hello, world!  It's"))
```

Output:

```
[('Hello', (0, 5)), (',', (5, 6)), ('world', (7, 12)), ('!', (12, 13)), ('It', (15, 17)), ("'", (17, 18)), ('s', (18, 19))]
[('Hello', (0, 5)), (',', (5, 6)), ('Ġworld', (6, 12)), ('!', (12, 13)), ('Ġ', (13, 14)), ('ĠIt', (14, 17)), ("'s", (17, 19))]
```

```python
for name in ["gpt2", "cl100k_base", "o200k_base"]:
    e = tiktoken.get_encoding(name)
    print(name, [e.decode([i]) for i in e.encode(" 1234567 and 2024-09-28")])
```

Output:

```
gpt2 [' 123', '45', '67', ' and', ' 2024', '-', '09', '-', '28']
cl100k_base [' ', '123', '456', '7', ' and', ' ', '202', '4', '-', '09', '-', '28']
o200k_base [' ', '123', '456', '7', ' and', ' ', '202', '4', '-', '09', '-', '28']
```

### A.4 Special tokens in user text

How `tiktoken` and a Hugging Face tokenizer treat special-token strings in the input.

```python
enc = tiktoken.get_encoding("gpt2")
try:
    enc.encode("hi <|endoftext|>")
except ValueError as err:
    print("ValueError:", str(err).splitlines()[0])
print(enc.encode("hi <|endoftext|>", allowed_special={"<|endoftext|>"}))
print(enc.encode("hi <|endoftext|>", disallowed_special=()))
```

Output:

```
ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'.
[5303, 220, 50256]
[5303, 1279, 91, 437, 1659, 5239, 91, 29]
```

```python
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
print(qwen("<|im_end|>")["input_ids"])
print(qwen("<|im_end|>", split_special_tokens=True)["input_ids"])
```

Output:

```
[151645]
[27, 91, 318, 6213, 91, 29]
```

### A.5 Chat templates

Apply Qwen2.5's chat template to a short conversation.

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a token?"},
    {"role": "assistant", "content": "A unit of text."},
    {"role": "user", "content": "Give an example."},
]
text = qwen.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
print(qwen.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"]))
```

Output:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is a token?<|im_end|>
<|im_start|>assistant
A unit of text.<|im_end|>
<|im_start|>user
Give an example.<|im_end|>
<|im_start|>assistant

[151644, 151645]
```

### A.6 Streaming decoding

Show tokens that split UTF-8 characters, then decode them incrementally.

```python
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("I ❤️ 東京")
print(ids)
print([enc.decode_single_token_bytes(i) for i in ids])
print([enc.decode([i]) for i in ids])
```

Output:

```
[40, 43074, 97, 37929, 10545, 251, 109, 12859, 105]
[b'I', b' \xe2\x9d', b'\xa4', b'\xef\xb8\x8f', b' \xe6', b'\x9d', b'\xb1', b'\xe4\xba', b'\xac']
['I', ' �', '�', '️', ' �', '�', '�', '�', '�']
```

```python
import codecs

def stream_decode(token_ids, enc):
    """Yield text pieces as tokens arrive, never splitting a UTF-8 character."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for t in token_ids:
        piece = decoder.decode(enc.decode_single_token_bytes(t))
        if piece:
            yield piece
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail

print(list(stream_decode(ids, enc)))
```

Output:

```
['I', ' ', '❤', '️', ' ', '東', '京']
```

### A.7 Offset mapping

Map each token back to its character span.

```python
gpt2 = AutoTokenizer.from_pretrained("gpt2")
out = gpt2("Tokenizers map text to IDs.", return_offsets_mapping=True)
print(list(zip(gpt2.convert_ids_to_tokens(out["input_ids"]), out["offset_mapping"])))
```

Output:

```
[('Token', (0, 5)), ('izers', (5, 10)), ('Ġmap', (10, 14)), ('Ġtext', (14, 19)), ('Ġto', (19, 22)), ('ĠIDs', (22, 26)), ('.', (26, 27))]
```
