# 5.2 Characters, Words, and Subwords

Before choosing an algorithm, we have to choose what kind of unit a token should be. There are three natural candidates. A token could be a **word**, the unit people think in. It could be a **character** (or a **byte**), the unit computers store. Or it could be something in between: a **subword**, a piece that is sometimes a whole word and sometimes a fragment. Each choice trades vocabulary size against sequence length, and each handles unseen text differently. This section works through the tradeoffs with measurements on a real corpus and explains why every modern LLM uses subwords.

Throughout this section we use the "tiny Shakespeare" corpus, about 1.1 million characters of Shakespeare's plays that is widely used for small language-model experiments. The file has 1,115,394 characters drawn from only 65 distinct characters. We hold out the last 10 percent of the file as a test set and build every vocabulary from the first 90 percent only. The appendix at the end of this section contains the code for every measurement quoted here.

## Word-level tokenization

The most obvious tokenizer splits text on whitespace and punctuation and gives every distinct word its own ID. Early neural language models worked this way, typically keeping the most frequent tens of thousands of words and mapping everything else to a single unknown token, often written `[UNK]` or `<unk>`.

Word tokens are attractive. Sequences are short, since one token covers a whole word, and each token carries a lot of meaning. But three problems make word-level vocabularies a poor fit for LLMs.

**The vocabulary is huge and still incomplete.** Natural language has a long tail: most distinct words are rare. Suppose we build a word vocabulary from the training portion of tiny Shakespeare, splitting on whitespace and punctuation, and then apply it to the held-out portion (Appendix A.2):

| Measurement | Value |
|---|---|
| Distinct words (types) in the training portion | 12,569 |
| Types that occur exactly once | 5,753 |
| Word and punctuation tokens in the held-out portion | 26,844 |
| Held-out tokens not in the vocabulary | 1,231 (4.59%) |

Even this small, single-author corpus has 12,569 distinct word types in its training portion, and nearly half of them appear exactly once. Despite that, 4.59 percent of the words in the held-out text never appeared in training. The most frequent unknown words are PROSPERO, SEBASTIAN, ANTONIO, GONZALO, and MIRANDA: names of characters from *The Tempest*, a play that happens to fall in the held-out portion. A word-level model can do nothing with them except emit `[UNK]`. For web-scale text, with its typos, names, URLs, code, numbers, and hundreds of languages, the open-ended tail is far larger. No fixed word list can cover it.

**Related forms share nothing.** "run", "running", and "runner" get three unrelated IDs. The model must learn from scratch, for each form separately, what it means, even though the forms share a root. Rare forms get little training signal.

**Splitting into words is itself language-specific.** Whitespace separates words in English, but not in Chinese, Japanese, or Thai, and many languages build long compound words. A word tokenizer needs rules for each language.

## Character-level and byte-level tokenization

At the other extreme, each character is a token. The vocabulary is tiny (tiny Shakespeare uses only 65 distinct characters), and there are no unknown words at all: any new word is just a new sequence of known characters.

The cost is sequence length. The held-out portion of tiny Shakespeare is 111,540 characters long, so a character-level model sees 111,540 positions, compared with 26,844 word tokens. Each position carries little information, so the model must spend depth and capacity composing characters into words before it can begin to work with meaning. Since the cost of processing a sequence grows with its length, and since a model's context window is a fixed number of positions, long sequences are expensive in both compute and memory.

Characters also do not fully solve the unknown-symbol problem. Unicode defines far more characters than any reasonable vocabulary would hold, spread across scripts, symbols, and emoji. A character vocabulary built from a training corpus will still meet characters it has never seen.

**Bytes** fix this. Every string can be encoded in UTF-8 as a sequence of bytes, and there are only 256 possible byte values. A tokenizer whose base units are bytes can represent any string in any language with a vocabulary of 256, with no unknown symbols ever. ASCII characters take one byte each, but other characters take two to four (Appendix A.3):

| Character | UTF-8 bytes |
|---|---|
| a | 97 |
| é | 195 169 |
| 你 | 228 189 160 |
| 🙂 | 240 159 153 130 |

So pure byte-level sequences are even longer than character sequences for most of the world's languages: a Chinese sentence is about three times as many bytes as characters. Models that operate directly on bytes exist (Section 8 mentions ByT5), but they pay for it in sequence length.

## Subword tokenization

**Subword tokenization** sits between the two extremes. Frequent strings, including most common words, get their own tokens. Rare words are split into smaller pieces that are themselves in the vocabulary, and in the worst case into single characters or bytes. With a vocabulary of a few tens of thousands of entries, a subword tokenizer can encode any text, never needs an unknown token (if it falls back to bytes), and keeps sequences short for common text.

GPT-2's tokenizer shows the pattern on a few related words, each with a leading space as it would appear mid-sentence (Appendix A.4):

| Word | GPT-2 tokens |
|---|---|
| run | `␣run` |
| running | `␣running` |
| runner | `␣runner` |
| runners | `␣runners` |
| tokenization | `␣token` `ization` |
| unhappiness | `␣unh` `appiness` |

Frequent words such as " running" are single tokens. Less frequent words are split: " tokenization" into " token" and "ization", pieces that also appear in many other words. The splits often line up with meaningful parts of words (morphemes), as in " token" + "ization", but not always: " unhappiness" became " unh" + "appiness", not " un" + "happiness". The pieces come from frequency statistics, not from linguistic knowledge. The model can still learn what " unh" + "appiness" means, because it sees that combination in context, but the split does not hand it the structure for free.

Chapter 4's fastText embeddings used a related idea from the other direction: they kept whole words as the units but built each word's vector partly from vectors for its character n-grams, so that rare words could borrow from common ones. Subword tokenization goes further and makes the pieces themselves the units the model reads and writes.

## The core tradeoff: vocabulary size vs. sequence length

All three choices sit on a single tradeoff curve. A bigger vocabulary means longer pieces, which means fewer tokens per text. A smaller vocabulary means shorter pieces and longer sequences. The standard way to measure where a tokenizer lands is its **compression**: the average number of bytes (or characters) of text per token on some representative corpus. Higher is shorter sequences.

On the held-out Shakespeare text, which is pure ASCII so that bytes and characters coincide:

| Tokenizer | Vocabulary size | Tokens | Bytes per token |
|---|---|---|---|
| Bytes (or characters) | 256 | 111,540 | 1.00 |
| Byte-level BPE trained on the training portion | 512 | 59,401 | 1.88 |
| Byte-level BPE trained on the training portion | 1,024 | 49,416 | 2.26 |
| GPT-2's tokenizer | 50,257 | 36,059 | 3.09 |
| Words and punctuation (with 4.59% unknown) | 12,569 | 26,844 | 4.16 |

The BPE rows come from the tokenizer we build in Section 10, trained on the first 90 percent of the file. The word-level row looks best on compression, but it achieves that by giving up on 4.59 percent of the words. GPT-2's tokenizer was trained on a very different corpus (web text, not Shakespeare) and still reaches about 3 bytes per token here while being able to encode any input at all.

Compression is not the only thing that matters. A larger vocabulary makes the embedding table and output layer larger, and each token is seen less often during training. Section 7 returns to this tradeoff in detail. But the table shows why subwords are the practical sweet spot: most of the compression of words, with none of the unknown-word problem.

## Why subwords won

Subword tokenization combines several advantages that no other unit offers at once:

1. **Open vocabulary.** Any string can be encoded, especially when the base units are bytes. There is no unknown-token problem.
2. **Good compression.** Common words and frequent word pieces are single tokens, so typical text needs only a few bytes' worth of sequence positions per token.
3. **Sharing between related forms.** Pieces such as "ization", "ing", or a common root recur across many words, so rare words are built from pieces the model has seen many times.
4. **Language independence.** The vocabulary is learned from data, so the same algorithm works for any language or for code, as long as the training corpus includes them.

## A brief history

Word-level vocabularies with an unknown token were standard in neural language models and neural machine translation until the mid-2010s. Translation made their weakness obvious: names, numbers, and rare words are exactly the things a translation system must copy or transliterate correctly. Sennrich et al. (2016) proposed segmenting rare words into subword units learned with byte pair encoding, and showed that this let a translation model handle rare and unseen words by translating their pieces. Around the same time, Google's neural machine translation system used a closely related method called WordPiece (Wu et al. 2016). Kudo (2018) introduced the unigram language model tokenizer, and the SentencePiece library (Kudo and Richardson 2018) made subword tokenization language-independent by working directly on raw text.

GPT-2 (Radford et al. 2019) took the next step that most LLMs have followed: byte-level BPE, which runs the BPE algorithm over UTF-8 bytes instead of characters so that no input is ever unknown. Sections 3 to 5 derive these algorithms in turn.

## Key takeaways

- Word-level vocabularies give short sequences but are huge, still meet unknown words (4.59 percent of held-out words even on tiny Shakespeare), and share nothing between related forms.
- Character- and byte-level tokenization never meets unknown input (bytes cover all of Unicode with 256 symbols) but produces long sequences with little meaning per token.
- Subword tokenization keeps frequent strings whole and splits rare ones into reusable pieces, giving an open vocabulary with good compression.
- Compression, measured in bytes or characters per token, summarizes where a tokenizer sits on the vocabulary size vs. sequence length tradeoff.
- Subword pieces come from frequency statistics and only sometimes match linguistic morphemes.

## Further reading

Kudo, Taku, et al. "SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing." In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 66–71, 2018. https://arxiv.org/abs/1808.06226.

Mielke, Sabrina J., et al. "Between Words and Characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP." arXiv preprint arXiv:2112.10508, 2021. https://arxiv.org/abs/2112.10508.

Radford, Alec, et al. "Language Models Are Unsupervised Multitask Learners." OpenAI technical report, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

Sennrich, Rico, et al. "Neural Machine Translation of Rare Words with Subword Units." In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715–1725, 2016. https://arxiv.org/abs/1508.07909.

Wu, Yonghui, et al. "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation." arXiv preprint arXiv:1609.08144, 2016. https://arxiv.org/abs/1609.08144.

## Appendix: Code for Section 5.2

These listings reproduce the measurements in this section. Run them in order in one Python session; they need the `tiktoken` package, and the first listing downloads tiny Shakespeare.

### A.1 Loading tiny Shakespeare

Download the corpus, split it 90/10, and count its characters.

Notebook: [5.2-A.1-loading-tiny-shakespeare.ipynb](../../code/05-tokenizer/5.2-A.1-loading-tiny-shakespeare.ipynb)

### A.2 A word-level vocabulary

Build a word vocabulary from the training portion and measure unknown words in the held-out portion.

Notebook: [5.2-A.2-a-word-level-vocabulary.ipynb](../../code/05-tokenizer/5.2-A.2-a-word-level-vocabulary.ipynb)

### A.3 UTF-8 bytes

Show the UTF-8 encoding of characters from different scripts.

Notebook: [5.2-A.3-utf-8-bytes.ipynb](../../code/05-tokenizer/5.2-A.3-utf-8-bytes.ipynb)

### A.4 Subword splits in GPT-2

Tokenize related words with GPT-2's tokenizer.

Notebook: [5.2-A.4-subword-splits-in-gpt-2.ipynb](../../code/05-tokenizer/5.2-A.4-subword-splits-in-gpt-2.ipynb)

