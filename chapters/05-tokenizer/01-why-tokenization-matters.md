# 5.1 Why Tokenization Matters

A language model is a function from numbers to numbers. It never reads the letters of a prompt, and it never writes letters in its answer. What it consumes is a sequence of integers, and what it produces, one step at a time, is a probability distribution over integers. Something has to sit between human text and those integers, and that something is the **tokenizer**. It **encodes** a string into a sequence of integer **token IDs** before the model runs, and it **decodes** the model's output IDs back into a string afterward.

Tokenization is easy to treat as plumbing: a preprocessing step you call once and forget. That is a mistake. The tokenizer decides what the model's basic units are, how long every input is, how much every request costs, and whether some tasks are easy or needlessly hard. This section explains where tokens go inside a model, why nearly every quantity in LLM work is measured in tokens, and why many surprising model behaviors turn out to be tokenization effects. The rest of the chapter then opens up the tokenizer itself.

## Encode, decode, and the vocabulary

A tokenizer is defined by a **vocabulary**: a fixed, finite list of strings (or byte sequences), each with an integer ID. Encoding cuts a string into pieces that are all in the vocabulary and replaces each piece with its ID. Decoding looks up the piece for each ID and joins the pieces back together.

GPT-2's tokenizer, which OpenAI's `tiktoken` library ships, makes a convenient first example. It encodes the sentence "Tokenization is the first step." as seven IDs:

| ID | 30642 | 1634 | 318 | 262 | 717 | 2239 | 13 |
|---|---|---|---|---|---|---|---|
| Piece | `Token` | `ization` | `␣is` | `␣the` | `␣first` | `␣step` | `.` |

Here `␣` marks a space character. Decoding the seven IDs gives back the original sentence exactly (code in Appendix A.1).

Three things are already visible. First, the pieces are neither characters nor words: "Tokenization" became two pieces, "Token" and "ization", while common words stayed whole. Second, the space before a word is part of that word's token: the model sees `' is'`, not `'is'` preceded by a separate space. Third, decoding reverses encoding exactly. The IDs themselves are arbitrary labels; ID 262 means " the" only because this particular vocabulary says so. A different tokenizer assigns different IDs to different pieces.

The size of the vocabulary, usually written $`V`$, is a design choice. GPT-2's vocabulary has 50,257 entries. The tokenizers used by more recent OpenAI models, `cl100k_base` and `o200k_base`, have about 100,000 and 200,000. Section 7 discusses how to choose $`V`$.

## Where token IDs go inside the model

Token IDs enter and leave the model at exactly two places.

**At the input**, each ID selects one row of an **embedding table**, a learned matrix $`E`$ with one row per vocabulary entry. Chapter 4 introduced embeddings as learned vectors for discrete symbols; the tokenizer decides what those symbols are. If the model width is $`d`$, then $`E`$ has shape $`V \times d`$, and a sequence of $`T`$ token IDs becomes a $`T \times d`$ matrix of vectors:

```math
\mathbf{x}_t = E[\,\text{id}_t\,], \qquad t = 1, \dots, T.
```

Looking up a row is equivalent to multiplying a one-hot vector of length $`V`$ by $`E`$, which is why the embedding table is trained by backpropagation like any other weight matrix.

**At the output**, the model produces a vector of $`V`$ scores (logits), one per vocabulary entry, and a softmax turns them into a probability distribution over the next token. Chapter 2 showed that next-token prediction is classification over the vocabulary and that the training loss is the cross-entropy of that softmax. The number of classes in that classification problem is exactly the tokenizer's vocabulary size.

So the tokenizer fixes the shape of both ends of the network. Change the tokenizer, and the embedding table and the output layer no longer mean anything: row 262 of the embedding table was trained to represent " the", and in a new vocabulary row 262 might be something else entirely.

## Everything is counted in tokens

Because the model processes one position per token, the token becomes the unit in which almost everything is measured.

- **Context length.** A model's context window, the maximum input it can attend to, is a number of tokens, not characters or words. A tokenizer that uses fewer tokens for the same text effectively gives the model a longer memory.
- **Training data.** Pretraining corpora are described in tokens: "trained on 15 trillion tokens" is a statement about both the text and the tokenizer.
- **Compute.** The cost of a forward pass grows with the number of tokens processed. So does the time to generate an answer, since a model generates one token per step.
- **Price.** Commercial APIs charge per input token and per output token.

A single sentence shows how much the tokenizer matters. Take the string "Tokenizers don't see words; they see 1,234 bytes of ünïcödé." and encode it with five tokenizers in wide use: the three `tiktoken` encodings `gpt2`, `cl100k_base`, and `o200k_base`, and the tokenizers of BERT and T5 from the Hugging Face `transformers` library (code in Appendix A.2):

| Tokenizer | Tokens | "don't" | "1,234" | "ünïcödé" |
|---|---|---|---|---|
| GPT-2 | 23 | `␣don` `'t` | `␣1` `,` `234` | 8 pieces; `␣ü` split into 2 byte tokens |
| `cl100k_base` | 23 | `␣don` `'t` | `␣` `1` `,` `234` | `␣ü` `n` `ï` `c` `ö` `d` `é` |
| `o200k_base` | 19 | `␣don't` | `␣` `1` `,` `234` | `␣ün` `ïc` `öd` `é` |
| BERT (uncased) | 18 | `don` `'` `t` | `1` `,` `234` | `unicode` |
| T5 | 26 | `▁don` `'` `t` | `▁1,` `2` `34` | `▁` `ü` `n` `ï` `c` `ö` `dé` |

The three `tiktoken` encodings split the remaining English words identically, and all five tokenizers keep short common words such as "see" and "words" whole. They disagree about almost everything else: whether "don't" is one piece or two, whether the space before a digit belongs to the digit, how to cut a number, and how to handle accented letters. In GPT-2's case, the letter "ü" was split into two tokens that each decode to the replacement character `�`. That is not a bug. GPT-2's tokenizer works on bytes, "ü" is two bytes in UTF-8, and each byte on its own is not a valid character. Section 3 explains why working on bytes is still a good idea, and Section 6 shows how to decode such tokens correctly.

BERT's tokenizer goes further than the others: it lowercases everything and strips accents, so "ünïcödé" became the ordinary word "unicode", and the original spelling is gone for good. It also marks word-internal pieces with `##` instead of marking spaces. T5's tokenizer marks the start of each word with `▁` instead. The same sentence costs anywhere from 18 to 26 tokens, and the tokenizers do not even agree on what the units are. Sections 4 and 5 explain where these conventions come from.

## The tokenizer is trained first and then frozen

A modern tokenizer is not written by hand. Its vocabulary is **learned** from a corpus of text by one of the algorithms in Sections 3 to 5. But it is learned separately from the model, by a much simpler procedure (counting, not gradient descent), and it is learned *before* the model. The pipeline is:

1. Collect a sample of the kind of text the model will see.
2. Train the tokenizer on that sample, producing a vocabulary and the rules for splitting text.
3. Tokenize the full training corpus.
4. Train the model on the resulting token IDs.

After step 2 the tokenizer is frozen. Every later use of the model, whether further training or answering a prompt, must use exactly the same tokenizer, because the model's embedding table and output layer are tied to that vocabulary. Changing the tokenizer after training means building a new embedding table and either retraining the model or carefully adapting it, which is why tokenizer design decisions are made early and are hard to undo. Section 9 covers the limited ways a vocabulary can be extended after the fact.

This separation has a practical consequence that is easy to miss: the tokenizer never sees the training loss. It does not know which splits would make the model's job easier. It only knows which strings were frequent in its training sample. Most of the tokenization problems in Section 8 follow from that gap between what the tokenizer optimizes and what the model needs.

## Many odd behaviors are tokenization effects

Once you know the model sees tokens, several well-known failures stop being mysterious.

- **Counting letters.** Asked how many times "r" appears in "strawberry", a model has to answer about characters it never directly sees. With the GPT-2 tokenizer, " strawberry" (with a leading space) is a single token, ID 41236. The model receives one opaque ID and must have learned, from indirect evidence during training, what letters that ID contains.
- **Arithmetic.** GPT-2's tokenizer splits " 1234" into " 12" and "34", but it keeps " 2024" as a single token. The model must learn addition over digit groups whose boundaries shift from number to number.
- **Uneven costs across languages.** A sentence in Hindi can take several times as many tokens as the same sentence in English under a tokenizer trained mostly on English text, which makes the model slower and more expensive for Hindi speakers and leaves less room in the context window.
- **Sensitivity to whitespace.** " hello" and "hello" are different tokens with different IDs, and a prompt that ends in a space asks the model to continue from a token sequence it rarely saw in training.

None of these are failures of reasoning in the usual sense. They are consequences of what the tokenizer handed the model. Section 8 examines each of them, with measurements, and the lesson throughout the chapter is the same: when a model behaves strangely on a particular string, look at its tokens first.

## What the rest of the chapter covers

Section 2 compares the three natural choices of unit (characters, words, and subwords) and explains why subwords won. Sections 3 to 5 derive the three main subword algorithms: byte pair encoding, WordPiece, and the unigram language model with SentencePiece. Section 6 walks through the full pipeline around the subword algorithm: normalization, pre-tokenization, special tokens, and decoding. Section 7 covers vocabulary size, Section 8 the failure modes, and Section 9 the libraries and the step from token IDs to model inputs. Section 10 builds a byte-level BPE tokenizer from scratch and checks that it reproduces GPT-2's tokenizer exactly.

## Key takeaways

- A tokenizer encodes strings into integer IDs from a fixed vocabulary and decodes IDs back into strings.
- Token IDs index rows of the embedding table at the input, and the output softmax is a distribution over the same vocabulary, so the tokenizer fixes the shape of both ends of the model.
- Context length, training data, compute, latency, and price are all measured in tokens, and different tokenizers can need noticeably different numbers of tokens for the same text.
- The tokenizer is trained separately from the model, before it, and is then frozen; changing it later invalidates the embedding table and output layer.
- Many surprising model behaviors (letter counting, arithmetic, multilingual cost, whitespace sensitivity) come from tokenization, so inspecting tokens is the first debugging step.

## Further reading

Mielke, Sabrina J., et al. "Between Words and Characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP." arXiv preprint arXiv:2112.10508, 2021. https://arxiv.org/abs/2112.10508.

Radford, Alec, et al. "Language Models Are Unsupervised Multitask Learners." OpenAI technical report, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

Sennrich, Rico, et al. "Neural Machine Translation of Rare Words with Subword Units." In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715–1725, 2016. https://arxiv.org/abs/1508.07909.

## Appendix: Code for Section 5.1

These listings reproduce the results quoted in this section. Run them in order in one Python session; they need the `tiktoken` and `transformers` packages, and the second listing downloads two tokenizers from the Hugging Face Hub.

### A.1 Encoding and decoding with tiktoken

Encode a sentence with GPT-2's tokenizer, show each ID's piece, and decode the IDs back to text.

Notebook: [5.1-A.1-encoding-and-decoding-with-tiktoken.ipynb](../../code/05-tokenizer/5.1-A.1-encoding-and-decoding-with-tiktoken.ipynb)

### A.2 One sentence, five tokenizers

Encode the same sentence with three `tiktoken` encodings, then with BERT's and T5's tokenizers. The printed lines are wrapped here.

Notebook: [5.1-A.2-one-sentence-five-tokenizers.ipynb](../../code/05-tokenizer/5.1-A.2-one-sentence-five-tokenizers.ipynb)

