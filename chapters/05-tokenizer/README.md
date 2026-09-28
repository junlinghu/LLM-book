# Chapter 5: Tokenizer

A language model never sees text. It sees a sequence of integers, and the tokenizer is the component that turns a string into those integers and back again. Chapter 4 showed how a model maps discrete symbols to learned vectors through an embedding table; this chapter decides what those symbols are. The choice sets the size of the output softmax, how many positions a document takes up, how much each request costs, and whether the model can spell, count digits, or read a language other than English efficiently. This chapter explains why modern LLMs settled on subword tokenization, derives the three main algorithms (byte pair encoding, WordPiece, and the unigram language model), walks through the full pipeline from normalization to decoding, and catalogs the failure modes that trace back to tokenization. Along the way we build a byte-level BPE tokenizer from scratch and check it against production libraries.

## Learning goals

- Explain what a tokenizer does, why models need one, and how it connects text to the embedding table and the output softmax.
- Compare character-, word-, and subword-level tokenization in terms of vocabulary size, sequence length, and out-of-vocabulary handling.
- Implement byte pair encoding (BPE) training and encoding from scratch, including byte-level BPE as used in GPT-2.
- Explain how WordPiece and the unigram language model differ from BPE, and what SentencePiece adds.
- Describe the full tokenization pipeline: normalization, pre-tokenization, the subword model, special tokens, and decoding.
- Reason about vocabulary size: its effect on sequence length, parameter count, compute, and rare-token training.
- Recognize tokenization artifacts (arithmetic, spelling, multilingual cost, whitespace and code, glitch tokens) and diagnose them by inspecting token boundaries.
- Use tiktoken and Hugging Face `tokenizers` in practice, and measure a tokenizer's compression on a corpus.

## Outline

### 1. Why tokenization matters
- Models operate on integer IDs; the tokenizer is the bridge between strings and IDs, in both directions (encode and decode)
- Where token IDs go: each ID selects a row of the embedding table (Chapter 4), and the model's output is a softmax over the same vocabulary (Chapter 2)
- Tokens are the unit of everything downstream: context length, training data size, compute per example, and API pricing are all counted in tokens
- A worked example: one sentence tokenized by several real tokenizers, with different splits and different token counts
- The tokenizer is trained separately, before the model, and then frozen: changing it later means changing the embedding table and retraining or adapting the model
- Many surprising model behaviors (miscounting letters, odd arithmetic errors, higher cost for some languages) are tokenization effects, not reasoning failures

### 2. Characters, words, and subwords
- **Word-level tokenization**: short sequences and meaningful units, but a huge vocabulary, no way to represent unseen words (the out-of-vocabulary or `[UNK]` problem), and no sharing between related forms ("run", "running", "runner")
- **Character-level tokenization**: a tiny vocabulary and no unknown words, but long sequences and little meaning per token, so the model must spend capacity composing characters into words
- **Byte-level tokenization**: 256 symbols cover every string in every language via UTF-8; the longest sequences of all (a single Chinese character is 3 bytes)
- **Subword tokenization**: frequent words stay whole, rare words split into reusable pieces ("tokenization" might become "token" + "ization"); a vocabulary of tens of thousands of entries covers open-ended text
- The core tradeoff: vocabulary size vs. sequence length, measured as compression (bytes or characters per token)
- Why subwords won: open vocabulary, good compression, and pieces that often line up with morphology
- A brief history: from word vocabularies in early neural language models, to subwords in neural machine translation (Sennrich et al. 2016; Wu et al. 2016), to byte-level BPE in GPT-2 (Radford et al. 2019)

### 3. Byte pair encoding (BPE)
- Origin: a data compression algorithm that repeatedly replaces the most frequent pair of bytes with a new symbol (Gage 1994), adapted for subword segmentation by Sennrich et al. (2016)
- **Training**: start from a base vocabulary of characters (or bytes); count all adjacent symbol pairs in the corpus; merge the most frequent pair into a new symbol; record the merge; repeat until the vocabulary reaches the target size
- Worked example on a tiny corpus ("low", "lower", "newest", "widest"), showing pair counts and the vocabulary after each merge
- Training on word frequencies, not raw text: pre-tokenize, count each distinct word once with its frequency, and merge within words only
- **Encoding** new text: apply the learned merges in the order they were learned (the merge rank), not greedy longest match
- **Decoding**: map IDs back to symbols and concatenate; why decoding is lossless for byte-level BPE
- Tie-breaking and determinism: the same text must always produce the same tokens
- Complexity: naive training recounts pairs after every merge; incremental pair-count updates make it practical on large corpora
- **Byte-level BPE (GPT-2)**: run BPE over UTF-8 bytes instead of Unicode characters, so the base vocabulary is exactly 256 symbols and no input is ever unknown; GPT-2 maps each byte to a printable character so merges can be stored as text (Radford et al. 2019)
- GPT-2's regex pre-tokenizer: split text into chunks (letters, digits, punctuation, whitespace, common English contractions) before BPE, so merges never cross those boundaries (for example, "dog." and "dog!" share the token "dog")
- **BPE-dropout**: randomly skip merges during training-time encoding to expose the model to multiple segmentations of the same word (Provilkov et al. 2020)

### 4. WordPiece
- Origin in Japanese and Korean voice search (Schuster and Nakajima 2012) and Google's neural machine translation system (Wu et al. 2016); the tokenizer used by BERT (Devlin et al. 2019)
- The training difference from BPE: choose the merge that most increases the likelihood of the training data, which amounts to scoring a pair by its count divided by the product of its parts' counts, rather than by raw frequency
- The `##` continuation prefix marks pieces that do not start a word ("playing" becomes "play" + "##ing")
- Encoding by greedy longest-match-first within each word, and falling back to `[UNK]` for a word that cannot be segmented
- Where WordPiece is still used today (BERT-family encoders) and why decoder LLMs mostly moved to byte-level BPE

### 5. The unigram language model and SentencePiece
- **Unigram LM** (Kudo 2018) works in the opposite direction from BPE: start with a large candidate vocabulary and prune it down
- The model: each token has a probability, and the probability of a segmentation is the product of its tokens' probabilities:

```math
P(\mathbf{x}) = \prod_{i=1}^{M} p(x_i), \qquad \mathbf{x}^* = \arg\max_{\mathbf{x} \in S(X)} P(\mathbf{x})
```

- where $`S(X)`$ is the set of all segmentations of the text $`X`$ into vocabulary tokens
- Encoding with the Viterbi algorithm: find the most probable segmentation by dynamic programming
- Training with expectation-maximization: estimate token probabilities, compute how much the total likelihood would drop if each token were removed, prune the least useful tokens (keeping single characters), and repeat
- **Subword regularization**: sample segmentations from the model during training instead of always using the best one, a form of data augmentation (Kudo 2018)
- **SentencePiece** (Kudo and Richardson 2018): a library that treats the input as a raw stream of Unicode characters, with whitespace encoded as an ordinary symbol (`▁`), so no language-specific pre-tokenizer is needed and decoding is exactly reversible
- SentencePiece implements both BPE and unigram models; byte fallback handles characters that are not in the vocabulary by encoding them as UTF-8 bytes
- Comparing the three algorithms side by side: training direction (merge up vs. prune down), selection criterion, encoding procedure, and typical users

### 6. The tokenization pipeline
- The stages of a production tokenizer: normalization, pre-tokenization, the subword model, post-processing (special tokens), and decoding
- **Normalization**: Unicode normalization (NFC vs. NFKC), lowercasing, accent stripping, and whitespace cleanup; what each one gains and what information it destroys (why most LLM tokenizers do little or no normalization)
- **Pre-tokenization**: splitting on whitespace and punctuation, regex-based splitting (GPT-2 and later tiktoken patterns), splitting numbers into single digits or groups of up to three digits, and how pre-tokenization choices limit what merges can learn
- **Special tokens**: beginning and end of sequence, padding, unknown, separator and classification tokens in BERT-style models, and the role-marker tokens used in chat templates
- Special tokens should be added by the application, not produced from user text: why a tokenizer must never turn a user string such as `<|endoftext|>` into the real special token, and how libraries control this
- Chat templates: how a conversation with system, user, and assistant turns is flattened into one token sequence
- **Decoding** back to text: joining pieces, handling the `▁` and `##` markers, and why decoding one token at a time can produce incomplete UTF-8 bytes (streaming output must buffer partial characters)
- Offsets and alignment: mapping tokens back to character spans, which matters for highlighting, span labeling, and citations

### 7. Choosing the vocabulary size
- What the vocabulary size $`V`$ controls: the embedding table and the output projection each have $`V \times d`$ parameters, where $`d`$ is the model width, and the softmax costs grow with $`V`$
- Larger vocabularies give shorter sequences, so more text fits in a fixed context and each training step or generated token covers more text
- Smaller vocabularies mean each token is seen more often in training; very large vocabularies contain rare tokens with poorly trained embeddings
- Typical sizes in practice: about 30K for BERT's WordPiece, 50,257 for GPT-2, 32K for the original LLaMA models, about 100K for tiktoken's `cl100k_base`, and 128K or more in several recent models; the trend toward larger vocabularies as models and multilingual coverage grow
- Padding the vocabulary size to a multiple of a power of two for hardware efficiency
- Measuring a tokenizer: compression (bytes per token) on held-out text in each target language and domain, vocabulary utilization, and fertility (tokens per word)
- The training corpus of the tokenizer matters as much as its size: a tokenizer trained mostly on English web text compresses code, math, and other languages poorly

### 8. Tokenization artifacts and failure modes
- **Spelling and character-level tasks**: a model that sees "strawberry" as two or three tokens has no direct view of its letters, so counting or reversing letters is hard
- **Numbers and arithmetic**: inconsistent splits of multi-digit numbers (for example "1234" vs. "12" + "34") make digit-wise algorithms hard to learn; single-digit or right-to-left grouping helps (Singh and Strouse 2024)
- **Multilingual cost**: the same content can take several times more tokens in some languages than in English, which means higher cost, less effective context, and slower generation for those users (Petrov et al. 2023)
- **Whitespace and code**: leading spaces are part of tokens (" hello" and "hello" are different tokens); indentation, tabs, and runs of spaces in code; why later tokenizers added tokens for runs of whitespace
- **Prompt boundary effects**: a prompt that ends with a trailing space, or in the middle of a word, gives the model a token sequence it rarely saw in training, which degrades completions
- **Glitch tokens**: tokens that appear in the tokenizer's training data but almost never in the model's training data get nearly untrained embeddings and can cause bizarre outputs; how to detect them (Land and Bartolo 2024)
- **Non-uniqueness**: the same string can be reached by many token sequences, but the model only ever saw the canonical one; consequences for scoring text and for constrained decoding
- **Comparing losses across tokenizers**: per-token loss and perplexity are not comparable between models with different tokenizers; normalize to bits per byte or per character
- Alternatives that avoid a learned vocabulary: byte-level models such as ByT5 (Xue et al. 2022), and their cost in sequence length

### 9. Tokenizers in practice
- **tiktoken**: OpenAI's fast BPE library; encodings such as `gpt2`, `cl100k_base`, and `o200k_base`; counting tokens before sending a request
- **Hugging Face `tokenizers`** and `transformers`: `AutoTokenizer`, fast (Rust) vs. slow (Python) tokenizers, the `tokenizer.json` format, and training a new tokenizer on your own corpus
- **SentencePiece** in practice: training with the `spm_train` command or the Python API, and the `.model` file
- Inspecting a tokenizer: listing the vocabulary and merges, visualizing token boundaries, and checking round-trip encode-decode on unusual inputs (emoji, mixed scripts, code, very long numbers)
- Common bugs: adding special tokens twice, mismatched tokenizer and model checkpoints, padding side for batched generation, and silently truncated inputs
- Extending a tokenizer with new tokens for a domain or language: resizing the embedding table and initializing the new rows (for example, as the average of the embeddings of the pieces the new token replaces)

### 10. Building a byte-level BPE tokenizer from scratch
- Goal: a tokenizer of about one hundred lines of Python, written for this book, that trains on a small corpus and reproduces the behavior of production byte-level BPE
- Step 1: convert text to UTF-8 bytes and split it with a regex pre-tokenizer
- Step 2: training loop: count pairs, merge the most frequent pair, record the merge, repeat until the target vocabulary size
- Step 3: encode by repeatedly applying the lowest-rank merge available in each chunk
- Step 4: decode by concatenating the bytes of each token and decoding UTF-8, replacing invalid sequences
- Step 5: add special tokens and make sure they are matched only when the application allows it
- Step 6: test round-tripping on English, other languages, emoji, and code; compare the compression with tiktoken's GPT-2 encoding on the same text
- Step 7: load GPT-2's published merges into our encoder and verify that it produces exactly the same token IDs as tiktoken

## Suggested code labs

1. **Tokenizer tour.** Tokenize the same paragraphs (English prose, another language, Python code, and a list of numbers) with GPT-2, `cl100k_base`, `o200k_base`, a BERT WordPiece tokenizer, and a SentencePiece model. Print the token boundaries and report tokens per word and bytes per token for each.
2. **BPE by hand, then in code.** Run five merges of BPE on a tiny word-frequency table on paper, then implement character-level BPE training and encoding and check that it reproduces your hand-computed merges.
3. **Byte-level BPE from scratch.** Build the byte-level BPE tokenizer of Section 10, train it on a few megabytes of text, and verify lossless round-tripping on arbitrary Unicode input, including emoji and mixed scripts.
4. **Match GPT-2 exactly.** Load GPT-2's vocabulary and merges into your encoder and confirm that it produces the same IDs as tiktoken on a large sample of text. Track down and fix every mismatch.
5. **Unigram LM segmentation.** Given a small vocabulary with token probabilities, implement Viterbi segmentation, then implement sampling of segmentations for subword regularization and compare the samples with the best segmentation.
6. **Vocabulary size sweep.** Train BPE tokenizers with vocabulary sizes from 1K to 64K on the same corpus. Plot bytes per token against vocabulary size on held-out text, and count how many tokens appear fewer than ten times in the training data.
7. **Multilingual cost.** Tokenize parallel sentences (the same content translated into several languages) with two or three tokenizers and plot the token count relative to English for each language.
8. **Numbers and arithmetic.** Show how GPT-2 and a digit-splitting tokenizer split the numbers from 1 to 10,000. Count how many distinct segmentation patterns appear, and discuss which scheme makes digit-by-digit addition easiest to learn.
9. **Find under-trained tokens.** For a small open model, compute the norm of each embedding row and list the tokens with the smallest norms. Try them in prompts and describe what happens.

## Key takeaways

- A tokenizer maps text to integer IDs and back; the IDs index the embedding table, and the model predicts a distribution over the same vocabulary.
- Subword tokenization balances vocabulary size against sequence length and removes out-of-vocabulary words, which is why every modern LLM uses it.
- BPE builds a vocabulary by merging frequent pairs; WordPiece merges by likelihood gain; the unigram LM prunes a large vocabulary using token probabilities.
- Byte-level BPE covers every possible input with a 256-symbol base vocabulary and decodes losslessly.
- Normalization, pre-tokenization, and special-token handling matter as much as the subword algorithm, and mistakes there cause subtle bugs.
- Vocabulary size trades sequence length against parameters and rare-token training; larger vocabularies have become common in recent models.
- Many odd model behaviors (spelling, arithmetic, multilingual cost, glitch tokens) come from tokenization, and inspecting token boundaries is the first step in diagnosing them.

## Further reading

Bostrom, Kaj, et al. "Byte Pair Encoding Is Suboptimal for Language Model Pretraining." In *Findings of the Association for Computational Linguistics: EMNLP 2020*, 2020. https://arxiv.org/abs/2004.03720.

Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics*, 4171–4186, 2019. https://arxiv.org/abs/1810.04805.

Gage, Philip. "A New Algorithm for Data Compression." *The C Users Journal* 12, no. 2 (1994): 23–38.

Kudo, Taku. "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates." In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics*, 66–75, 2018. https://arxiv.org/abs/1804.10959.

Kudo, Taku, et al. "SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing." In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 66–71, 2018. https://arxiv.org/abs/1808.06226.

Land, Sander, et al. "Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models." In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 2024. https://arxiv.org/abs/2405.05417.

Mielke, Sabrina J., et al. "Between Words and Characters: A Brief History of Open-Vocabulary Modeling and Tokenization in NLP." arXiv preprint arXiv:2112.10508, 2021. https://arxiv.org/abs/2112.10508.

Petrov, Aleksandar, et al. "Language Model Tokenizers Introduce Unfairness Between Languages." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2305.15425.

Provilkov, Ivan, et al. "BPE-Dropout: Simple and Effective Subword Regularization." In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 1882–1892, 2020. https://arxiv.org/abs/1910.13267.

Radford, Alec, et al. "Language Models Are Unsupervised Multitask Learners." OpenAI technical report, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

Schuster, Mike, et al. "Japanese and Korean Voice Search." In *2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 5149–5152, 2012. https://doi.org/10.1109/ICASSP.2012.6289079.

Sennrich, Rico, et al. "Neural Machine Translation of Rare Words with Subword Units." In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 1715–1725, 2016. https://arxiv.org/abs/1508.07909.

Singh, Aaditya K., et al. "Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs." arXiv preprint arXiv:2402.14903, 2024. https://arxiv.org/abs/2402.14903.

Wu, Yonghui, et al. "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation." arXiv preprint arXiv:1609.08144, 2016. https://arxiv.org/abs/1609.08144.

Xue, Linting, et al. "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models." *Transactions of the Association for Computational Linguistics* 10 (2022): 291–306. https://arxiv.org/abs/2105.13626.
