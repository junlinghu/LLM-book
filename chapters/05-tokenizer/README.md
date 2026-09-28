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

## Sections

1. **[Why Tokenization Matters](01-why-tokenization-matters.md)**

   What encoding and decoding do, how token IDs select rows of the embedding table and index the output softmax, why context length, compute, and price are all counted in tokens, how the same sentence splits under five real tokenizers, and why many odd model behaviors are really tokenization effects.

2. **[Characters, Words, and Subwords](02-characters-words-and-subwords.md)**

   The tradeoffs of word-, character-, byte-, and subword-level units measured on tiny Shakespeare (including a 4.59 percent unknown-word rate for a word vocabulary), compression in bytes per token, and a short history from subword translation to GPT-2's byte-level BPE.

3. **[Byte Pair Encoding (BPE)](03-byte-pair-encoding.md)**

   BPE training and rank-ordered encoding, worked by hand on the classic "low, lower, newest, widest" corpus, then byte-level BPE, GPT-2's printable byte map (`Ġ` for a space), its regex pre-tokenizer, and BPE-dropout.

4. **[WordPiece](04-wordpiece.md)**

   The likelihood-based merge criterion and how it differs from BPE on the same toy corpus, `##` continuation pieces, greedy longest-match encoding checked against BERT's tokenizer, `[UNK]` and its costs, and where WordPiece is still used.

5. **[The Unigram Language Model and SentencePiece](05-unigram-and-sentencepiece.md)**

   Segmentation as a probabilistic model, Viterbi encoding, EM training with pruning, subword regularization, and SentencePiece's `▁` whitespace symbol, byte fallback, and default normalization, with a side-by-side comparison of the three algorithms.

6. **[The Tokenization Pipeline](06-the-tokenization-pipeline.md)**

   Normalization (NFC vs. NFKC and what each destroys), pre-tokenization and digit grouping, special tokens and why user text must never produce them, chat templates, streaming decoding that buffers partial UTF-8 characters, and offset mappings.

7. **[Choosing the Vocabulary Size](07-choosing-the-vocabulary-size.md)**

   What the vocabulary size costs in parameters and compute, the diminishing returns of larger vocabularies measured with a sweep from 512 to 16,000 entries, rare and intermediate tokens, sizes used by real models, hardware padding, and measuring compression across ten languages and on code.

8. **[Tokenization Artifacts and Failure Modes](08-tokenization-artifacts.md)**

   Spelling and letter counting, inconsistent digit splits and arithmetic, multilingual cost, whitespace and code, prompts that end in a space, glitch tokens such as " SolidGoldMagikarp", non-canonical token sequences, bits per byte for comparing models, and byte-level models.

9. **[Tokenizers in Practice](09-tokenizers-in-practice.md)**

   Using tiktoken, Hugging Face tokenizers, and SentencePiece, training a new tokenizer, inspecting and round-trip testing, common bugs, turning token IDs into packed training blocks, padded batches with attention masks, and embedding lookups, and extending a vocabulary.

10. **[Building a Byte-Level BPE Tokenizer from Scratch](10-bpe-from-scratch.md)**

    A complete byte-level BPE tokenizer in about a hundred lines of Python, trained on tiny Shakespeare, with special tokens and round-trip tests, then loaded with GPT-2's published merges and verified to reproduce tiktoken's GPT-2 token IDs exactly on test strings and on the whole corpus.

## Suggested code labs

1. **Byte-level BPE from scratch.** Build the byte-level BPE tokenizer of Section 10: first check your merge loop against five merges computed by hand on a tiny word-frequency table, then train on a few megabytes of text and verify lossless round-tripping on arbitrary Unicode input, including emoji, mixed scripts, and code. (This is original code written for this book.)
2. **Match GPT-2 exactly.** Load GPT-2's published vocabulary and merges into your encoder and confirm that it produces the same token IDs as tiktoken's `gpt2` encoding on a large sample of text. Track down and fix every mismatch, so your tokenizer can stand in for GPT-2's in any later code.
3. **Special tokens and chat templates.** Add end-of-text, padding, and chat role-marker tokens to your tokenizer. Encode a multi-turn conversation into one token sequence, decode it back, and show that a user string containing `<|endoftext|>` is encoded as ordinary text unless special tokens are explicitly allowed. Finish with a streaming decoder that buffers incomplete UTF-8 bytes when decoding one token at a time.
4. **From token IDs to model inputs.** Turn a tokenized corpus into training batches: concatenate documents with end-of-text separators, cut the stream into fixed-length blocks, and build input and target tensors shifted by one position. Separately, pad a batch of variable-length prompts and build its attention mask. Pass the IDs through an `nn.Embedding` table to get tensors of shape (batch, sequence length, model width), and save the tokenized corpus to disk for reuse.
5. **Tokenizer tour.** Tokenize the same inputs (English prose, parallel sentences in several other languages, Python code, and numbers from 1 to 10,000) with GPT-2, `cl100k_base`, `o200k_base`, and a SentencePiece model. Print token boundaries, report bytes per token for each language and domain, and count the distinct ways each tokenizer splits numbers.

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

Grattafiori, Aaron, et al. "The Llama 3 Herd of Models." arXiv preprint arXiv:2407.21783, 2024. https://arxiv.org/abs/2407.21783.

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

Song, Xinying, et al. "Fast WordPiece Tokenization." In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, 2021. https://arxiv.org/abs/2012.15524.

Touvron, Hugo, et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971, 2023. https://arxiv.org/abs/2302.13971.

Wu, Yonghui, et al. "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation." arXiv preprint arXiv:1609.08144, 2016. https://arxiv.org/abs/1609.08144.

Xue, Linting, et al. "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models." *Transactions of the Association for Computational Linguistics* 10 (2022): 291–306. https://arxiv.org/abs/2105.13626.
