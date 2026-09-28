# Chapter 7: Generative Pretraining (GPT)

_Draft in progress._

## Outline

### 7.1 Language modeling as next-token prediction
- Factoring the probability of a text into next-token predictions, one token at a time
- The training loss: cross-entropy on the next token at every position, and perplexity
- From the encoder-decoder of Chapter 6 to a decoder-only model: drop the encoder and cross-attention, keep the causal mask

### 7.2 The GPT architecture
- A stack of decoder blocks: masked self-attention and a feed-forward network, with residual connections
- Pre-layer normalization, GELU, learned position embeddings, and tied input and output embeddings (Chapter 4)
- Common changes in recent models: RMSNorm, SwiGLU, and rotary position embeddings
- Counting parameters: about 12 × layers × width² plus the embedding table

### 7.3 Pretraining data
- Where the text comes from: web crawls, books, code, and reference text
- Cleaning the data: quality filtering, deduplication, and removing personal information
- Choosing the data mixture
- Turning documents into training examples: tokenizing (Chapter 5), end-of-text separators, and packing into fixed-length blocks

### 7.4 Training at scale
- The optimizer setup: AdamW, learning-rate warmup and cosine decay, and gradient clipping
- Fitting large models on hardware: mixed precision, gradient accumulation, and data parallelism
- Estimating compute: about 6 × parameters × training tokens
- Scaling laws and compute-optimal training: how to split a budget between model size and data

### 7.5 Generating text
- Sampling one token at a time from the model's output distribution
- Greedy decoding, temperature, top-k, and top-p (nucleus) sampling
- Repetition and how sampling settings affect it
- Caching keys and values so each new token is cheap (details in Chapter 13)

### 7.6 What pretraining produces
- In-context learning: zero-shot and few-shot prompting
- What a base model can and can't do: it continues text rather than following instructions
- Why pretraining is followed by fine-tuning (preview of Chapter 8)
- Summary and exercises

## Code Labs

| Lab | Topic | Section |
|-----|-------|---------|
| Lab 7.1 | Build a bigram language model as a baseline and measure its loss and perplexity | 7.1 |
| Lab 7.2 | Build a mini GPT from scratch in PyTorch and check its parameter count against the formula | 7.2 |
| Lab 7.3 | Tokenize and pack a small corpus, then pretrain the mini GPT with warmup and cosine decay | 7.3, 7.4 |
| Lab 7.4 | Generate text with greedy decoding, temperature, top-k, and top-p, and compare the outputs | 7.5 |
| Lab 7.5 | Load pretrained GPT-2 and test zero-shot and few-shot prompting on a simple task | 7.6 |

## Code

Code for this chapter will live in this folder.
