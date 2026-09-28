# Chapter 7: Generative Pretraining (GPT)

A GPT learns from a task that sounds almost too simple: read some text and guess the next token. No labels are needed, because the text supplies its own answers, so any text is training data. Chapter 6 built the encoder-decoder Transformer for translation; a GPT keeps only the decoder, drops cross-attention, and reuses nearly everything else, from the causal mask to the token-by-token generation loop. Scaled up in model size, data, and compute, this **generative pretraining** (Radford et al. 2018; Radford et al. 2019; Brown et al. 2020) produced models that can continue text, answer questions, and learn a new task from a few examples in the prompt. This chapter defines the objective, turns the Chapter 6 decoder into the GPT architecture, prepares pretraining data, trains at scale guided by scaling laws, samples text, and shows what a base model cannot yet do without fine-tuning (Chapter 8). Along the way we build and pretrain a mini GPT from scratch and check it against GPT-2's released weights.

## Learning goals

- Write the probability of a text as a product of next-token predictions, and compute the pretraining loss and perplexity.
- Explain how a decoder-only Transformer differs from the encoder-decoder of Chapter 6, and why no labeled data is needed to train it.
- Assemble the GPT architecture from Chapter 6's parts (pre-norm decoder blocks without cross-attention, learned positions, tied output projection) and describe the changes in recent models: RMSNorm, SwiGLU, and rotary position embeddings.
- Count a GPT model's parameters and training compute from its configuration, using $`P \approx 12Nd^2 + Vd`$ and $`C \approx 6PD`$.
- Describe how a pretraining corpus is collected, filtered, deduplicated, tokenized, and packed into fixed-length training blocks.
- Explain the pretraining optimization recipe and how scaling laws guide the split of a compute budget between model size and data.
- Generate text with temperature, top-k, and top-p sampling, and explain why greedy decoding and beam search work poorly for open-ended text.
- Explain in-context learning, and why a base model continues text rather than following instructions.

## Outline

### 1. Language modeling as next-token prediction
- A language model assigns a probability to any token sequence $`\mathbf{x} = (x_1, \dots, x_n)`$ by the chain rule: the factorization of Section 6.1 with the source sentence removed

```math
p_\theta(x_1, \dots, x_n) = \prod_{t=1}^{n} p_\theta(x_t \mid x_{\lt t})
```

- From encoder-decoder to decoder-only: drop the encoder and cross-attention (Section 6.12), keep masked self-attention, and the "target" becomes the text itself
- Teacher forcing on plain text: a block of $`n + 1`$ tokens gives inputs $`x_1, \dots, x_n`$ and labels $`x_2, \dots, x_{n+1}`$, the shifted-right arrangement of Section 6.2; with the causal mask (Section 6.4), one forward pass trains all $`n`$ positions
- The loss: cross-entropy (Chapter 2) averaged over positions, with no label smoothing, and **perplexity** as its exponential

```math
\mathcal{L}(\theta) = -\frac{1}{n} \sum_{t=1}^{n} \log p_\theta(x_{t+1} \mid x_{\le t}), \qquad \mathrm{PPL} = \exp(\mathcal{L})
```

- Reading the numbers: an untrained model should start near $`\ln V`$ (about 10.8 nats for GPT-2's 50,257-token vocabulary), and per-token losses are comparable only between models with the same tokenizer (bits per byte, Chapter 5)
- Why this objective scales: every token of any text is a label, so the data is limited only by how much text exists, unlike the sentence pairs of Chapter 6; compare BERT's masked language modeling, which trains on only 15% of positions (Section 6.12)
- The pretrain-then-fine-tune idea: learn general-purpose representations from unlabeled text, then adapt to tasks (Radford et al. 2018)

### 2. The GPT architecture
- The model: token embeddings plus learned absolute position embeddings (Section 6.6), a stack of $`N`$ decoder blocks, a final LayerNorm, and an output projection tied to the token embedding (Section 6.2; Press and Wolf 2017)
- Each block is the decoder block of Section 6.8 without cross-attention, so it has the encoder block's two sublayers (Section 6.7) but a causal mask: masked multi-head self-attention and a position-wise FFN
- GPT-2's arrangement (Radford et al. 2019): pre-norm blocks (Section 6.7), an extra LayerNorm after the last block, and residual-branch weights scaled at initialization by $`1/\sqrt{2N}`$, since each of the $`N`$ blocks adds two branches to the residual stream (Chapter 3); like the first GPT (Radford et al. 2018), it uses GELU in the FFN in place of ReLU (Hendrycks and Gimpel 2016)

```math
H \leftarrow H + \mathrm{MultiHead}\big(\mathrm{LayerNorm}(H);\ \text{causal mask}\big), \qquad H \leftarrow H + \mathrm{FFN}\big(\mathrm{LayerNorm}(H)\big)
```

- Counting parameters with Section 6.11: removing cross-attention's $`4d^2`$ from the $`16d^2`$ decoder block leaves $`12d^2`$ per block, so

```math
P \approx 12Nd^2 + Vd + n_{\max} d
```

- Checking it on GPT-2 small ($`N = 12`$, $`d = 768`$, $`h = 12`$, $`d_{\text{ff}} = 3072`$, $`V = 50{,}257`$, $`n_{\max} = 1{,}024`$): about 124 million with biases and LayerNorms, the size of the released checkpoint, although the paper reported 117 million; the same formula comes within 1% of GPT-3's 175 billion ($`N = 96`$, $`d = 12{,}288`$; Brown et al. 2020)
- Common changes in recent models, each a small swap inside the same block (Touvron et al. 2023): **RMSNorm** instead of LayerNorm (Zhang and Sennrich 2019; Chapter 3), **rotary position embeddings** instead of learned absolute positions (Su et al. 2024; Section 6.6), and a **SwiGLU** FFN instead of the two-layer ReLU or GELU network (Shazeer 2020)

```math
\mathrm{FFN}_{\text{SwiGLU}}(\mathbf{x}) = W_2\big(\mathrm{SiLU}(W_1 \mathbf{x}) \odot W_3 \mathbf{x}\big), \qquad d_{\text{ff}} = \tfrac{8}{3} d
```

- SwiGLU has three weight matrices instead of two, so shrinking $`d_{\text{ff}}`$ from $`4d`$ to $`\tfrac{8}{3} d`$ keeps the FFN at about $`8d^2`$ weights and the $`12d^2`$ count unchanged
- Implementing the model in about a hundred lines of PyTorch by reusing the attention and block code of Chapter 6 (Phuong and Hutter 2022 give pseudocode for the decoder-only Transformer)

### 3. Pretraining data
- Where the text comes from: filtered web crawls (such as Common Crawl), books, Wikipedia and other reference text, and code
- An example: GPT-2's WebText, built from the pages linked from Reddit posts with at least 3 karma, about 8 million documents and 40 GB of text after deduplication and cleaning (Radford et al. 2019)
- Cleaning: language identification, heuristic and classifier-based quality filtering (Raffel et al. 2020; Brown et al. 2020), removing personal information and unwanted content, and removing benchmark test sets to avoid contamination
- **Deduplication**, exact and near-duplicate: repeated text wastes compute and is more likely to be memorized (Lee et al. 2022)
- The data mixture: sampling sources in proportions that differ from their sizes, so small high-quality sources are seen more often than large noisy ones (Brown et al. 2020)
- From documents to training examples with the Chapter 5 tokenizer: append an end-of-text token to each document, concatenate, and cut the stream into blocks of $`n + 1`$ tokens; packed blocks need no padding, so the padding masks of Section 6.4 and the length bucketing of Section 6.9 are not needed
- A detail of packing: the causal mask lets a token attend across an end-of-text token into the previous document; a block-diagonal mask (Section 6.4) prevents this, at some cost in simplicity
- Measuring data in tokens, the unit that sets both the training compute and the scaling laws of Section 4

### 4. Training at scale
- The optimization recipe, compared with the original Transformer's (Section 6.9): AdamW (Loshchilov and Hutter 2019; Chapter 3) with decoupled weight decay, linear warmup followed by cosine decay instead of the inverse-square-root schedule, gradient-norm clipping (Chapter 3), and typically little or no dropout because the model sees most text only once; GPT-3, for example, used $`\beta_1 = 0.9`$, $`\beta_2 = 0.95`$, clipping at 1.0, and weight decay 0.1 (Brown et al. 2020)
- Batches measured in tokens, as in Section 6.9 but far larger, reaching millions of tokens per step in large models
- Fitting the model on hardware: mixed-precision (BF16) training (Chapter 3), gradient accumulation to reach the target batch size, and data parallelism across devices; the memory for parameters, Adam's two moments, and activations follows from Section 6.11
- Estimating compute: Section 6.11's rule of about 6 FLOPs per parameter per token gives, for $`P`$ parameters and $`D`$ training tokens,

```math
C \approx 6PD
```

- The attention term is small at GPT context lengths: it exceeds the weight term only when $`n \gt 6d`$ (Section 6.11), which is 4,608 tokens for GPT-2 small; for GPT-3 the rule gives $`6 \cdot 175 \times 10^{9} \cdot 300 \times 10^{9} \approx 3.15 \times 10^{23}`$ FLOPs, matching the reported $`3.14 \times 10^{23}`$
- **Scaling laws**: test loss falls as a smooth power law in model size, data, and compute over many orders of magnitude (Kaplan et al. 2020)
- **Compute-optimal training**: for a fixed $`C`$, model size and training tokens should grow in about equal proportion, roughly 20 tokens per parameter; Chinchilla (70 billion parameters, 1.4 trillion tokens) outperformed the 280-billion-parameter Gopher trained with the same compute (Hoffmann et al. 2022)

```math
\mathcal{L}(P, D) \approx E + \frac{A}{P^{\alpha}} + \frac{B}{D^{\beta}}
```

- Why many models are trained on far more tokens than the compute-optimal amount: a smaller model is cheaper to serve at inference (Touvron et al. 2023; Chapter 13)
- Monitoring a run: training and validation loss against tokens seen, gradient norms, and loss spikes

### 5. Generating text
- The autoregressive loop of Section 6.10 without an encoder: process the prompt, take the distribution at the last position, choose a token, append it, and repeat until an end-of-text token, a stop condition, or the context limit $`n_{\max}`$
- Greedy decoding and beam search (Section 6.10) find high-probability continuations, but for open-ended text they tend to produce dull, repetitive loops (Holtzman et al. 2020); sampling from the model's distribution gives more varied and more human-like text
- **Temperature** rescales the logits $`\mathbf{o}`$ before the softmax: $`\tau \lt 1`$ sharpens the distribution toward greedy decoding, and $`\tau \gt 1`$ flattens it toward uniform

```math
p_\tau(i) = \frac{\exp(o_i / \tau)}{\sum_{j=1}^{V} \exp(o_j / \tau)}
```

- **Top-k sampling** samples only from the $`k`$ most probable tokens, renormalized (Fan et al. 2018)
- **Top-p (nucleus) sampling** samples from the smallest set of tokens whose total probability reaches $`p`$, so the number of candidates adapts to how confident the model is (Holtzman et al. 2020)

```math
V^{(p)} = \text{smallest set such that} \sum_{i \in V^{(p)}} p(i) \ge p
```

- Combining the settings, how they trade diversity against coherence, and fixing the random seed for reproducible outputs
- Reusing work: the key-value caching of Section 6.10 makes each new token cheap; its memory cost, and other ways to make generation fast, are the subject of Chapter 13

### 6. What pretraining produces
- A **base model** continues text: given a prompt, it produces what is likely to come next in its training data
- Zero-shot behavior from prompt design alone, for example adding "TL;DR:" after an article to elicit a summary (Radford et al. 2019)
- **In-context learning**: zero-, one-, and few-shot prompting, where examples of a task in the prompt improve performance without any weight updates, and more so for larger models (Brown et al. 2020)
- What a base model does not do reliably: follow instructions (it may continue a question with more questions), refuse harmful requests, or avoid stating falsehoods and reproducing biases in its data
- Evaluating a base model: validation loss or perplexity, bits per byte across tokenizers (Chapter 5), and few-shot benchmarks (Chapter 12)
- From a base model to an assistant: supervised fine-tuning on demonstrations (Chapter 8) and learning from human preferences (Chapters 9 to 11; Ouyang et al. 2022)

## Suggested code labs

1. **A bigram baseline.** Tokenize a small corpus (for example tiny Shakespeare) with your Chapter 5 tokenizer and fit a bigram model by counting. Report its validation loss and perplexity, check that a model that predicts uniformly scores $`\ln V`$, and generate a few samples. This is the baseline the mini GPT must beat.
2. **A mini GPT from scratch.** Build a decoder-only model from your Chapter 6 code: pre-norm blocks with masked self-attention and an FFN (no cross-attention), learned position embeddings, a final LayerNorm, a tied output projection, and scaled initialization of the residual branches. Rerun the causal leak test of Section 6.4, and check the parameter count against $`12Nd^2 + Vd + n_{\max} d`$ plus biases and normalization parameters. Then load GPT-2 small's released weights into your model and confirm that its logits match Hugging Face's `GPT2LMHeadModel` on the same input.
3. **Pretrain the mini GPT.** Pack the tokenized corpus into blocks of $`n + 1`$ tokens with end-of-text separators, and train with AdamW, warmup plus cosine decay, gradient clipping, and BF16 autocast. Check the initial loss against $`\ln V`$ and that the model can overfit a single batch, then plot training and validation loss against tokens seen and compare with the bigram baseline. Estimate the run's compute with $`6PD`$ and compare it with the measured time and hardware throughput. As an extension, train three model sizes on the same token budget and plot validation loss against compute.
4. **Decoding strategies.** Implement greedy decoding, temperature, top-k, and top-p sampling over a model's logits. Generate continuations of the same prompts from your mini GPT and from GPT-2 with each setting, and measure repetition (the fraction of repeated n-grams) and diversity (distinct n-grams across samples). Plot how the size of the top-p candidate set changes from step to step.
5. **In-context learning with GPT-2.** Pick a simple task such as sentiment labeling or word translation, and measure GPT-2's accuracy with 0, 1, 4, and 8 examples in the prompt, for two or more model sizes. Compare several prompt formats, and show that asking the base model to follow an instruction often produces a continuation instead of an answer.

## Key takeaways

- A GPT is the Chapter 6 decoder without cross-attention, trained to predict the next token; the causal mask lets every position of every block be trained in one parallel pass.
- Next-token prediction needs no labels, so the amount of training data is limited only by how much usable text exists.
- The GPT block is a pre-norm decoder block with $`12d^2`$ weights, so a model has about $`12Nd^2 + Vd`$ parameters; recent models keep the design and swap in RMSNorm, SwiGLU, and rotary position embeddings.
- Pretraining data is collected, filtered, deduplicated, mixed, tokenized, and packed into fixed-length blocks; data quality matters as much as quantity.
- Training uses AdamW with warmup and cosine decay, gradient clipping, and mixed precision; compute is about $`6PD`$, and compute-optimal training grows model size and data together.
- Sampling with temperature, top-k, or top-p produces better open-ended text than greedy decoding or beam search.
- Pretraining produces a base model that continues text and can learn tasks in context, but does not reliably follow instructions; fine-tuning (Chapter 8) addresses that.

## Further reading

Brown, Tom B., et al. "Language Models Are Few-Shot Learners." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2005.14165.

Fan, Angela, Mike Lewis, and Yann Dauphin. "Hierarchical Neural Story Generation." In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics*, 2018. https://arxiv.org/abs/1805.04833.

Hendrycks, Dan, and Kevin Gimpel. "Gaussian Error Linear Units (GELUs)." arXiv preprint arXiv:1606.08415, 2016. https://arxiv.org/abs/1606.08415.

Hoffmann, Jordan, et al. "Training Compute-Optimal Large Language Models." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.15556.

Holtzman, Ari, et al. "The Curious Case of Neural Text Degeneration." In *International Conference on Learning Representations*, 2020. https://arxiv.org/abs/1904.09751.

Kaplan, Jared, et al. "Scaling Laws for Neural Language Models." arXiv preprint arXiv:2001.08361, 2020. https://arxiv.org/abs/2001.08361.

Lee, Katherine, et al. "Deduplicating Training Data Makes Language Models Better." In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*, 2022. https://arxiv.org/abs/2107.06499.

Loshchilov, Ilya, and Frank Hutter. "Decoupled Weight Decay Regularization." In *International Conference on Learning Representations*, 2019. https://arxiv.org/abs/1711.05101.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Phuong, Mary, and Marcus Hutter. "Formal Algorithms for Transformers." arXiv preprint arXiv:2207.09238, 2022. https://arxiv.org/abs/2207.09238.

Press, Ofir, and Lior Wolf. "Using the Output Embedding to Improve Language Models." In *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics*, 2017. https://arxiv.org/abs/1608.05859.

Radford, Alec, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. "Improving Language Understanding by Generative Pre-Training." OpenAI technical report, 2018. https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.

Radford, Alec, et al. "Language Models Are Unsupervised Multitask Learners." OpenAI technical report, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

Raffel, Colin, et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *Journal of Machine Learning Research* 21, no. 140 (2020): 1–67. https://arxiv.org/abs/1910.10683.

Shazeer, Noam. "GLU Variants Improve Transformer." arXiv preprint arXiv:2002.05202, 2020. https://arxiv.org/abs/2002.05202.

Su, Jianlin, et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." *Neurocomputing* 568 (2024): 127063. https://arxiv.org/abs/2104.09864.

Touvron, Hugo, et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971, 2023. https://arxiv.org/abs/2302.13971.

Zhang, Biao, and Rico Sennrich. "Root Mean Square Layer Normalization." In *Advances in Neural Information Processing Systems 32*, 2019. https://arxiv.org/abs/1910.07467.
