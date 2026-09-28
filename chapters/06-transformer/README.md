# Chapter 6: Transformer

Chapter 3 ended with the limits of recurrent networks: they process a sequence one step at a time, so training cannot be parallelized across positions, and everything the network knows about the past must squeeze through a fixed-size hidden state. The transformer removes both limits. Its core operation, **attention**, lets every position look directly at every other position and decide, based on content, which ones matter. All positions are computed in parallel with large matrix multiplications, which is exactly what GPUs do best. Stacked with the tools from Chapter 3 (residual connections, normalization, careful initialization, and AdamW with warmup), attention gives an architecture that trains stably at great depth and scale. Almost every large language model is a transformer. This chapter builds one from scratch: token embeddings and positional information, scaled dot-product attention, causal masking, multi-head attention, the feed-forward block, and the full decoder-only language model. It then trains the model on next-token prediction, generates text with it, counts its parameters and compute, and surveys the main variants: encoder and encoder-decoder models, efficient attention, and mixture-of-experts.

## Learning goals

- Explain the problem attention solves: direct, content-based access to every position, computed in parallel.
- Derive scaled dot-product attention, including why the scores are divided by $`\sqrt{d_k}`$, and implement it with causal and padding masks.
- Implement multi-head self-attention and explain what multiple heads buy, along with multi-query and grouped-query variants.
- Explain why attention needs positional information, and compare sinusoidal, learned, rotary (RoPE), and ALiBi position schemes.
- Assemble a pre-norm transformer block (attention, feed-forward network, residual connections, normalization) and a full decoder-only language model.
- Train a small GPT-style model with the next-token cross-entropy loss and generate text from it.
- Count the parameters, FLOPs, and memory of a transformer from its configuration, and explain why attention costs grow quadratically with sequence length.
- Distinguish encoder-only, decoder-only, and encoder-decoder transformers, and describe the main architectural variants used in modern LLMs.

## Outline

### 1. From recurrence to attention
- Recap of the limits of recurrent networks (Chapter 3): sequential computation and a fixed-size bottleneck between distant positions
- Attention in encoder-decoder translation: letting the decoder look back at every encoder state instead of one summary vector (Bahdanau et al. 2015)
- The key idea of the transformer: drop recurrence entirely and build the whole network from attention plus position-wise layers (Vaswani et al. 2017)
- What is gained: every position reaches every other in one step, and all positions are computed in parallel during training
- What it costs: compute and memory that grow quadratically with sequence length, and no built-in notion of order
- A map of the chapter: inputs, attention, blocks, the full model, training, cost, and variants

### 2. Inputs: token embeddings and the residual stream
- From token IDs (Chapter 5) to vectors: the embedding matrix $`E \in \mathbb{R}^{V \times d}`$ as a lookup table (Chapter 4)
- The model width $`d`$ and the notation used throughout the chapter: $`L`$ layers, $`h`$ heads of dimension $`d_h`$ (usually $`h \, d_h = d`$), feed-forward width $`d_{\text{ff}}`$, vocabulary size $`V`$, and context length $`n`$
- The input as a matrix $`X \in \mathbb{R}^{n \times d}`$, one row per token, and the residual stream view (Chapter 3): every block reads from this matrix and adds its output back
- The output side: a final normalization and an unembedding matrix $`W_U \in \mathbb{R}^{V \times d}`$ that turns each position's vector into logits over the vocabulary
- Weight tying: sharing the embedding and unembedding matrices (Press and Wolf 2017), and why some models do and others do not

### 3. Scaled dot-product attention
- Queries, keys, and values as three learned projections of the same input: $`Q = XW_Q`$, $`K = XW_K`$, $`V = XW_V`$
- Attention as a soft, differentiable dictionary lookup: each query scores every key, the scores become weights through a softmax, and the output is a weighted average of the values

```math
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
```

- Why divide by $`\sqrt{d_k}`$: if query and key entries are independent with mean 0 and variance 1, their dot product has variance $`d_k`$; without scaling, large scores saturate the softmax and its gradients vanish
- Shapes written out for every tensor, and the $`n \times n`$ attention matrix
- Attention as a weighted average: each row of weights sums to 1, the output is a convex combination of value vectors, and attention by itself is permutation-equivariant
- Self-attention (queries, keys, and values from the same sequence) vs. cross-attention (queries from one sequence, keys and values from another)
- Implementing it in a few lines, and checking it against PyTorch's `F.scaled_dot_product_attention`

### 4. Masking: causal and padding masks
- Why a language model must not see the future: position $`t`$ may attend only to positions $`\le t`$
- The causal mask: set scores for future positions to $`-\infty`$ before the softmax, so their weights become exactly 0

```math
M_{ts} = \begin{cases} 0 & s \le t \\ -\infty & s \gt t \end{cases}, \qquad
\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
```

- Training all positions in parallel: one forward pass produces a next-token prediction at every position, with no leakage, which is what makes transformer training so efficient compared with RNNs
- Padding masks for batches of sequences with different lengths (Chapter 5), and combining them with the causal mask
- A test for leaks: changing a future token must not change any earlier output

### 5. Multi-head attention
- One head computes one pattern of weights; different relationships (the previous token, a matching bracket, the subject of a verb) need different patterns
- Multi-head attention: $`h`$ heads, each with its own projections into a $`d_h`$-dimensional subspace, run in parallel; their outputs are concatenated and mixed by an output projection $`W_O`$

```math
\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W_O, \qquad \mathrm{head}_i = \mathrm{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)})
```

- Cost: with $`h \, d_h = d`$, multi-head attention has about the same parameters and FLOPs as a single head of full width, namely $`4d^2`$ parameters for $`W_Q, W_K, W_V, W_O`$
- Implementation with reshapes and transposes instead of a loop over heads
- Sharing keys and values across heads: multi-query attention (Shazeer 2019) and grouped-query attention (Ainslie et al. 2023), which shrink the memory needed to cache keys and values during generation
- What heads learn: attention patterns as a partial window into the model, and the caveat that attention weights are not a complete explanation of model behavior

### 6. Positional information
- Why attention needs it: without positions, self-attention treats its input as an unordered set, so "dog bites man" and "man bites dog" look the same
- **Sinusoidal encodings**: fixed sines and cosines of different frequencies added to the embeddings (Vaswani et al. 2017)

```math
\mathrm{PE}_{(p,\, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad \mathrm{PE}_{(p,\, 2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right)
```

- **Learned absolute embeddings**: one trainable vector per position (as in GPT-2 and BERT), and their limit: no positions beyond the training length
- **Rotary position embeddings (RoPE)**: rotate each pair of query and key dimensions by an angle proportional to the position, so the score $`\mathbf{q}_m^\top \mathbf{k}_n`$ depends on positions only through the offset $`m - n`$ (Su et al. 2024); the default in many recent LLMs
- **ALiBi**: no position vectors at all; add a head-specific penalty proportional to the distance to each attention score (Press et al. 2022)
- Relative vs. absolute position, and extrapolation to sequences longer than those seen in training

### 7. The transformer block
- The position-wise feed-forward network (FFN): the same two-layer MLP applied to every position independently, typically with $`d_{\text{ff}} = 4d`$

```math
\mathrm{FFN}(\mathbf{x}) = W_2\, \phi(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
```

- Division of labor: attention moves information between positions; the FFN transforms information within each position and holds much of the model's parameters
- Activations: ReLU in the original transformer, GELU in GPT-2 and BERT (Hendrycks and Gimpel 2016), and gated variants such as SwiGLU in many recent LLMs (Shazeer 2020), with $`d_{\text{ff}}`$ reduced to about $`\tfrac{8}{3} d`$ to keep the parameter count the same

```math
\mathrm{SwiGLU}(\mathbf{x}) = W_2\big(\mathrm{SiLU}(W_1 \mathbf{x}) \odot W_3 \mathbf{x}\big)
```

- Residual connections and normalization around each sublayer (Chapter 3): post-norm in the original transformer vs. pre-norm in most modern models (Xiong et al. 2020), with LayerNorm or RMSNorm

```math
\mathbf{h} \leftarrow \mathbf{h} + \mathrm{MultiHead}\big(\mathrm{Norm}(\mathbf{h})\big), \qquad \mathbf{h} \leftarrow \mathbf{h} + \mathrm{FFN}\big(\mathrm{Norm}(\mathbf{h})\big)
```

- Dropout placement (on attention weights and sublayer outputs) and bias terms, which many recent models drop
- Initialization: small normal weights and scaled-down projections that write into the residual stream (Chapter 3)

### 8. The full model: a decoder-only language model
- Stacking $`L`$ blocks between the embedding and the unembedding: the complete GPT-style architecture in about 100 lines of PyTorch
- The next-token objective: at every position, predict the following token with cross-entropy (Chapter 2), averaged over positions

```math
\mathcal{L}(\theta) = -\frac{1}{n} \sum_{t=1}^{n} \log p_\theta(x_{t+1} \mid x_{\le t})
```

- Teacher forcing: the model always conditions on the true previous tokens, so a sequence of $`n+1`$ tokens gives $`n`$ training examples in one parallel forward pass
- Training in practice: the Chapter 3 recipe (AdamW, warmup plus cosine decay, gradient clipping, mixed precision), fixed-length training blocks from a token stream (Chapter 5), and checking that the initial loss is close to $`\ln V`$
- Generating text: run the model, take the logits at the last position, choose a token (greedy or sampling with temperature), append it, and repeat
- The key-value (KV) cache: during generation, the keys and values of earlier positions never change, so store them instead of recomputing them at every step
- A worked configuration: GPT-2 small (12 layers, $`d = 768`$, 12 heads, context 1,024, vocabulary 50,257)

### 9. Counting parameters, compute, and memory
- Parameters per block: about $`4d^2`$ in attention and $`2 d \, d_{\text{ff}} = 8d^2`$ in the FFN when $`d_{\text{ff}} = 4d`$, for about $`12d^2`$ per block and $`12 L d^2`$ in total, plus $`Vd`$ for the embeddings
- Checking the formula against a real model's parameter count
- FLOPs: about $`2N`$ per token for a forward pass through a model with $`N`$ parameters (and about $`6N`$ per token for training), plus attention's cost, which grows with the context length
- The quadratic cost of attention: the $`n \times n`$ score matrix takes $`O(n^2 d)`$ compute per layer and, if materialized, $`O(n^2)`$ memory per head
- Memory-aware attention kernels: FlashAttention computes exact attention block by block without storing the full score matrix (Dao et al. 2022)
- Activation memory in training, and why context length is expensive

### 10. Transformer families and variants
- **Encoder-only** models: bidirectional attention with no causal mask, trained by masked-token prediction, used for classification and embeddings (BERT; Devlin et al. 2019)
- **Encoder-decoder** models: an encoder over the input and a causal decoder with cross-attention, as in the original transformer and T5 (Raffel et al. 2020)
- **Decoder-only** models: one causal stack for everything; why this simplest design became the standard for LLMs
- Efficient attention for long sequences: sparse and sliding-window patterns (Child et al. 2019; Beltagy et al. 2020) and linear attention (Katharopoulos et al. 2020)
- **Mixture-of-experts (MoE)** FFNs: a router sends each token to a few of many expert FFNs, adding parameters without adding proportional compute per token (Shazeer et al. 2017; Fedus et al. 2022)
- Beyond text: the same architecture on image patches (Dosovitskiy et al. 2021), audio, and other modalities
- A modern LLM block in one line: pre-norm RMSNorm, RoPE, grouped-query attention, SwiGLU FFN, no biases (for example Llama; Touvron et al. 2023)
- Looking inside: the residual stream as a communication channel between heads and FFNs, and induction heads as a mechanism for in-context copying (Elhage et al. 2021; Olsson et al. 2022)

## Suggested code labs

1. **Attention from scratch.** Implement scaled dot-product attention in PyTorch with optional causal and padding masks. Check it against `torch.nn.functional.scaled_dot_product_attention` on random inputs. Then remove the $`1/\sqrt{d_k}`$ scaling, increase $`d_k`$ from 16 to 1,024, and plot how the entropy of the attention weights and the size of their gradients change.
2. **Multi-head causal self-attention and a leak test.** Implement multi-head attention with reshapes (no loop over heads) and compare its output and parameter count with `nn.MultiheadAttention`. Write a test that changes the token at position $`t`$ and asserts that the outputs at all positions before $`t`$ are unchanged. Extend it to grouped-query attention and confirm that with one key-value group per head it matches standard multi-head attention.
3. **Positional encodings.** Implement sinusoidal encodings and RoPE. Verify numerically that with RoPE the score between a query at position $`m`$ and a key at position $`n`$ depends only on $`m - n`$. Then train the same small model on a task that needs order (for example reversing or sorting a short sequence of digits) with no positional information, learned absolute embeddings, and RoPE, and compare accuracy.
4. **Build and train a mini GPT.** Assemble embeddings, $`L`$ pre-norm blocks, and a tied unembedding into a decoder-only model. Check that the initial loss is close to $`\ln V`$ and that the model can overfit a single batch. Train it on a small text corpus (for example a character- or BPE-level Shakespeare dataset) with AdamW, warmup plus cosine decay, and gradient clipping, and sample text with different temperatures during training.
5. **Count parameters and FLOPs.** Write a function that computes the parameter count and forward FLOPs per token from $`(L, d, h, d_{\text{ff}}, V, n)`$. Check the parameter count against your mini GPT and against the GPT-2 small checkpoint from Hugging Face `transformers`. Plot how attention's share of the FLOPs changes as the context length grows.
6. **Generation with a KV cache.** Add a key-value cache to your mini GPT's attention layers. Confirm that cached generation produces exactly the same tokens as uncached generation under greedy decoding, and measure the speedup as the generated sequence gets longer.
7. **Ablations and attention maps.** Using your mini GPT, compare pre-norm and post-norm blocks (with and without warmup) and GELU vs. SwiGLU FFNs at equal parameter count. Then visualize the attention weights of each head on a sequence of random tokens repeated twice, and look for heads that attend to the token after the previous occurrence of the current token (induction heads).

## Key takeaways

- Attention lets each position gather information from every other position by content, in one step and in parallel, removing the sequential bottleneck of recurrent networks.
- Scaled dot-product attention is a softmax-weighted average of values; dividing by $`\sqrt{d_k}`$ keeps the softmax out of saturation, and a causal mask lets a model train on every position at once without seeing the future.
- Multiple heads attend to different relationships at little extra cost, and sharing keys and values across heads (MQA, GQA) saves memory during generation.
- Attention is order-blind, so position must be added: sinusoidal or learned embeddings, or relative schemes such as RoPE and ALiBi.
- A transformer block is attention plus a position-wise FFN, each wrapped in a residual connection with normalization; pre-norm, RMSNorm, RoPE, GQA, and SwiGLU describe many modern LLMs.
- A decoder-only transformer trained with next-token cross-entropy is a complete language model; it has about $`12Ld^2`$ parameters, costs about $`2N`$ FLOPs per token to run, and pays a cost quadratic in context length for attention.

## Further reading

Ainslie, Joshua, et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 2023. https://arxiv.org/abs/2305.13245.

Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural Machine Translation by Jointly Learning to Align and Translate." In *International Conference on Learning Representations*, 2015. https://arxiv.org/abs/1409.0473.

Beltagy, Iz, Matthew E. Peters, and Arman Cohan. "Longformer: The Long-Document Transformer." arXiv preprint arXiv:2004.05150, 2020. https://arxiv.org/abs/2004.05150.

Child, Rewon, et al. "Generating Long Sequences with Sparse Transformers." arXiv preprint arXiv:1904.10509, 2019. https://arxiv.org/abs/1904.10509.

Dao, Tri, et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2205.14135.

Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics*, 2019. https://arxiv.org/abs/1810.04805.

Dosovitskiy, Alexey, et al. "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale." In *International Conference on Learning Representations*, 2021. https://arxiv.org/abs/2010.11929.

Elhage, Nelson, et al. "A Mathematical Framework for Transformer Circuits." *Transformer Circuits Thread*, 2021. https://transformer-circuits.pub/2021/framework/index.html.

Fedus, William, Barret Zoph, and Noam Shazeer. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." *Journal of Machine Learning Research* 23, no. 120 (2022): 1–39. https://arxiv.org/abs/2101.03961.

Hendrycks, Dan, and Kevin Gimpel. "Gaussian Error Linear Units (GELUs)." arXiv preprint arXiv:1606.08415, 2016. https://arxiv.org/abs/1606.08415.

Katharopoulos, Angelos, et al. "Transformers Are RNNs: Fast Autoregressive Transformers with Linear Attention." In *Proceedings of the 37th International Conference on Machine Learning*, 2020. https://arxiv.org/abs/2006.16236.

Olsson, Catherine, et al. "In-Context Learning and Induction Heads." arXiv preprint arXiv:2209.11895, 2022. https://arxiv.org/abs/2209.11895.

Phuong, Mary, and Marcus Hutter. "Formal Algorithms for Transformers." arXiv preprint arXiv:2207.09238, 2022. https://arxiv.org/abs/2207.09238.

Press, Ofir, and Lior Wolf. "Using the Output Embedding to Improve Language Models." In *Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics*, 2017. https://arxiv.org/abs/1608.05859.

Press, Ofir, Noah A. Smith, and Mike Lewis. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." In *International Conference on Learning Representations*, 2022. https://arxiv.org/abs/2108.12409.

Radford, Alec, et al. "Language Models Are Unsupervised Multitask Learners." OpenAI technical report, 2019. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.

Raffel, Colin, et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *Journal of Machine Learning Research* 21, no. 140 (2020): 1–67. https://arxiv.org/abs/1910.10683.

Shazeer, Noam. "Fast Transformer Decoding: One Write-Head Is All You Need." arXiv preprint arXiv:1911.02150, 2019. https://arxiv.org/abs/1911.02150.

Shazeer, Noam. "GLU Variants Improve Transformer." arXiv preprint arXiv:2002.05202, 2020. https://arxiv.org/abs/2002.05202.

Shazeer, Noam, et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." In *International Conference on Learning Representations*, 2017. https://arxiv.org/abs/1701.06538.

Su, Jianlin, et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." *Neurocomputing* 568 (2024): 127063. https://arxiv.org/abs/2104.09864.

Touvron, Hugo, et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv preprint arXiv:2302.13971, 2023. https://arxiv.org/abs/2302.13971.

Vaswani, Ashish, et al. "Attention Is All You Need." In *Advances in Neural Information Processing Systems 30*, 2017. https://arxiv.org/abs/1706.03762.

Xiong, Ruibin, et al. "On Layer Normalization in the Transformer Architecture." In *Proceedings of the 37th International Conference on Machine Learning*, 2020. https://arxiv.org/abs/2002.04745.

Zhang, Biao, and Rico Sennrich. "Root Mean Square Layer Normalization." In *Advances in Neural Information Processing Systems 32*, 2019. https://arxiv.org/abs/1910.07467.
