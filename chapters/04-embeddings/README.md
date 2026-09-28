# Chapter 4: Embeddings

_Draft in progress._

## Outline

### 4.1 [Why embeddings](01-why-embeddings.md)
- One-hot vectors are huge, sparse, and carry no notion of similarity
- The idea behind dense vectors: words that appear in similar contexts should have similar vectors

### 4.2 [Learning word embeddings with Word2Vec](02-learning-word-embeddings-with-word2vec.md)
- Skip-gram and CBOW as shallow neural networks (ties back to Chapters 2 and 3)
- Negative sampling to make training fast
- fastText's subword n-grams for rare words (a bridge to Chapter 5)

### 4.3 [The geometry of embedding space](03-the-geometry-of-embedding-space.md)
- Cosine similarity and nearest neighbors
- Analogies ("king − man + woman ≈ queen") and where they break down
- Bias carried in embeddings

### 4.4 [From static to contextual embeddings](04-from-static-to-contextual-embeddings.md)
- One vector per word can't tell "river bank" from "bank account"
- How transformers give each occurrence its own vector (preview of Chapter 6)

### 4.5 [Embeddings inside LLMs](05-embeddings-inside-llms.md)
- The embedding layer as a lookup table of vocabulary size × model dimension
- Learned end to end with the rest of the model
- Weight tying with the output layer
- Adding position information (preview of positional encodings)

### 4.6 [Sentence embeddings and applications](06-sentence-embeddings-and-applications.md)
- From word vectors to sentence and document embeddings
- Semantic search, clustering, and retrieval-augmented generation (RAG)
- Summary and exercises

## Code Labs

| Lab | Topic | Section |
|-----|-------|---------|
| Lab 4.1 | Train a skip-gram model with negative sampling from scratch in PyTorch | 4.2 |
| Lab 4.2 | Explore pretrained word vectors: nearest neighbors, analogies, and a 2D plot | 4.3 |
| Lab 4.3 | Compare static and contextual embeddings for the same word in different sentences | 4.4 |
| Lab 4.4 | Build an `nn.Embedding` layer, inspect its weights, and tie it to an output layer | 4.5 |
| Lab 4.5 | Build a tiny semantic search with a sentence embedding model | 4.6 |

## Code

Code for this chapter will live in this folder.
