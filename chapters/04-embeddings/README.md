# Chapter 4: Embeddings

_Draft in progress._

## Outline

### 4.1 [Why embeddings](01-why-embeddings.md)
- One-hot vectors are huge, sparse, and carry no notion of similarity
- Multiplying a one-hot vector by a weight matrix selects a row: the embedding layer is a lookup table (`nn.Embedding`)
- Learned end to end with the rest of the network; only the rows looked up in a batch receive gradients
- The idea behind dense vectors: words that appear in similar contexts should have similar vectors

### 4.2 [Learning word embeddings with Word2Vec](02-learning-word-embeddings-with-word2vec.md)
- Skip-gram and CBOW as shallow neural networks (ties back to Chapters 2 and 3)
- Negative sampling to make training fast
- fastText's subword n-grams for rare words (a bridge to Chapter 5)

### 4.3 [The geometry of embedding space](03-the-geometry-of-embedding-space.md)
- Cosine similarity and nearest neighbors
- Analogies ("king − man + woman ≈ queen") and where they break down
- Bias carried in embeddings

### 4.4 [The limits of static embeddings](04-the-limits-of-static-embeddings.md)
- One vector per word can't tell "river bank" from "bank account"; in pretrained vectors the frequent sense dominates
- What a contextual representation would need, and where the book builds one (Chapter 6)
- Why static embeddings are still useful
- Chapter summary and exercises

## Code Labs

| Lab | Topic | Section |
|-----|-------|---------|
| Lab 4.1 | Train a skip-gram model with negative sampling from scratch in PyTorch | 4.2 |
| Lab 4.2 | Explore pretrained word vectors: nearest neighbors, analogies, and a 2D plot | 4.3 |
| Lab 4.3 | Ambiguous words in pretrained static vectors: find the nearest neighbors of words such as "bank" and "apple" and measure which sense dominates | 4.4 |
| Lab 4.4 | Build an `nn.Embedding` layer, check that lookup equals one-hot multiplication, and inspect which rows receive gradients | 4.1 |

## Code

Code for this chapter will live in this folder.
