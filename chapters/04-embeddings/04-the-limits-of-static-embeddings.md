# 4.4 The Limits of Static Embeddings

Word2Vec and fastText give each word exactly one vector, and so does the embedding layer of Section 4.1. Once training ends, the vector for "bank" is fixed: the same numbers represent "bank" in "we sat on the river bank" and in "I opened a bank account." Embeddings of this kind are called **static**. They were a large step beyond one-hot vectors, but language does not work one-word-one-meaning. This section shows concretely what goes wrong when every occurrence of a word shares a single vector, both in a toy example and in real pretrained vectors, and describes what a better representation would have to do. Building that representation takes the attention mechanism of Chapter 6. The section ends with a summary of the chapter and exercises.

## One vector per word is not enough

Many common words have several unrelated meanings. "bank" can be the side of a river or a financial institution; "bat" can be an animal or a piece of sports equipment; "spring" can be a season, a coil of metal, or a source of water. Linguists distinguish *homonyms*, whose meanings are unrelated, from *polysemous* words, whose senses are related ("paper" as a material, a newspaper, or an academic article), but for a static embedding the problem is the same: one vector has to serve all of them.

What does that one vector end up representing? Recall how skip-gram trains it (Section 4.2). Every occurrence of "bank" pulls its input vector toward the output vectors of its context words. Occurrences in the financial sense pull toward "money," "loan," and "account"; occurrences in the river sense pull toward "river," "water," and "muddy." The final vector is a compromise between these pulls, weighted by how often each sense occurs in the corpus. In a news corpus, where financial banks are far more common than river banks, the vector sits mostly on the financial side, and the river sense is poorly represented.

A toy example makes the compromise visible. Use 3-dimensional vectors whose axes loosely mean "finance," "water and nature," and "action," and give "bank" a vector partway between the two senses:

```python
import torch
import torch.nn.functional as F

static = {
    "bank":    torch.tensor([0.6, 0.5, 0.0]),   # a blend of both senses
    "river":   torch.tensor([0.0, 1.0, 0.1]),
    "muddy":   torch.tensor([0.0, 0.8, 0.0]),
    "account": torch.tensor([1.0, 0.0, 0.1]),
    "opened":  torch.tensor([0.3, 0.0, 0.9]),
    "the":     torch.tensor([0.1, 0.1, 0.1]),
}
cos = lambda a, b: F.cosine_similarity(a, b, dim=0).item()
print(f"static bank vs river: {cos(static['bank'], static['river']):.3f}, "
      f"vs account: {cos(static['bank'], static['account']):.3f}")
```

```text
static bank vs river: 0.637, vs account: 0.764
```

The static vector is moderately similar to both senses and strongly similar to neither. Any model that reads "bank" through this vector starts out not knowing which bank is meant, and must work it out from the other words using its later layers, if it can.

## What pretrained vectors show

Real embeddings trained on a large corpus show the frequency effect directly. The code below loads the 100,000 most frequent words of a set of pretrained GloVe vectors (Pennington et al. 2014), 50-dimensional vectors trained on Wikipedia and newswire text, and lists the nearest neighbors (Section 4.3) of two ambiguous words:

```python
import gzip
import numpy as np

# GloVe vectors (50 dimensions, trained on Wikipedia and newswire text), top 100,000 words
words, vecs = [], []
with gzip.open("glove50.gz", "rt", encoding="utf8") as f:
    next(f)                                             # header line: "400000 50"
    for i, line in enumerate(f):
        if i == 100_000:
            break
        parts = line.rstrip().split(" ")
        words.append(parts[0])
        vecs.append(np.array(parts[1:], dtype=np.float32))
E = np.stack(vecs)
E /= np.linalg.norm(E, axis=1, keepdims=True)          # unit length: dot product = cosine
idx = {w: i for i, w in enumerate(words)}

def neighbors(word, k=10):
    scores = E @ E[idx[word]]
    order = [i for i in np.argsort(-scores) if words[i] != word][:k]
    return [words[i] for i in order]

def cos(a, b):
    return float(E[idx[a]] @ E[idx[b]])

for w in ["bank", "apple"]:
    print(f"{w}: {', '.join(neighbors(w, 8))}")
for a, b in [("bank", "loan"), ("bank", "river"), ("apple", "microsoft"), ("apple", "pear")]:
    print(f"cos({a}, {b}) = {cos(a, b):.3f}")
```

```text
bank: banks, securities, banking, investment, exchange, financial, credit, lender
apple: blackberry, chips, iphone, microsoft, ipad, pc, ipod, intel
cos(bank, loan) = 0.676
cos(bank, river) = 0.414
cos(apple, microsoft) = 0.733
cos(apple, pear) = 0.543
```

(The file `glove50.gz` is the `glove-wiki-gigaword-50` vectors in word2vec text format, as distributed by the `gensim-data` project.) Every one of the eight neighbors of "bank" is financial; "river" does not appear even among its 100 nearest neighbors. The neighbors of "apple" are all about computers and phones, because in news text the company is mentioned far more often than the fruit, and the vector for "apple" is closer to "microsoft" than to "pear." The less frequent sense has not vanished, since "bank" is still somewhat similar to "river," but it has been outvoted. A model that reads "apple" in a recipe gets a vector that mostly says "technology company."

## Context changes every word

Ambiguous words are only the most visible symptom. Almost every word shifts its meaning with context. "Heavy" means different things in "a heavy box," "heavy rain," and "a heavy heart." "Run" in "run a company" and "run a marathon" shares a spelling and a loose idea but little else. A word's grammatical role also depends on context: "book" is a noun in "read the book" and a verb in "book a flight." And the meaning of a sentence depends on word order and structure that no bag of static word vectors captures: "dog bites man" and "man bites dog" contain the same three vectors.

## What a better representation would do

The fix is to stop treating the representation of a word as a property of the word alone. Instead, let the vector for the word at position $`t`$ in a sentence be a function of the *whole* sentence:

```math
\mathbf{h}_t = f_\theta(w_1, w_2, \dots, w_T)_t .
```

Here $`f_\theta`$ is a neural network with parameters $`\theta`$, and $`\mathbf{h}_t`$ is a **contextual embedding** of the $`t`$-th word. The same word in two different sentences would then get two different vectors, and ideally the two occurrences of "bank" in our examples would end up near "river" and near "account," respectively.

Static embeddings would not disappear. A natural design for $`f_\theta`$ starts by looking up a static vector for each input word in an embedding table, exactly as in Section 4.1, and then transforms those vectors using information from the other positions. The static embedding is the *starting point* for each occurrence, and the context moves it to the right place.

We have in fact already seen a network that computes vectors of this kind: the recurrent network of Section 3.10, whose hidden state $`\mathbf{h}_t`$ depends on the current word and, through $`\mathbf{h}_{t-1}`$, on every word before it. Section 3.10 also explained the limits of recurrence: computation proceeds one position at a time, and information from distant words has to survive many updates of a fixed-size state. What we want is a way for every position to look *directly* at every other position, in parallel. That is the **attention** mechanism, and the network built from it, the Transformer, is the subject of Chapter 6. Section 6.13 returns to the "bank" example with a pretrained Transformer encoder and shows the river and financial senses receiving clearly different vectors.

> **Code Lab 4.4** explores ambiguous words in pretrained static vectors: it lists the nearest neighbors of words such as "bank," "apple," "bat," and "spring," checks which sense dominates, and measures how similar each word's vector is to words from each of its senses.

## Are static embeddings obsolete?

For most language tasks, contextual vectors from a pretrained network beat static ones. Static embeddings still have a place. They are tiny and fast: looking up a vector costs nothing, while a contextual vector requires running a network over the whole sentence. They are easy to inspect, which makes them good for teaching and for studying how meaning is reflected in text statistics, as in Section 4.3. And the ideas behind them, prediction-based training, negative sampling, and subword pieces, live on in the models that replaced them. Most directly, the first layer of the networks in Chapters 6 and 7 is still a static embedding table, the `nn.Embedding` layer of Section 4.1, trained end to end.

## Key takeaways

- A static embedding gives every occurrence of a word the same vector, which blends all of the word's senses in proportion to their frequency and ignores word order.
- In pretrained vectors the frequent sense dominates: the nearest neighbors of "bank" in GloVe are all financial, and those of "apple" are about technology companies.
- A contextual embedding $`\mathbf{h}_t = f_\theta(w_1, \dots, w_T)_t`$ would give each occurrence its own vector, computed from the whole sentence and starting from a static table lookup. Recurrent networks compute such vectors sequentially; the Transformer of Chapter 6 computes them in parallel with attention.
- Static embeddings remain useful: they are cheap, easy to inspect, and survive as the first layer of larger networks.

## Chapter summary

This chapter followed a single idea: **represent discrete symbols as dense, learned vectors whose geometry reflects how the symbols are used.**

- **Why embeddings (4.1).** One-hot vectors are $`V`$-dimensional, sparse, and make every pair of words equally unrelated. Multiplying a one-hot vector by a weight matrix selects a row, so the first layer of a network is really a lookup table of dense vectors, `nn.Embedding` in PyTorch, trained end to end with the rest of the network; only the rows of words in a batch receive gradients. The distributional hypothesis, that words in similar contexts have similar meanings, turns raw text into a training signal for those vectors.
- **Word2Vec (4.2).** Skip-gram predicts context words from a center word, and CBOW predicts the center word from its context; both are two-layer networks trained with cross-entropy. Negative sampling replaces the $`V`$-way softmax with $`k + 1`$ binary classifications, making training fast. fastText builds word vectors from character n-grams, which handles rare and unseen words and anticipates subword tokenization.
- **Geometry (4.3).** Cosine similarity compares directions; nearest neighbors are words used in similar contexts, including antonyms. Some relations appear as consistent vector offsets, but analogies are fragile and depend on excluding the input words. Embeddings absorb social biases from their training text, which can be measured but not simply removed.
- **The limits of static embeddings (4.4).** A static vector blends all senses of a word, dominated by the most frequent one, and a bag of static vectors ignores word order. Fixing this requires a vector for each occurrence, computed from the whole sentence.

Two threads lead forward. Chapter 5 decides what the rows of the embedding table stand for, by splitting text into subword tokens. Chapter 6 builds the Transformer, whose attention layers turn a sequence of looked-up token vectors into contextual vectors, and Section 6.13 turns those vectors into embeddings of whole sentences for semantic search.

## Exercises

1. **One-hot versus lookup.** For a vocabulary of $`V = 50{,}000`$ and $`d = 512`$, how many multiply-adds does computing $`\mathbf{e}_i^\top W`$ as a dense matrix-vector product take, and how many does a table lookup take? For a batch of 32 sequences of 1,024 tokens, how much memory would the one-hot input tensor occupy in 32-bit floats? Then verify in PyTorch that `F.one_hot(ids, V).float() @ emb.weight` equals `emb(ids)` for a random `ids` tensor.

2. **Deriving negative sampling.** Starting from $`\ell_{\text{NEG}}`$ in Section 4.2, derive $`\partial \ell / \partial \mathbf{v}_c`$ using $`\sigma'(z) = \sigma(z)(1 - \sigma(z))`$. Show that when all output vectors are zero, the loss equals $`(k + 1) \ln 2`$. Then explain in words why a negative sample that the model already scores as clearly fake contributes almost nothing to the gradient.

3. **The 3/4 power.** For a toy vocabulary with counts $`(10{,}000, 1{,}000, 100, 10, 1)`$, compute the noise distribution $`P_n`$ for exponents $`\alpha = 0, 0.5, 0.75, 1`$. Plot each word's share against $`\alpha`$, and describe what goes wrong at the two extremes $`\alpha = 0`$ (uniform) and $`\alpha = 1`$ (unigram). If you have completed Lab 4.2, retrain with $`\alpha = 0`$ and $`\alpha = 1`$ and compare the nearest neighbors of a few words.

4. **Analogies with and without filtering.** Using the pretrained vectors from Lab 4.3, evaluate 20 analogies of your choice from at least three relation types (for example gender, country-capital, and verb tense). For each, report whether the correct answer is top-1 when the three input words are excluded and when they are not. Which relation types work best, and how often is the unfiltered top answer one of the inputs?

5. **Which rows learn.** Build an `nn.Embedding` table with 1,000 rows and $`d = 16`$, followed by a linear layer that predicts one of 1,000 classes with cross-entropy. Run one forward and backward pass on a batch that contains only IDs 0 through 9, some of them repeated. Count how many rows of the embedding table and how many rows of the output layer's weight receive a nonzero gradient, explain the difference, and check that a repeated ID's gradient is the sum of its per-position gradients.

6. **Senses in static vectors.** Using the pretrained vectors from Lab 4.3, pick five ambiguous words. For each, choose three words that clearly belong to each of two senses (for example "loan," "deposit," "account" and "river," "shore," "muddy" for "bank"), and compute the word's average cosine similarity to each group. Which sense dominates? Then compute the vector of the word minus the average of the dominant-sense group, and list its nearest neighbors. Does the other sense appear, and what does that suggest about how a static vector stores a mixture?

## Further reading

Bojanowski, Piotr, Edouard Grave, Armand Joulin, and Tomas Mikolov. "Enriching Word Vectors with Subword Information." *Transactions of the Association for Computational Linguistics* 5 (2017): 135–146. https://arxiv.org/abs/1607.04606.

Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. "Distributed Representations of Words and Phrases and Their Compositionality." In *Advances in Neural Information Processing Systems 26*, 2013. https://arxiv.org/abs/1310.4546.

Pennington, Jeffrey, Richard Socher, and Christopher D. Manning. "GloVe: Global Vectors for Word Representation." In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, 1532–1543, 2014. https://aclanthology.org/D14-1162/.
