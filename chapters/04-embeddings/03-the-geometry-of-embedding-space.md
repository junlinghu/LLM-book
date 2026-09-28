# 4.3 The Geometry of Embedding Space

After training, a Word2Vec model is just a table: one row of $`d`$ numbers per word. The individual numbers mean nothing on their own. There is no "animal" coordinate or "plural" coordinate, and retraining with a different random seed would produce a completely different-looking table. What carries meaning is the *geometry* of the table: which vectors are close to which, and in what directions they differ. This section develops the tools for reading that geometry. We define cosine similarity and use it to find nearest neighbors, look at the famous vector analogies and the ways they mislead, and then face an uncomfortable fact: because embeddings learn from how people actually write, they also learn people's stereotypes.

## Measuring similarity

Given two word vectors $`\mathbf{a}`$ and $`\mathbf{b}`$ in $`\mathbb{R}^d`$, there are three natural ways to compare them.

The **dot product** $`\mathbf{a}^\top \mathbf{b} = \sum_i a_i b_i`$ is what the skip-gram model itself uses as a score (Section 4.2). It is large when the vectors point in similar directions *and* are long.

The **Euclidean distance** $`\lVert \mathbf{a} - \mathbf{b} \rVert`$ is the straight-line distance between the two points. It is small when the vectors are close in both direction and length.

The **cosine similarity** is the cosine of the angle $`\theta`$ between the vectors, which is the dot product after both vectors have been scaled to unit length:

```math
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\lVert \mathbf{a} \rVert \, \lVert \mathbf{b} \rVert} = \cos\theta \in [-1, 1] .
```

A cosine of 1 means the vectors point in the same direction, 0 means they are orthogonal (as every pair of one-hot vectors is, Section 4.1), and $`-1`$ means they point in opposite directions. The length of either vector plays no role.

For comparing word embeddings, cosine similarity is the standard choice. The reason is that a vector's length tends to reflect *how much* training signal the word received, which depends on how often and in how many different contexts it appears, rather than *what* the word means. The direction is where the meaning lives. Dot products and Euclidean distances mix the two; cosine keeps only the direction.

A useful fact connects all three. If we normalize every vector to unit length, $`\hat{\mathbf{a}} = \mathbf{a} / \lVert \mathbf{a} \rVert`$, then cosine similarity is just the dot product, and the squared Euclidean distance is a simple function of it:

```math
\lVert \hat{\mathbf{a}} - \hat{\mathbf{b}} \rVert^2 = \lVert \hat{\mathbf{a}} \rVert^2 + \lVert \hat{\mathbf{b}} \rVert^2 - 2\, \hat{\mathbf{a}}^\top \hat{\mathbf{b}} = 2 - 2 \cos(\mathbf{a}, \mathbf{b}) .
```

So for unit vectors, "highest cosine," "highest dot product," and "smallest distance" all pick out the same neighbors. This is why embedding tables are often normalized once, after training, and then searched with plain dot products, which are fast to compute with a single matrix multiplication. Section 4.6 relies on the same fact for semantic search.

## A toy embedding space

Real embeddings have hundreds of dimensions, which makes them hard to reason about by eye. To see the mechanics, we use a hand-made table of 4-dimensional vectors. Think of the axes loosely as "royalty," "maleness," "fruitness," and "personhood." Learned embeddings never have axes this clean, but the computations are exactly the same.

```python
import torch
import torch.nn.functional as F

words = ["king", "queen", "man", "woman", "prince", "apple", "banana"]
E = torch.tensor([
    [0.90,  0.75, 0.02, 0.70],   # king
    [0.88, -0.70, 0.05, 0.72],   # queen
    [0.08,  0.80, 0.01, 0.90],   # man
    [0.10, -0.78, 0.03, 0.88],   # woman
    [0.70,  0.70, 0.00, 0.55],   # prince
    [0.00,  0.02, 0.95, 0.05],   # apple
    [0.03, -0.01, 0.90, 0.02],   # banana
])
idx = {w: i for i, w in enumerate(words)}

def cosine(a, b):
    return (a @ b) / (a.norm() * b.norm())

print(f"cos(king, queen) = {cosine(E[idx['king']], E[idx['queen']]):.3f}")
print(f"cos(king, apple) = {cosine(E[idx['king']], E[idx['apple']]):.3f}")
```

```text
cos(king, queen) = 0.423
cos(king, apple) = 0.053
```

"king" and "queen" share royalty and personhood but differ in gender, so their cosine is moderate; "king" and "apple" have almost nothing in common. PyTorch also provides `F.cosine_similarity`, which does the same computation along a chosen dimension.

## Nearest neighbors

The simplest question to ask an embedding space is "which words are closest to this one?" To answer it for every word at once, normalize the rows of the embedding matrix, then multiply by the normalized query vector. The result is a vector of $`V`$ cosine similarities, one per word, and the top entries are the **nearest neighbors**:

```python
def nearest(query, E, k=3, exclude=()):
    sims = F.normalize(E, dim=1) @ F.normalize(query, dim=0)   # cosine with every row
    for i in exclude:
        sims[i] = -float("inf")                                  # never return these
    top = sims.topk(k)
    return [(words[i], round(s.item(), 3)) for s, i in zip(top.values, top.indices)]

print(nearest(E[idx["banana"]], E, exclude=[idx["banana"]]))
```

```text
[('apple', 0.998), ('queen', 0.077), ('woman', 0.052)]
```

We exclude the query word itself, which would otherwise always be its own nearest neighbor with cosine 1. For a real vocabulary of $`V`$ words, this is one matrix-vector product with $`V \times d`$ multiply-adds, which takes well under a second even for millions of words. For billions of vectors, Section 4.6 introduces approximate search.

What do nearest neighbors look like in a real embedding space trained on a large corpus? You can explore this yourself in Lab 4.2, but some patterns are consistent. The neighbors of a word typically include:

- **Synonyms and near-synonyms**, which appear in nearly identical contexts.
- **Other members of the same category**: the neighbors of a country name are other countries, and the neighbors of a first name are other first names.
- **Morphological variants**: other forms of the same word, such as plurals and different verb tenses.
- **Antonyms.** "hot" and "cold," or "good" and "bad," appear in very similar contexts ("the water is ___"), so the distributional hypothesis puts them close together, even though their meanings are opposite.

The last point deserves emphasis. Embedding similarity measures **relatedness of use**, not identity of meaning. Two words are close if they can stand in the same places in text. That is often what we want, for instance when grouping topics, but a system that treats nearest neighbors as synonyms will sometimes swap a word for its opposite.

## Analogies: directions carry meaning

The most striking property of Word2Vec embeddings is that some *differences* between vectors are meaningful too. Mikolov et al. (2013a) showed that the offset from "man" to "woman" is roughly the same as the offset from "king" to "queen," so that

```math
\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}} .
```

More generally, to solve an analogy "$`a`$ is to $`b`$ as $`c`$ is to ?", compute the vector $`\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c`$ and return the word whose vector has the highest cosine similarity with it:

```math
d^* = \arg\max_{d \,\notin\, \{a, b, c\}} \cos\!\left(\mathbf{v}_d,\; \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c\right) .
```

The idea is that the offset $`\mathbf{v}_b - \mathbf{v}_a`$ captures a *relation*, such as "female counterpart of," "capital city of," or "past tense of," and adding it to $`\mathbf{v}_c`$ applies the same relation to a new word. Mikolov et al. built a test set of such questions, semantic ones like country-capital pairs and syntactic ones like adjective-adverb or singular-plural pairs, and it became a standard way to evaluate word embeddings for several years.

In our toy space:

```python
def analogy(a, b, c, E, k=1, exclude_inputs=True):
    """a is to b as c is to ?  Searches for the word nearest to b - a + c."""
    q = E[idx[b]] - E[idx[a]] + E[idx[c]]
    ex = [idx[a], idx[b], idx[c]] if exclude_inputs else []
    return nearest(q, E, k=k, exclude=ex)

print(analogy("man", "king", "woman", E))
print(analogy("man", "king", "woman", E, k=3, exclude_inputs=False))
```

```text
[('queen', 0.996)]
[('queen', 0.996), ('woman', 0.802), ('king', 0.354)]
```

Why would a model trained only to predict context words produce parallel offsets? Consider what distinguishes "king" from "queen" in text: the contexts of "king" include more "he," "his," and "father," and the contexts of "queen" include more "she," "her," and "mother." The contexts of "man" and "woman" differ in much the same way. Since each word's vector is shaped by its contexts, the same context difference produces roughly the same vector difference. The analogy works to the extent that the relation shows up as a *consistent* shift in contexts across many word pairs.

### Where analogies break down

Analogies make a memorable demonstration, and they are easy to over-interpret. Several caveats apply.

**The inputs are excluded, and that matters.** The $`\arg\max`$ above skips $`a`$, $`b`$, and $`c`$. Most analogy code does this silently. In the gensim library, for example, `most_similar(positive=["king", "woman"], negative=["man"])` never returns any of the three query words. Without that filter, the nearest word to $`\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c`$ in a real embedding space frequently turns out to be one of the inputs, often $`b`$ ("king") itself, because the offset $`\mathbf{v}_c - \mathbf{v}_a`$ is small compared with $`\mathbf{v}_b`$. Our toy space is too clean to show this: its second-best answer without filtering is the input "woman," but "queen" still wins. With real vectors, the "correct" answer is often only the best answer *among the words that remain*, which is a weaker claim than the equation suggests.

**Only some relations are linear.** Relations with a single consistent contextual signature, such as gender, singular-plural, or country-capital for well-known countries, tend to work. Many others do not: relations that differ across word pairs, relations involving rare words, and relations with more than one valid answer. Averaged accuracy on a benchmark hides how uneven the results are.

**Frequency and neighborhoods dominate.** The answer is often simply a close neighbor of $`c`$ or $`b`$ that happens to fit. A word that is near "queen" for unrelated reasons can win the $`\arg\max`$ over the intended answer.

**Analogy accuracy is not usefulness.** Embeddings that score well on analogies do not necessarily work best for the downstream tasks we care about, and vice versa. Treat analogies as an illustration of structure in the space, not as a measure of quality.

## Seeing the space in two dimensions

Plots of embedding spaces, with countries in one cluster and fruits in another, are a staple of talks and textbooks. They are made by projecting the $`d`$-dimensional vectors down to two dimensions. The simplest projection is **principal component analysis (PCA)**, which keeps the two directions along which the vectors vary most:

```python
def pca_2d(X):
    Xc = X - X.mean(dim=0)                    # center the cloud of points
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vh[:2].T                      # coordinates along the top 2 directions
```

PCA is a linear projection, so it preserves offsets: if "king − queen" and "man − woman" are parallel in the full space, their projections are parallel too, which makes PCA the right tool for plotting analogy offsets. Nonlinear methods such as t-SNE and UMAP are better at revealing clusters, but they distort distances and directions freely, so do not read analogies or global distances off a t-SNE plot.

Either way, remember that a 2-D plot shows a sliver of a space with hundreds of dimensions. Two words that look close in the plot may be far apart in the full space, and vice versa. Use plots to form hypotheses and cosine similarities to check them.

> **Code Lab 4.2** loads a set of pretrained word vectors and explores their geometry: nearest neighbors for words of your choice, analogies with and without excluding the input words, and a 2D PCA plot of several word categories and analogy pairs.

In the lab, loading pretrained vectors takes one call with the gensim library:

```python
import gensim.downloader as api

wv = api.load("word2vec-google-news-300")      # 300-d skip-gram vectors; a large download
wv.most_similar("coffee", topn=5)              # nearest neighbors by cosine
wv.most_similar(positive=["king", "woman"], negative=["man"], topn=5)   # analogy
wv.similarity("hot", "cold")                   # cosine similarity of two words
```

## Bias carried in embeddings

The distributional hypothesis has a consequence that is easy to state and hard to live with: embeddings reflect how words are *used* in the training text, and human text is full of stereotypes. If a corpus mentions nurses mostly alongside "she" and engineers mostly alongside "he," skip-gram learns exactly that, because predicting those co-occurrences is its whole job. The model is not malfunctioning; it is faithfully compressing the statistics it was given, including the ones we would rather it did not.

This bias is measurable with the geometric tools of this section. Define a **gender direction** as the normalized difference of a pair of gendered words, or better, the average over several pairs:

```math
\mathbf{g} = \frac{1}{|P|} \sum_{(m, f) \in P} \left(\mathbf{v}_m - \mathbf{v}_f\right), \qquad P = \{(\text{he}, \text{she}), (\text{man}, \text{woman}), (\text{father}, \text{mother}), \dots\} .
```

Then project words that *should* be gender-neutral onto it: the score $`\cos(\mathbf{v}_w, \mathbf{g})`$ is positive for words that lean toward the male side and negative for words that lean toward the female side. When this is done for occupation words in embeddings trained on large web and news corpora, many occupations turn out to lean strongly one way or the other, in line with common stereotypes. The analogy machinery shows the same thing more vividly: an analogy query such as "man is to [occupation] as woman is to ?" can return a stereotyped completion. The same approach works for other attributes, such as ethnicity, religion, or age, by choosing different word sets; association tests compare the average similarity between groups of target words (for example, names associated with different groups) and groups of attribute words (for example, pleasant and unpleasant words).

Why this matters:

- **Downstream systems inherit it.** A résumé-screening or search system built on biased embeddings can rank people differently based on words correlated with gender or ethnicity, even if those attributes are never used directly.
- **It is not only a word-embedding problem.** LLMs are trained on the same kind of text with objectives built on the same co-occurrence statistics. Their embedding tables and internal representations absorb the same associations, and those can surface in generated text.

Can the bias be removed? A simple idea is **projection-based debiasing**: estimate the bias direction $`\mathbf{g}`$, then remove each neutral word's component along it,

```math
\mathbf{v}_w \leftarrow \mathbf{v}_w - \left(\mathbf{v}_w^\top \hat{\mathbf{g}}\right) \hat{\mathbf{g}}, \qquad \hat{\mathbf{g}} = \mathbf{g} / \lVert \mathbf{g} \rVert .
```

After this step, the words have zero projection on $`\hat{\mathbf{g}}`$. But bias is not confined to a single direction. Words that were stereotypically associated with one gender still tend to cluster together after the projection, and a simple classifier can often still recover the original association from the remaining dimensions. Projection makes the most obvious measurement look clean without removing the underlying structure. The honest summary is that bias in embeddings can be measured and reduced, but not simply switched off, and that measuring it should be a routine part of building any system on top of learned representations.

## Key takeaways

- Individual embedding coordinates are meaningless; similarity and directions carry the meaning.
- Cosine similarity compares directions and ignores length; for unit-normalized vectors it equals the dot product, and squared Euclidean distance is $`2 - 2\cos`$.
- Nearest neighbors in embedding space are words used in similar contexts: synonyms, category members, morphological variants, and also antonyms.
- Some relations appear as consistent vector offsets, so $`\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}}`$ lands near $`\mathbf{v}_{\text{queen}}`$, but the result depends on excluding the input words and holds only for relations with a consistent contextual signature.
- 2-D plots are useful sketches; PCA preserves offsets, while t-SNE and UMAP preserve clusters but distort geometry.
- Embeddings absorb social stereotypes from their training text. Bias can be measured with directions and association tests, but projecting out a single direction does not remove it.

## Further reading

Jurafsky, Daniel, and James H. Martin. *Speech and Language Processing*. 3rd ed. draft. See the chapter "Vector Semantics and Embeddings." https://web.stanford.edu/~jurafsky/slp3/.

Mikolov, Tomas, Kai Chen, Greg Corrado, and Jeffrey Dean. "Efficient Estimation of Word Representations in Vector Space." arXiv preprint arXiv:1301.3781, 2013. https://arxiv.org/abs/1301.3781.
