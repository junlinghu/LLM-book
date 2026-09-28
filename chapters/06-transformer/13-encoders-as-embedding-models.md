# 6.13 Encoders as Embedding Models

Section 12 used an encoder-only model such as BERT by attaching a small output layer and fine-tuning the whole network on a labeled task. An encoder is useful in a second way, with no task-specific head at all: as an **embedding model**. Its final layer produces one vector per token, and each vector depends on the entire input, so an encoder gives exactly the *contextual* embeddings that Chapter 4 (Section 4.4) asked for, where the same word gets different vectors in different sentences. Pooling those vectors and training the encoder so that similar texts get similar vectors turns it into a function from any piece of text to a single vector. This section looks at contextual token vectors from a pretrained BERT, shows how to build sentence embeddings from them, and uses those embeddings for semantic search and clustering. Chapter 7 (Section 7.6) connects them to a generative model in retrieval-augmented generation.

## Contextual vectors from a pretrained encoder

Chapter 4 ended with a problem that no static embedding can solve: "bank" in "we sat on the river bank" and in "I opened a bank account" gets one and the same vector, a blend dominated by the more frequent financial sense. In an encoder, the token embedding lookup of Section 2 still gives both occurrences the same starting vector, but every layer then mixes in information from the other positions through bidirectional self-attention (Section 3), with no causal mask (Section 4). By the last layer, each occurrence has its own vector, shaped by its context on both sides.

[Code 6.13.1](#code-6131-contextual-vectors-for-bank) extracts the last-layer vector for "bank" from BERT-base in three sentences and compares them by cosine similarity:

| Sentences compared | Cosine similarity of the two "bank" vectors |
|---|---|
| "we sat on the river bank" vs. "i opened a bank account" | 0.332 |
| "i opened a bank account" vs. "the bank approved my loan" | 0.746 |

The two financial occurrences come out much more similar to each other than the river occurrence is to either of them, even though all three start from the same row of BERT's embedding table. (Exact values can differ slightly across library versions.)

Two practical details matter when using these vectors. First, the model sees tokens, not words (Chapter 5), and a long or rare word may be split into several tokens. "bank" happens to be a single token in BERT's vocabulary; for a word that is split, a common approach is to average its tokens' vectors. Second, the model produces vectors at every layer, not just the last (in Hugging Face `transformers`, pass `output_hidden_states=True`), and different layers hold different information. Vectors from lower layers stay closer to the static embedding of the token, and vectors from higher layers are more shaped by context. Which layer works best depends on the task.

## From token vectors to one vector per text

Many practical problems are about whole pieces of text. Which help-center article answers this customer's question? Which of these million support tickets are about the same problem? Each calls for a single vector per sentence, paragraph, or document, arranged so that texts with similar meanings are close together: a function that maps a text of any length to a fixed-size **sentence embedding** $`\mathbf{s} \in \mathbb{R}^d`$, such that the cosine similarity of two texts' vectors reflects how similar their meanings are.

### Averaging static word vectors

The simplest approach averages the static embeddings of the words in the text:

```math
\mathbf{s} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{v}_{w_t} .
```

This is fast, needs no extra training, and works surprisingly well as a baseline for grouping texts by topic: a text about cooking averages many cooking-related vectors and lands in the cooking region of the space. Its weaknesses are those of static embeddings, made worse by averaging. Word order disappears completely, so "the dog chased the cat" and "the cat chased the dog" get identical vectors. Each word keeps only its one blended sense. And very frequent words such as "the" and "of" contribute as much as the informative ones; a common improvement is a weighted average that gives rare words more weight.

### Pooling contextual token vectors

An encoder gives us something better to average: its last-layer vectors $`\mathbf{h}_1, \dots, \mathbf{h}_T`$, which already reflect word order and context. The most common pooling methods are:

- **Mean pooling**: average the contextual vectors of all real tokens.
- **The `[CLS]` vector**: BERT prepends `[CLS]` to every input (Section 12), and its final vector can serve as a summary of the whole text.
- **Last token**: in a decoder-only model (Chapter 7), only the final position has seen the entire text, because of the causal mask, so its vector is the natural summary.

Mean pooling needs one piece of care. Sentences in a batch have different lengths, so shorter ones are padded to a common length, and the padding positions must not be counted in the average. The tokenizer's **attention mask**, 1 for real tokens and 0 for padding, is the same information as the padding mask of Section 4, and weighting the sum by it removes the padding. In the toy batch of [Code 6.13.2](#code-6132-mean-pooling-with-a-padding-mask), the first sentence has two real tokens, $`(1, 0)`$ and $`(3, 2)`$, and one padding position; mean pooling with the mask gives $`(2, 1)`$, while a plain average over all three positions would have given $`(1.33, 0.67)`$.

There is a catch. A model pretrained with masked language modeling was never asked to make whole-sentence vectors comparable by cosine similarity. Reimers and Gurevych (2019) found that sentence vectors pooled from an off-the-shelf BERT model performed poorly on semantic similarity benchmarks, in some settings worse than simply averaging static word vectors. The pretrained representations contain the needed information, but not in a geometry where cosine similarity reads it out.

### Training for similarity

The fix is to train the encoder directly for the property we want. **Sentence-BERT** (Reimers and Gurevych 2019) runs two texts through the *same* encoder (a *siamese* setup, since the two branches share all their weights), mean-pools each into a vector, and fine-tunes the encoder so that the two vectors are close when the texts have similar meanings and far apart when they do not. Because both texts are encoded independently by the same network, the result is a single function from text to vector: every document can be embedded once, and any two vectors can be compared with a cheap cosine.

Most sentence embedding models today are trained with a **contrastive** objective that resembles Word2Vec's negative sampling (Chapter 4). Training data consists of pairs of texts that belong together: a question and its answer, a search query and the page that was clicked, a title and its article, two paraphrases. For a batch of $`B`$ pairs $`(q_i, p_i)`$, embed all of them, and treat each query's own partner as the positive and the other $`B - 1`$ partners in the batch as negatives. The loss is a softmax cross-entropy over cosine similarities:

```math
\ell_i = -\log \frac{\exp\!\left(\cos(\mathbf{q}_i, \mathbf{p}_i) / \tau\right)}{\sum_{j=1}^{B} \exp\!\left(\cos(\mathbf{q}_i, \mathbf{p}_j) / \tau\right)} ,
```

where $`\tau`$ is a small **temperature** (such as 0.05) that sharpens the softmax, since cosines lie only between $`-1`$ and $`1`$. This is a $`B`$-way classification: for each query, pick out its true partner among the passages in the batch. It is the same idea as negative sampling, "score real pairs above fake ones," but with the other examples in the batch serving as the negatives for free, and with a softmax instead of independent sigmoids. The appendix implements it in a few lines ([Code 6.13.3](#code-6133-in-batch-contrastive-loss)).

Larger batches provide more negatives and generally train better embeddings. Adding **hard negatives**, passages that look relevant but are not (for instance, retrieved by keyword overlap), forces the model to learn finer distinctions than random negatives do.

### Longer texts and document embeddings

An encoder has a maximum input length (BERT's learned position embeddings stop at 512 positions), and a single vector can only hold so much. For long documents, the standard practice is to split the document into **chunks** of a few hundred tokens, often with some overlap so that no sentence is cut off from its context, and embed each chunk separately. A search then returns chunks, and the document containing the best chunk can be shown to the user. Averaging all chunk vectors into one document vector is possible, and works for coarse topic grouping, but it blurs together the different things a long document says.

## Semantic search

With a sentence embedding model, search by meaning takes a few lines. Embed every document once and store the vectors. At query time, embed the query with the same model and return the documents with the highest cosine similarity. [Code 6.13.4](#code-6134-semantic-search-and-clustering) does this with `all-MiniLM-L6-v2`, a small Sentence-BERT-style encoder from the `sentence-transformers` library that produces 384-dimensional vectors, over six short documents. The top two results for three queries:

| Query | Top results (cosine similarity) |
|---|---|
| "I can't log in to my mailbox" | 0.537 "How to reset a forgotten email password."<br>0.443 "Steps for recovering access to your account." |
| "caring for a young cat" | 0.437 "Kittens need to be fed several times a day."<br>0.430 "The cat sat on the windowsill, watching birds." |
| "monetary policy news" | 0.509 "The central bank raised interest rates again."<br>0.192 "Our quarterly revenue grew by twelve percent." |

Look at what matched. "I can't log in to my mailbox" shares no content words with "How to reset a forgotten email password," yet it is the top result. "Monetary policy news" finds the sentence about interest rates, and the revenue sentence, related to money but not to monetary policy, comes a distant second. Keyword search would have found none of these. Because the vectors are normalized to unit length, the dot product is the cosine similarity (Chapter 4, Section 4.3), and scoring the whole collection is a single matrix-vector product.

Two practical points apply to every embedding-based search system:

- **Queries and documents must be embedded by the same model.** Vectors from different models live in unrelated spaces, and their cosine similarities are meaningless. If you change the embedding model, you must re-embed the whole collection.
- **Absolute scores depend on the model.** A cosine of 0.5 may be a strong match for one model and a weak one for another. Compare scores within a model, and calibrate any threshold for "relevant enough" on examples.

### Searching millions of vectors

Exact search compares the query with every stored vector, at a cost of $`N \times d`$ multiply-adds for $`N`$ documents. On a modern CPU or GPU that is fast for thousands or even millions of vectors, but it becomes too slow for billions of vectors or thousands of queries per second. **Approximate nearest neighbor (ANN)** indexes trade a small loss in accuracy for large speedups. Two families are widely used: *clustering-based* indexes partition the vectors into groups and search only the few groups nearest to the query, and *graph-based* indexes connect each vector to its near neighbors and search by walking the graph toward the query. Libraries such as FAISS implement both, and **vector databases** wrap them with storage, filtering by metadata, and updates. Many production systems also combine embedding search with traditional keyword search, because embeddings can miss exact matches that matter, such as product codes, names, and rare technical terms.

## Clustering and other uses

Once texts are vectors, the standard tools of machine learning apply directly.

**Clustering** groups texts by meaning without any labels. Running k-means with three clusters on the six normalized document vectors from the search example (Code 6.13.4) separates them into the three topics, cats, account access, and finance, two documents each, without supervision. The same approach organizes support tickets by issue, groups news articles into stories, or reveals what topics a large dataset contains. A closely related use is **near-duplicate detection**: pairs of texts with very high cosine similarity are likely paraphrases or copies, which is useful when cleaning the enormous text collections used to pretrain language models (Chapter 7).

**Classification** with few labels is another. Training a small classifier, even logistic regression (Chapter 2), on top of frozen sentence embeddings often works well with only a few hundred labeled examples, because the embedding model has already done the hard work of representing meaning.

This completes the families of Section 12 as users meet them: encoder-decoder models that map one sequence to another, and encoder-only models used either with a task head or as embedding models. The third family keeps only the decoder, trains it to predict the next token of ordinary text, and is the subject of the next chapter.

## Key takeaways

- An encoder's last-layer vectors are contextual embeddings: in BERT, the two financial occurrences of "bank" have cosine similarity 0.746, while the river and financial occurrences have 0.332, although all start from the same embedding row.
- A sentence embedding maps a text of any length to one vector. It can be built by averaging static word vectors, or by pooling an encoder's token vectors (mean pooling with the padding mask, the `[CLS]` vector, or the last token of a decoder).
- Pooled vectors from a model pretrained only with masked language modeling compare poorly by cosine; Sentence-BERT fine-tunes a siamese encoder for similarity, and contrastive training with in-batch negatives is the standard objective.
- Semantic search embeds documents once and ranks them by cosine similarity to the embedded query; it matches meaning rather than keywords, requires one model for queries and documents, and scales with approximate nearest neighbor indexes.
- The same vectors support clustering, near-duplicate detection, and classification with few labels.

## Appendix: Code

The snippets below reproduce the results described in this section. They need PyTorch, Hugging Face `transformers` (Code 6.13.1), `sentence-transformers` and scikit-learn (Code 6.13.4); the pretrained models download on first use, and everything runs on a CPU. Printed output is shown in comments.

### Code 6.13.1: Contextual vectors for "bank"

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").eval()
cos = lambda a, b: F.cosine_similarity(a, b, dim=0).item()

def word_vector(sentence, word):
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state[0]           # (T, 768): one vector per token
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    return hidden[tokens.index(word)]                        # first occurrence of the word

a = word_vector("we sat on the river bank", "bank")
b = word_vector("i opened a bank account", "bank")
c = word_vector("the bank approved my loan", "bank")
print(f"river vs finance:   {cos(a, b):.3f}")
print(f"finance vs finance: {cos(b, c):.3f}")
# river vs finance:   0.332
# finance vs finance: 0.746
```

### Code 6.13.2: Mean pooling with a padding mask

```python
import torch

def mean_pool(token_vecs, mask):
    """token_vecs: (B, T, d) contextual vectors; mask: (B, T), 1 for real tokens, 0 for padding."""
    m = mask.unsqueeze(-1).float()                    # (B, T, 1)
    summed = (token_vecs * m).sum(dim=1)              # padding contributes nothing
    counts = m.sum(dim=1).clamp(min=1e-9)             # number of real tokens per sentence
    return summed / counts                            # (B, d)

token_vecs = torch.tensor([[[1., 0.], [3., 2.], [0., 0.]],     # 2 real tokens + 1 pad
                           [[2., 2.], [0., 4.], [4., 0.]]])    # 3 real tokens
mask = torch.tensor([[1, 1, 0],
                     [1, 1, 1]])
print(mean_pool(token_vecs, mask))
# tensor([[2., 1.],
#         [2., 2.]])
```

### Code 6.13.3: In-batch contrastive loss

```python
import torch
import torch.nn.functional as F

def in_batch_contrastive_loss(q, p, temperature=0.05):
    """q, p: (B, d) embeddings of B matching (query, passage) pairs.
    Row i of q should match row i of p; the other B - 1 passages act as negatives."""
    q, p = F.normalize(q, dim=-1), F.normalize(p, dim=-1)
    scores = q @ p.T / temperature                    # (B, B) scaled cosine similarities
    labels = torch.arange(q.shape[0])                 # the correct passage is on the diagonal
    return F.cross_entropy(scores, labels)
```

### Code 6.13.4: Semantic search and clustering

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

docs = [
    "The cat sat on the windowsill, watching birds.",
    "Our quarterly revenue grew by twelve percent.",
    "How to reset a forgotten email password.",
    "Kittens need to be fed several times a day.",
    "The central bank raised interest rates again.",
    "Steps for recovering access to your account.",
]
doc_emb = model.encode(docs, normalize_embeddings=True)       # (6, 384), unit length

def search(query, k=2):
    q = model.encode(query, normalize_embeddings=True)        # (384,)
    scores = doc_emb @ q                                      # cosine similarities
    top = scores.argsort()[::-1][:k]
    return [(float(scores[i]), docs[i]) for i in top]

for query in ["I can't log in to my mailbox", "caring for a young cat", "monetary policy news"]:
    print(query)
    for score, doc in search(query):
        print(f"   {score:.3f}  {doc}")
# I can't log in to my mailbox
#    0.537  How to reset a forgotten email password.
#    0.443  Steps for recovering access to your account.
# caring for a young cat
#    0.437  Kittens need to be fed several times a day.
#    0.430  The cat sat on the windowsill, watching birds.
# monetary policy news
#    0.509  The central bank raised interest rates again.
#    0.192  Our quarterly revenue grew by twelve percent.

labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(doc_emb)
for c in range(3):
    print(f"cluster {c}:", [d for d, l in zip(docs, labels) if l == c])
# cluster 0: ['The cat sat on the windowsill, watching birds.', 'Kittens need to be fed several times a day.']
# cluster 1: ['How to reset a forgotten email password.', 'Steps for recovering access to your account.']
# cluster 2: ['Our quarterly revenue grew by twelve percent.', 'The central bank raised interest rates again.']
```

## Further reading

Devlin, Jacob, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 4171–4186, 2019. https://arxiv.org/abs/1810.04805.

Reimers, Nils, and Iryna Gurevych. "Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks." In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing*, 2019. https://arxiv.org/abs/1908.10084.
