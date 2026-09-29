# Code

Section labs from the chapter drafts live here as Jupyter notebooks. Each `### Code X.Y.Z` listing under `chapters/` has one notebook:

```text
code/<chapter-slug>/X.Y.Z-<kebab-title>.ipynb
```

For example, Code 2.5.1 in Chapter 2 is [`02-neural-network-basics/2.5.1-gradient-descent-on-an-elongated-quadratic-bowl.ipynb`](02-neural-network-basics/2.5.1-gradient-descent-on-an-elongated-quadratic-bowl.ipynb).

The chapter section keeps the heading and a short description, and links to the notebook. In-text references such as `[Code 2.5.1](#code-251-...)` still point at that heading.

A notebook starts with the listing title and the prose that introduced it, then the Python from the chapter. Where a later listing reused an import or a definition from an earlier one, that dependency is included so the notebook runs on its own. Results that the chapter printed are in a following markdown cell labeled **Expected output**; the notebooks are not pre-executed.

Figure-generation scripts and the images they write stay under `chapters/<chapter>/figures/`. They are not moved here.

Notebooks are grouped by chapter:

| Folder | Listings |
|--------|----------|
| [`02-neural-network-basics/`](02-neural-network-basics/) | Code 2.2–2.9 |
| [`07-training-a-transformer/`](07-training-a-transformer/) | Code 7.7 |
| [`08-generative-pretraining-gpt/`](08-generative-pretraining-gpt/) | Code 8.1–8.6 |
| [`10-reinforcement-learning-basics/`](10-reinforcement-learning-basics/) | Code 10.1–10.7 |
| [`11-rlhf/`](11-rlhf/) | Code 11.2, 11.4–11.7 |
| [`14-more-rl-methods-for-llms/`](14-more-rl-methods-for-llms/) | Code 14.2–14.5 |

Dependencies differ by notebook (NumPy, PyTorch, Gymnasium, Hugging Face `transformers`, and others). The chapter section that links the notebook says which packages that listing needs.
