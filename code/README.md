# code

Runnable examples, labs, and figure-generation scripts, one folder per chapter. Folder names match the slugs under [`chapters/`](../chapters/).

Prose and rendered figures stay in `chapters/<chapter-slug>/`. Scripts here write PNG files into the matching `chapters/<chapter-slug>/figures/` directory.

Section labs from the chapter drafts also live here as Jupyter notebooks. Numbered `### Code X.Y.Z` listings, appendix listings such as `### A.1`, and other multi-line Python examples each have one notebook:

```text
code/<chapter-slug>/X.Y.Z-<kebab-title>.ipynb
code/<chapter-slug>/<section>-A.1-<kebab-title>.ipynb
```

For example, Code 2.5.1 in Chapter 2 is [`02-neural-network-basics/2.5.1-gradient-descent-on-an-elongated-quadratic-bowl.ipynb`](02-neural-network-basics/2.5.1-gradient-descent-on-an-elongated-quadratic-bowl.ipynb). The chapter section keeps the heading and a short description, and links to the notebook. In-text references such as `[Code 2.5.1](#code-251-...)` still point at that heading.

A notebook starts with the listing title and the prose that introduced it, then the Python from the chapter. Where a later listing reused an import or a definition from an earlier one, that dependency is included so the notebook runs on its own. Results that the chapter printed are in a following markdown cell labeled **Expected output**; the notebooks are not pre-executed.

## Layout

| Folder | Contents |
|--------|----------|
| [`01-introduction`](01-introduction/) | No code yet |
| [`02-neural-network-basics`](02-neural-network-basics/) | From-scratch autograd, NumPy MLP, `fig_*.py` figure scripts, and notebooks for Code 2.2–2.9 |
| [`03-deep-neural-networks`](03-deep-neural-networks/) | Notebooks for the worked examples in Sections 3.1–3.10 |
| [`04-embeddings`](04-embeddings/) | Notebooks for Sections 4.1–4.4 |
| [`05-tokenizer`](05-tokenizer/) | Notebooks for Sections 5.1–5.10, including the byte-level BPE tokenizer |
| [`06-transformer-basic-architecture`](06-transformer-basic-architecture/) | `make_figures.py` and notebooks for Sections 6.2–6.6 |
| [`07-training-a-transformer`](07-training-a-transformer/) | `make_figures.py` and notebooks for Sections 7.1–7.7 |
| [`08-generative-pretraining-gpt`](08-generative-pretraining-gpt/) | `make_figures.py` and notebooks for Code 8.1–8.6 |
| [`09-supervised-fine-tuning-sft`](09-supervised-fine-tuning-sft/) | Notebooks for Sections 9.1–9.7 |
| [`10-reinforcement-learning-basics`](10-reinforcement-learning-basics/) | Gridworld, bandits, DQN, REINFORCE, PPO, figure scripts, and notebooks for Code 10.1–10.7 |
| [`11-rlhf`](11-rlhf/) | `make_figures.py` and notebooks for Code 11.2 and 11.4–11.7 |
| [`12-evaluation`](12-evaluation/) | Notebooks for Sections 12.2–12.8 |
| [`13-inference`](13-inference/) | Notebooks for Sections 13.1–13.9 |
| [`14-more-rl-methods-for-llms`](14-more-rl-methods-for-llms/) | Notebooks for Code 14.2–14.5 |

Chapter 1 does not have labs yet.

## Running scripts

From the repository root:

```bash
cd code/02-neural-network-basics
for f in fig_*.py; do python "$f"; done
```

```bash
python code/06-transformer-basic-architecture/make_figures.py
python code/11-rlhf/make_figures.py
```

Chapter 2 needs NumPy, Matplotlib, and PyTorch. Chapter 10 also needs Gymnasium. Notebook dependencies vary (NumPy, PyTorch, Gymnasium, Hugging Face `transformers`, and others). See each chapter’s `README.md` under `chapters/` for that chapter’s labs and dependencies.
