# A Concise Guide to Large Language Models

Companion repository for the book by **Junling Hu**.

This repo holds the book’s chapter drafts, figures, and accompanying code. It is meant as a practical path from neural-network basics to how modern LLMs are trained, fine-tuned, evaluated, and served—with enough explanation that you can follow the ideas and enough code that you can try them.

## What this repo is for

- **Read the book in progress.** Chapter text lives under [`chapters/`](chapters/) as Markdown.
- **Run the examples.** Code that supports a chapter (from-scratch labs, figure scripts, notebooks) lives under [`code/`](code/), in a folder with the same slug as the chapter (`code/02-neural-network-basics/`, and so on).
- **Regenerate figures.** Where a chapter ships plotting scripts, you can rebuild the images from those scripts rather than editing PNGs by hand.

It is **not** a pretrained-model zoo, a production serving stack, or a complete copy of every commercial LLM paper. The focus is understanding the pipeline end to end.

## How the repo is organized

```text
LLM-book/
├── README.md                 ← you are here
├── chapters/                 ← prose and rendered figures
│   ├── README.md             ← short chapter index
│   ├── 01-introduction/
│   ├── 02-neural-network-basics/
│   ├── …
│   └── 14-more-rl-methods-for-llms/
└── code/                     ← runnable examples, labs, and figure scripts
    ├── README.md
    ├── 01-introduction/
    ├── 02-neural-network-basics/
    ├── …
    └── 14-more-rl-methods-for-llms/
```

Each chapter folder under `chapters/` typically contains:

| Item | Role |
|------|------|
| `README.md` | Chapter overview, section list, labs, takeaways |
| Numbered `.md` files | The section prose (e.g. `01-….md`, `02-….md`) |
| `figures/` | Rendered images used in the chapter |

The matching folder under [`code/`](code/) holds the Python that generates those figures and any from-scratch labs. See [`code/README.md`](code/README.md). Chapters that do not yet have code contain a short note that labs will be added as the chapter is written.

Start from a chapter’s `README.md`, then open the numbered sections in order.

## Chapter outline

1. [Introduction](chapters/01-introduction/)
2. [The Basics of Neural Networks](chapters/02-neural-network-basics/)
3. [Deep Neural Networks](chapters/03-deep-neural-networks/)
4. [Embeddings](chapters/04-embeddings/)
5. [Tokenizer](chapters/05-tokenizer/)
6. [Transformer: Basic Architecture](chapters/06-transformer-basic-architecture/)
7. [Training a Transformer](chapters/07-training-a-transformer/)
8. [Generative Pretraining (GPT)](chapters/08-generative-pretraining-gpt/)
9. [Supervised Fine-Tuning (SFT)](chapters/09-supervised-fine-tuning-sft/)
10. [Reinforcement Learning Basics](chapters/10-reinforcement-learning-basics/)
11. [RLHF](chapters/11-rlhf/)
12. [Evaluation](chapters/12-evaluation/)
13. [Inference](chapters/13-inference/)
14. [More RL Methods for LLMs](chapters/14-more-rl-methods-for-llms/)

Chapters early in the sequence (especially neural networks and deep networks) are the most complete. Later chapters may still be stubs or drafts while the book is written.

## Who it is for

Readers who want a clear picture of LLMs without unnecessary jargon: engineers, students, teachers, and practitioners who need to evaluate or build on these systems. Early chapters assume comfort with basic Python and high-school linear algebra; later chapters build on that foundation.

## Working with the code

Requirements vary by chapter. A typical setup for the neural-network chapters is Python 3 with **NumPy**, **Matplotlib**, and **PyTorch**.

Example (Chapter 2 figures). From the repository root:

```bash
cd code/02-neural-network-basics
for f in fig_*.py; do python "$f"; done
```

Those scripts write PNGs into `chapters/02-neural-network-basics/figures/`. Other chapters follow the same split: run the script from `code/<chapter-slug>/`, and the images land in `chapters/<chapter-slug>/figures/`. A single script can also be run by path:

```bash
python code/06-transformer-basic-architecture/make_figures.py
```

See each chapter’s `README.md` for that chapter’s labs and dependencies.

## Status

This is an active book draft. Content, figure numbering, and folder layout may change as chapters are revised. Prefer the latest `main` branch.

## License and attribution

Unless a file says otherwise, treat the materials as the author’s book draft. If you reuse excerpts or figures, credit **Junling Hu** and link back to this repository.
