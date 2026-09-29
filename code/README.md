# code

Runnable examples, labs, and figure-generation scripts, one folder per chapter. Folder names match the slugs under [`chapters/`](../chapters/).

Prose and rendered figures stay in `chapters/<chapter-slug>/`. Scripts here write PNG files into the matching `chapters/<chapter-slug>/figures/` directory.

## Layout

| Folder | Contents |
|--------|----------|
| [`01-introduction`](01-introduction/) | No code yet |
| [`02-neural-network-basics`](02-neural-network-basics/) | From-scratch autograd, NumPy MLP, and `fig_*.py` figure scripts |
| [`03-deep-neural-networks`](03-deep-neural-networks/) | No code yet |
| [`04-embeddings`](04-embeddings/) | No code yet |
| [`05-tokenizer`](05-tokenizer/) | No code yet |
| [`06-transformer-basic-architecture`](06-transformer-basic-architecture/) | `make_figures.py` |
| [`07-training-a-transformer`](07-training-a-transformer/) | `make_figures.py` |
| [`08-generative-pretraining-gpt`](08-generative-pretraining-gpt/) | `make_figures.py` |
| [`09-supervised-fine-tuning-sft`](09-supervised-fine-tuning-sft/) | No code yet |
| [`10-reinforcement-learning-basics`](10-reinforcement-learning-basics/) | Gridworld, bandits, DQN, REINFORCE, PPO, and figure scripts |
| [`11-rlhf`](11-rlhf/) | `make_figures.py` |
| [`12-evaluation`](12-evaluation/) | No code yet |
| [`13-inference`](13-inference/) | No code yet |
| [`14-more-rl-methods-for-llms`](14-more-rl-methods-for-llms/) | No code yet |

Folders marked “no code yet” will gain labs as those chapters are written.

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

Chapter 2 needs NumPy, Matplotlib, and PyTorch. Chapter 10 also needs Gymnasium. See each chapter’s `README.md` under `chapters/` for that chapter’s labs and dependencies.
