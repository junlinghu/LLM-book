# 10.4 Collecting Human Preference Data

The reward model can only be as good as the judgments it learns from. This section is about the data that feeds Stage 2 of the pipeline: what format the judgments take, how labelers are instructed and how often they agree, and which public preference datasets you can use to experiment. Much of it is less about machine learning than about careful data collection, but decisions made here, such as which responses are compared and what labelers are told to value, shape the final model as much as any hyperparameter in Section 7.

## Anatomy of a preference example

A preference example consists of a prompt $x$, two or more candidate responses, and a human judgment among them. In its simplest and most common form it is a triple

```math
(x,\ y_c,\ y_r),
```

where $`y_c`$ is the *chosen* response and $`y_r`$ the *rejected* one. The prompt may be a single user message or a whole conversation up to the turn being judged.

Where do the candidate responses come from? Usually they are sampled from the model being improved: the SFT model at first, then later RL policies (Section 3). Sampling several responses to the same prompt at temperature 1 gives variety. Some pipelines also include responses from other models, from earlier checkpoints, or written by people, so that the comparison covers a wider range of quality. The choice matters because the reward model learns to discriminate *among the kinds of responses it is shown*. A reward model trained only on pairs of fluent, on-topic responses may have no idea what to do with a degenerate one, which is exactly what an RL policy might produce.

## Pairwise comparisons vs. rankings vs. ratings

There are three main ways to ask for a judgment.

**Pairwise comparisons.** Show two responses and ask which is better. This is the format of Christiano et al. (2017), of Anthropic's helpful and harmless data, in which crowdworkers chose the more helpful of two responses at each turn of a conversation (Bai et al. 2022), and of Llama 2 (Touvron et al. 2023). A comparison can carry a strength: Llama 2's annotators also said whether their choice was significantly better, better, slightly better, or negligibly better or unsure, and Section 5 shows how that strength can enter the loss as a margin.

**Rankings.** Show $K$ responses and ask for an ordering. A ranking of $K$ responses implies $`\binom{K}{2}`$ pairwise comparisons, so it extracts more information per prompt that the labeler has read and understood. InstructGPT's labelers ranked between $K = 4$ and $K = 9$ responses per prompt (Ouyang et al. 2022). A related format asks only for the best of $K$: Ziegler et al. (2019) asked labelers to pick the best of four continuations, noting that more options let a labeler amortize the cost of reading the prompt.

**Ratings.** Score each response on its own, for example on a 1-to-7 Likert scale or on several attributes. InstructGPT's labelers gave each output an overall 1-to-7 quality rating alongside the ranking, plus binary labels such as whether it hallucinated or followed the instructions. NVIDIA's HelpSteer2 rates each response from 0 to 4 on five attributes: helpfulness, correctness, coherence, complexity, and verbosity (Wang et al. 2024). Ratings can always be converted into comparisons by comparing scores for the same prompt.

| Format | Information per prompt | Strengths | Weaknesses |
|---|---|---|---|
| Pairwise comparison | 1 comparison (plus optional strength) | Simple, fast, relative judgments are consistent | Little information per prompt read |
| Ranking of $K$ | $`\binom{K}{2}`$ correlated comparisons | Amortizes reading the prompt; InstructGPT's choice | Slower per task; comparisons within one ranking are not independent |
| Rating | 1 score per response, possibly per attribute | Absolute and reusable; multi-attribute detail | Scales drift between labelers and over time; coarse |

Most RLHF pipelines prefer comparisons or rankings, because relative judgments are easier to make consistently than absolute ones. Two labelers may disagree about whether a response is a 5 or a 6, yet agree about which of two responses is better. The same observation led Chatbot Arena to rank models from pairwise votes rather than scores (Section 12.3).

A practical detail with rankings: the $`\binom{K}{2}`$ comparisons from one ranking share responses and are strongly correlated. InstructGPT found that shuffling them into the dataset as independent examples made the reward model overfit within a single pass, and instead put all the comparisons from one prompt into the same batch element (Ouyang et al. 2022). Section 5 shows the corresponding loss.

Ties are another detail. Labelers sometimes cannot pick a winner. InstructGPT dropped ties from reward model training; Llama 2 kept a "negligibly better or unsure" category and gave it the smallest margin.

## Labeling guidelines

"Which response is better?" has no single answer until someone specifies *better for what*. Guidelines turn the goals of Section 1 into instructions a labeler can apply in a few minutes per example.

Good guidelines typically:

- **Define the criteria and their priority.** InstructGPT's labelers were asked to prioritize helpfulness to the user during training data collection, while in final evaluations they were asked to prioritize truthfulness and harmlessness (Ouyang et al. 2022). Llama 2 collected helpfulness and safety preferences separately, with separate guidelines, and trained a separate reward model for each (Touvron et al. 2023).
- **Give worked examples**, especially of hard cases: a harmful request, an ambiguous prompt, a response that is well written but wrong.
- **Say how to handle what the labeler cannot check.** A labeler cannot verify every fact in a medical answer. Should an unverifiable but confident answer beat a hedged one? Unless guidelines say otherwise, confident-sounding responses tend to win, which teaches the model to sound confident (Section 12.4 connects this to hallucination).
- **Break big judgments into smaller ones.** Sparrow broke the requirements for good dialogue into natural-language rules, such as not pretending to have a human identity, and asked raters about each rule separately (Glaese et al. 2022).

Not every project writes detailed guidelines. Bai et al. (2022) deliberately left crowdworkers to interpret "helpful" and "harmful" largely for themselves, trading consistency for breadth.

Guidelines are not neutral. They encode whose preferences count and what trade-offs are acceptable, and the reward model will learn them faithfully, including their blind spots.

## Who labels, and how often they agree

Labelers range from crowdworkers to contractors to domain experts. InstructGPT hired about 40 contractors, selected by a screening test of how well they could identify and respond to sensitive prompts and by their agreement with researchers on a labeling task; the authors noted that such a small group is not representative of everyone who will use the model (Ouyang et al. 2022). Stiennon et al. (2020) monitored agreement between labelers and researchers throughout data collection and gave labelers feedback. Bai et al. (2022) used crowdworkers from Amazon Mechanical Turk and Upwork.

However carefully they are chosen, people disagree. Some reported agreement rates:

- InstructGPT's training labelers agreed with each other on 72.6 ± 1.5% of comparisons, and held-out labelers, who produced no training data, on 77.3 ± 1.3% (Ouyang et al. 2022).
- In the summarization work, researcher-researcher agreement was 73 ± 4% (as reported by Ouyang et al. 2022), and on CNN/DM summaries the inter-labeler agreement was 66.9% (Stiennon et al. 2020).
- Bai et al. (2022) found average agreement of only about 63% between Anthropic researchers and their crowdworkers on a sample of comparisons.

These numbers set expectations for Section 5. If two careful people agree on only about 70 to 75% of comparisons, a reward model cannot be expected to "agree" with a held-out label much more often than that; a reward model accuracy of 70% can be excellent. Raw agreement also overstates consistency, because two people choosing at random agree half the time on a binary choice. Cohen's kappa (Section 12.5) corrects for chance: for a binary choice with balanced labels, the chance agreement is 0.5, and $`\kappa = (p_o - 0.5) / 0.5`$, so 72.6% observed agreement corresponds to $\kappa \approx 0.45$ ([Code 10.4.2](#code-1042-agreement-and-cohens-kappa-for-pairwise-labels)).

Disagreement is not only noise. Some comparisons are genuinely close, and for some prompts reasonable people want different things. A single reward model averages over all of this, so it learns the preferences of a typical labeler under the guidelines, not those of any particular user.

## Biases in human judgments

Beyond random disagreement, human judgments have systematic biases that a reward model will learn and an RL policy will amplify.

- **Length and thoroughness.** People often prefer longer responses that look more comprehensive. OpenAI noted that ChatGPT was "often excessively verbose," and attributed this partly to biases in the training data, "trainers prefer longer answers that look more comprehensive," and partly to over-optimization (OpenAI 2022). Stiennon et al. (2020) found that their reward models preferred longer summaries, and Singhal et al. (2023) found that response length explains much of the reward gain in several RLHF setups.
- **Confidence and style.** A fluent, confident, well-formatted answer can win over a correct but hesitant one, especially when the labeler cannot check the facts.
- **Agreement with the user.** Responses that agree with the user's stated views can be preferred to ones that correct them, a source of sycophancy (Chapter 11).
- **Position.** When two responses are shown side by side, the order can influence the choice, which is why interfaces randomize it (Section 12.5 discusses the same bias in evaluation).

A useful check on any preference dataset is to measure such features directly. In the helpful subset of Anthropic's HH-RLHF data, for example, the chosen response is the longer of the two in about 59% of training pairs ([Code 10.4.1](#code-1041-loading-hh-rlhf-and-checking-a-length-bias)). Length is thus mildly predictive of preference on its own, and a reward model can pick up that shortcut.

## Scale and cost

Preference datasets have grown by orders of magnitude. Ziegler et al. (2019) used 5,000 comparisons for stylistic tasks and 60,000 for summarization. Stiennon et al. (2020) collected 64,832 summary comparisons. Bai et al. (2022) collected a base dataset of 44,000 helpfulness and 42,000 harmlessness (red-teaming) comparisons, followed by further rounds from rejection sampling and online RLHF. For Llama 2, Meta collected over 1.4 million binary comparisons of its own (Touvron et al. 2023).

Because human labels are expensive, a growing share of preference data is produced by AI models acting as judges. UltraFeedback, for example, is labeled entirely by GPT-4 (Cui et al. 2024). AI feedback and the Constitutional AI approach are the subject of Chapter 11 (Section 6). This chapter assumes human labels, but the algorithms are the same whoever provides the comparisons.

## Public datasets

Several public preference datasets are widely used to train and study reward models.

| Dataset | Source of responses | Source of labels | Format and size |
|---|---|---|---|
| Anthropic HH-RLHF (Bai et al. 2022) | A 52B assistant model and variants | Crowdworkers, choosing the more helpful (or, for red-teaming, more harmful) response | Pairs of full dialogues (`chosen`, `rejected`); helpful and harmless subsets |
| OpenAI Summarize from Feedback (Stiennon et al. 2020) | Summarization policies and references | Trained labelers | 64,832 TL;DR comparisons, plus Likert ratings for evaluation |
| Stanford Human Preferences, SHP (Ethayarajh et al. 2022) | Human-written Reddit comments | Reddit votes, using a comment's later timing but higher score to infer preference | 385,000 pairs across 18 subject areas |
| UltraFeedback (Cui et al. 2024) | 17 open and proprietary models, over 60,000 instructions | GPT-4, on instruction-following, truthfulness, honesty, and helpfulness | Over 1 million ratings; often binarized into pairs |
| HelpSteer2 (Wang et al. 2024) | Model responses to real prompts | Human annotators | 21,362 responses, each rated 0 to 4 on five attributes |

HH-RLHF is the most common starting point for experiments and is used in this chapter's labs. Each record contains a `chosen` and a `rejected` transcript that share every turn except the final assistant response. Its dataset card warns that the data are meant for training preference models, not for supervised training of dialogue agents, because the harmlessness data include responses to harmful requests; the `chosen` response is only the *less* bad of two, not necessarily a good one.

Rating datasets are commonly *binarized* into pairs. The UltraFeedback Binarized version used to train the Zephyr model, for example, takes the response with the highest overall score as chosen and one of the other three at random as rejected (Tunstall et al. 2023). Binarization discards information, and different binarization rules give different datasets, so it is worth checking which rule a dataset used.

With a dataset of chosen and rejected responses in hand, the next step is to fit a function that scores responses so that chosen ones score higher. What model should that be, and what loss makes its scores mean something?

## Code for this section

The listings below need the `datasets` library.

### Code 10.4.1: Loading HH-RLHF and checking a length bias

Loads the helpful subset of HH-RLHF, splits each record into the shared prompt and the two final responses, and measures how often the chosen response is longer. The prompt ends at the last `"\n\nAssistant:"`, where the two transcripts diverge. A few dozen records do not share a prompt under this rule and are skipped. On the training split, this prints about 43,800 usable pairs, with the chosen response longer (in characters) in about 59% of them.

```python
from datasets import load_dataset

ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")

def split_prompt(chosen, rejected):
    """Both transcripts share the conversation so far; they differ only in the last assistant turn."""
    marker = "\n\nAssistant:"
    cut = chosen.rfind(marker) + len(marker)
    prompt = chosen[:cut]
    if not rejected.startswith(prompt):
        raise ValueError("chosen and rejected do not share the prompt")
    return prompt, chosen[cut:], rejected[cut:]

ex = ds["train"][0]
prompt, y_c, y_r = split_prompt(ex["chosen"], ex["rejected"])
print(prompt[-200:], "\nCHOSEN:", y_c[:200], "\nREJECTED:", y_r[:200])

n = longer = skipped = 0
for ex in ds["train"]:
    try:
        _, y_c, y_r = split_prompt(ex["chosen"], ex["rejected"])
    except ValueError:
        skipped += 1
        continue
    n += 1
    longer += len(y_c) > len(y_r)
print(f"pairs: {n}, skipped: {skipped}, chosen is longer: {longer / n:.3f}")
```

### Code 10.4.2: Agreement and Cohen's kappa for pairwise labels

Given two labelers' choices on the same comparisons (0 for the first response, 1 for the second), computes raw agreement and Cohen's kappa. With balanced labels, 72.6% agreement gives a kappa of about 0.45.

```python
import numpy as np

def agreement_and_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    p_o = np.mean(a == b)                                    # observed agreement
    p_a, p_b = a.mean(), b.mean()                            # how often each labeler picks response 2
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)                  # agreement expected by chance
    return p_o, (p_o - p_e) / (1 - p_e)

rng = np.random.default_rng(0)
truth = rng.integers(0, 2, 10_000)
flip = lambda p: np.where(rng.random(truth.size) < p, 1 - truth, truth)
a, b = flip(0.15), flip(0.15)                                # each labeler errs 15% of the time
print(agreement_and_kappa(a, b))                             # about (0.745, 0.49)
print((0.726 - 0.5) / 0.5)                                   # kappa for 72.6% agreement, balanced labels
```

## Key takeaways

- A preference example is a prompt with a chosen and a rejected response, usually sampled from the model being trained; what gets compared determines what the reward model learns to distinguish.
- Pairwise comparisons, rankings of $K$ responses, and ratings each trade speed, information, and consistency; relative judgments are the most consistent, and rankings yield $`\binom{K}{2}`$ correlated pairs.
- Guidelines define "better," including priorities between helpfulness, honesty, and harmlessness, and how to treat claims labelers cannot verify.
- Careful labelers agree on only about 65 to 77% of comparisons in published reports, which caps achievable reward model accuracy.
- Human judgments are biased toward length, confidence, and agreement; measure these features in any dataset you use.
- Public datasets include HH-RLHF, Summarize from Feedback, SHP, UltraFeedback (AI-labeled), and HelpSteer2.

## Further reading

Bai, Yuntao, et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." arXiv preprint arXiv:2204.05862, 2022. https://arxiv.org/abs/2204.05862.

Cui, Ganqu, et al. "UltraFeedback: Boosting Language Models with Scaled AI Feedback." In *Proceedings of the 41st International Conference on Machine Learning*, 2024. https://arxiv.org/abs/2310.01377.

Ethayarajh, Kawin, et al. "Understanding Dataset Difficulty with V-Usable Information." In *Proceedings of the 39th International Conference on Machine Learning*, 2022. https://arxiv.org/abs/2110.08420.

Glaese, Amelia, et al. "Improving Alignment of Dialogue Agents via Targeted Human Judgements." arXiv preprint arXiv:2209.14375, 2022. https://arxiv.org/abs/2209.14375.

OpenAI. "Introducing ChatGPT." November 30, 2022. https://openai.com/index/chatgpt/.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Singhal, Prasann, et al. "A Long Way to Go: Investigating Length Correlations in RLHF." In *Conference on Language Modeling*, 2024. https://arxiv.org/abs/2310.03716.

Stiennon, Nisan, et al. "Learning to Summarize from Human Feedback." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2009.01325.

Touvron, Hugo, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv preprint arXiv:2307.09288, 2023. https://arxiv.org/abs/2307.09288.

Tunstall, Lewis, et al. "Zephyr: Direct Distillation of LM Alignment." arXiv preprint arXiv:2310.16944, 2023. https://arxiv.org/abs/2310.16944.

Wang, Zhilin, et al. "HelpSteer2: Open-Source Dataset for Training Top-Performing Reward Models." arXiv preprint arXiv:2406.08673, 2024. https://arxiv.org/abs/2406.08673.

Ziegler, Daniel M., et al. "Fine-Tuning Language Models from Human Preferences." arXiv preprint arXiv:1909.08593, 2019. https://arxiv.org/abs/1909.08593.
