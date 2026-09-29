# 12.3 Benchmarks and Task Suites

A *benchmark* is a fixed collection of tasks with an agreed way of scoring them. Benchmarks are the common currency of LLM evaluation: they appear in every model release, drive research agendas, and fill leaderboards. They are also frequently misread. This section surveys the benchmark families you will see most often, explains how each is scored, describes how the field's favorite benchmarks have changed as models improved, and ends with a guide to reading a leaderboard critically.

We describe methods and design choices rather than quoting leaderboard numbers. Scores change every few weeks, depend heavily on evaluation settings, and are best looked up at the source when you need them.

## How a benchmark score is produced

Before looking at specific benchmarks, it helps to see how many choices hide behind a single score. To evaluate a model on a benchmark you must decide:

1. **The prompt template.** How is the question presented? Is there a system prompt? Are the answer options labeled "A." or "(A)"?
2. **Few-shot or zero-shot.** Are worked examples included in the prompt? How many, and which ones?
3. **Reasoning.** Is the model allowed or encouraged to think step by step before answering (chain of thought)? For reasoning models, how large is the thinking budget?
4. **Scoring method.** For multiple choice: compare log-likelihoods of the options (Section 2), or let the model generate a letter and parse it? For free-form answers: exact string match, normalized match, symbolic equivalence, unit tests, or a judge model?
5. **Decoding.** Greedy decoding or sampling? At what temperature? How many samples, and how are they combined (first sample, majority vote, best-of-$`n`$)?
6. **Answer extraction.** If the model writes "The answer is probably (C), though (B) is also plausible," what counts as its answer?

Each choice can move a score by several points, and different papers make different choices. When two reports disagree about the same model on the same benchmark, the cause is usually one of these settings rather than a mistake. Evaluation *harnesses* exist to standardize them.

**Evaluation harnesses.** EleutherAI's `lm-evaluation-harness` is an open-source framework that implements hundreds of benchmarks with shared, versioned prompt templates and scoring code; Biderman et al. describe the lessons its developers learned about reproducibility. Stanford's HELM (Holistic Evaluation of Language Models) takes a broader view: it evaluates many models on many scenarios and reports multiple metrics per scenario (accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency) under standardized conditions, making trade-offs visible rather than collapsing everything into one number. Model developers also publish their own evaluation code (OpenAI's `simple-evals` is one example). Whenever you report a number, say which harness and version produced it.

## Knowledge and reasoning

### MMLU and its successors

**MMLU** (Massive Multitask Language Understanding), introduced by Hendrycks et al. in 2020, contains four-option multiple-choice questions across 57 subjects, from elementary mathematics and US history to law, medicine, and computer science. Its breadth made it the default headline number for general knowledge for several years.

MMLU illustrates the full life cycle of a benchmark. At release it was very hard for models. By 2025, the authors of Humanity's Last Exam noted that frontier LLMs were achieving over 90 percent accuracy on it. At that level the benchmark stops discriminating between strong models, and its flaws start to dominate. Gema et al. manually re-annotated a sample of 5,700 MMLU questions across all subjects (MMLU-Redux) and estimated that about 6.5 percent of MMLU questions contain errors, with some subsets far worse. When a meaningful fraction of "correct answers" are themselves wrong, the last few points of accuracy measure agreement with label errors, not ability.

**MMLU-Pro** (Wang et al., 2024) responds to saturation in three ways: it removes trivial and noisy questions, adds harder reasoning-focused questions, and expands the answer options from four to ten, which lowers the accuracy of random guessing from 25 percent to 10 percent. The authors report that scores dropped substantially relative to MMLU and became less sensitive to prompt variations, and that chain-of-thought reasoning helped on MMLU-Pro, suggesting it tests reasoning more than recall.

**GPQA** (Graduate-level Google-Proof Q&A; Rein et al., 2023) goes further in difficulty. Its 448 multiple-choice questions in biology, physics, and chemistry were written by domain experts. The authors report that experts with or pursuing PhDs in the relevant domain reached 65 percent accuracy, while highly skilled non-experts reached only 34 percent despite spending over 30 minutes per question with unrestricted web access. That is what "Google-proof" means. A curated, higher-quality subset called GPQA Diamond is the version most often reported today.

**Humanity's Last Exam** (HLE; Phan et al., 2025) was designed explicitly as a response to saturated benchmarks. It consists of 2,500 questions across dozens of subjects, contributed by subject-matter experts worldwide, in multiple-choice and short-answer formats suitable for automatic grading. Each question has an unambiguous, verifiable answer that cannot be quickly found by searching the internet, and candidate questions were filtered against frontier models before inclusion. The creators also keep a private held-out set to detect overfitting. HLE reports calibration error alongside accuracy, because a model that is confidently wrong on expert questions is a worse assistant than one that knows when it does not know (Section 4).

### ARC and HellaSwag: early reasoning benchmarks

Two older benchmarks from the Allen Institute and collaborators appear in many papers, especially for smaller base models.

**ARC** (the AI2 Reasoning Challenge; Clark et al., 2018) contains grade-school science multiple-choice questions. Its *Challenge* set contains questions that simple retrieval and word co-occurrence baselines answered incorrectly, so solving them requires more than keyword matching. (Do not confuse this ARC with ARC-AGI, discussed below.)

**HellaSwag** (Zellers et al., 2019) tests commonsense inference: given the start of a description of an everyday event, choose the most plausible continuation from four options. The wrong options were generated by a language model and then filtered adversarially, keeping those that fooled models of the time while remaining obviously wrong to humans. HellaSwag was hard for models in 2019 and is essentially saturated for modern LLMs. It remains useful for tracking progress during pretraining of small models, where it still discriminates.

Both are usually scored by comparing option log-likelihoods, and both illustrate a general lesson: adversarial filtering against the models of one era produces a benchmark that is hard *for that era*.

### ARC-AGI: fluid intelligence on novel tasks

François Chollet's **ARC-AGI** (originally the Abstraction and Reasoning Corpus, 2019) takes a different approach. Each task shows a few input-output pairs of small colored grids that follow some hidden rule; the solver must infer the rule and apply it to a new input. The tasks require only "core knowledge" priors such as objects, counting, and symmetry, and each task is novel, so memorized knowledge helps little. The goal is to measure how efficiently a system acquires new skills, not how many skills it already has.

The benchmark has gone through versions as systems improved. ARC-AGI-2 (2025) kept the grid format with harder tasks calibrated against human test-takers. ARC-AGI-3, launched in March 2026, is interactive: agents are dropped into hundreds of turn-based game-like environments with no instructions, and must explore, infer the goal, and plan, with scores based on action efficiency relative to human baselines. Its authors report that humans can solve all of the environments, while frontier AI systems scored below 1 percent as of March 2026.

## Mathematics

### GSM8K and MATH

**GSM8K** (Cobbe et al., 2021) contains 8.5K grade-school math word problems requiring a few steps of arithmetic. Each has a numeric final answer, so scoring is simple: extract the final number from the model's output and compare it with the reference. The GSM8K paper is also historically important for proposing *verifiers*: sample many candidate solutions and let a trained model pick the best one, an idea that runs through process reward models and RLVR (Chapter 14).

**MATH** (Hendrycks et al., 2021) contains 12,500 problems from high school mathematics competitions, spanning algebra, geometry, number theory, counting and probability, and more, with full step-by-step solutions. Final answers can be expressions like $\frac{\sqrt{3}}{2}$ or $x \in (1, 4]$, so scoring requires normalizing and checking *mathematical equivalence*, not string equality: $\frac{1}{2}$, $0.5$, and $\tfrac{2}{4}$ should all match. Evaluation code typically asks the model to put its final answer in a `\boxed{}` expression, extracts it, and compares it to the reference with a combination of string normalization and symbolic algebra (for example, with SymPy). A widely used 500-problem subset, often called MATH-500, is common in reasoning-model reports.

Both benchmarks are now largely saturated at the frontier, and GSM8K is a known contamination target (Section 2 described the GSM1k study).

### Competition math: AIME and beyond

Reasoning models are now commonly evaluated on problems from the American Invitational Mathematics Examination (AIME), a competition with two 15-question exams each year and integer answers from 0 to 999. Using the most recent year's problems reduces contamination, because they were written after most training cutoffs. But a year's AIME has only 30 problems, so each problem is worth more than 3 percentage points, and a single lucky or unlucky sample changes the score noticeably. Reports therefore often average accuracy over many samples per problem (for example, "avg@16" or "pass@1 averaged over 64 samples"). Section 8 explains why small benchmarks need this treatment and wide confidence intervals.

## Code

### HumanEval, MBPP, and the pass@k metric

**HumanEval** (Chen et al., 2021) contains 164 hand-written Python programming problems. Each gives a function signature and docstring; the model writes the body; the result is run against hidden unit tests. **MBPP** (Mostly Basic Python Problems; Austin et al., 2021) is a similar set of about 1,000 crowd-sourced entry-level problems. Both use *functional correctness* rather than text similarity: code that passes the tests is correct, however it is written. This is far better than comparing code with a reference by string overlap, because two correct programs can look completely different.

The standard metric is **pass@k**: the probability that at least one of $k$ sampled solutions passes all the tests. The naive estimate (sample exactly $k$ solutions per problem and check whether any pass) has high variance. Chen et al. proposed an unbiased estimator that uses $n \geq k$ samples per problem. If $c$ of the $n$ samples are correct, then

```math
\text{pass@}k = \mathbb{E}_{\text{problems}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right].
```

The intuition: $\binom{n-c}{k} / \binom{n}{k}$ is the probability that a random subset of $k$ of the $n$ samples contains *no* correct solution. One minus that is the probability that at least one of the $k$ is correct. Averaging over all subsets uses all $n$ samples and lowers variance.

For example, with $n = 10$ samples of which $c = 3$ are correct, pass@1 is $1 - \binom{7}{1}/\binom{10}{1} = 1 - 7/10 = 0.3$, just the fraction correct. Pass@5 is $1 - \binom{7}{5}/\binom{10}{5} = 1 - 21/252 \approx 0.917$.

The binomial coefficients get huge quickly, so it is better to compute the ratio as a product:

```python
import numpy as np

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem: n samples, c correct."""
    if n - c < k:
        return 1.0        # every k-subset must contain a correct sample
    # C(n-c, k) / C(n, k) = prod_{i=n-c+1}^{n} (1 - k / i)
    ratio = np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    return 1.0 - ratio

results = [(10, 3), (10, 0), (10, 10), (10, 1)]      # (n, c) per problem
print(np.mean([pass_at_k(n, c, k=1) for n, c in results]))   # 0.35
print(np.mean([pass_at_k(n, c, k=5) for n, c in results]))
```

Pass@1 measures the chance of getting it right on the first try, which is what most users experience. Pass@k for larger $k$ measures what is reachable with repeated sampling and a way to pick a working answer (such as running tests), which matters for RL (a problem with pass@k of zero gives no learning signal) and for agent systems that can retry. The HumanEval paper itself showed how much repeated sampling helps: the original Codex model solved 28.8 percent of problems with one sample but 70.2 percent with 100 samples.

**Test quality matters.** A handful of unit tests may accept incorrect code that happens to pass them. EvalPlus (Liu et al., 2023) augmented HumanEval and MBPP with many more automatically generated tests and found that the original tests missed a substantial number of wrong solutions, lowering measured pass rates. The general lesson: a functional-correctness benchmark is only as good as its tests.

HumanEval and MBPP are small, self-contained, and saturated at the frontier. **LiveCodeBench** addresses contamination by continuously collecting new problems from competitive programming contests and letting you evaluate only on problems released after a model's training cutoff.

### Repository-level coding: the SWE-bench story

Real software engineering means editing large codebases, not writing isolated functions. **SWE-bench** (Jimenez et al., 2023) turned real GitHub issues from 12 popular Python repositories into tasks. The model (usually an agent that can browse files and run commands) receives the repository and the issue text and must produce a patch. The patch is graded by running tests from the pull request that originally fixed the issue: "fail-to-pass" tests that must now pass, and "pass-to-pass" tests that check nothing else broke.

The history of SWE-bench is a compact lesson in benchmark maintenance:

1. **Original SWE-bench (2023).** Many tasks turned out to be underspecified or to have tests that rejected valid solutions, so the benchmark *underestimated* capability.
2. **SWE-bench Verified (August 2024).** OpenAI, working with the SWE-bench authors, had professional developers screen 1,699 tasks and released a subset of 500 that annotators judged well specified with fair tests. It became the standard coding-agent number in frontier model releases for the next year and a half.
3. **Retirement (February 2026).** OpenAI announced that it had stopped reporting SWE-bench Verified. It audited 138 problems that its o3 model did not consistently solve and found that 59.4 percent had material issues in test design or problem description. It also found that all frontier models it tested could reproduce the gold patch or verbatim problem details for some tasks, indicating contamination. Its conclusion: gains on the benchmark increasingly reflected exposure during training rather than real improvements in software engineering. It recommended SWE-bench Pro (Deng et al., 2025), which uses harder, longer-horizon tasks and keeps part of its data private, while noting that Pro is not perfect either.

The arc from "too hard because of bad tests" to "too easy because of contamination" took about two and a half years. Expect similar arcs for today's benchmarks.

## Chat and preference evaluation

Knowledge, math, and code benchmarks have checkable answers. Most of what people ask a chat assistant does not: "help me write a toast for my sister's wedding" has no reference answer. Evaluating open-ended responses requires judgments of preference, from humans or from models.

### MT-Bench

**MT-Bench** (Zheng et al., 2023) contains 80 challenging multi-turn questions across eight categories, including writing, roleplay, reasoning, math, coding, and knowledge. Each model's answers are graded by a strong LLM (originally GPT-4) on a 1 to 10 scale, or compared pairwise against another model's answers. The same paper studied *LLM-as-a-judge* systematically, reporting that GPT-4 judgments agreed with human preferences over 80 percent of the time, about the same level at which humans agree with each other, while also documenting position, verbosity, and self-enhancement biases. Section 6 is devoted to that topic.

### Win rates and AlpacaEval

A simpler preference metric is the **win rate** against a fixed baseline: for each prompt, show a judge the candidate's response and the baseline's response and ask which is better. The win rate is the fraction of prompts the candidate wins (ties often count as half).

**AlpacaEval** computes win rates against a reference model with an LLM judge on a fixed set of instructions. Because LLM judges prefer longer answers, the original version could be gamed by verbosity. **Length-controlled AlpacaEval** (Dubois et al., 2024) fixes this with a regression: it fits a generalized linear model predicting the judge's preference from the length difference and other features, and then reports the predicted win rate *as if the two responses had the same length*. The authors report that length control raised the benchmark's Spearman correlation with Chatbot Arena rankings from 0.94 to 0.98. **Arena-Hard** (Li et al., 2024) similarly uses an LLM judge, but on difficult prompts automatically selected from real Arena conversations.

### Arena-style ratings: Elo and Bradley-Terry

**Chatbot Arena** (Chiang et al., 2024), launched by LMSYS in 2023, collects preferences at scale from real users. A user types any prompt, two anonymous models answer side by side, and the user votes for the better one (or a tie) before the models' names are revealed. The platform moved to its own site as LMArena in 2024, incorporated as a company in 2025, and rebranded as Arena in January 2026. Its leaderboard is one of the most cited in the field.

How do you turn millions of pairwise votes between many models into a ranking? The classic answer is the **Elo** rating system from chess. Each model $i$ has a rating $R_i$. The expected probability that $A$ beats $B$ is

```math
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}.
```

A 400-point gap corresponds to 10:1 odds. After a game with outcome $S_A$ (1 for a win, 0 for a loss, 0.5 for a tie), the ratings update online:

```math
R_A \leftarrow R_A + K\,(S_A - E_A), \qquad R_B \leftarrow R_B - K\,(S_A - E_A),
```

where $K$ controls the step size. A win against a stronger opponent (low $E_A$) earns many points; a win against a weaker one earns few.

Online Elo was designed for players whose skill changes over time, and its results depend on the order in which games are processed. A model's quality does not change between votes, so a better approach fits all votes at once with the **Bradley-Terry** model, which Chapter 11 used for reward models. Each model has a strength $\beta_i$, and

```math
P(i \text{ beats } j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}} = \sigma(\beta_i - \beta_j),
```

where $\sigma$ is the logistic sigmoid. Fitting the $\beta$'s by maximum likelihood is just logistic regression: each vote is a training example whose features are $+1$ for model $i$, $-1$ for model $j$, and 0 elsewhere. The Chatbot Arena paper describes this approach, reports scores on an Elo-like scale for familiarity, and computes confidence intervals by bootstrap resampling of the votes. The Arena has also added "style control," which adds features such as response length and markdown formatting to the regression so that rankings reflect content more than presentation, the same idea as length-controlled AlpacaEval.

A small Bradley-Terry fit takes a few lines:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

models = ["A", "B", "C"]
# Each vote: (winner, loser)
votes = [("A", "B")] * 60 + [("B", "A")] * 40 + [("B", "C")] * 55 + \
        [("C", "B")] * 45 + [("A", "C")] * 70 + [("C", "A")] * 30

idx = {m: i for i, m in enumerate(models)}
X, y = [], []
for w, l in votes:
    row = np.zeros(len(models)); row[idx[w]], row[idx[l]] = 1, -1
    X.append(row); y.append(1)
    X.append(-row); y.append(0)          # mirrored copy keeps the problem symmetric
clf = LogisticRegression(fit_intercept=False, C=1e6).fit(np.array(X), np.array(y))
beta = clf.coef_[0]
elo_like = 1000 + 400 / np.log(10) * (beta - beta.mean())   # rescale to an Elo-style scale
print(dict(zip(models, elo_like.round())))
```

The rescaling factor $400/\ln 10$ converts natural-log odds to Elo's base-10, 400-point convention.

**Strengths and weaknesses.** Arena ratings reflect real prompts from many users and are hard to game by training on a fixed test set. But the prompt distribution is whatever visitors happen to type, voters are not experts and rarely verify facts, and presentation influences votes. Singh et al.'s "The Leaderboard Illusion" identified further structural problems: some providers privately tested many variants before release and published only the best scores (the authors identified 27 private variants tested by Meta before the Llama 4 release), and proprietary models received more battles, and hence more data, than open ones. Arena responded with policy changes, but the lesson generalizes: any leaderboard where participants control what gets submitted is exposed to selection effects.

## Beyond single-turn tasks

As of 2026, many headline evaluations test *agents*: models that use tools, browse the web, run code, and take multi-step actions. Examples include SWE-bench-style coding agents, web-browsing and computer-use benchmarks, and interactive environments like ARC-AGI-3. Agent evaluations add new sources of variance: the scaffold (the software around the model), the tools available, time and cost budgets, and the stochasticity of long interactions. OpenAI noted, for instance, that GPT-4's score on SWE-bench Lite varied from 2.7 percent to 28.3 percent depending on the scaffold. When comparing agent results, compare scaffolds and budgets, not just models.

## Reading a leaderboard critically

Leaderboards compress many choices into one column of numbers. Before drawing a conclusion from one, ask:

| Question | Why it matters |
|---|---|
| Which exact benchmark version and subset? | "MATH" vs. "MATH-500," "SWE-bench" vs. "Verified" vs. "Pro" are different tests |
| What prompting, shots, reasoning budget, and decoding? | Settings can move scores by several points |
| Who ran the evaluation? | Self-reported numbers are chosen by the party that benefits |
| How big is the benchmark, and what are the error bars? | On 30 AIME problems, a 3-point gap is one problem |
| Is the benchmark saturated or contaminated? | Near the ceiling, noise and label errors dominate |
| Was the model or prompt tuned on this benchmark? | Adversarial Goodhart (Section 1) |
| Does the benchmark resemble your use case? | A high GPQA score says little about customer-support quality |
| Is the scaffold or tool setup the same across entries? | Agent results depend heavily on scaffolding |

A good habit is to look at the *pattern* of results across several independent benchmarks rather than any single number, and to confirm important decisions with an evaluation built from your own tasks (Section 8).

## Key takeaways

- A benchmark score depends on many hidden choices (prompt, shots, reasoning, scoring, decoding); harnesses such as lm-evaluation-harness and HELM standardize them.
- Benchmarks follow a life cycle: hard at release, then saturated, error-limited, and contaminated. MMLU gave way to MMLU-Pro, GPQA, and Humanity's Last Exam; SWE-bench Verified was retired in favor of SWE-bench Pro.
- Code is scored by functional correctness with the unbiased pass@k estimator; test quality limits what a code benchmark can tell you.
- Open-ended chat quality is measured by preferences: LLM-judged benchmarks (MT-Bench, length-controlled AlpacaEval, Arena-Hard) and human-voted arenas ranked with Bradley-Terry models.
- Read leaderboards critically: check settings, sample size, contamination, who ran the evaluation, and whether the benchmark matches your use case.

## Further reading

Chen, Mark, et al. "Evaluating Large Language Models Trained on Code." arXiv preprint arXiv:2107.03374, 2021. https://arxiv.org/abs/2107.03374.

Chiang, Wei-Lin, et al. "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference." In *Proceedings of the 41st International Conference on Machine Learning*, 2024. https://arxiv.org/abs/2403.04132.

Dubois, Yann, et al. "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." arXiv preprint arXiv:2404.04475, 2024. https://arxiv.org/abs/2404.04475.

Hendrycks, Dan, et al. "Measuring Massive Multitask Language Understanding." In *International Conference on Learning Representations*, 2021. https://arxiv.org/abs/2009.03300.

Jimenez, Carlos E., et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" In *International Conference on Learning Representations*, 2024. https://arxiv.org/abs/2310.06770.

OpenAI. "Why SWE-bench Verified No Longer Measures Frontier Coding Capabilities." February 23, 2026. https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/.

Phan, Long, et al. "Humanity's Last Exam." arXiv preprint arXiv:2501.14249, 2025. https://arxiv.org/abs/2501.14249.

Wang, Yubo, et al. "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark." arXiv preprint arXiv:2406.01574, 2024. https://arxiv.org/abs/2406.01574.
