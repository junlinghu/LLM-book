# 12.8 Reporting Results Honestly

Every section of this chapter has ended with a warning about some way evaluation results can mislead: contamination, saturated benchmarks, prompt sensitivity, judge biases, annotator disagreement, selective disclosure. This final section is about the discipline that ties those warnings together. It covers how to quantify the uncertainty in an evaluation result, how to avoid fooling yourself and your readers with selected examples and selected numbers, and a checklist for evaluating a new model or fine-tune.

The goal is simple to state: **a reported result should let a reader predict what they would see if they ran the evaluation themselves.** That requires reporting not just a number but how uncertain it is, how it was produced, and what was tried along the way.

## Every score is an estimate

When we report that a model scores 70 percent on a benchmark, we are really estimating something we cannot observe directly: the model's accuracy on the whole population of questions the benchmark is meant to represent. The benchmark's questions are one sample from that population. A different sample of equally valid questions would give a somewhat different score. Miller makes this framing explicit for LLM evaluations: treat the evaluation questions as drawn from an unseen "super-population," and analyze the results with the same statistical tools that other experimental sciences use.

### Confidence intervals for accuracy

For a benchmark with $n$ independent questions scored right or wrong, the observed accuracy $\hat{p}$ is a sample mean of Bernoulli variables. Its standard error is

```math
\text{SE}(\hat{p}) = \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}},
```

and an approximate 95 percent confidence interval is $`\hat{p} \pm 1.96\,\text{SE}`$.

**Worked example.** A model answers 140 of 200 questions correctly, so $\hat{p} = 0.70$. The standard error is $\sqrt{0.7 \times 0.3 / 200} \approx 0.032$, and the 95 percent interval is $0.70 \pm 0.064$, or roughly 64 to 76 percent. A second model scoring 73 percent on the same 200 questions is well inside that range.

The size of these intervals surprises many people. Some illustrative cases, all computed with the formula above at the stated accuracy:

| Benchmark size $n$ | Accuracy | 95% CI half-width | Example |
|---|---|---|---|
| 30 | 50% | ±17.9 points | One year of AIME problems |
| 200 | 70% | ±6.4 points | A small custom evaluation |
| 500 | 70% | ±4.0 points | A typical curated subset |
| 2,500 | 50% | ±2.0 points | A benchmark the size of Humanity's Last Exam |
| 10,000 | 70% | ±0.9 points | A large benchmark |

For small $n$ or accuracies near 0 or 1, the normal approximation is poor; the *Wilson score interval* or a bootstrap interval (below) behaves better.

### Clustered questions

The formula above assumes questions are independent. Many benchmarks violate this. Reading-comprehension benchmarks ask several questions about the same passage; some coding benchmarks have several tasks from the same repository; multilingual benchmarks translate the same question into many languages. Questions in a cluster tend to be answered correctly or incorrectly together, so the effective sample size is smaller than $n$. Miller recommends *clustered standard errors* in this case, which can be substantially larger than the naive ones. As a simple rule, if you have 1,000 questions built from 100 passages, your uncertainty is closer to that of a benchmark with somewhere between 100 and 1,000 independent questions, not 1,000.

### Comparing two models: use paired differences

The question we usually care about is not "What is model A's accuracy?" but "Is model B better than model A?" If both models were evaluated on the *same* questions, the right analysis is **paired**: compute the difference on each question and analyze those differences.

Pairing matters because most of the variation in benchmark scores comes from question difficulty, which affects both models alike. Hard questions are hard for both; easy ones are easy for both. Pairing cancels that shared variation.

**Worked example.** Models A and B are evaluated on the same 500 questions. A gets 350 right (70.0 percent) and B gets 366 right (73.2 percent), a difference of 3.2 points. Is B better?

- *Unpaired analysis (wrong here).* Treating the two scores as independent, the standard error of the difference is $\sqrt{0.700 \times 0.300/500 + 0.732 \times 0.268/500} \approx 0.0285$, giving a 95 percent interval of $0.032 \pm 0.056$, or $[-0.024, 0.088]$. That includes zero, so we cannot conclude that B is better.
- *Paired analysis.* Suppose the models agree on 440 questions and disagree on 60: B is right and A wrong on 38, and A is right and B wrong on 22. Define $d_i = \mathbf{1}[B \text{ correct}] - \mathbf{1}[A \text{ correct}] \in \lbrace -1, 0, 1 \rbrace$. The mean difference is $\bar{d} = 16/500 = 0.032$. The variance of $d_i$ is $\overline{d^2} - \bar{d}^2 = 60/500 - 0.032^2 \approx 0.119$, so $\text{SE}(\bar{d}) = \sqrt{0.119/500} \approx 0.0154$ and the 95 percent interval is $0.032 \pm 0.030$, or about $[0.002, 0.062]$.

The same data give an inconclusive answer when analyzed as if unpaired and a (just barely) significant one when analyzed correctly. Pairing roughly halved the standard error here. The practical lesson: **always evaluate models you want to compare on the same questions, save per-question results, and compare them per question.**

### The bootstrap

When formulas are awkward, for example for win rates with ties, pass@k, F1 scores, or medians, the **bootstrap** gives confidence intervals by resampling. Draw $n$ questions with replacement from the $n$ you have, recompute the metric, repeat thousands of times, and take the 2.5th and 97.5th percentiles of the resulting distribution. For comparing two models, resample *questions* and recompute the difference on each resample, which preserves the pairing.

```python
import numpy as np

def paired_bootstrap_ci(scores_a, scores_b, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """95% CI for mean(scores_b) - mean(scores_a), resampling questions (keeps pairing)."""
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return b.mean() - a.mean(), (lo, hi)

# Per-question correctness for the worked example above
a = np.array([1] * 328 + [0] * 112 + [1] * 22 + [0] * 38)   # agree-correct, agree-wrong, A-only, B-only
b = np.array([1] * 328 + [0] * 112 + [0] * 22 + [1] * 38)
print(paired_bootstrap_ci(a, b))   # difference 0.032, CI roughly (0.002, 0.062)
```

The Chatbot Arena leaderboard (Section 3) uses exactly this idea, bootstrapping over votes, to put confidence intervals on its ratings.

### How many questions do you need?

Turning the confidence-interval formula around tells you how big an evaluation must be to detect a given difference. For an unpaired comparison of two accuracies near $p$, with significance level 0.05 (two-sided) and 80 percent power, a standard approximation for the number of questions per model is

```math
n \approx \frac{(z_{0.975} + z_{0.8})^2 \cdot 2p(1-p)}{\Delta^2} = \frac{(1.96 + 0.84)^2 \cdot 2p(1-p)}{\Delta^2},
```

where $\Delta$ is the difference you want to detect. At $p = 0.5$, detecting a 2-point difference ($\Delta = 0.02$) needs about 9,800 questions; detecting a 5-point difference needs about 1,600. Pairing reduces these numbers, often substantially, but the lesson stands: small evaluations can detect only large differences. Card et al. showed that many NLP experiments are *underpowered*, meaning they are too small to reliably detect the effects they claim, and that underpowered studies that do report significant results tend to exaggerate effect sizes.

## Multiple seeds and other sources of variance

Question sampling is only one source of randomness. Others include:

**Sampling randomness at inference.** If the model samples with temperature above zero, rerunning the same evaluation gives a different score. For small benchmarks like AIME, reports often average over many samples per question (for example, 16 or 64) to reduce this variance. Note that averaging samples reduces *sampling* noise but does nothing about *question* noise: 30 AIME problems are still only 30 problems, however many times you sample each.

**Training randomness.** Two fine-tuning runs with identical data and hyperparameters but different random seeds (for initialization of new parameters, data order, dropout) can produce models whose benchmark scores differ by more than the effect you are trying to measure. Bouthillier et al. analyzed the sources of variance in machine-learning benchmarks and found that data sampling, parameter initialization, and hyperparameter choices all contribute, and that ignoring them leads to unreliable conclusions. Madaan et al. quantified variance in LLM evaluation benchmarks specifically, including the spread across pretraining runs that differed only in their random seed, and urged practitioners to factor this variance in before concluding that one training choice beats another. When you claim that a training *method* is better, rather than that a specific *checkpoint* is better, you should train with several seeds (three to five is common when affordable) and report the mean and spread.

**Prompt and format sensitivity.** Sclar et al. showed that meaning-preserving formatting changes can produce large accuracy differences in few-shot settings, and that the ranking of models can change with the format. A robust report either uses a standardized harness (Section 3) or reports results across several reasonable prompt formats.

**Hyperparameter search.** Dodge et al. argued that reporting only the best result after tuning hides the cost of finding it, and proposed reporting *expected validation performance* as a function of the number of hyperparameter trials. The general principle: report how much search went into a result, because more search buys higher numbers through selection alone (Section 1).

**Judge randomness.** If an LLM judge scores the outputs, the judge's sampling and its version are additional sources of variance. Fix the judge version, use temperature zero where possible, and report the judge.

A reasonable minimum for a paper or internal report comparing methods: several training seeds where feasible, a fixed and reported evaluation configuration, per-question results for paired comparisons, and confidence intervals on every headline number.

### Multiple comparisons

If you compare a new model with a baseline on 20 benchmarks and test each difference at the 0.05 level, you expect about one "significant" result by chance even if the models are identical. Report all the benchmarks you ran, not just the ones where you won. When the claim is "better on benchmark X," a correction such as Bonferroni (divide the significance level by the number of comparisons) or a pre-registered primary metric keeps the error rate honest.

## Avoiding cherry-picked examples

Qualitative examples, a few sample outputs shown in a paper or a launch post, are persuasive and useful for building intuition. They are also easy to abuse. A model that produces a brilliant answer 1 time in 20 can be made to look brilliant by showing that one answer.

Good practice for examples:

- **Say how examples were selected.** "The first five prompts from our test set," "randomly sampled," or "chosen to illustrate a failure mode" are all honest; unlabeled selection is not.
- **Show random samples**, not just the best ones, and show them for both the new model and the baseline on the same prompts.
- **Show failures.** A report that shows only successes invites readers to assume there are no failures. Showing representative failures builds trust and helps others.
- **Don't regenerate until it looks good.** If you sampled several responses and picked one, say so, and ideally report how many samples it took.
- **Use examples to illustrate, not to prove.** A claim of improvement needs the quantitative evaluation behind it.

The same principles apply to numbers. Reporting the best of several runs, the best of several prompts, or only the benchmarks where the new model wins are all forms of cherry-picking. The Leaderboard Illusion study (Section 3) documented how testing many private variants and disclosing selectively distorts a public leaderboard; the same effect distorts papers and launch posts.

## What to report

To make an evaluation reproducible and interpretable, report:

1. **The model(s)**: exact versions or checkpoints, including the date for API models, whose behavior can change over time.
2. **The evaluation data**: benchmark names and versions, the exact subset, the number of items, and any filtering. For custom sets, how items were collected.
3. **The configuration**: harness and version, prompt templates, number of shots, chain-of-thought or reasoning budget, decoding parameters (temperature, top-p, max tokens), number of samples, and answer extraction.
4. **The scoring**: exact-match rules, the judge model and prompt, human annotation protocol and agreement.
5. **Uncertainty**: confidence intervals, number of seeds, and how intervals were computed.
6. **Contamination checks**: what was done to check that test data did not appear in training data, and the results (Section 2).
7. **What was tried**: how many configurations, prompts, or checkpoints were evaluated before the reported one was chosen, and on which data the choice was made.
8. **Limitations**: what the evaluation does not cover.

Mitchell et al.'s *model cards* proposal is a useful template for the model-facing part of this information: intended use, evaluation data, metrics, results disaggregated across relevant groups and conditions, and caveats. Modern system cards for frontier models extend the idea to safety evaluations (Section 7).

## An evaluation checklist for a new model or fine-tune

Pulling the whole chapter together, here is a checklist you can adapt when evaluating a new model or fine-tune.

**Before you start**

- [ ] Write down the decision the evaluation must support (ship or not; method A vs. B) and the primary metric, before seeing results.
- [ ] Define the task distribution you care about, and build or choose test sets that represent it, including a custom set from real or realistic usage.
- [ ] Split data into a development set (for tuning prompts, checkpoints, and hyperparameters) and a test set (touched rarely).
- [ ] Check test data for contamination against your training data with n-gram overlap; prefer fresh or private items where possible.
- [ ] Estimate how many items you need to detect the difference you care about.

**Capability and quality**

- [ ] Intrinsic metrics (perplexity or bits per byte on a named corpus) for base-model changes, with the same tokenizer or byte-normalized.
- [ ] Standard benchmarks relevant to your use case, run through a standard harness with fixed, reported settings; avoid saturated benchmarks as primary evidence.
- [ ] Task-specific evaluation on your custom set, with automatic checks where answers are verifiable.
- [ ] Open-ended quality via pairwise comparisons against the previous model, using an LLM judge with order swapping and length control, validated against human labels on a sample.
- [ ] A human evaluation for key decisions, with a rubric, blind randomized presentation, and measured agreement.

**Hallucination and calibration**

- [ ] A factuality set graded correct / incorrect / abstained; report accuracy, hallucination rate, and abstention together.
- [ ] Faithfulness checks if the system uses retrieval or summarizes documents.
- [ ] Calibration (ECE or a reliability diagram) if the system exposes or acts on confidence.

**Safety**

- [ ] Harmful-request tests and adversarial variants, judged by a validated classifier; report attack success rate per category.
- [ ] Over-refusal tests on benign borderline prompts; report the false refusal rate alongside.
- [ ] Informal red teaming, with every failure turned into a test case.
- [ ] A regression check that fine-tuning did not erode the base model's safety behavior.

**Analysis and reporting**

- [ ] Paired comparisons on the same items, with confidence intervals (analytic or bootstrap).
- [ ] Multiple training seeds for method comparisons where affordable; mean and spread reported.
- [ ] Sensitivity to prompt format checked on at least the primary metric.
- [ ] All benchmarks run are reported, not just the favorable ones.
- [ ] Random samples of outputs (including failures) read by a person, and representative examples shared with their selection method stated.
- [ ] Full configuration, contamination checks, search effort, and limitations documented.

No checklist makes an evaluation perfect. But working through one forces the questions that most often go unasked, and it produces results that others can trust, reproduce, and build on.

## Key takeaways

- Every benchmark score is an estimate with uncertainty; report confidence intervals, and remember that small benchmarks (like a year of AIME problems) have very wide ones.
- Compare models on the same items with paired analyses or a paired bootstrap; pairing can substantially shrink uncertainty, and clustered questions enlarge it.
- Training seeds, sampling, prompt format, hyperparameter search, and judge randomness all add variance; report several seeds for method claims and state how much search was done.
- Avoid cherry-picking in both examples and numbers: state how examples were chosen, show random samples and failures, and report every benchmark you ran.
- Report enough detail (model versions, data, configuration, scoring, uncertainty, contamination checks, limitations) that a reader could reproduce the result, and use a checklist to cover capability, hallucination, safety, and analysis.

## Further reading

Biderman, Stella, et al. "Lessons from the Trenches on Reproducible Evaluation of Language Models." arXiv preprint arXiv:2405.14782, 2024. https://arxiv.org/abs/2405.14782.

Bouthillier, Xavier, et al. "Accounting for Variance in Machine Learning Benchmarks." arXiv preprint arXiv:2103.03098, 2021. https://arxiv.org/abs/2103.03098.

Card, Dallas, et al. "With Little Power Comes Great Responsibility." arXiv preprint arXiv:2010.06595, 2020. https://arxiv.org/abs/2010.06595.

Dodge, Jesse, et al. "Show Your Work: Improved Reporting of Experimental Results." arXiv preprint arXiv:1909.03004, 2019. https://arxiv.org/abs/1909.03004.

Madaan, Lovish, et al. "Quantifying Variance in Evaluation Benchmarks." arXiv preprint arXiv:2406.10229, 2024. https://arxiv.org/abs/2406.10229.

Miller, Evan. "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations." arXiv preprint arXiv:2411.00640, 2024. https://arxiv.org/abs/2411.00640.

Mitchell, Margaret, et al. "Model Cards for Model Reporting." In *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 2019. https://arxiv.org/abs/1810.03993.
