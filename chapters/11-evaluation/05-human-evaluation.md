# 11.5 Human Evaluation

For open-ended tasks, such as writing, advice, explanation, and conversation, there is often no reference answer and no automatic check. The final judge of whether a response is good is a person. Human evaluation is therefore often described as the "gold standard" for LLM evaluation, and human preferences are also the raw material of RLHF (Chapter 10).

But human evaluation is not automatically gold. People disagree, get tired, have biases, and can be fooled by fluent text. The quality of a human evaluation depends almost entirely on how it is designed. This section covers the two main formats (rubric scoring and pairwise comparison), how to measure whether annotators agree, and the practical issues of cost, bias, and expertise.

## What human evaluation is for

It helps to be clear about what question a human evaluation answers. Common goals include:

- **Comparing two systems.** Is the new fine-tune better than the old one? This is the most common use, and pairwise comparisons suit it well.
- **Measuring absolute quality.** Is the model good enough to ship? What fraction of responses are acceptable? Rubric scoring suits this.
- **Diagnosing failures.** What kinds of mistakes does the model make? This calls for error annotation with a taxonomy of error types rather than a single score.
- **Validating automatic metrics.** Does an LLM judge (Section 6) or an automatic metric agree with people? Here the human labels are the reference against which the metric is measured.

Each goal leads to a different design. A single study that tries to serve all of them usually serves none well.

## Rubrics and absolute scoring

In **absolute** (or *pointwise*) evaluation, an annotator sees one response at a time and rates it against a set of criteria. The criteria and scale are written down in a **rubric**.

A good rubric has three properties:

1. **Separate dimensions.** Rather than one overall "quality" score, rate distinct aspects that can vary independently: correctness, helpfulness, completeness, harmlessness, clarity, adherence to instructions. A response can be correct but unhelpful, or helpful but unsafe.
2. **Anchored scale points.** Each score is defined by a concrete description, ideally with examples. "4 = good" is vague; "4 = fully correct and addresses every part of the request, with at most minor stylistic issues" is usable.
3. **Decision rules for hard cases.** What if the response is correct but refuses part of the request? What if the prompt was ambiguous? Writing these rules down is what makes different annotators agree.

Here is an example rubric for factual correctness on a 1 to 5 scale:

| Score | Description |
|---|---|
| 5 | All factual claims are correct and verifiable; no misleading omissions |
| 4 | All central claims correct; one minor inaccuracy that does not change the answer |
| 3 | Mostly correct, but at least one inaccuracy a user might act on |
| 2 | A central claim is wrong, or several secondary claims are wrong |
| 1 | The main answer is wrong or fabricated |

The number of scale points is a trade-off. Binary scales (acceptable / not acceptable) are fast and give high agreement but lose information. Five- or seven-point *Likert* scales capture more nuance but invite each annotator to use the scale differently: one annotator's 4 is another's 3. Many practical evaluations use a binary or three-level scale per dimension, with free-text comments for context.

Absolute scores also drift. Annotators calibrate against the responses they have recently seen, so the same response can be rated higher after a run of bad ones. Karpinska et al. found that crowd workers' judgments of story quality improved when they were shown model outputs alongside human-written references, which gave them an anchor for calibration.

## Pairwise comparisons

In **pairwise** evaluation, an annotator sees one prompt and two responses, A and B, and says which is better (often with options for "tie" or "both bad"). This is the format used for RLHF preference data (Chapter 10) and for Chatbot Arena (Section 3).

Pairwise judgments have practical advantages:

- **They are easier.** People are much better at saying which of two things is better than at placing one thing on an absolute scale.
- **They are more consistent.** Relative judgments cancel out individual differences in how annotators use a scale.
- **They are sensitive to small differences.** Two responses that would both get a "4" can still be compared.

They also have limits. A pairwise preference tells you which response is *better*, not whether either is *good*; two terrible responses still produce a winner. It also collapses all dimensions into one decision, so it does not tell you *why* one response was preferred. Hosking et al. found that single preference scores under-represent important aspects like factuality, and that more assertive responses were perceived as having fewer factual errors, meaning a confident tone can win preferences it does not deserve. A common compromise is to ask for a pairwise preference *and* a short rubric rating or error flags on each response.

### From pairwise votes to scores

Pairwise results can be summarized as a **win rate**: the fraction of comparisons the new system wins, with ties counted as half.

```math
\text{win rate} = \frac{W + \tfrac{1}{2} T}{W + L + T},
```

where $W$, $L$, and $T$ are the numbers of wins, losses, and ties. A win rate of 50 percent means the systems are indistinguishable on this evaluation. With many systems, the **Bradley-Terry model** from Section 3 turns all the pairwise results into a single strength score per system, $P(i \succ j) = \sigma(\beta_i - \beta_j)$. This is the same model used for reward modeling in Chapter 10: the reward model is essentially a Bradley-Terry model whose strengths are predicted from the text of each response.

Pairwise evaluations must randomize which response appears on the left and which on the right, because people (like LLM judges, Section 6) have position biases. They should also hide which system produced which response, so that annotators are not influenced by brand or expectations.

## Inter-annotator agreement

If two careful annotators, given the same rubric, give very different labels to the same responses, then either the task is ambiguous, the rubric is unclear, or the annotators are not reliable. Before trusting any human evaluation, measure agreement.

### Raw percent agreement and why it misleads

The simplest measure is **percent agreement**: the fraction of items on which two annotators give the same label. It is easy to understand but misleading, because some agreement happens by chance. If both annotators label 90 percent of responses as "acceptable" regardless of content, they will agree on at least 82 percent of items (both say acceptable with probability $0.9 \times 0.9 = 0.81$, both say unacceptable with probability $0.1 \times 0.1 = 0.01$) without reading anything.

### Cohen's kappa

**Cohen's kappa** corrects for chance agreement. For two annotators labeling the same items with categorical labels,

```math
\kappa = \frac{p_o - p_e}{1 - p_e},
```

where $p_o$ is the observed agreement and $p_e$ is the agreement expected by chance if each annotator labeled independently according to their own label frequencies:

```math
p_e = \sum_{c} p_{1,c}\, p_{2,c},
```

with $p_{a,c}$ the fraction of items annotator $a$ assigned to category $c$. Kappa is 1 for perfect agreement, 0 for agreement no better than chance, and negative for systematic disagreement.

**A worked example.** Two annotators label 100 responses as "acceptable" or "unacceptable":

| | Annotator 2: acceptable | Annotator 2: unacceptable | Total |
|---|---|---|---|
| **Annotator 1: acceptable** | 70 | 10 | 80 |
| **Annotator 1: unacceptable** | 5 | 15 | 20 |
| **Total** | 75 | 25 | 100 |

Observed agreement is $p_o = (70 + 15) / 100 = 0.85$. Annotator 1 says "acceptable" 80 percent of the time and annotator 2 says it 75 percent of the time, so chance agreement is $p_e = 0.80 \times 0.75 + 0.20 \times 0.25 = 0.60 + 0.05 = 0.65$. Therefore

```math
\kappa = \frac{0.85 - 0.65}{1 - 0.65} = \frac{0.20}{0.35} \approx 0.57.
```

An 85 percent agreement rate sounds high, but after correcting for the fact that most responses are acceptable, agreement is only moderate.

```python
import numpy as np

def cohens_kappa(labels_a, labels_b) -> float:
    labels_a, labels_b = np.asarray(labels_a), np.asarray(labels_b)
    cats = np.union1d(labels_a, labels_b)
    p_o = np.mean(labels_a == labels_b)
    p_e = sum(np.mean(labels_a == c) * np.mean(labels_b == c) for c in cats)
    return (p_o - p_e) / (1 - p_e)

a = ["ok"] * 70 + ["ok"] * 10 + ["bad"] * 5 + ["bad"] * 15
b = ["ok"] * 70 + ["bad"] * 10 + ["ok"] * 5 + ["bad"] * 15
print(round(cohens_kappa(a, b), 2))   # 0.57
```

How high should kappa be? A widely cited convention from Landis and Koch labels values of 0.41 to 0.60 "moderate," 0.61 to 0.80 "substantial," and above 0.80 "almost perfect." These labels are rough guides, not laws; what counts as acceptable depends on the task. Subjective judgments like "which response is more helpful" often have only moderate agreement even among careful annotators, and that is itself informative: it tells you how much any single label can be trusted.

### Beyond two annotators and binary labels

- **Weighted kappa** is used for ordinal scales like 1 to 5, where disagreeing by one point should count less than disagreeing by four. Disagreements are weighted by their distance (linearly or quadratically).
- **Fleiss' kappa** generalizes the idea to more than two annotators who each label every item.
- **Krippendorff's alpha** handles any number of annotators, missing labels (not every annotator labels every item), and different kinds of scales. It is a good default for real annotation projects, where coverage is rarely complete.

### Using agreement in practice

Agreement is a diagnostic tool, not just a number to report.

1. **Pilot first.** Have several annotators label the same 50 to 100 items. Compute agreement.
2. **Inspect disagreements.** Look at the items where annotators disagreed. Usually a pattern emerges: an ambiguous rubric term, an unanticipated case, or a prompt that is genuinely unclear.
3. **Revise the guidelines** and repeat until agreement stabilizes.
4. **Keep overlap in production.** Have a fraction of items (for example, 10 to 20 percent) labeled by multiple annotators throughout the project, to monitor drift and annotator quality.
5. **Report agreement** alongside results. A win rate of 58 percent from annotators with kappa of 0.2 is weak evidence.

Agreement also sets a ceiling on what any automatic metric can achieve. If humans agree with each other 80 percent of the time on a pairwise task, an LLM judge that agrees with humans 80 percent of the time is performing about as well as another human would. This is how Zheng et al. framed their MT-Bench results (Section 6).

## Cost, bias, and annotator expertise

### Cost and speed

Human evaluation is slow and expensive. A careful rating of a long response against a detailed rubric can take several minutes, and expert annotators (physicians, lawyers, senior engineers) can cost many times more than general crowd workers. As a result, human evaluations are usually small, often a few hundred comparisons, which means their statistical uncertainty is large (Section 8). They are also hard to repeat for every checkpoint, which is why teams lean on automatic metrics and LLM judges during development and reserve human evaluation for key decisions.

### Biases

Annotators are human, and their judgments are shaped by more than content:

- **Length and style.** People tend to prefer longer, more detailed, well-formatted responses, even when the extra content adds nothing. This is the same verbosity bias that affects LLM judges and that Chatbot Arena's style control tries to adjust for.
- **Assertiveness.** As noted above, Hosking et al. found that confident-sounding responses are perceived as more factual. Annotators who cannot easily verify claims reward confidence, which is one route by which RLHF can increase hallucination (Section 4).
- **Position and order.** The first or second response in a pair may be favored; responses rated early in a session may be rated differently from those rated late.
- **Fatigue.** Quality drops over long annotation sessions.
- **Annotator background.** Cultural background, language, political views, and personal values affect judgments about tone, appropriateness, and helpfulness. Who the annotators are shapes what the model learns to be.
- **Anchoring and expectations.** If annotators know which system is the "new" one, they may favor it.

Design choices mitigate some of these: blind and randomized presentation, short sessions, attention checks (items with known answers), clear rubrics that separate style from substance, and diverse annotator pools.

### Can annotators even tell?

As models improve, a deeper problem arises: non-expert annotators may be unable to judge the quality of what they are reading. Clark et al. found that untrained evaluators distinguished GPT-3-written text from human-written text at chance level, and that brief training helped only modestly. Karpinska et al. found that crowd workers on Amazon Mechanical Turk, unlike English teachers, failed to distinguish model-generated stories from human references, even with strict qualification filters.

For technical content the problem is sharper. A non-programmer cannot tell whether code is correct; a non-physician cannot tell whether medical advice is sound. For these tasks, human evaluation requires **domain experts**, and even experts need time and tools (running the code, checking references). This is the core motivation for research on *scalable oversight*: how can humans supervise AI systems whose outputs they cannot easily check? GPQA (Section 3) was built partly as a testbed for this question.

### Is human data still human?

A newer concern is that crowd workers may use LLMs to do the annotation. Veselovsky et al. re-ran a summarization task on Mechanical Turk and, using keystroke detection and a synthetic-text classifier, estimated that 33 to 46 percent of workers used LLMs to complete it. Their task was particularly LLM-friendly, so the rate may be lower elsewhere, but the risk is clear: a "human" evaluation may secretly be an LLM evaluation. Mitigations include interfaces that detect pasting, tasks that require judgment rather than text production, trusted annotator pools, and spot checks.

## Designing a small human evaluation

Putting these ideas together, a practical recipe for comparing two models:

1. **Define the question** (for example, "Is fine-tune B more helpful than A for our customer-support prompts?").
2. **Sample prompts** randomly from real or realistic usage, stratified by category if needed. Fix the set before looking at outputs.
3. **Generate responses** from both systems with the same settings.
4. **Write a rubric** with a pairwise preference plus two or three dimension ratings or error flags. Include examples and tie-breaking rules.
5. **Pilot** with a few annotators on 50 items; measure agreement; revise.
6. **Run** with blind, randomized left/right order, multiple annotators on a subset, and attention checks.
7. **Analyze** win rates with confidence intervals (Section 8), agreement statistics, and per-category breakdowns. Read a sample of disagreements.
8. **Report** the rubric, annotator pool and expertise, number of items and annotators, agreement, and results with uncertainty.

## Key takeaways

- Human evaluation is the reference for open-ended quality, but its value depends entirely on design: clear goals, good rubrics, blind and randomized presentation, and qualified annotators.
- Rubric (absolute) scoring measures whether responses are good enough; pairwise comparison is easier and more sensitive for comparing systems but hides why one response won.
- Measure inter-annotator agreement with chance-corrected statistics such as Cohen's kappa or Krippendorff's alpha; use disagreements to improve guidelines, and treat human agreement as a ceiling for automatic metrics.
- Human judgments are biased by length, style, assertiveness, position, and annotator background, and non-experts often cannot judge technical or factual correctness.
- Human evaluation is expensive and therefore usually small; report its uncertainty and check that "human" labels are really human.

## Further reading

Clark, Elizabeth, et al. "All That's 'Human' Is Not Gold: Evaluating Human Evaluation of Generated Text." In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing*, 2021. https://arxiv.org/abs/2107.00061.

Cohen, Jacob. "A Coefficient of Agreement for Nominal Scales." *Educational and Psychological Measurement* 20, no. 1 (1960): 37–46.

Hosking, Tom, et al. "Human Feedback Is Not Gold Standard." arXiv preprint arXiv:2309.16349, 2023. https://arxiv.org/abs/2309.16349.

Karpinska, Marzena, et al. "The Perils of Using Mechanical Turk to Evaluate Open-Ended Text Generation." arXiv preprint arXiv:2109.06835, 2021. https://arxiv.org/abs/2109.06835.

Landis, J. Richard, et al. "The Measurement of Observer Agreement for Categorical Data." *Biometrics* 33, no. 1 (1977): 159–174.

Stiennon, Nisan, et al. "Learning to Summarize from Human Feedback." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2009.01325.

Veselovsky, Veniamin, et al. "Artificial Artificial Artificial Intelligence: Crowd Workers Widely Use Large Language Models for Text Production Tasks." arXiv preprint arXiv:2306.07899, 2023. https://arxiv.org/abs/2306.07899.
