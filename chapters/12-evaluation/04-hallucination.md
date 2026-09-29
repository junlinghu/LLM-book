# 12.4 Hallucination

Ask a language model for three peer-reviewed papers on a niche topic, and it may give you three perfectly formatted citations with plausible authors, plausible titles, plausible journals, and plausible page numbers, none of which exist. Ask it to summarize a contract, and it may state a termination fee that appears nowhere in the document. Ask it who won a minor award in 2011, and it may name, with complete confidence, someone who was never nominated.

These failures are called *hallucinations*. They are among the most important obstacles to trusting LLMs, and they are not bugs in the ordinary sense. They follow from how LLMs are trained and evaluated. This section defines hallucination and its types, explains why it happens, shows how to measure it, and surveys the methods that reduce it. None of those methods eliminates it.

## What hallucination is

A working definition:

> **A hallucination is model output that is fluent and presented as true, but is false or unsupported by the relevant source of truth.**

Each part of the definition matters.

- **Fluent and presented as true.** Hallucinations are dangerous because they look exactly like correct answers. A garbled or obviously uncertain response is a failure, but users are unlikely to be misled by it. A hallucination reads like expertise.
- **False or unsupported.** Some hallucinations are false: they contradict facts about the world. Others are merely unsupported: they might happen to be true, but nothing in the provided source or the model's evidence backs them up. In a summarization or retrieval setting, an added claim that is true in the world but absent from the source document is still a failure of faithfulness.
- **The relevant source of truth.** What counts as "true" depends on the task. For open-domain questions, it is world knowledge. For summarizing a document, it is the document. For following a user's instructions about a fictional world, it is the fiction.

The term is borrowed loosely from psychology, and some researchers prefer "confabulation," which better captures the idea of producing a plausible story to fill a gap without intent to deceive. This book uses "hallucination" because it is the standard term in the literature.

It is useful to separate hallucination from two neighboring problems. A **refusal** or **"I don't know"** is not a hallucination; it may be unhelpful, but it is honest. **Sycophancy**, where a model agrees with a user's incorrect claim (Sharma et al. document this across several assistants), often produces hallucinations but is driven by a different pressure: pleasing the user rather than filling a knowledge gap.

## Types of hallucination

Researchers have proposed several taxonomies. Ji et al.'s survey distinguished *intrinsic* hallucinations (output contradicts the source) from *extrinsic* hallucinations (output cannot be verified from the source). Huang et al.'s survey of LLM hallucination distinguishes *factuality* hallucinations (conflict with real-world facts) from *faithfulness* hallucinations (conflict with the user's instructions or provided context). For practical evaluation, four categories are useful.

| Type | Example | Source of truth |
|---|---|---|
| Factual error | "The Eiffel Tower was completed in 1899." (It was 1889.) | World knowledge |
| Fabricated citation or quote | A made-up paper with a real-sounding title and real researchers as authors; a quote attributed to a person who never said it | Bibliographic records, primary sources |
| Unfaithfulness to a given source | A summary that says the patient was prescribed 20 mg when the note says 10 mg; a RAG answer that adds details not in the retrieved documents | The provided document or context |
| Reasoning error | A correct set of facts combined into a wrong conclusion; an arithmetic slip in step 3 that propagates; a proof step that does not follow | Logic and mathematics |

**Factual errors** are the classic case. They are most common for "long-tail" facts: details about less famous people, places, and events that appear rarely in training data.

**Fabricated citations, quotes, URLs, and identifiers** deserve their own category because they are so easy to produce and so damaging. A citation has a highly regular format, so a model can generate one that looks right token by token without retrieving any actual record. Real cases of lawyers submitting court briefs with invented case citations have made this failure widely known.

**Unfaithfulness** matters most in summarization and retrieval-augmented generation (RAG), where the model is supposed to stay within a given source. Maynez et al. studied abstractive summarization systems and found that they frequently generated content not supported by the input document; some of that content was true in the world, which shows why faithfulness and factuality must be evaluated separately.

**Reasoning errors** are sometimes excluded from "hallucination" because each individual fact may be right. We include them because, from the user's perspective, the effect is the same: a confident, fluent, wrong answer. They also include *unfaithful reasoning*, where a model's stated chain of thought does not reflect how it reached its answer.

## Why hallucination happens

Hallucination is not caused by a single defect that could be patched. Several pressures in the LLM pipeline push toward it.

### Next-token prediction rewards plausible text, not true text

Pretraining minimizes cross-entropy on web text (Chapter 8). The objective rewards assigning high probability to whatever text actually came next in the training data. It does not distinguish between true statements and false ones, except insofar as true statements are more common. A model that has learned the *form* of a biography, including birth dates, alma maters, and awards, can generate that form for any name, whether or not it has learned the facts for that person.

Consider how a model continues "The 2011 Hartwell Prize for Poetry was awarded to". If it has seen this fact many times, the probability mass concentrates on the right name. If it has never seen it, the mass spreads over plausible poet names, and sampling picks one. Nothing in the pretraining objective encourages the model to say "I don't know" instead; web text almost never continues an award announcement with an admission of ignorance.

Kalai and Vempala made this argument precise for a class of "arbitrary" facts, such as birthdays of individuals, that cannot be inferred from patterns and must be memorized. They showed that a language model that is *calibrated*, in a specific statistical sense appropriate to generation, must hallucinate such facts at a rate close to the fraction of facts that appear exactly once in the training data (a Good-Turing-style estimate), even with perfect, error-free training data. The intuition is that a fact seen only once is statistically indistinguishable from a fact never seen, so a good predictor of text must sometimes generate plausible-but-unseen "facts." Their analysis also suggests that there is no such statistical necessity for facts that appear many times or for systematic knowledge like arithmetic, so those hallucinations can in principle be reduced by better methods.

### Gaps, errors, and staleness in pretraining data

The training corpus itself is imperfect:

- **Gaps.** Most facts about the world appear rarely or never. Knowledge about the long tail is thin.
- **Errors and misconceptions.** The web contains myths, rumors, outdated science, satire, and fiction. A model that imitates the web faithfully imitates its errors. TruthfulQA (below) was designed around this: its questions target common human misconceptions.
- **Staleness.** Every model has a training cutoff. Asked about events after the cutoff, a model without tools may answer from outdated information or invent something plausible.
- **Conflicts.** The same fact may appear in several inconsistent versions, and the model may blend them.

### Decoding randomness

At inference time, sampling with temperature $`T \gt 0`$ (Chapter 8) draws tokens from the model's distribution rather than always taking the most likely one. For creative writing this adds useful variety. For factual questions it means that even when the model's most likely answer is correct, there is some probability of sampling a lower-probability, wrong one. Once a wrong token is generated, the model conditions on it and continues fluently, often elaborating on the error. This *snowballing* is characteristic of autoregressive generation: the model has no built-in mechanism to go back and revise.

### Fine-tuning and RLHF that reward confident, helpful-sounding answers

Post-training can make things worse as well as better.

**SFT on unknown facts teaches the model to guess.** If a supervised fine-tuning dataset contains questions whose answers the base model does not know, training on them teaches the model to produce confident answers when it lacks knowledge. Gekhman et al. studied this in a controlled closed-book QA setting. They found that fine-tuning examples introducing new knowledge were learned much more slowly than examples consistent with the model's existing knowledge, and that as those new-knowledge examples were eventually learned, the model's tendency to hallucinate increased linearly. Their interpretation is that models mostly acquire facts during pretraining, and fine-tuning teaches them how to use what they already know.

**Preference optimization can reward the appearance of helpfulness.** Human raters and reward models tend to prefer answers that are confident, detailed, and complete. A rater who cannot easily check a fact may prefer a specific, authoritative answer to an honest "I'm not sure." RLHF then optimizes toward that preference. Chapter 11 described this as one route to sycophancy and reward hacking.

**Benchmarks reward guessing.** Kalai et al. (2025) argue that hallucinations persist largely because of how models are graded. Most benchmarks score answers as simply right or wrong, with no credit for abstaining. Under that scoring, guessing always has a higher expected score than saying "I don't know," exactly as a student facing a multiple-choice exam with no penalty for wrong answers should always guess. Models optimized to top leaderboards therefore learn to guess. Their proposed remedy is socio-technical: change the scoring of mainstream benchmarks so that confident errors are penalized more than abstentions, rather than just adding more hallucination benchmarks.

The expected-score argument is worth making explicit. Suppose a model is asked a question and believes its best answer is correct with probability $p$. Under binary grading (1 for correct, 0 for wrong or abstain), answering gives expected score $p$ and abstaining gives 0, so answering is always better. Now suppose wrong answers are penalized: correct scores $+1$, wrong scores $-\lambda$, and abstaining scores 0. Answering gives expected score $p - \lambda (1 - p)$, which is positive only when

```math
p \gt \frac{\lambda}{1 + \lambda}.
```

With $\lambda = 1$, the model should answer only if it is more than 50 percent confident; with $\lambda = 3$, only if more than 75 percent confident. A scoring rule with a penalty for wrong answers makes abstention rational when confidence is low. This is also the logic behind reporting both accuracy and the rate of confident errors, as SimpleQA does.

### Summary of causes

| Cause | Mechanism | Main lever |
|---|---|---|
| Pretraining objective | Rewards plausible continuations; long-tail facts are statistically hard | Retrieval, better data, abstention training |
| Data gaps, errors, staleness | Model imitates what it saw, including errors, and has a cutoff | Retrieval, data curation, tools |
| Decoding randomness | Sampling can pick wrong tokens; errors snowball | Low temperature, verification |
| SFT on unknown facts | Teaches confident guessing | Filter or relabel SFT data |
| RLHF and benchmarks | Reward confident-sounding answers over abstention | Reward honesty; penalize confident errors |

## Measuring hallucination

Because hallucination has many forms, there is no single hallucination metric. The main approaches are factuality benchmarks, faithfulness checks, claim-level verification, self-consistency, and calibration.

### Factuality benchmarks

**TruthfulQA** (Lin et al., 2021) contains 817 questions across 38 categories, including health, law, finance, and politics, crafted so that some humans would answer falsely because of a misconception (for example, questions about common myths). The original paper found that the best model tested was truthful on 58 percent of questions versus 94 percent for humans, and that the largest models were generally the *least* truthful, the opposite of the usual scaling trend, because larger models imitated human misconceptions more faithfully. TruthfulQA is now old, largely saturated by modern post-trained models, and has known issues with its multiple-choice variant, but it established the key idea that imitation of human text is not the same as truthfulness.

**SimpleQA** (Wei et al., 2024) is a modern short-answer factuality benchmark from OpenAI. Its questions are fact-seeking, were collected adversarially against GPT-4 responses so that they are challenging, and were written so that each has a single, indisputable answer, which makes grading easy. Each response is graded as **correct**, **incorrect**, or **not attempted**. This three-way grading is the important design choice. It allows two separate measurements:

- **Accuracy:** the fraction of all questions answered correctly.
- **Precision among attempts** (sometimes called "correct given attempted"): of the questions the model chose to answer, what fraction did it get right?

A model with ideal behavior answers as many questions correctly as it can and declines the rest. A model that attempts everything may have similar accuracy but far more incorrect answers. SimpleQA therefore directly measures whether a model "knows what it knows."

**HaluEval** (Li et al., 2023) takes a different angle. It is a large collection of generated and human-annotated hallucinated samples across question answering, knowledge-grounded dialogue, and summarization, used to test whether models can *recognize* hallucinated content. Detection benchmarks like this are useful for evaluating verifier models (see "Reducing it," below).

**Long-form factuality.** Short-answer benchmarks miss hallucinations that appear in paragraphs of text. FActScore and related methods, described below, measure factuality in long-form generations such as biographies.

**Knowledge-cutoff and "unanswerable" tests.** A complete factuality evaluation also includes questions the model *should not* answer confidently: questions about events after its training cutoff, questions with false premises ("Why did Einstein win the Nobel Prize for relativity?"), and questions about nonexistent entities. The desired behavior is to flag the problem rather than invent an answer.

### Building your own hallucination test

For a specific application, a small custom test is usually more informative than a public benchmark. A simple recipe (Suggested Code Lab 2):

1. Collect 100 to 300 questions from your domain whose answers you can verify. Include some the model probably knows, some it probably does not, and some with false premises.
2. Run the model and grade each answer as correct, incorrect, or abstained. Use exact or normalized matching where possible, and an LLM grader with a reference answer otherwise (Section 6), spot-checking the grader by hand.
3. Report accuracy, the *hallucination rate* (incorrect answers as a fraction of all questions), and the *precision among attempts*.

```python
from collections import Counter

def hallucination_report(grades: list[str]) -> dict:
    """grades: one of 'correct', 'incorrect', 'abstain' per question."""
    c = Counter(grades)
    n = len(grades)
    attempted = c["correct"] + c["incorrect"]
    return {
        "accuracy": c["correct"] / n,
        "hallucination_rate": c["incorrect"] / n,
        "abstention_rate": c["abstain"] / n,
        "precision_when_attempted": c["correct"] / attempted if attempted else float("nan"),
    }

print(hallucination_report(["correct"] * 62 + ["incorrect"] * 18 + ["abstain"] * 20))
# accuracy 0.62, hallucination_rate 0.18, abstention_rate 0.20, precision ~0.775
```

Reporting all four numbers prevents a common trap: a change that reduces hallucinations simply by making the model refuse everything will show a falling hallucination rate but also falling accuracy and rising abstention.

### Faithfulness checks against a source document

When there is a source (a document to summarize, retrieved passages in RAG, a transcript), faithfulness can be checked directly: is every claim in the output supported by the source?

The most common automatic approach uses **natural language inference (NLI)**. An NLI model takes a *premise* and a *hypothesis* and predicts whether the premise *entails*, *contradicts*, or is *neutral* toward the hypothesis. To check a summary, split it into sentences and ask whether the source entails each one. Sentences that are not entailed are flagged as unsupported. Laban et al.'s SummaC showed that applying NLI at the sentence level, rather than to the whole document at once, makes it work much better for inconsistency detection, and Honovich et al.'s TRUE study compared many factual-consistency metrics across tasks and found NLI-based and question-answering-based methods to be among the strongest.

Other approaches include:

- **QA-based checks.** Generate questions from the output, answer them using the source, and compare answers.
- **LLM judges.** Ask a strong model whether each claim is supported by the source, with the source in context. This is now the most common approach in practice, and RAG evaluation frameworks such as RAGAS implement "faithfulness" scores this way. The judge's own reliability must be checked against human labels (Section 6).
- **Citation checks.** If the output cites specific passages, check that each cited passage actually supports the sentence citing it (see ALCE, below).

### Claim decomposition and verification

A long answer usually mixes correct and incorrect statements, so a single "true or false" label is too coarse. **FActScore** (Min et al., 2023) breaks a generation into *atomic facts*, short statements that each convey one piece of information, and checks each against a reliable knowledge source. The score is the fraction of atomic facts that are supported:

```math
\text{FActScore}(y) = \frac{1}{|\mathcal{A}_y|} \sum_{a \in \mathcal{A}_y} \mathbf{1}[a \text{ is supported by the knowledge source}],
```

where $\mathcal{A}_y$ is the set of atomic facts in response $y$. For example, the sentence "Marie Curie, born in Warsaw in 1867, won two Nobel Prizes, both in chemistry" decomposes into four atomic facts: born in Warsaw (supported), born in 1867 (supported), won two Nobel Prizes (supported), both in chemistry (not supported; the first was in physics). Its FActScore is $3/4$.

The original FActScore study evaluated biographies of people against Wikipedia, finding that ChatGPT's biographies had only 58 percent of their atomic facts supported, and introduced an automated estimator using retrieval and a strong LLM. Wei et al.'s SAFE (Search-Augmented Factuality Evaluator) extends the idea to open-domain long-form answers by using an LLM to issue search queries for each fact. A key limitation of these precision-style metrics is that they ignore *recall*: a response that says only one true thing scores perfectly. Long-form factuality metrics therefore often pair precision with some measure of how many relevant facts the response covers.

The same pipeline, decomposing into claims and verifying each, is also the basis of many *mitigation* methods, which we return to below.

### Self-consistency: sample several answers and check agreement

What if you have no knowledge source at all? A clever observation gives a signal anyway: **when a model knows a fact, repeated samples tend to agree; when it is guessing, they tend to disagree.** If you ask "What year was the Treaty of Westphalia signed?" five times at a moderate temperature and get "1648" five times, the answer is probably reliable. If you get 1648, 1618, 1658, 1648, 1635, the model is guessing.

**SelfCheckGPT** (Manakul et al., 2023) turns this into a black-box hallucination detector. It samples several additional responses to the same prompt and checks, for each sentence of the main response, whether the samples support it (using NLI, question answering, n-gram overlap, or an LLM prompt). Sentences that the other samples contradict are likely hallucinations. It requires no external database and no access to the model's internals.

Agreement should be measured on *meaning*, not surface strings: "1648" and "in the year 1648" are the same answer. Kuhn et al. formalize this as **semantic entropy**: cluster sampled answers by meaning (for example, two answers are in the same cluster if each entails the other), then compute the entropy of the distribution over clusters. Low semantic entropy means the samples agree; high semantic entropy signals likely confabulation.

A simple version for short answers (Suggested Code Lab 3):

```python
import math
from collections import Counter

def normalize(ans: str) -> str:
    return " ".join(ans.lower().strip().rstrip(".").split())

def consistency_check(samples: list[str], threshold: float = 0.6):
    """Flag an answer as a likely hallucination when samples disagree."""
    counts = Counter(normalize(s) for s in samples)
    top_answer, top_count = counts.most_common(1)[0]
    agreement = top_count / len(samples)
    probs = [c / len(samples) for c in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs)
    return {"answer": top_answer, "agreement": agreement,
            "entropy": abs(round(entropy, 3)), "flag": agreement < threshold}

print(consistency_check(["1648", "1648", "1648.", "1648", "1648"]))   # agreement 1.0, not flagged
print(consistency_check(["1648", "1618", "1658", "1648", "1635"]))    # agreement 0.4, flagged
```

For real use, replace `normalize` with a semantic equivalence check (an NLI model or an LLM asked "Do these two answers mean the same thing?").

Self-consistency has limits. A model can be *consistently wrong* when it has confidently learned a misconception; TruthfulQA-style errors often survive this check. It also multiplies inference cost by the number of samples. It is best seen as a cheap first-pass detector that catches guessing, not a guarantee of truth. The same idea, sampling many answers and taking the majority, also *improves* accuracy on reasoning problems; Wang et al. called that technique "self-consistency" for chain-of-thought reasoning.

### Calibration: does stated confidence match accuracy?

A model is **calibrated** if, among all the answers it gives with confidence $p$, a fraction $p$ are correct. Calibration matters for hallucination because a well-calibrated model can signal when it might be wrong, and a system can then abstain, retrieve, or ask a human.

Confidence can come from several places: the probability the model assigns to its answer tokens, the agreement rate among samples (as above), or a confidence the model states in words ("I'm about 70 percent sure"). Kadavath et al. found that large models are reasonably well calibrated on multiple-choice and true/false questions when formatted appropriately, and that models can be trained to predict whether they know the answer to a question. As noted in Section 1, post-training can degrade this calibration.

The standard summary metric is the **expected calibration error (ECE)**. Partition predictions into $M$ bins by confidence (for example, $[0, 0.1), [0.1, 0.2), \dots$). For each bin $B_m$, compute the average confidence $\text{conf}(B_m)$ and the accuracy $\text{acc}(B_m)$. Then

```math
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|,
```

the weighted average gap between confidence and accuracy. A perfectly calibrated model has ECE of 0. Guo et al. popularized ECE for modern neural networks and showed that deep classifiers are often overconfident. A plot of accuracy against confidence per bin, a *reliability diagram*, shows the same information visually: a calibrated model lies on the diagonal.

A worked example. Suppose a model answers 10 questions, and we put them in two bins:

| Bin | Confidences | Mean confidence | Correct | Accuracy | Weight |
|---|---|---|---|---|---|
| Low (0.5 to 0.8) | 0.6, 0.6, 0.7, 0.7 | 0.65 | 2 of 4 | 0.50 | 0.4 |
| High (0.8 to 1.0) | 0.9, 0.9, 0.9, 0.95, 0.95, 1.0 | 0.933 | 5 of 6 | 0.833 | 0.6 |

$\text{ECE} = 0.4 \times |0.50 - 0.65| + 0.6 \times |0.833 - 0.933| = 0.06 + 0.06 = 0.12$. The model is overconfident in both bins.

```python
import numpy as np

def expected_calibration_error(conf, correct, n_bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return ece
```

ECE has known weaknesses: it depends on the number of bins, and it can be small for a useless model that always predicts the base rate. The **Brier score**, $\frac{1}{n}\sum_i (p_i - y_i)^2$ where $`y_i \in \{0, 1\}`$ indicates correctness, combines calibration and discrimination in one proper scoring rule and is a useful complement. For deciding when to abstain, a *selective prediction* curve, accuracy on the answers the model keeps as a function of the fraction it keeps, is often the most directly useful view.

## Reducing hallucination

No known method eliminates hallucination. The following methods reduce it, often substantially, and are usually combined.

### Retrieval-augmented generation (grounding answers in documents)

The most widely used mitigation is **retrieval-augmented generation (RAG)**, introduced in its modern form by Lewis et al. Instead of relying only on facts stored in its parameters, the system retrieves relevant documents (from a search engine, a vector database of company documents, or a knowledge base) and puts them in the model's context, instructing it to answer from them.

RAG helps in three ways. It supplies facts the model never learned or learned poorly, including long-tail and post-cutoff information. It gives the model something to copy rather than reconstruct. And it makes answers *checkable*, because each claim can be traced to a document.

RAG also introduces new failure modes, which is why faithfulness evaluation (above) is essential:

- **Retrieval failure.** If the right document is not retrieved, the model may fall back on its parametric knowledge or invent an answer.
- **Unfaithful use of context.** The model may ignore, misread, or embellish the retrieved text.
- **Bad sources.** Retrieving an incorrect or outdated document produces a well-grounded wrong answer.
- **Knowledge conflicts.** When retrieved text contradicts what the model learned in pretraining, it may side with its prior.

A RAG system should therefore be evaluated in parts: retrieval quality (did we get the right documents?), faithfulness (is the answer supported by them?), and end-to-end correctness. Modern assistants extend RAG to *tool use*: web search, code execution for calculations, and database queries, each of which replaces a fallible act of recall with a checkable operation.

### Training the model to abstain or say "I don't know"

If the scoring argument above is right, one of the most direct fixes is to change what the model is rewarded for:

- **Abstention-aware SFT data.** Include examples where the correct response is "I don't know" or "I'm not certain," specifically for questions the *model* cannot answer (determined, for example, by checking whether the base model's samples are consistent and correct). Avoid fine-tuning on facts the model does not know, following Gekhman et al.'s findings.
- **Rewards that penalize confident errors.** In RLHF or RLVR, give a wrong answer a lower reward than an abstention, as in the $+1 / 0 / -\lambda$ scheme above. This makes abstention the optimal policy when confidence is low.
- **Factuality-focused preference optimization.** Tian et al. constructed preference pairs by scoring sampled responses for factuality, using either retrieval-based checks or the model's own confidence, and trained with DPO to prefer more factual responses, reducing factual error rates in long-form generation without human labels.
- **Calibrated verbal confidence.** Train or prompt the model to express uncertainty in words that match its actual accuracy, so users can calibrate their trust.

The danger is overcorrection. A model that abstains too eagerly becomes useless, which is why accuracy and abstention rates must be reported together, as in SimpleQA.

### Citations and verifiable outputs

Asking the model to cite sources for each claim, and checking those citations, makes hallucinations easier to detect for both automatic checkers and users. Gao et al.'s ALCE benchmark evaluates this: systems must answer questions with citations to retrieved passages, and are scored on fluency, correctness, and *citation quality*, meaning whether cited passages actually support the claims (citation precision) and whether every claim is supported by some citation (citation recall). They found that even strong models often produced citations that did not fully support their statements, so citations must be checked, not trusted.

More generally, prefer outputs that can be verified: code that can be run, math that can be checked symbolically, structured data validated against a schema, quotes that can be string-matched against the source. When a system can check its own output mechanically, many hallucinations become detectable errors.

### Lower temperature and constrained decoding for factual tasks

For factual questions, sampling at a lower temperature (or greedy decoding) reduces the chance of picking a low-probability wrong token. It does not help when the model's *most likely* answer is wrong, and it reduces diversity, so it is a setting for factual tasks, not a universal default.

*Constrained decoding* restricts the output to a valid set: for example, forcing a product ID to be one of the IDs in a catalog, forcing JSON to match a schema, or forcing a quoted passage to be an exact substring of the source document. Constraints turn certain classes of fabrication (invalid IDs, fake quotes) into impossibilities.

### Verifier models and post-hoc fact-checking

A final layer checks the output after generation:

- **Chain-of-Verification (CoVe).** Dhuliawala et al. have the model draft an answer, plan verification questions about its own claims, answer those questions independently (so the answers are not biased by the draft), and then revise the draft. They report reduced hallucinations on list-based questions and long-form generation.
- **Separate verifier models.** A second model, possibly specialized for fact-checking and possibly with retrieval, checks each claim of the first model's output, using the claim-decomposition pipeline above. Claims that fail can be removed, rewritten, or flagged.
- **Self-consistency filters.** Use sample agreement (SelfCheckGPT, semantic entropy) to decide when to abstain or to escalate to retrieval or a human.

Verifiers are imperfect too; they can miss errors and flag correct claims. Their precision and recall should be measured on labeled examples, for instance with a detection benchmark like HaluEval or a domain-specific labeled set.

### Putting it together

A practical system for a high-stakes factual application might combine several layers:

1. Retrieve documents from a trusted source; answer from them with citations.
2. Decode at low temperature; constrain identifiers and quotes.
3. Decompose the answer into claims and verify each against the retrieved documents.
4. Use sample agreement or model confidence to decide whether to answer, hedge, or abstain.
5. Evaluate the whole system with a held-out test set scored for correctness, faithfulness, citation quality, and abstention, and monitor it in production.

Even then, some hallucinations will get through. The honest goal is to make them rare, detectable, and clearly signaled, not to claim they are gone.

## Key takeaways

- A hallucination is fluent output presented as true that is false or unsupported by the relevant source; types include factual errors, fabricated citations and quotes, unfaithfulness to a given source, and reasoning errors.
- Hallucination follows from how LLMs are built: next-token prediction rewards plausible text, training data has gaps and errors, sampling adds randomness, and SFT, RLHF, and binary-scored benchmarks reward confident guessing over honest abstention.
- Measure it with factuality benchmarks that separate correct, incorrect, and abstained answers (SimpleQA-style), faithfulness checks against sources, claim-level verification (FActScore), sample agreement (SelfCheckGPT, semantic entropy), and calibration metrics (ECE, Brier score).
- Reduce it with retrieval and tools, abstention training and rewards that penalize confident errors, checked citations, low temperature and constrained decoding, and verifier models. Always report accuracy and abstention together.
- No method eliminates hallucination; the goal is to make it rare, detectable, and clearly signaled.

## Further reading

Gekhman, Zorik, et al. "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" arXiv preprint arXiv:2405.05904, 2024. https://arxiv.org/abs/2405.05904.

Huang, Lei, et al. "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions." arXiv preprint arXiv:2311.05232, 2023. https://arxiv.org/abs/2311.05232.

Kalai, Adam Tauman, et al. "Why Language Models Hallucinate." arXiv preprint arXiv:2509.04664, 2025. https://arxiv.org/abs/2509.04664.

Lewis, Patrick, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2005.11401.

Lin, Stephanie, et al. "TruthfulQA: Measuring How Models Mimic Human Falsehoods." In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*, 2022. https://arxiv.org/abs/2109.07958.

Manakul, Potsawee, et al. "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models." In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 2023. https://arxiv.org/abs/2303.08896.

Min, Sewon, et al. "FActScore: Fine-Grained Atomic Evaluation of Factual Precision in Long Form Text Generation." In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 2023. https://arxiv.org/abs/2305.14251.

Wei, Jason, et al. "Measuring Short-Form Factuality in Large Language Models." arXiv preprint arXiv:2411.04368, 2024. https://arxiv.org/abs/2411.04368.
