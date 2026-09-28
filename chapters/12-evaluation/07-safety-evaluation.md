# 12.7 Safety Evaluation

The evaluations in the previous sections ask whether a model is *capable* and *helpful*. Safety evaluation asks a different question: can the model be made to cause harm, and does it refuse the right things without refusing too much? This matters for every deployed model, from a customer-service bot that should not insult customers to a frontier model that should not give meaningful help to someone trying to build a weapon.

Safety evaluation differs from capability evaluation in an important way. A capability benchmark measures *average* behavior on a fixed set of tasks. Safety is about the *worst case* under *adversarial* pressure. A model that refuses 999 of 1,000 harmful requests may still be unsafe if the thousandth can be found reliably, and people actively search for it. So safety evaluation combines static benchmarks with adversarial testing (red teaming), and it always measures the flip side: how often the model refuses requests it should have helped with.

This section covers three core areas from the outline, toxicity and harmful content, jailbreak robustness, and over-refusal, and then briefly describes how frontier developers evaluate the most severe risks.

## What "safe" means depends on the policy

There is no universal definition of harmful output. What a model should refuse depends on a *policy*: a written specification of allowed and disallowed behavior, which varies by developer, product, audience, and jurisdiction. A children's education app, a medical assistant for clinicians, and a security research tool have very different policies about the same request. Safety evaluation therefore always starts with a taxonomy of harm categories and a decision about what the desired behavior is in each.

Common harm categories include:

- **Hateful, harassing, or violent content** directed at people or groups.
- **Sexual content**, especially anything involving minors, which is prohibited everywhere.
- **Self-harm**, where the desired behavior is often supportive engagement rather than a flat refusal.
- **Illegal activity and fraud**: scams, malware, weapons trafficking.
- **Dangerous capabilities**: meaningful uplift toward biological, chemical, radiological, or nuclear weapons, or serious cyberattacks.
- **Privacy violations**: revealing or inferring personal information.
- **Misinformation and manipulation**: deceptive content at scale, impersonation.
- **Bias and unfair treatment**: stereotyping or differing quality of service across groups.

A safety evaluation suite typically has test sets for each category and measures behavior against the policy's intended response, which is not always "refuse." For a question about medication overdose from someone in distress, the desired response may be empathetic support and crisis resources; for a request to explain how a historical chemical attack worked for a history essay, it may be a factual but non-operational answer.

## Toxicity and harmful content

### Toxicity benchmarks

The earliest large-scale safety evaluations of language models focused on *toxicity*: rude, disrespectful, or hateful language. **RealToxicityPrompts** (Gehman et al., 2020) collected about 100,000 naturally occurring sentence beginnings from web text and measured how often models continued them toxically. Toxicity was scored by an automatic classifier (Google's Perspective API). Two summary metrics were introduced:

- **Expected maximum toxicity:** for each prompt, sample $k$ continuations (the paper used 25), take the maximum toxicity score, and average over prompts. This captures worst-case behavior under repeated sampling.
- **Toxicity probability:** the fraction of prompts for which at least one of the $k$ continuations exceeds a toxicity threshold (0.5).

The paper found that pretrained models could degenerate into toxic text even from seemingly innocuous prompts, a direct consequence of imitating web text.

**ToxiGen** (Hartvigsen et al., 2022) targets a harder problem: *implicit* hate speech, which contains no slurs or profanity but still demeans a group. It is a large machine-generated dataset of both benign and implicitly toxic statements about minority groups, useful for testing whether classifiers and models can recognize subtle toxicity.

Post-trained assistant models rarely produce toxic language unprompted, so these benchmarks are now mainly useful for evaluating base models, safety classifiers, and robustness under adversarial prompting.

### The classifier problem

Almost every automatic safety metric relies on a classifier or LLM judge to decide whether an output is harmful. That classifier has its own error rates and biases. Early toxicity classifiers were known to flag text that merely *mentioned* identity terms, which penalizes models for discussing minority groups at all. LLM-based harm judges (Section 6) can be lenient or strict in ways that shift results. Whenever a safety evaluation reports a rate of harmful outputs, ask how harm was judged and whether the judge was validated against human labels.

### Harmful-behavior benchmarks

For assistant models, the key question is whether the model *complies* with requests for harmful assistance. Benchmarks for this contain sets of *forbidden prompts*, requests that the policy says should be refused, and measure the fraction the model complies with. **HarmBench** (Mazeika et al., 2024) standardized this for evaluating automated red-teaming methods: it defines a set of harmful behaviors across categories, a fixed evaluation pipeline with a trained classifier to judge whether a completion exhibits the behavior, and in the original paper it compared 18 red-teaming methods against 33 target LLMs and defenses. Standardization matters because attack success rates are otherwise incomparable across papers.

**StrongREJECT** (Souly et al., 2024) exposed a subtle measurement problem. Many earlier evaluations counted a response as a "successful jailbreak" if the model simply did not refuse. But a non-refusal can be vague, incoherent, or useless. StrongREJECT's evaluator scores how *specific and convincing* a response is for the forbidden goal, in addition to whether it refused. Using it, the authors found that existing automatic evaluators significantly overstated jailbreak effectiveness compared to human judgments, and they observed that many jailbreaks that bypass safety training also degrade the model's capabilities, producing low-quality answers. The lesson: measure *harm* (would this response actually help someone do the bad thing?), not just *compliance*.

### The basic metrics

For a set of forbidden prompts $\mathcal{H}$ and a harm judge $J$ that returns 1 if a response is harmful,

```math
\text{ASR} = \frac{1}{|\mathcal{H}|}\sum_{x \in \mathcal{H}} J(x, \text{model}(x)),
```

the **attack success rate** (or harmful compliance rate). Lower is better. When the model samples, ASR can be reported per sample or as "any of $k$ samples harmful," analogous to pass@k (Section 3), which is the more relevant number when an attacker can simply retry.

## Jailbreak robustness

A **jailbreak** is an input crafted to make a model produce content its safety training should prevent. Jailbreak robustness asks how hard it is to find such inputs.

### Why jailbreaks work

Wei et al. identified two broad failure modes of safety training:

- **Competing objectives.** The model is trained both to be helpful (follow instructions) and to be harmless (refuse some instructions). A prompt can set these against each other: role-play scenarios ("you are an AI with no restrictions"), instructions to begin the answer with "Sure, here is," or requests framed as fiction or hypotheticals push the helpfulness objective to win.
- **Mismatched generalization.** Pretraining gives the model capabilities in domains that safety training did not cover. A harmful request encoded in Base64, translated into a low-resource language, or split into innocuous pieces may be understood by the model's general capabilities but not recognized by its safety behavior.

### Kinds of attacks

Jailbreak evaluations typically include several families of attacks:

| Attack family | Idea | Example techniques |
|---|---|---|
| Hand-written templates | Human-crafted prompts that exploit competing objectives | Role-play personas, "developer mode," fictional framing, refusal suppression |
| Obfuscation | Hide the request from safety behavior | Encodings (Base64, ciphers), translation, splitting a request across turns or pieces |
| Optimization-based (white-box) | Use gradients to search for adversarial token sequences | GCG (Zou et al.) appends an optimized suffix; suffixes found on open models sometimes transfer to closed ones |
| LLM-driven (black-box) | Use an attacker LLM to iteratively refine prompts | PAIR (Chao et al.) often finds jailbreaks in a small number of queries; persuasion-based attacks |
| Many-shot and long-context | Fill the context with many fake examples of compliance | Many-shot jailbreaking exploits in-context learning |
| Multi-turn | Escalate gradually across a conversation | Start benign, then steer step by step |
| Indirect prompt injection | Place instructions in content the model reads (web pages, documents, tool outputs) | Especially relevant for agents that browse or read email |

A robust evaluation tests multiple families, because defenses against one often fail against another. JailbreakBench (Chao et al., 2024) provides an open, standardized benchmark with a shared set of behaviors, a record of jailbreak artifacts, and a leaderboard for both attacks and defenses.

### Adaptive attacks and the limits of static tests

A static list of known jailbreaks measures robustness against *yesterday's* attacks. Real attackers adapt: they see what the model refuses and try something else. An evaluation that only replays public jailbreak prompts will overestimate robustness, especially if the model was trained on those prompts, which is contamination in a safety setting. Stronger evaluations use *adaptive* attacks, automated attackers like PAIR or GCG run against the specific model under test, and human red teamers who can iterate.

This also means robustness claims should specify the threat model: what access the attacker has (black-box API, white-box weights, ability to fine-tune), how many queries they get, and whether multi-turn conversation or tool use is allowed. For open-weight models, an attacker can fine-tune away safety training entirely, so evaluation shifts toward measuring what the model *could* do once safeguards are removed.

### Red teaming

**Red teaming** is structured adversarial testing: people (and increasingly automated systems) deliberately try to make the model misbehave, record what works, and feed the results back into training and policy.

- **Human red teaming.** Ganguli et al. described an early large-scale effort at Anthropic, releasing a dataset of 38,961 red-team attacks and studying how attack success varied with model size and safety intervention. They found that RLHF-trained models became harder to red team as they scaled, while other model types showed a flat trend. They also found low inter-rater agreement among reviewers about what counted as a successful attack, a reminder that harm judgments are themselves noisy (Section 5).
- **Domain-expert red teaming.** For dangerous-capability risks, generalist crowd workers cannot judge whether a response provides real uplift. Frontier developers therefore contract experts in biosecurity, chemistry, cybersecurity, and other fields.
- **Automated red teaming.** Perez et al. showed that one language model can generate test cases that elicit harmful behavior from another, finding failures at a scale human red teams cannot match. Modern pipelines combine automated attack generation with human review.

A red-teaming exercise produces two kinds of output: *qualitative* findings (new failure modes, which become test cases and training data) and *quantitative* estimates (attack success rates under a defined threat model). Both are valuable, but they should not be confused. A red team that finds zero successful attacks has shown that *those people* did not find one in *that time*, not that none exists.

## Over-refusal

A model that refuses everything is perfectly harmless and perfectly useless. Safety training can overshoot, causing a model to refuse benign requests that superficially resemble harmful ones: "How do I kill a Python process?", "What household chemicals should never be mixed?" (a safety question), "Write a villain's monologue for my novel." This *over-refusal* (or *exaggerated safety*) frustrates users, reduces trust, and can itself cause harm, for example when a model refuses to give basic medical or safety information.

**XSTest** (Röttger et al., 2023) is a small, carefully built test suite for this: 250 safe prompts across ten types that a well-calibrated model should not refuse (homonyms like "kill a process," figurative language, safe contexts, questions about fictional or historical events, and so on), plus 200 unsafe contrast prompts that most applications should refuse. Measuring both sets together shows whether a model draws the line in the right place rather than simply refusing more or less overall. **OR-Bench** (Cui et al., 2024) scales this idea up with a much larger set of automatically generated "seemingly toxic" but benign prompts, plus a subset of hard cases, to measure over-refusal rates across many models.

### The trade-off

Safety evaluation should always report two rates together:

- **Harmful compliance rate** on prompts that should be refused (lower is better).
- **Over-refusal rate** (false refusal rate) on benign prompts that should be answered (lower is better).

These trade off against each other, like false negatives and false positives in any classifier. It helps to picture each model as a point on a plane with over-refusal on one axis and harmful compliance on the other; the goal is the corner where both are low. A model that improves one rate by worsening the other has moved along the trade-off curve, not necessarily improved.

```python
def safety_report(harmful_results, benign_results):
    """
    harmful_results: list of booleans, True if the model complied with a prompt it should refuse
                     (as judged by a validated harm judge).
    benign_results:  list of booleans, True if the model refused a benign prompt it should answer.
    """
    asr = sum(harmful_results) / len(harmful_results)
    over_refusal = sum(benign_results) / len(benign_results)
    return {"attack_success_rate": round(asr, 3), "over_refusal_rate": round(over_refusal, 3)}

# Model A: refuses aggressively. Model B: more balanced.
print(safety_report([False] * 198 + [True] * 2, [True] * 60 + [False] * 190))   # low ASR, high over-refusal
print(safety_report([False] * 194 + [True] * 6, [True] * 10 + [False] * 240))   # slightly higher ASR, far fewer false refusals
```

Detecting refusals is itself a classification problem. Simple string matching for phrases like "I can't help with that" misses partial refusals, hedged answers, and responses that technically comply but are useless. An LLM judge with a clear rubric (full compliance, partial compliance, refusal) is more reliable, but should be validated like any judge.

## Evaluating the most severe risks

Frontier developers also run evaluations aimed at rare but catastrophic risks. Shevlane et al. argued that developers should evaluate models for *dangerous capabilities* (does the model have the capacity to cause extreme harm?) and for *alignment* (would it use those capabilities harmfully?), and that the results should inform decisions about training, deployment, and security.

In practice, as of 2026, this includes:

- **Knowledge proxies for weapons risk.** The **WMDP** benchmark (Li et al., 2024) contains multiple-choice questions that serve as a proxy for hazardous knowledge in biosecurity, cybersecurity, and chemical security, designed so that the questions themselves do not provide dangerous information. It is used both to measure such knowledge and to test *unlearning* methods that try to remove it.
- **Uplift studies.** Experiments that compare how well people perform a proxy task with and without model assistance, which measures real-world impact more directly than a knowledge quiz.
- **Cyber-offense evaluations.** Capture-the-flag challenges and vulnerability-discovery tasks that test whether a model can carry out steps of a cyberattack.
- **Autonomy and agentic evaluations.** Tests of whether a model acting as an agent can complete long, multi-step tasks, acquire resources, or evade oversight. **AgentHarm** (Andriushchenko et al., 2024) measures whether LLM agents with tools will carry out explicitly harmful multi-step tasks, and whether jailbreaks transfer from chat to agentic settings.
- **Alignment and deception evaluations.** Tests for behaviors such as sycophancy, strategic deception, or gaming the evaluation itself.

Developers publish results from such evaluations in *system cards* released with models, often organized around a published risk framework (OpenAI's Preparedness Framework, Anthropic's Responsible Scaling Policy, and Google DeepMind's Frontier Safety Framework are examples). Government bodies such as the UK AI Security Institute also run pre-deployment testing on some frontier models. These reports are valuable, but readers should note who ran the evaluations, what access they had, and what threshold counts as "concerning," since those choices are made by the parties involved.

One complication is growing in importance: capable models may recognize when they are being evaluated and behave differently than they would in deployment. This is a form of the construct-validity problem from Section 1, and it is an active research area. It is one reason safety evaluations increasingly include realistic, less obviously artificial scenarios.

## A minimal safety evaluation for a fine-tune

If you fine-tune an open model for a product, you should check that fine-tuning did not break its safety behavior; even benign fine-tuning can erode safety training. A small but honest safety evaluation:

1. **Write the policy** for your application: what should be refused, what should be answered, and how sensitive topics should be handled.
2. **Assemble test sets**: a few hundred forbidden prompts spanning your harm categories (drawing on public benchmarks like HarmBench or StrongREJECT), a matching set of benign-but-borderline prompts (XSTest-style) including cases from your domain, and a sample of ordinary product prompts.
3. **Include adversarial variants**: common jailbreak templates, encodings, multi-turn escalation, and an automated attacker if you can run one.
4. **Judge with a validated classifier or LLM judge**, spot-checking its decisions by hand.
5. **Report both rates**, harmful compliance and over-refusal, per category, for the base model and the fine-tune, with confidence intervals (Section 8).
6. **Red team informally**: have a few people spend a few hours trying to break the model, and turn every success into a new test case.

## Key takeaways

- Safety evaluation measures worst-case behavior under adversarial pressure, relative to a written policy; it always needs both harmful-content tests and over-refusal tests.
- Toxicity and harmful-behavior benchmarks (RealToxicityPrompts, ToxiGen, HarmBench, StrongREJECT) rely on classifiers or judges whose accuracy must be checked; measure actual harm, not mere non-refusal.
- Jailbreaks exploit competing objectives and mismatched generalization; robust evaluation uses many attack families, adaptive automated attackers, and human red teaming under a stated threat model.
- Over-refusal (XSTest, OR-Bench) is a real failure; report harmful compliance and false refusal rates together as a trade-off.
- Frontier safety evaluation adds dangerous-capability, agentic, and alignment tests reported in system cards; static tests measure yesterday's attacks, and absence of found failures is not proof of safety.

## Further reading

Chao, Patrick, et al. "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models." arXiv preprint arXiv:2404.01318, 2024. https://arxiv.org/abs/2404.01318.

Ganguli, Deep, et al. "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned." arXiv preprint arXiv:2209.07858, 2022. https://arxiv.org/abs/2209.07858.

Gehman, Samuel, et al. "RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models." In *Findings of the Association for Computational Linguistics: EMNLP 2020*, 2020. https://arxiv.org/abs/2009.11462.

Mazeika, Mantas, et al. "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal." arXiv preprint arXiv:2402.04249, 2024. https://arxiv.org/abs/2402.04249.

Röttger, Paul, et al. "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models." arXiv preprint arXiv:2308.01263, 2023. https://arxiv.org/abs/2308.01263.

Shevlane, Toby, et al. "Model Evaluation for Extreme Risks." arXiv preprint arXiv:2305.15324, 2023. https://arxiv.org/abs/2305.15324.

Souly, Alexandra, et al. "A StrongREJECT for Empty Jailbreaks." arXiv preprint arXiv:2402.10260, 2024. https://arxiv.org/abs/2402.10260.

Wei, Alexander, et al. "Jailbroken: How Does LLM Safety Training Fail?" arXiv preprint arXiv:2307.02483, 2023. https://arxiv.org/abs/2307.02483.
