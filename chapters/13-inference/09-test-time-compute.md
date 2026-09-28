# 13.9 Test-Time Compute

Most of this chapter has been about spending *less* computation per token. This final section turns the question around. A model's quality is usually thought of as fixed once training ends: a bigger model, trained on more data, gives better answers. But the same trained model can often give much better answers if it is allowed to spend more computation at inference time, by writing out its reasoning, by trying several answers and picking the best, or by searching over possible solutions. This idea, called **test-time compute** (or *inference-time scaling*), has become a second axis for improving LLMs alongside scaling training, and it changes the economics of inference that the previous sections described.

## Spending more computation at inference to get better answers

Section 2 observed that a forward pass costs about $2N$ FLOPs per token, regardless of how hard the token is. A model that answers a hard math problem with a single token spends exactly as much computation on it as on the word "the". Humans do not work this way: we think longer about harder problems. Test-time compute gives models a way to do the same.

There are two basic ways to spend more inference computation on a problem:

- **Sequential (depth):** generate a longer response, such as a chain of reasoning before the answer, or revise an answer repeatedly. Each additional token is another forward pass, so a longer response is more computation spent serially on the problem.
- **Parallel (width):** generate several independent responses and aggregate them, by voting or by choosing the best according to some verifier.

These can be combined, for example by sampling several long reasoning chains, or by searching a tree of partial reasoning steps. In all cases the cost of answering a query grows with the number of generated tokens, while the model's parameters stay the same.

The central empirical finding is that, for many tasks with checkable answers, such as math, coding, and logic puzzles, accuracy improves steadily as test-time compute grows. Snell et al. studied how best to allocate a fixed inference budget and found that the best strategy depends on problem difficulty, and that in some settings, spending extra compute at test time was more effective than using a much larger model. OpenAI's o1 announcement in 2024 reported that performance improved with both more reinforcement-learning training compute and more "thinking" time at test time, and the DeepSeek-R1 paper (2025) described in detail how reinforcement learning with verifiable rewards (Chapter 11) produces models that learn on their own to generate long reasoning chains.

## Chain of thought

### Prompting for reasoning

**Chain-of-thought (CoT) prompting**, introduced by Wei et al., asks the model to produce intermediate reasoning steps before its final answer, originally by including worked examples with step-by-step solutions in a few-shot prompt. Kojima et al. showed that simply appending an instruction such as "Let's think step by step" elicits similar behavior without examples. On multi-step problems, such as arithmetic word problems, reasoning chains substantially improved the accuracy of sufficiently large models.

Why does writing out reasoning help? Two complementary explanations are useful:

- **More serial computation.** A transformer performs a fixed number of layers of computation per token. Writing intermediate results as tokens lets the model use its output as a scratchpad, feeding the results of one forward pass into the next. A problem that needs more sequential steps than the network has layers can be spread across many tokens.
- **Decomposition into likely steps.** Each individual reasoning step (such as "48 divided by 2 is 24") is a simpler prediction, closer to patterns seen in training, than jumping straight to a final answer.

### Reasoning models

Prompted chain of thought uses whatever reasoning ability the model has. **Reasoning models** are trained specifically to reason at length before answering. As Chapter 11 describes, they are typically trained with reinforcement learning on problems with verifiable answers, such as math with known solutions and code with test cases, which rewards the final answer and lets the model discover useful reasoning behaviors such as trying an approach, checking intermediate results, noticing mistakes, and backtracking. Such models often produce thousands or tens of thousands of "thinking" tokens before a final answer, and many APIs expose a setting that controls how much reasoning effort the model spends.

Sequential test-time compute can also be controlled directly. Muennighoff et al. introduced *budget forcing*: to cap reasoning, the decoder forces the end of the thinking phase once a token budget is reached; to extend it, the decoder suppresses the end-of-thinking marker and appends a word such as "Wait", prompting the model to keep reasoning. They reported that accuracy on some math benchmarks rose as the forced thinking budget increased.

From an inference perspective, reasoning models shift the workload heavily toward **decode**. A short question may produce a very long output, which is the expensive, memory-bound phase (Section 3), with a KV cache that grows throughout the response (Section 4).

## Self-consistency, best-of-N, and search over candidate answers

### Self-consistency

**Self-consistency** (Wang et al.) is the simplest parallel method. Sample $N$ reasoning chains at a nonzero temperature, extract the final answer from each, and return the most common answer:

$$
\hat{y} = \arg\max_{y} \sum_{i=1}^{N} \mathbb{1}[\text{answer}(r_i) = y], \qquad r_i \sim p_\theta(\cdot \mid x).
$$

The intuition is that there are many ways to reason to a correct answer but errors tend to scatter across different wrong answers, so the correct answer is often the plurality even if no single chain is reliable. Self-consistency needs no extra model, and the vote share also serves as a confidence signal (Chapter 12 uses the same idea to flag likely hallucinations). It requires answers that can be compared, such as numbers, multiple-choice letters, or short strings; for free-form text, a model can be asked to pick the most consistent response.

```python
import re
from collections import Counter

def extract_answer(text):
    """Take the last number in the response as its final answer."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None

def self_consistency(generate, question, n=8, temperature=0.8):
    """generate(prompt, temperature) -> str. Returns the majority answer and its vote share."""
    prompt = f"{question}\nThink step by step, then give the final answer as a number."
    answers = [extract_answer(generate(prompt, temperature)) for _ in range(n)]
    votes = Counter(a for a in answers if a is not None)
    if not votes:
        return None, 0.0
    answer, count = votes.most_common(1)[0]
    return answer, count / n
```

In a serving system the $N$ samples share the same prompt, so a paged KV cache (Section 4) stores the prompt's cache once and the samples can run in the same batch, making parallel sampling cheaper than $N$ separate requests.

A simple model shows both the power and the limits of voting. Suppose each sample is correct independently with probability $p$, and consider a question with two possible answers. Majority voting over $N$ samples is correct with probability

$$
\Pr[\text{majority correct}] = \sum_{k \gt N/2} \binom{N}{k} p^k (1-p)^{N-k}.
$$

If $p = 0.6$, this rises from 0.60 with one sample to about 0.68 with five and 0.85 with 25. But if $p \lt 0.5$, voting makes things *worse* as $N$ grows, converging on the wrong answer. With many possible answers, plurality voting only needs the correct answer to be the most common one, which is a weaker requirement, but the lesson stands: voting amplifies what the model already tends to do. It cannot find answers the model rarely produces, and real samples are correlated, so gains are smaller than this independence model suggests.

### Best-of-N with a verifier

Instead of voting, we can **score** each candidate and pick the best. Given a scoring function $v(x, y)$,

$$
\hat{y} = \arg\max_{i \in \{1, \dots, N\}} v(x, y_i), \qquad y_i \sim p_\theta(\cdot \mid x).
$$

The scorer can be:

- **An exact verifier.** For code, run the unit tests; for math with a checkable result, check it; for a formal proof, run the proof checker. If a correct candidate exists and the verifier is exact, best-of-N succeeds.
- **A learned verifier or reward model.** Cobbe et al. trained verifiers to judge the correctness of solutions to grade-school math problems and found that generating many candidates and choosing the one the verifier ranked highest improved accuracy substantially over the generator alone. The reward models of RLHF (Chapter 10) can be used the same way.
- **An LLM judge** (Chapter 12), with all its biases.

When an exact verifier exists, the relevant quantity is **coverage**: the probability that at least one of $N$ samples is correct. With independent samples each correct with probability $p$,

$$
\text{coverage}(N) = 1 - (1 - p)^N,
$$

which is the pass@$k$ metric of Chapter 12 with $k = N$. Even a small $p$ grows quickly: with $p = 0.4$, five samples give 92 percent coverage. Brown et al. found that coverage keeps rising over several orders of magnitude of samples on coding and math tasks, so that repeated sampling from a weaker model can match or beat single attempts from a stronger one when answers can be verified automatically. Without an exact verifier, the benefit is limited by the verifier's accuracy: a learned verifier can be fooled, and optimizing hard against it runs into the same reward over-optimization seen in RLHF.

### Search over reasoning steps

Best-of-N scores only complete answers. **Process reward models** (PRMs) score each intermediate step of a solution. Lightman et al. collected human labels on the correctness of individual reasoning steps and found that a process-supervised reward model selected correct solutions to challenging math problems more reliably than an outcome-supervised one that only judged final answers.

With a step-level scorer, generation becomes a **search**. Instead of sampling complete solutions independently, the system expands partial solutions step by step and allocates more computation to the promising ones:

- **Step-level beam search.** Keep the $B$ best partial solutions according to the PRM, extend each with several candidate next steps, score them, and keep the best $B$ again. This is the beam search of Section 1, but over reasoning steps rather than tokens, and scored by a verifier rather than likelihood.
- **Tree search.** Tree of Thoughts (Yao et al.) prompts the model to propose several possible next "thoughts", evaluate them (with the model itself as evaluator), and explore the resulting tree with breadth-first or depth-first search, allowing lookahead and backtracking. Monte Carlo tree search and related methods from game playing have also been applied.
- **Sequential revision.** The model critiques and revises its own previous answer, with or without feedback from tools such as a code interpreter. This is sequential rather than parallel compute, and it works best when the feedback carries real information, such as a failing test.

Snell et al. compared such strategies and found that which one works best depends on difficulty: on easier problems, sequential revision of a single answer was more efficient, while on harder problems, broader parallel search with a verifier helped more. This suggests *adaptive* allocation: estimate how hard a question is, and spend compute accordingly.

## The tradeoff between answer quality and latency or cost

Test-time compute is not free. Every method above buys quality by spending tokens, and the rest of this chapter tells us exactly what tokens cost.

**Cost.** Parallel methods multiply the output tokens by $N$; sequential methods multiply the length of each response. With input price $c_{\text{in}}$ and output price $c_{\text{out}}$ per token (Section 8), a query with $n_{\text{in}}$ prompt tokens and $N$ samples of $n_{\text{out}}$ tokens each costs approximately

$$
\text{cost} \approx n_{\text{in}} \, c_{\text{in}} + N \, n_{\text{out}} \, c_{\text{out}},
$$

assuming the prompt is processed once and shared across samples (prefix caching). Because output tokens are the expensive kind, and reasoning chains are long, test-time compute can multiply the cost of a query many times over. A reasoning model that thinks for 10,000 tokens to answer a question that a standard model answers in 200 uses about 50 times as many output tokens.

**Latency.** Parallel samples can run concurrently in the same batch, so their wall-clock cost is modest when there is spare capacity, though each sample then decodes a little more slowly (Section 5). Sequential reasoning adds latency directly: 10,000 thinking tokens at a TPOT of 20 ms take over three minutes. Long chains also grow the KV cache for the entire response, reducing how many requests fit in memory and making each decode step slower (Section 4). This pushes serving systems toward the long-output regime where speculative decoding, KV cache compression, and efficient attention kernels matter most.

**Diminishing returns.** Accuracy usually improves roughly with the *logarithm* of test-time compute: each doubling helps less in absolute terms. Majority voting saturates once the plurality answer is stable. Best-of-N with a learned verifier eventually starts selecting answers that fool the verifier. Very long reasoning can also go wrong: models sometimes "overthink" easy questions, spending many tokens with no benefit, or talk themselves out of correct answers.

**When it helps.** Test-time compute pays off most when:

- The task has a verifiable or at least comparable answer (math, code, structured decisions).
- The problem is hard enough that the model is uncertain, but not so hard that correct answers are essentially never sampled.
- The value of a correct answer is high relative to the cost and delay of computing it.

It helps least for simple lookups, casual conversation, and latency-critical interactive uses. Deployed systems increasingly route queries adaptively: answering easy ones quickly with little or no reasoning and spending more on hard ones, sometimes letting the user choose the reasoning effort.

**A new scaling axis.** The broader lesson is that inference cost and model quality are no longer separable. Training compute and inference compute can partly substitute for each other: a smaller model that thinks longer can sometimes match a larger model that answers immediately. This makes every efficiency technique in this chapter more valuable, because each token saved by quantization, batching, speculative decoding, or better kernels can be reinvested in more reasoning. It also means that the cost of an LLM application is increasingly determined not only by which model is used, but by how much it is allowed to think.

## Key takeaways

- Test-time compute improves answers by spending more inference computation: sequentially (longer reasoning) or in parallel (more samples), or both through search.
- Chain of thought gives the model serial computation through its own generated tokens; reasoning models trained with RL on verifiable rewards learn to produce long reasoning chains, and budget forcing can control their length.
- Self-consistency takes the majority answer over sampled chains; best-of-N picks the candidate a verifier scores highest; process reward models enable search over reasoning steps.
- Voting amplifies what the model already tends to answer; verifiers let coverage, $1 - (1-p)^N$, drive accuracy, but learned verifiers can be gamed.
- Test-time compute multiplies output tokens, the expensive kind, increasing cost, latency, and KV cache use, with diminishing returns; it is most valuable for hard problems with checkable answers.

## Further reading

Brown, Bradley, et al. "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling." arXiv preprint arXiv:2407.21787, 2024. https://arxiv.org/abs/2407.21787.

Cobbe, Karl, et al. "Training Verifiers to Solve Math Word Problems." arXiv preprint arXiv:2110.14168, 2021. https://arxiv.org/abs/2110.14168.

DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv preprint arXiv:2501.12948, 2025. https://arxiv.org/abs/2501.12948.

Kojima, Takeshi, et al. "Large Language Models Are Zero-Shot Reasoners." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2205.11916.

Lightman, Hunter, et al. "Let's Verify Step by Step." In *International Conference on Learning Representations*, 2024. https://arxiv.org/abs/2305.20050.

Muennighoff, Niklas, et al. "s1: Simple Test-Time Scaling." arXiv preprint arXiv:2501.19393, 2025. https://arxiv.org/abs/2501.19393.

OpenAI. "Learning to Reason with LLMs." September 12, 2024. https://openai.com/index/learning-to-reason-with-llms/.

Snell, Charlie, et al. "Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters." arXiv preprint arXiv:2408.03314, 2024. https://arxiv.org/abs/2408.03314.

Wang, Xuezhi, et al. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." In *International Conference on Learning Representations*, 2023. https://arxiv.org/abs/2203.11171.

Wei, Jason, et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2201.11903.

Yao, Shunyu, et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." In *Advances in Neural Information Processing Systems 36*, 2023. https://arxiv.org/abs/2305.10601.
