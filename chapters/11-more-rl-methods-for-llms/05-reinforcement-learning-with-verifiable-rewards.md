# 11.5 Reinforcement Learning with Verifiable Rewards (RLVR)

A learned reward model is a proxy, and Section 1 described how a policy exploits it. Some tasks do not need a proxy. A math answer is equal to a known number or it is not. A program passes its tests or it does not. A response contains the required boxed expression or it does not. Reinforcement learning with verifiable rewards (RLVR) replaces $`r_{\phi}`$ with a program that checks the response. The name and the open recipe were set out for general post-training by Lambert et al. (2024). The same idea, a checker instead of a preference model, is what the reasoning models of 2025 optimize with the critic-free updates of Section 2.

## A checker in place of a reward model

The policy still samples a response $`y`$ to a prompt $`x`$. The reward is a function $`v(x, y)`$ that a program computes, typically 1 or 0:

- **Math.** Extract a final answer from the response and compare it with the ground truth. Equality has to be more than string match: $`0.5`$ and $`1/2`$ are the same answer, and a checker that does not know that will punish a correct solution. A small computer-algebra comparison, or a normalized string format, is the usual tool. The grade is on the extracted answer, not on whether a human likes the prose.
- **Code.** Run the program against unit tests. Passing every test returns 1, failing any test or failing to compile returns 0. The tests are part of the training data, and they have to be written as carefully as labels.
- **Format.** A regular expression checks that the response obeys a contract the rest of the reward depends on: a final answer inside `\boxed{...}`, a reasoning trace inside a designated tag, a valid JSON object. Format is often a second, smaller reward added to the correctness reward, because a correct answer that the extractor cannot find is indistinguishable from a wrong one.

The training loop is then any policy gradient from Section 2, most often GRPO. For each problem, sample a group of solutions, score each with $`v`$, compute group-relative advantages, and update the policy. No Bradley-Terry model is fit. The dataset is a set of problems with answers, not a set of ranked essays.

[Code 11.5.1](#code-1151-a-verifier-and-the-advantages-it-induces) is the whole reward function for a toy arithmetic task: find a `\boxed{...}` span, compare it with the gold answer, and turn the bits into GRPO advantages. On four samples with rewards $`1, 0, 0, 1`$, the advantages are $`+1, -1, -1, +1`$. The two samples that never boxed an answer get the same zero as the sample that boxed the wrong number. The checker cannot tell "almost" from "wrong," and the advantage does not pretend it can.

## Why this resists reward hacking

Gao et al. (2023) studied a gap between two models: a proxy reward that the policy optimizes, and a gold reward that people trust more. The gap exists because the proxy is a learned function with its own errors, and optimization looks for those errors. A verifier does not have that gap when it computes the quantity the evaluation will also compute. If the test set grades math answers by exact match against a boxed expression, and the training reward does the same, then raising the training reward is raising the evaluation, at least on problems drawn from the same pool.

Two caveats keep this from being a free pass.

The verifier can be wrong about the task even when it is deterministic. A test suite that does not cover the specification pays the policy for passing the tests, not for solving the problem. An extractor that searches the whole response for the gold number pays the policy for mentioning that number anywhere, including in a sentence that says the number is incorrect. A format reward that is large compared with the correctness reward pays the policy for boxes and not for arithmetic. These are bugs in $`v`$, and they are reward hacking in the ordinary sense: the policy did what the reward said, and the reward said the wrong thing. They are also inspectable. A learned reward model's failure mode is a high score on a strange paragraph. A verifier's failure mode is a unit test you can read.

The policy can also overfit the training problems without breaking the checker. A model that memorizes the answers to the ten thousand problems in the training file will score perfectly on them and will not have learned to add. The defense is the same as in supervised learning: hold out problems, and watch the held-out accuracy rather than the training reward. Because the reward and the held-out metric are the same function, a divergence between them is memorization or leakage, not the proxy-gold curve of Gao et al.

What verifiable rewards genuinely remove is the failure in which the training number and the true objective come apart while both are being measured on the same prompts. That is the failure Section 1 documented for learned reward models, and it is why reasoning recipes prefer a checker whenever a checker exists.

## Training a reasoning model with GRPO

DeepSeek-R1 is the public example of running this at scale (DeepSeek-AI 2025). The report describes two related models, and the difference between them is the point of the recipe.

**R1-Zero** starts from a pretrained base model and applies GRPO with rule-based rewards only. There is no supervised fine-tuning stage. The rewards are an accuracy reward, from a checker of the kind above, and a format reward that asks the model to put its reasoning in a designated tag so the answer can be extracted. The report is explicit about not using a neural reward model for this stage. A learned score would reintroduce the proxy that the rule was meant to avoid, and on long reasoning traces a neural reward is itself something the policy can learn to flatter.

**R1** adds stages around that core, because R1-Zero's reasoning, while effective, is not always easy to read and is not by itself a general assistant. The report describes a cold-start supervised stage on a small set of long chain-of-thought traces written to be readable, then a reasoning-oriented RL stage with the same style of rule-based rewards, then rejection sampling and supervised fine-tuning that mixes reasoning traces with ordinary instruction data, and finally a second RL stage that covers both reasoning and general helpfulness. The second stage is where a learned preference signal comes back, for qualities a checker cannot see, such as whether an open-ended answer is useful and harmless. Verifiable rewards and preference rewards are stages, not rival religions.

The GRPO details that matter for this kind of run are the ones Section 2 already set up. Groups in which every sample is correct, or every sample is wrong, have zero advantage and teach nothing. Early in training, a base model fails almost every hard problem, so most groups are all zeros and the gradient is carried by the few problems the model sometimes solves. Late in training, easy problems become all ones and also drop out. The useful prompts are the ones at the boundary, and a large group size $`G`$ is how the run keeps some successes and some failures in the same group. Section 8 describes later algorithms that resample until a group is mixed, instead of hoping the batch contains enough mixed groups.

Binary rewards also make the scale of $`\hat{A}_i`$ depend only on how many of the $`G`$ samples succeeded. That is a feature: the optimizer does not need reward whitening to reconcile a math checker with a code checker. It is a bias if the standardization is done per prompt, which Dr. GRPO discusses, and it is a reason to log accuracy by source of problem rather than one global reward.

## What shows up during training

The behavior the R1 report describes is a change in *how* the model spends tokens, not only in how often the final answer is right.

The traces get longer. A model paid only for the final answer, and allowed to generate for many tokens, learns to use the tokens. Intermediate calculations, restatements of the problem, and checks of earlier steps all have a chance of raising the probability of eventually boxing the right number, and the outcome reward reinforces the whole trace when the number is right. Average length is therefore a training curve, not just a decoding parameter. A length that rises while held-out accuracy rises is the model spending test-time compute (Section 13.9). A length that rises while accuracy is flat is the model filling the context window without being paid for it, and it is the first thing the length-normalization fixes in Section 8 are aimed at.

The traces also change shape. The R1 report describes the model beginning to reflect: to notice a contradiction, to back up, and to try a different step, without anyone having supervised those phrases. This is less mysterious than it sounds, and it is not a guarantee. Reflection is one of the behaviors that can precede a correct answer in a long sample. Outcome RL reinforces every token in a successful sample equally (Section 2), so a pattern that correlates with eventual success becomes more likely. If "wait, let me recompute" sometimes saves a wrong derivation, that phrase is reinforced on the samples where the derivation is then fixed. The pattern is emergent in the weak sense that it was not a line in the reward function. It is not emergent in the sense that the reward was indifferent to it. The reward paid for correct finals, and reflection was a means that showed up in the winning samples.

Two limits follow. First, the means and the ends are not separated. A superstitious phrase that happened to appear in correct samples is reinforced too. Second, the model can learn to produce the *appearance* of checking without a check that constrains the answer. A verifier of the final number does not read the trace. Process rewards, next, are the attempt to score the trace itself.

## Outcome rewards and process rewards

An **outcome reward** scores the finished response. Everything in this section so far is an outcome reward, whether it is a program or a reward model. A **process reward** scores the steps. The response is split into lines or into reasoning steps, and each step gets its own label: correct given the previous steps, or not.

The labels are expensive, because a person or a strong model has to read each step, and the split into steps is itself a choice. Lightman et al. (2023) collected human judgments of this kind and found that a reward model trained on step-level correctness selected correct solutions to challenging math problems more reliably than a reward model that saw only the final answer. Uesato et al. (2022) compared process feedback with outcome feedback on math word problems and studied how each one locates the step where a solution failed. Cobbe et al. (2021) is the earlier form of the outcome side: train a verifier on finished solutions, and search against it at test time. Section 13.9 uses these models as inference-time scorers. Here they are training signals.

Inside GRPO, a process reward changes the advantage from one number per response to one number per step. Tokens inside a step share that step's reward, and the return can be the sum of later step rewards, so an early mistake is blamed on the early step rather than spread over the whole trace. Credit assignment, which Section 2 gave up on when it multiplied every token by the same outcome advantage, comes back in a coarse form. The group baseline still applies: a step is good or bad relative to other samples, or relative to a value model if one has been put back.

The DeepSeek-R1 report discusses process reward models as an alternative and stays with outcome checkers for the large reasoning run. The objections are practical. Defining a step so that different samples can be compared is brittle. A learned process reward is a proxy again, and a policy that generates thousands of tokens has a long time in which to find text the step-level model overrates. An outcome checker has no opinion about the prose, so it cannot be flattered by it. The cost of that purity is the crude credit assignment above, and the reliance on at least occasional correct samples to produce a positive advantage.

The split used in practice follows the split in the R1 pipeline. Use a verifier for anything the verifier can see: the number, the tests, the format. Use a process reward or a rubric when the thing you care about is the shape of the reasoning and you are willing to audit the reward for hacking. Use a preference model or a judge, as in Section 6, for qualities that are real but not checkable. The algorithms of Section 2 do not change. Only $`r_i`$ changes.

## Code for this section

### Code 11.5.1: A verifier and the advantages it induces

```python
import re
import torch

def boxed_answer(text):
    found = re.findall(r"\\boxed\{([^}]*)\}", text)
    return found[-1].strip() if found else None

def verify(text, gold):
    """1 if the last boxed span matches the gold answer, else 0."""
    return 1.0 if boxed_answer(text) == gold else 0.0

samples = [
    r"17*20=340 and 17*3=51, so 391. \boxed{391}",
    r"17*23 = 17*25 - 34 = 391. The answer is 391.",   # right, but no box
    r"\boxed{381}",
    r"20*23 - 3*23 = 460 - 69 = \boxed{391}",
]
rewards = torch.tensor([verify(s, "391") for s in samples])
std = rewards.std(unbiased=False)
advantages = torch.zeros_like(rewards) if std == 0 else (rewards - rewards.mean()) / std
print("extracted:", [boxed_answer(s) for s in samples])
print("rewards:  ", rewards.tolist())
print("GRPO adv: ", [round(a, 4) for a in advantages.tolist()])
```

Output:

```text
extracted: ['391', None, '381', '391']
rewards:   [1.0, 0.0, 0.0, 1.0]
GRPO adv:  [1.0, -1.0, -1.0, 1.0]
```

The second sample states the right number and scores zero, because the extractor found no box. A format reward exists so that this sample is not the maximum of the group's reward.

## Key takeaways

- RLVR replaces the learned reward with a program: a matched math answer, a passed test suite, or a matched format. The policy gradient is unchanged.
- Because the training reward and the evaluation compute the same function, the proxy-versus-gold gap of Gao et al. (2023) does not open. Broken checkers and memorization of the training problems still do.
- The DeepSeek-R1 recipe runs GRPO against rule-based outcome rewards. R1-Zero skips supervised fine-tuning; R1 wraps that RL stage in a cold start, a rejection-sampling SFT stage, and a final RL stage that also uses preference rewards.
- Training on outcome correctness lengthens traces and can produce self-checking, because those patterns appear in samples that reach the right answer. The checker does not read the trace.
- Process rewards score each step and improve credit assignment, at the cost of defining steps and of reintroducing a learned proxy. Outcome checkers remain the default when a checker exists.

## Further reading

Cobbe, Karl, et al. "Training Verifiers to Solve Math Word Problems." arXiv preprint arXiv:2110.14168, 2021. https://arxiv.org/abs/2110.14168.

DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv preprint arXiv:2501.12948, 2025. https://arxiv.org/abs/2501.12948.

Gao, Leo, et al. "Scaling Laws for Reward Model Overoptimization." In *Proceedings of the 40th International Conference on Machine Learning*, 2023. https://arxiv.org/abs/2210.10760.

Lambert, Nathan, et al. "Tulu 3: Pushing Frontiers in Open Language Model Post-Training." arXiv preprint arXiv:2411.15124, 2024. https://arxiv.org/abs/2411.15124.

Lightman, Hunter, et al. "Let's Verify Step by Step." arXiv preprint arXiv:2305.20050, 2023. https://arxiv.org/abs/2305.20050.

Uesato, Jonathan, et al. "Solving Math Word Problems with Process- and Outcome-based Feedback." arXiv preprint arXiv:2211.14275, 2022. https://arxiv.org/abs/2211.14275.
