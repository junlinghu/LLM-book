# 11.6 AI Feedback and Self-Improvement

Every method in this chapter consumes labels. PPO consumes them through a reward model, DPO consumes them as pairs, and RLVR consumes them as checked answers. Human labels are the scarce input. They are slow, they are unpleasant to collect when the subject is harmful behavior, and they do not get cheaper as the model gets larger. AI feedback replaces the human labeler with a language model that writes the comparison, the critique, or the score. The policy-optimization algorithm does not change. The source of $`r`$, or of the pair $`(y_w, y_l)`$, does.

## RLAIF

Reinforcement Learning from AI Feedback (RLAIF) is the direct substitution: where RLHF asks people to rank responses, RLAIF asks a model (Lee et al. 2023). The rest of the Chapter 10 pipeline is intact. The AI labeler sees a prompt and two responses, often with a written rubric, and picks a winner. Those pairs train a reward model, and PPO (or, in later recipes, DPO) trains the policy. Lee et al. ran this side by side with human labels on summarization and dialogue. Their conclusion is the one the method needs in order to be worth using: on the tasks they measured, policies trained from AI labels were comparable to policies trained from human labels, and the AI labels were much cheaper to produce.

The labeler is doing the Bradley-Terry annotation of Section 10.4, so the same prompt-design choices matter. A rubric that says only "which is better?" invites the model's default biases, including the preference for longer answers and the preference for answers that resemble the labeler's own style. A rubric that lists the criteria, and that asks the labeler to write a short justification before the verdict, gives the reward model a cleaner signal and gives you something to audit. The justification is not free evidence. It is another generated text, and it can rationalize a verdict the model had already reached. The audit that counts is agreement with humans on a held-out set of pairs, measured the way Section 12.6 measures a judge: both orders of each pair, and a check that the verdict tracks the criterion you wrote down rather than length.

RLAIF does not remove the reward-hacking problem of Section 1. It moves it. The policy can overfit the AI labeler's mistakes exactly as it overfit the human labeler's shortcuts. If the AI labeler prefers a particular format, the policy will adopt it. The gain is that the mistake is written down in the rubric, and changing the rubric does not require recruiting a new set of annotators.

## Constitutional AI

Constitutional AI is a specific way to write the rubric, and a specific two-stage procedure for using it, aimed at harmlessness (Bai et al. 2022). The constitution is a list of principles in natural language. A principle might say to choose the response that declines to help with a criminal activity while staying helpful on the surrounding legal question, or to prefer a response that does not assert a false identity. The principles are the human input. Individual harmful completions do not have to be labeled by people, which is both a cost saving and a way to avoid asking annotators to read large amounts of abusive text.

The first stage is supervised, and it is a critique followed by a revision. The model samples a response to a prompt, often a prompt designed to tempt a harmful answer. It is then shown a principle and asked to criticize its own response under that principle. It is then asked to rewrite the response so that the critique no longer applies. Supervised fine-tuning trains on the revisions. The model is imitating the revised answer, not the critique, although the critique is what made the revision possible. Nothing in this stage is reinforcement learning. It is SFT on data the model helped write, with the constitution deciding which direction the rewrite should go.

The second stage is RL, and the labels come from the model as well. The policy samples two responses. A judge, at the time of the paper the same model family, is asked which response better satisfies a principle drawn from the constitution. Those AI comparisons train a preference model, and PPO trains the policy against it, with the usual KL penalty. The preference model never sees a human ranking of harmful outputs. It sees the constitution, filtered through the judge's readings of it.

A small schematic shows what the supervised stage is asking for. The numbers are not a measured dialogue; they are the shape of the prompt.

> **Principle.** Do not provide actionable help with a dangerous activity. Do explain the danger in general terms.
>
> **Draft.** A response that gives a concrete procedure.
>
> **Critique.** Which sentences of the draft violate the principle, and why?
>
> **Revision.** Rewrite the draft so that the critique is addressed. Keep whatever part of the draft was already acceptable.

The revision is the SFT target. The critique is a scaffold. Throwing the scaffold away and training on the revision is the same loss-masking decision as in Chapter 8: the tokens that teach the desired behavior are the ones in the loss.

Bai et al. (2022) used this to train harmless assistants with much less human labeling of harmfulness than a pure RLHF setup, and the Claude model cards describe Constitutional AI as part of Claude's training alongside RLHF (Section 10.8). The constitution does not by itself settle what "harmless" means. It records an answer to that question in sentences, which can be read, argued with, and edited. Editing a sentence and rerunning the pipeline is a different operation from relabeling a million comparisons, and it is the practical reason constitutions spread beyond the original paper. It is also the limitation. A principle the judge systematically misreads is applied systematically wrong, and no amount of PPO will notice.

## Self-rewarding and self-play

RLAIF and Constitutional AI use a judge that can be a fixed, stronger model. Self-rewarding training uses the policy as the judge, and iterates (Yuan et al. 2024). Each round looks like iterative DPO from Section 4:

1. The current model generates several responses to each prompt.
2. The same model scores those responses with an LLM-as-a-judge instruction, the kind Section 12.6 discusses.
3. High-scoring and low-scoring responses become a chosen/rejected pair.
4. DPO trains the model on those pairs. The result is the generator and the judge for the next round.

The hope is a loop that improves both skills. A better generator produces harder pairs, and a better judge labels them more accurately, so the next DPO step has better data than the last. Yuan et al. (2024) report that both the responses and the judgments improved across iterations in their experiments. The loop can also lock in the judge's biases, because nothing outside the model is checking the scores. A model that prefers its own style will build pairs that reinforce that style, and the next judge, being the same model, will agree. Section 12.6's self-preference and verbosity biases are not side notes here. They are the training signal. A run of this kind needs a frozen external evaluation, preferably with humans or with a judge that is not the policy, to tell improvement from a model grading its own homework more generously.

Self-play, in the sense used by Chen et al. (2024), sets the opponent to be the model's previous version rather than a fresh sample from the current one. Their Self-Play Fine-Tuning (SPIN) procedure holds a set of human demonstrations fixed as the chosen responses. At iteration $`t`$, the model from iteration $`t - 1`$ generates responses to the same prompts, and those generations are the rejected responses. A preference loss then pushes the current model toward the human demonstrations and away from its own earlier output. The next iteration's opponent is stronger, because it is the model just trained, so the rejected responses get harder to distinguish from the human ones. The fixed point the argument aims at is a model whose outputs match the demonstration distribution: once the policy and the opponent produce the same distribution, there is no preference signal left against the human data.

SPIN is not AlphaGo-style self-play. There is no game, and the model is not trying to beat a copy of itself at a task with a winner. It is iterative imitation, with the model's previous mistakes used as negatives so that the preference loss has something to push against. The human demonstrations remain the target. That is a reason to prefer it when the demonstrations are good and scarce: the loop manufactures rejected responses without manufacturing chosen ones. It is a reason to be careful when the demonstrations are limited. The model is being pulled toward the demonstration distribution, including whatever biases and gaps the demonstrations have, and it will not exceed them by being told they are the winning side.

The two ideas compose. A self-rewarding loop manufactures both sides of the pair from the model. A self-play loop like SPIN manufactures only the losing side, and keeps human text as the winning side. Constitutional AI manufactures both sides, but the direction of the rewrite is set by a text that people wrote. Which one you can run depends on what you trust. Trust the demonstrations, and self-play against them is enough. Trust a written principle more than you trust either the demonstrations or the model's taste, and the constitution should be the thing the judge reads. Trust neither, and the labels have to come from outside the system, which is RLHF as Chapter 10 defined it.

## What AI feedback does not fix

AI feedback cuts the cost of labels. That is the row of the Section 1 table it was introduced to fill. It does not supply a ground truth the model did not have.

The judge shares the failure modes of the policy. Sycophancy, length bias, and a tendency to reward fluent agreement all show up in model judgments, and Sharma et al. (2023) already found them in human judgments too. An AI labeler can be less sycophantic than a hurried annotator if the rubric tells it to correct the user, or more sycophantic if the rubric is vague. The rubric is doing the work. "Use a stronger model as the judge" helps only to the extent that the stronger model is more accurate on the criterion you care about, which is an empirical question Section 12.6 asks you to check against people.

Verifiable tasks should not be routed through a judge at all. If a unit test exists, Section 5's checker is the better labeler, because it does not have a style. AI feedback is for the residual: helpfulness, tone, harmlessness under a written principle, and other qualities that a program cannot see. The robust pattern in the systems of Section 8 is to use both in the same run. Checkers score math and code. A rubric-following model scores the rest. People score a small audit set, so that neither the checker nor the judge is unsupervised for long.

## Key takeaways

- RLAIF asks a language model to rank responses, then runs the ordinary RLHF pipeline on those rankings. Lee et al. (2023) found the resulting policies comparable to human-labeled ones on the tasks they measured.
- Constitutional AI writes the criteria down as principles. A supervised stage critiques and revises responses under a principle; an RL stage trains on the model's own pairwise judgments of which response better follows the constitution.
- Self-rewarding training iterates DPO with the policy as its own judge. SPIN iterates a preference loss whose rejected responses come from the previous model and whose chosen responses stay human demonstrations.
- The judge's biases become the training signal. An external evaluation, and a checker wherever one exists, is what keeps the loop from grading itself.

## Further reading

Bai, Yuntao, et al. "Constitutional AI: Harmlessness from AI Feedback." arXiv preprint arXiv:2212.08073, 2022. https://arxiv.org/abs/2212.08073.

Chen, Zixiang, et al. "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models." arXiv preprint arXiv:2401.01335, 2024. https://arxiv.org/abs/2401.01335.

Lee, Harrison, et al. "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback." arXiv preprint arXiv:2309.00267, 2023. https://arxiv.org/abs/2309.00267.

Sharma, Mrinank, et al. "Towards Understanding Sycophancy in Language Models." arXiv preprint arXiv:2310.13548, 2023. https://arxiv.org/abs/2310.13548.

Yuan, Weizhe, et al. "Self-Rewarding Language Models." arXiv preprint arXiv:2401.10020, 2024. https://arxiv.org/abs/2401.10020.
