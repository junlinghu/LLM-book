# 14.4 DPO Variants and Other Preference Methods

DPO turns a preference pair into a classification loss. The variants in this section change one assumption at a time: what happens when the labels are deterministic, what happens when the data are single thumbs rather than pairs, whether a reference model is required, whether length is normalized, and whether the pairs stay frozen while the policy moves. Rejection sampling fine-tuning sits at the end as the baseline that does none of this machinery and is still hard to beat.

## IPO: a finite target margin

The Bradley-Terry model behind DPO treats preferences as noisy. Even a response that is truly better wins only with probability $`\sigma(\Delta r)`$, which is strictly less than one. Maximum likelihood is happy to drive the reward gap to infinity, because that is how the model explains a pair that is *always* labeled the same way. Offline datasets are full of such pairs. Each prompt appears once, with one chosen response and one rejected response, and nothing in the file says the label was a close call.

Azar et al. (2024) show that fitting a deterministic preference with a Bradley-Terry loss pushes the policy toward a degenerate optimum: the rejected response's probability is driven toward zero, and the policy overfits the particular wording of the chosen response. Identity Preference Optimization (IPO) replaces the logistic loss with a squared distance from a finite margin. With $`h_{\theta}`$ the log-ratio gap,

```math
h_{\theta}(x, y_w, y_l) = \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)},
```

the IPO loss is

```math
\mathcal{L}_{\mathrm{IPO}}(\theta) = \mathbb{E}\left[ \left( h_{\theta}(x, y_w, y_l) - \frac{1}{2\beta} \right)^2 \right].
```

The target $`1/(2\beta)`$ is the margin at which IPO wants the chosen response to beat the rejected one, measured in nats of log-ratio against the reference. Once $`h_{\theta}`$ reaches that value, the gradient on the pair vanishes. DPO has no such point. Its logistic loss keeps rewarding a wider margin forever, with a weight that only decays as a sigmoid.

$\beta$ still means what it meant. A smaller $\beta$ is a larger target margin and a more aggressive policy, just as a smaller $\beta$ in the KL-regularized objective allows more divergence. The difference is that the target is finite. On a three-way softmax with $`\beta = 0.1`$, the target gap is $`1/(2 \times 0.1) = 5`$ nats. [Code 14.4.1](#code-1441-dpo-does-not-stop-ipo-stops-at-the-target-margin) runs both losses from a uniform start. IPO settles at a gap of 5. DPO passes 5 and is still widening the gap when the run stops, with the rejected response's probability down to about $`0.001`$. That is the overfitting Azar et al. (2024) derive, visible on a problem small enough to solve by hand.

IPO is the right variant when the labels are clean and nearly hard, and when DPO runs show the chosen and rejected log-probabilities separating without bound while evaluations flatten. It is still an offline pair loss with a reference model. It does not fix distribution shift, and it does not accept unpaired labels.

## KTO: learning from thumbs

Most production feedback is not a pair. A user clicks a thumbs-up or a thumbs-down on one response, and there is no sibling response sitting next to it. Building pairs means sampling a second response and paying someone to compare them. Kahneman-Tversky Optimization (KTO) trains on the binary signal directly (Ethayarajh et al. 2024).

The implicit reward is the same log-ratio as in DPO, $`r_{\theta}(x, y) = \log(\pi_{\theta}(y \mid x) / \pi_{\mathrm{ref}}(y \mid x))`$, without a $\beta$ inside it. What changes is the loss, which Ethayarajh et al. take from prospect theory. People judge an outcome relative to a reference point, and they weigh losses and gains differently. KTO's reference point $`z_0`$ is an estimate of the KL divergence between the policy and the reference, and the value of a completion is

```math
v(x, y) =
\begin{cases}
\lambda_D\, \sigma\big(\beta\,(r_{\theta}(x, y) - z_0)\big) & \text{if } y \text{ is desirable}, \\[4pt]
\lambda_U\, \sigma\big(\beta\,(z_0 - r_{\theta}(x, y))\big) & \text{if } y \text{ is undesirable}.
\end{cases}
```

The training loss is $`\lambda_y - v(x, y)`$, a constant shift of maximizing $`v`$. Desirable responses are pulled above $`z_0`$; undesirable responses are pulled below it. The coefficients $`\lambda_D`$ and $`\lambda_U`$ are the prospect-theory weights. Setting $`\lambda_U > \lambda_D`$ says that showing the user a bad answer hurts more than showing a good one helps, which is the loss-aversion asymmetry the name refers to. In practice the two coefficients are also the knob for class imbalance: if thumbs-down are rare, a larger $`\lambda_U`$ keeps them from being ignored.

The reference point has to be estimated without using the response being scored, or the loss can reward a trivial rescaling. Ethayarajh et al. (2024) estimate it from mismatched pairs inside the minibatch: pair each prompt with some other prompt's response, average those log-ratios, and clamp at zero,

```math
\hat{z}_0 = \max\left(0,\ \frac{1}{m} \sum_{i \neq j} \log \frac{\pi_{\theta}(y_j \mid x_i)}{\pi_{\mathrm{ref}}(y_j \mid x_i)}\right).
```

A mismatched response is not a meaningful answer to $`x_i`$, so $`\hat{z}_0`$ measures how far the policy has drifted in general, not whether this particular answer was good.

A one-line numerical example, with $\beta = 1$ and $`z_0 = 0.02`$. A desirable response has log-ratio $`r = \log(0.40 / 0.25) \approx 0.470`$; an undesirable one has $`r = \log(0.15 / 0.30) \approx -0.693`$. With $`\lambda_D = \lambda_U = 1`$,

```math
v_{\mathrm{des}} = \sigma(0.470 - 0.02) \approx 0.611, \qquad v_{\mathrm{und}} = \sigma(0.02 - (-0.693)) \approx 0.671,
```

and the losses are $`1 - v`$, about 0.389 and 0.329. Both are below the $`0.5`$ of a response sitting on the reference point, which is what we want: the good answer is already above $`z_0`$ and the bad answer is already below it. [Code 14.4.2](#code-1442-kto-simpo-and-orpo-on-toy-numbers) evaluates the same inputs.

KTO is the method to reach for when the data are binary and unpaired. It still uses a reference model. It does not need a second response, which is the expensive object. Ethayarajh et al. (2024) report it matching or exceeding pair-based losses on the datasets in their paper, including settings where they throw away the pairing and keep only the binary label. The comparison is the reason to try it, not a guarantee: a carefully collected pair still contains more information than a lone thumb, and if you have the pairs, DPO or IPO can use them.

## ORPO: one stage, no reference

Odds Ratio Preference Optimization (ORPO) folds the SFT term and the preference term into a single loss and drops the reference model (Hong et al. 2024). The SFT term is ordinary next-token loss on the chosen response, $`\mathcal{L}_{\mathrm{SFT}} = -\log \pi_{\theta}(y_w \mid x)`$. The preference term is a logistic loss on the log odds ratio between the chosen and rejected responses,

```math
\operatorname{odds}_{\theta}(y \mid x) = \frac{\pi_{\theta}(y \mid x)}{1 - \pi_{\theta}(y \mid x)}, \qquad
\mathcal{L}_{\mathrm{OR}} = -\log \sigma\left( \log \frac{\operatorname{odds}_{\theta}(y_w \mid x)}{\operatorname{odds}_{\theta}(y_l \mid x)} \right).
```

The combined loss is $`\mathcal{L}_{\mathrm{SFT}} + \lambda\, \mathcal{L}_{\mathrm{OR}}`$. The coefficient $\lambda$ balances "imitate the chosen response" against "prefer the chosen response to the rejected one." There is no $`\pi_{\mathrm{ref}}`$. The SFT term is the anchor that the reference model provided in DPO: without it, nothing would stop the policy from lowering both probabilities as long as the ratio moved the right way.

For a full response, $`\pi_{\theta}(y \mid x)`$ is a product of token probabilities and is tiny, so $`1 - \pi_{\theta}(y \mid x)`$ is indistinguishable from 1 at any precision that matters. On a pair with sequence probabilities $`10^{-8}`$ and $`10^{-9}`$, the log odds ratio and the plain log probability ratio differ by about $`10^{-8}`$ nats ([Code 14.4.2](#code-1442-kto-simpo-and-orpo-on-toy-numbers)). The odds formula is the theoretically complete one, and it reduces to a reference-free logistic loss on $`\log \pi(y_w) - \log \pi(y_l)`$ for real sequences. The piece that actually replaces the reference is the SFT term, not the difference between odds and probabilities.

ORPO is attractive when the starting point is not already a good instruction model, or when storing a second copy of the weights is the constraint. It is one stage instead of SFT-then-DPO. The cost is a loss with two jobs. If $\lambda$ is too small, the run is plain SFT and the rejected responses are ignored. If $\lambda$ is too large, the preference term dominates and the likelihood-displacement problem of Section 3 returns, because the odds ratio, like the DPO margin, is a comparison.

## SimPO: length-normalized and reference-free

SimPO also drops the reference model, and it changes the reward that the comparison uses (Meng et al. 2024). DPO's implicit reward is a *sum* of per-token log-ratios. A small positive log-ratio on every token adds up, so a longer response earns a larger reward from the same per-token improvement. Human raters separately tend to prefer longer answers (Section 11.4). The two biases point the same way, and DPO policies often grow verbose.

SimPO scores a response by its average log-probability, scaled by a temperature $\beta$, and it requires the chosen response to win by a margin $\gamma$:

```math
p_{\theta}(x, y) = \frac{\beta}{|y|} \sum_{t=1}^{|y|} \log \pi_{\theta}(y_t \mid x, y_{\lt t}), \qquad
\mathcal{L}_{\mathrm{SimPO}} = -\,\mathbb{E}\left[ \log \sigma\big( p_{\theta}(x, y_w) - p_{\theta}(x, y_l) - \gamma \big) \right].
```

The average removes the mechanical advantage of length. The margin $\gamma$ plays a role related to IPO's target: a zero gap is not enough, so the loss does not treat a pair of identical average log-probabilities as solved. Both $\beta$ and $\gamma$ are tuned. $\beta$ here is a scale on the reward, not a KL coefficient, because there is no reference distribution in the loss.

A five-token response and a two-token response show the discrepancy the average is aimed at. Suppose the policy improves on the reference by $`+0.05`$ nats per token on the long response and by $`+0.08`$ nats per token on the short one. DPO's implicit rewards, at the usual $\beta = 0.1$, are $`0.1 \times 0.05 \times 5 = 0.025`$ and $`0.1 \times 0.08 \times 2 = 0.016`$. The long response wins the comparison despite being worse on every token. SimPO at $\beta = 2$ scores them $`2 \times 0.05 = 0.10`$ and $`2 \times 0.08 = 0.16`$. The short response wins, which matches the per-token fact. [Code 14.4.2](#code-1442-kto-simpo-and-orpo-on-toy-numbers) evaluates both.

Length normalization is not free. If the true preference really is "say more, because the extra sentences are useful," an average can under-reward the longer good answer. $\gamma$ and the data have to carry that judgment. Meng et al. (2024) report that the combination, a reference-free average plus a margin, improves length-controlled win rates against DPO on the chat benchmarks in their paper. As with the other variants, that is a reason to include SimPO in a comparison on your own pairs, especially if DPO's outputs are getting longer and the evaluations are not getting better.

## Online and iterative DPO

Every loss above reads a fixed file of pairs. Section 3's derivation assumed something stronger: that the policy being optimized is the policy the reward was defined for, and that the comparisons reflect that reward. A fixed file was sampled from whatever model built the dataset. After enough gradient steps the policy has moved, and the pairs are off-policy.

Online DPO closes the loop. Each round:

1. Sample one or more prompts. Generate two or more responses from the *current* policy.
2. Label the responses. A reward model, a verifier, or an LLM judge (Section 6) picks a winner and a loser. Ties are dropped.
3. Take a DPO step on those fresh pairs. Optionally set the reference model to the policy at the start of the round, so the KL anchor moves forward instead of staying at the original SFT model.
4. Repeat.

The labels are now on-policy, which is the distribution PPO was sampling from, and the update is still the DPO loss rather than a clipped surrogate. Generation is back in the inner loop, so the cheap part of offline DPO is spent. What remains cheap is the absence of a value model and, if the judge is the reward, the absence of a separately trained reward head.

Iterative DPO is the same idea in larger rounds. Generate a dataset from the current policy, label it, run offline DPO to convergence, and use the result as the next generator. Yuan et al. (2024) do this with the model judging its own outputs, which Section 6 covers as self-rewarding. The Llama 3 post-training recipe uses rejection sampling and DPO in place of PPO, choosing them as more stable and easier to scale (Grattafiori et al. 2024). The pattern across these systems is that "DPO" in a modern report often means a DPO *loss* inside a loop that refreshes the data, not a single offline pass.

The failure mode that remains is the judge. Online DPO will cheerfully fit whatever the labeler rewards, including length, sycophancy, and the quirks of an LLM judge (Section 12.6). Refreshing the pairs removes distribution shift. It does not remove reward hacking of the labeler.

## Rejection sampling fine-tuning

The simplest use of a reward is not a new loss at all. Sample $`n`$ responses to a prompt from the current model, score them, throw away all but the best, and run supervised fine-tuning on what remains. This is rejection sampling fine-tuning, also called reward-ranked fine-tuning or RAFT (Dong et al. 2023). Llama 2's early RLHF rounds did exactly this, and only the later rounds added PPO (Touvron et al. 2023; Section 11.8).

The method is best-of-$`n`$ (Section 13.9) used as a data filter rather than as an inference procedure. At inference, best-of-$`n`$ spends extra samples on every user request and does not change the weights. Rejection sampling spends the samples once, during training, and the fine-tuned model then produces a single answer at ordinary cost. The training objective on the kept answers is the SFT loss of Chapter 9. There is no KL term, no preference margin, and no importance ratio. The hope is that the best of $`n`$ draws from a decent model is a better demonstration than the average draw, so imitating it moves the model toward the reward.

It works when the model already puts some probability on good answers, so that $`n`$ samples contain at least one. It cannot discover a behavior the sampler never produces. It also inherits the reward's mistakes directly: if the highest-scoring of $`n`$ answers is long and empty, that is the new demonstration. A KL penalty or a cap on how far the selected answers may be from the sampler reduces that, at which point the method starts to resemble the KL-regularized objective again.

Rejection sampling is the right baseline whenever a paper claims a complicated preference loss is necessary. If SFT on the best of $`n`$ closes most of the gap, the extra machinery has to earn its keep on the remainder. Several of the systems in Section 8 still use it as a stage, sandwiched between SFT and a heavier RL run.

## Which loss uses what

| Method | Data | Reference model | What is constrained |
|---|---|---|---|
| DPO | Chosen and rejected pairs | Yes | Logistic margin of log-ratios, no finite target |
| IPO | Pairs | Yes | Squared error toward a margin of $`1/(2\beta)`$ |
| KTO | Binary labels, one response each | Yes | Desirable above $`z_0`$, undesirable below |
| ORPO | Pairs, and the chosen response as SFT data | No | SFT on the chosen response, plus an odds ratio |
| SimPO | Pairs | No | Logistic margin of length-averaged log-probs, with offset $\gamma$ |
| Online DPO | Pairs regenerated from the current policy | Yes, sometimes refreshed | The DPO margin, on fresh labels |
| Rejection sampling FT | Prompts and a scorer | No | SFT on the winning sample only |

## Code for this section

### Code 14.4.1: DPO does not stop; IPO stops at the target margin

A three-way softmax (chosen, rejected, other), $`\beta = 0.1`$, reference equal to the uniform initial policy. IPO's target gap is $`1/(2\beta) = 5`$ nats. DPO is run with a larger step size so that it is visible, on this convex toy, that the logistic loss is still increasing the gap after it has passed 5.

```python
import torch
import torch.nn.functional as F

def run(kind, steps, lr, beta=0.1):
    logits = torch.zeros(3, requires_grad=True)
    ref = F.log_softmax(logits.detach(), dim=0)
    opt = torch.optim.SGD([logits], lr=lr)
    for _ in range(steps):
        logp = F.log_softmax(logits, dim=0)
        h = (logp[0] - ref[0]) - (logp[1] - ref[1])
        loss = -F.logsigmoid(beta * h) if kind == "dpo" else (h - 1 / (2 * beta)) ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logp = F.log_softmax(logits, dim=0)
        h = (logp[0] - ref[0]) - (logp[1] - ref[1])
        pi = logp.exp()
        print(f"{kind:4s}  pi {[round(p, 4) for p in pi.tolist()]}  h {h.item():.3f}  "
              f"target {1 / (2 * beta):.1f}")

run("dpo", steps=40, lr=2.0)
run("ipo", steps=80, lr=0.05)
```

Output:

```text
dpo   pi [0.964, 0.0013, 0.0348]  h 6.644  target 5.0
ipo   pi [0.9184, 0.0062, 0.0754]  h 5.000  target 5.0
```

### Code 14.4.2: KTO, SimPO, and ORPO on toy numbers

```python
import math
import torch

beta, z0 = 1.0, 0.02
r_des = math.log(0.40) - math.log(0.25)
r_und = math.log(0.15) - math.log(0.30)
v_des = torch.sigmoid(torch.tensor(beta * (r_des - z0)))
v_und = torch.sigmoid(torch.tensor(beta * (z0 - r_und)))
print(f"KTO  r_des {r_des:.3f} loss {1 - v_des:.3f}   r_und {r_und:.3f} loss {1 - v_und:.3f}")

# Per-token log-ratio +0.05 on a 5-token response, +0.08 on a 2-token one.
dpo_w, dpo_l = 0.1 * 0.05 * 5, 0.1 * 0.08 * 2
sim_w, sim_l = 2.0 * 0.05, 2.0 * 0.08
print(f"DPO implicit rewards  long {dpo_w:.3f} short {dpo_l:.3f}  (long wins)")
print(f"SimPO rewards         long {sim_w:.3f} short {sim_l:.3f}  (short wins)")

pw, pl = 1e-8, 1e-9
log_odds = math.log((pw / (1 - pw)) / (pl / (1 - pl)))
log_prob = math.log(pw / pl)
print(f"ORPO log-odds {log_odds:.6f}  log-prob ratio {log_prob:.6f}  "
      f"difference {log_odds - log_prob:.3e}")
```

Output:

```text
KTO  r_des 0.470 loss 0.389   r_und -0.693 loss 0.329
DPO implicit rewards  long 0.025 short 0.016  (long wins)
SimPO rewards         long 0.100 short 0.160  (short wins)
ORPO log-odds 2.302585  log-prob ratio 2.302585  difference 9.000e-09
```

## Key takeaways

- IPO keeps DPO's log-ratio gap but regresses it to a finite target, $`1/(2\beta)`$, so deterministic labels cannot drive the rejected probability to zero.
- KTO replaces the pair with a binary label and a reference point $`z_0`$: desirable responses are pulled above it, undesirable ones below it.
- ORPO adds an odds-ratio penalty to the SFT loss and needs no reference model. On full sequences the odds ratio is numerically a probability ratio; the SFT term is what anchors the policy.
- SimPO compares length-averaged log-probabilities and demands a margin $\gamma$, which removes the sum-of-tokens bonus that longer answers get under DPO.
- Online and iterative DPO regenerate pairs from the current policy so the labels stay on-policy. Rejection sampling fine-tuning simply does SFT on the best of $`n`$ scored samples, and it is the baseline the fancier losses have to beat.

## Further reading

Azar, Mohammad Gheshlaghi, et al. "A General Theoretical Paradigm to Understand Learning from Human Preferences." In *Proceedings of the 27th International Conference on Artificial Intelligence and Statistics*, 2024. https://arxiv.org/abs/2310.12036.

Dong, Hanze, et al. "RAFT: Reward Ranked FineTuning for Generative Foundation Model Alignment." *Transactions on Machine Learning Research*, 2023. https://arxiv.org/abs/2304.06767.

Ethayarajh, Kawin, et al. "KTO: Model Alignment as Prospect Theoretic Optimization." In *Proceedings of the 41st International Conference on Machine Learning*, 2024. https://arxiv.org/abs/2402.01306.

Grattafiori, Aaron, et al. "The Llama 3 Herd of Models." arXiv preprint arXiv:2407.21783, 2024. https://arxiv.org/abs/2407.21783.

Hong, Jiwoo, et al. "ORPO: Monolithic Preference Optimization without Reference Model." In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 2024. https://arxiv.org/abs/2403.07691.

Meng, Yu, et al. "SimPO: Simple Preference Optimization with a Reference-Free Reward." In *Advances in Neural Information Processing Systems 37*, 2024. https://arxiv.org/abs/2405.14734.

Touvron, Hugo, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv preprint arXiv:2307.09288, 2023. https://arxiv.org/abs/2307.09288.

Yuan, Weizhe, et al. "Self-Rewarding Language Models." arXiv preprint arXiv:2401.10020, 2024. https://arxiv.org/abs/2401.10020.
