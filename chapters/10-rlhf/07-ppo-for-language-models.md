# 10.7 PPO for Language Models

Section 9.7 derived Proximal Policy Optimization and implemented it for CartPole: collect rollouts with the current policy, estimate advantages with GAE, and take several epochs of clipped updates. That algorithm is unchanged here. What changes when the policy is a language model with billions of parameters, the environment is text generation, and the reward comes from another large model? This section covers only those differences: the four models involved, how rollouts are generated, how the KL penalty of Section 6 becomes a per-token reward, how advantages and losses are computed over tokens, the practical details that make training stable, and what it all costs in memory and compute. It ends with the chapter's third code lab, a minimal PPO implementation for a small language model.

## What stays the same

As a reminder, PPO from Section 9.7 has four ingredients:

- the probability ratio between the current policy and the policy that collected the data, $`\rho_t(\theta) = \pi_{\theta}(a_t \mid s_t) / \pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)`$;
- the clipped surrogate objective, $`L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t[\min(\rho_t \hat{A}_t,\ \mathrm{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon) \hat{A}_t)]`$;
- advantages from generalized advantage estimation, $`\hat{A}_t = \sum_{l \ge 0} (\gamma \lambda)^l \delta_{t+l}`$ with $`\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)`$;
- a loop that collects a batch of rollouts, computes advantages once, and then runs a few epochs of minibatch updates.

In the token-level MDP of Section 2, a time step is a token: $`s_t = (x, y_{\lt t})`$ and $`a_t = y_t`$. Every quantity above is computed per response token. Everything else in this section is about how to feed this algorithm.

## Four models

PPO for language models involves four networks, two trained and two frozen.

| Model | Role | Trained? | Typical initialization | Output per response |
|---|---|---|---|---|
| Policy (actor) $`\pi_{\theta}`$ | Generates responses; the model we want | Yes | SFT model | Log-probability of each token |
| Value model (critic) $`V_{\psi}`$ | Estimates expected future reward from each prefix | Yes | Reward model | A value at each token |
| Reward model $`r_{\phi}`$ | Scores complete responses (Section 5) | No | (Trained in Stage 2) | One scalar |
| Reference model $`\pi_{\mathrm{ref}}`$ | Anchor for the KL penalty (Section 6) | No | SFT model | Log-probability of each token |

```mermaid
flowchart LR
    P["prompts"] --> A["Policy π_θ<br/>(trained)"]
    A -- "sampled responses" --> RM["Reward model r_φ<br/>(frozen)"]
    A -- "responses" --> REF["Reference π_ref<br/>(frozen)"]
    A -- "responses" --> V["Value model V_ψ<br/>(trained)"]
    RM -- "score at last token" --> SH["shaped per-token rewards"]
    REF -- "log π_ref per token" --> SH
    A -- "log π_θ per token" --> SH
    SH --> GAE["GAE advantages and returns"]
    V -- "V(s_t) per token" --> GAE
    GAE --> U["clipped PPO update<br/>of policy and value model"]
    U --> A
```

*Figure 10.7.1. The four models of PPO-based RLHF and the data flowing between them in one iteration.*

The **value model** is the one genuinely new component relative to the REINFORCE lab of Section 2. (Chapter 9 wrote its parameters as $\phi$; here $\phi$ already names the reward model $`r_{\phi}`$, so the value model gets $\psi$.) It is a transformer with a scalar head at *every* position, not just the last: $`V_{\psi}(s_t)`$ is read from the hidden state at the position that predicts $`y_t`$, and estimates the reward the response will end up with, given the prompt and the tokens so far. It is usually initialized from the reward model, which already maps text to a reward-like scalar (Ouyang et al. 2022; Huang et al. 2024). Some implementations instead put a value head on the policy's own backbone, which saves memory. Stiennon et al. (2020) used a value network with completely separate parameters, because updates to a shared value head could partially destroy the pretrained policy early in training.

## Rollouts

A rollout in RLHF is a batch of prompts and the responses the current policy generates for them. Collecting one involves:

1. **Generation.** Sample one response per prompt from $`\pi_{\theta}`$, token by token, at temperature 1 and without top-$k$ or top-$p$ truncation, up to a maximum length (Section 2).
2. **Scoring.** Run four forward passes over the prompt-plus-response sequences, all without gradients: the policy, for $`\log \pi_{\theta_{\mathrm{old}}}(y_t \mid s_t)`$, which PPO's ratio needs; the reference model, for $`\log \pi_{\mathrm{ref}}(y_t \mid s_t)`$; the value model, for $`V(s_t)`$; and the reward model, for the scalar score $`r_{\phi}(x, y)`$.

**Generation often dominates training time.** Scoring is a single parallel forward pass over each sequence, like training. Generation is autoregressive: one full forward pass of the policy per generated token, each limited by memory bandwidth rather than compute (Section 13.3). For long responses, generation can take most of each iteration's wall-clock time. Production systems therefore generate with optimized inference engines, such as vLLM with its paged KV cache (Section 13.4), and copy the latest policy weights into the engine after each update; frameworks such as OpenRLHF and HybridFlow (veRL) are organized around this split between a generation engine and a training engine (Hu et al. 2024; Sheng et al. 2025).

The reward model may use a different tokenizer from the policy, or expect a different format. The simplest robust approach, used in this section's code, is to decode the policy's tokens to text and re-tokenize for the reward model.

## Token-level reward shaping

Section 6 wrote the RLHF objective as an expected reward $`r_{\phi}(x, y) - \beta \log(\pi_{\theta}(y \mid x) / \pi_{\mathrm{ref}}(y \mid x))`$ and showed that the log-ratio is a sum over tokens. PPO for language models distributes this reward over the tokens of the response. The KL part is charged at every token, and the reward model's score is added at the last token:

```math
r_t = -\beta \log \frac{\pi_{\theta}(y_t \mid x, y_{\lt t})}{\pi_{\mathrm{ref}}(y_t \mid x, y_{\lt t})} + \mathbf{1}[t = T]\, r_{\phi}(x, y).
```

Here $T$ is the index of the last response token, the EOS token if the response ended, and $\beta$ is the KL coefficient (fixed, or adapted as in Section 6). The log-probabilities in the penalty are those recorded at rollout time, so within one iteration $`r_t`$ is a fixed number, like the rewards from a game environment in Chapter 9. With $\gamma = 1$, the sum of the $`r_t`$ over a response equals the sequence-level penalized reward of Section 6 exactly. InstructGPT describes this as adding "a per-token KL penalty from the SFT model at each token to mitigate over-optimization of the reward model" (Ouyang et al. 2022).

Why distribute the penalty rather than subtract it all at the end? The sum is the same, but the per-token form gives the value model and GAE a dense signal: a token that the policy makes much more likely than the reference would is penalized right where it occurs, rather than being blamed together with every other token in the response. The *quality* signal, however, still arrives only at the end, and the credit-assignment problem of Section 2 remains.

## Advantages and returns over tokens

With per-token rewards and per-token values in hand, GAE runs backward over each response exactly as in Section 9.7:

```math
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad \hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}, \qquad \hat{R}_t = \hat{A}_t + V(s_t),
```

with $`V(s_{T+1}) = 0`$ and $`\hat{A}_{T+1} = 0`$, since the episode ends after the last token. The returns $`\hat{R}_t`$ are the regression targets for the value model.

Two settings are standard for text. The discount is $\gamma = 1$: the response is short compared with a game episode, and there is no reason to prefer reward earlier in a response. InstructGPT reports that "no discount is applied when estimating the generalized advantage" (Ouyang et al. 2022). The GAE parameter $\lambda$ is commonly 0.95, as in Chapter 9, trading the bias of the value model against the variance of the single terminal reward.

Positions after the end of a response (padding) must be excluded everywhere: they get no reward, no value, no advantage, and no loss. In code, a response mask that is 1 on real response tokens and 0 elsewhere multiplies every per-token quantity, and averages are taken over masked tokens only ([Code 10.7.1](#code-1071-shaped-rewards-gae-and-the-ppo-losses)).

## The update

After the rollout is scored and advantages computed, PPO performs a few epochs of minibatch updates. For each minibatch, it recomputes $`\log \pi_{\theta}(y_t \mid s_t)`$ and $`V_{\psi}(s_t)`$ with gradients, and minimizes

```math
\mathcal{L}(\theta, \psi) = \underbrace{\frac{1}{N} \sum_{i,t} \max\Big( -\hat{A}_{i,t}\, \rho_{i,t}(\theta),\ -\hat{A}_{i,t}\, \mathrm{clip}\big(\rho_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon\big) \Big)}_{\text{clipped policy loss}} + c_v\, \underbrace{\frac{1}{N} \sum_{i,t} \tfrac{1}{2} \big( V_{\psi}(s_{i,t}) - \hat{R}_{i,t} \big)^2}_{\text{value loss}},
```

where the sums run over the response tokens $t$ of each response $i$ in the minibatch, $N$ is the number of such tokens, and $`c_v`$ weights the value loss. This is the clipped objective of Section 9.7 written as a loss to minimize. Many implementations also clip the value prediction to stay within a range of its rollout-time value, as the reference PPO implementations do (Huang et al. 2022). An explicit entropy bonus is usually omitted, because the KL penalty already rewards entropy (Section 6).

Two different constraints are now active, and it is worth keeping them apart:

- The **clip** limits how far one *update* moves the policy from $`\pi_{\theta_{\mathrm{old}}}`$, the policy that generated this rollout. It is PPO's trust region and resets every iteration.
- The **KL penalty** limits how far the policy has moved *in total* from the fixed $`\pi_{\mathrm{ref}}`$. It is part of the reward and accumulates over training.

Putting it together, one iteration of PPO-based RLHF is:

1. Sample a batch of prompts; generate responses with $`\pi_{\theta}`$.
2. Compute $`\log \pi_{\theta_{\mathrm{old}}}`$, $`\log \pi_{\mathrm{ref}}`$, $V$, and $`r_{\phi}`$ for the batch, without gradients.
3. Form the shaped rewards $`r_t`$; compute $`\hat{A}_t`$ and $`\hat{R}_t`$ with GAE; whiten the advantages.
4. For a few epochs, for each minibatch: recompute log-probabilities and values, compute the clipped policy loss and the value loss, and take a gradient step on the policy and value model.
5. Optionally update $\beta$ with the adaptive controller; log reward, KL, response length, clip fraction, and approximate KL.

## LLM-specific practice

PPO has many small implementation details that matter (Huang et al. 2022 counted 37 of them for standard RL benchmarks). RLHF adds its own. Huang et al. (2024) reproduced the summarization results of Stiennon et al. (2020) and documented the details needed to do so; Xu et al. (2024) identified advantage normalization, large batch sizes, and an exponential moving average of the reference model as key factors for PPO's performance in their experiments. The most important are these.

**Reward normalization and whitening.** The reward model's offset is arbitrary and its scale sets the effective $\beta$ (Section 5). Ziegler et al. (2019) normalized the reward model to mean 0 and variance 1 on samples from the initial policy; Stiennon et al. (2020) and InstructGPT set the offset so that reference or demonstration responses scored 0 on average. Llama 2 went further and whitened its reward scores during PPO, after first undoing the reward model's sigmoid, "in order to increase stability and balance properly with the KL penalty term" (Touvron et al. 2023). Whitening the *advantages* within each batch, subtracting the mean and dividing by the standard deviation over all response tokens, is standard, as it is in Chapter 9 (Huang et al. 2024).

**Handling responses that never end.** A response that hits the maximum length without an EOS token is cut off mid-thought. The reward model was trained on complete responses with the reward read at the final token, so its score for a truncated one is unreliable, and a policy may learn to exploit it. A common remedy, the "EOS trick," replaces the reward model's score with a fixed low value, such as $-1$, for any response without an EOS token (Huang et al. 2024, describing the procedure of Ziegler et al. 2019 and Stiennon et al. 2020). This keeps rewards well defined and discourages rambling. It also requires that the SFT model reliably learned to emit EOS in the first place, which is why padding must not be confused with EOS during SFT (Section 8.2).

**Small learning rates.** The policy starts at a good solution and should move gently. Llama 2 used a constant learning rate of $`10^{-6}`$ for PPO (Touvron et al. 2023), and InstructGPT used $`9 \times 10^{-6}`$ for its value function (Ouyang et al. 2022). Gradient clipping (Llama 2 clipped at 1.0) and, in InstructGPT's case, an exponential moving average of the policy weights add further stability.

**Large prompt batches.** Each rollout's reward signal is noisy: one sample per prompt, one scalar per sample. Large batches average out the noise. InstructGPT and Llama 2 both used 512 prompts per PPO iteration, split into minibatches of 64, with a single pass over the batch (Ouyang et al. 2022; Touvron et al. 2023). Ziegler et al. (2019) used four PPO epochs per batch. Few epochs keep the data close to on-policy.

**Determinism.** Dropout must be disabled in all four models. With dropout, the ratio $`\rho_t`$ is not 1 at the start of the first epoch even though the policy has not changed, and the KL penalty becomes noisy (Huang et al. 2024).

**Monitoring.** The quantities worth watching are the mean reward-model score, the KL divergence from the reference, the mean response length (a rising length with rising reward is a warning sign of length exploitation), the fraction of responses that end with EOS, and, from Chapter 9, the approximate KL between old and new policies and the clip fraction. Sampling and reading responses regularly is irreplaceable.

## Memory and compute

The four models make PPO for language models expensive. Consider a 7-billion-parameter policy with mixed-precision training and Adam. Chapter 8 estimated about 16 bytes per trained parameter for weights, gradients, and optimizer states, about 112 GB for the policy alone. If the value model is also a separate 7B model, it needs another 112 GB. The frozen reference and reward models need only their 16-bit weights, 14 GB each. Before any activations or KV cache for generation, that is roughly 250 GB, spread across several GPUs.

Common ways to reduce this:

- **A smaller reward model and value model.** InstructGPT used a 6B reward model and value function even for its 175B policy (Ouyang et al. 2022).
- **Sharing a backbone.** A value head on the policy's backbone removes one large model, at some risk to stability.
- **Parameter-efficient training.** With LoRA (Section 8.6), the policy is the frozen SFT weights plus small adapters, and the reference model is the same weights with the adapters switched off, so one copy of the base weights serves both roles.
- **Offloading and sharding.** Frozen models can be offloaded to CPU memory when not in use, and all models can be sharded across devices (Section 13.7).

The loop itself alternates between two very different workloads: generation, which is latency- and memory-bandwidth-bound and benefits from inference optimizations, and training, which is compute-bound and needs gradients and optimizer states. Systems such as DeepSpeed-Chat switch a single set of GPUs between the two modes (Yao et al. 2023); others place generation and training on separate GPUs and synchronize weights between them (Hu et al. 2024; Sheng et al. 2025). The cost and complexity of this machinery is one of the main motivations for the simpler methods of Chapter 11.

## Lab: PPO on a small language model

The third suggested code lab puts the pieces together ([Code 10.7.2](#code-1072-a-minimal-ppo-loop-for-a-small-language-model)). It uses DistilGPT-2 as a stand-in for an SFT model and the reward model from the Section 5 lab, and runs PPO with per-token KL shaping, GAE with $\gamma = 1$ and $\lambda = 0.95$, advantage whitening, the EOS trick, and two epochs of minibatch updates per iteration. It prints, for each iteration, the mean reward-model score, the summed KL divergence from the reference, the mean response length, the fraction of responses that ended, and the approximate KL and clip fraction of the last update. Libraries such as TRL, OpenRLHF, and veRL implement the same algorithm with many more options and at much larger scale; after writing it once yourself, their configuration options will be easy to map onto this section.

Our three-iteration CPU run of this listing, with the reward model from Section 5, is a useful cautionary tale rather than a success story. In the first iteration, only 2 of the 16 sampled responses (12%) ended with an end-of-sequence token within the 32-token budget, so most of the batch received the fixed no-EOS score of $`-1`$ and the mean score was about $`-0.6`$. In the next two iterations, no response ended, every score was $`-1`$, the whitened advantages carried almost no signal, and the approximate KL and clip fraction of the updates fell toward zero. The summed KL from the reference stayed below about 0.25 nats and was even slightly negative in one iteration, which is possible because the sum of log-ratios over 16 samples is a noisy estimate of the KL divergence, not the divergence itself. The machinery worked (the losses were finite, the clipping and value updates ran, and nothing diverged), but the reward signal was dominated by the EOS trick. The reason is that DistilGPT-2 is not an SFT model: it was never trained to end an assistant turn, so it almost never emits the end-of-sequence token after a single reply. A real SFT model, trained on demonstrations that end with that token, does not have this problem. With a base model as the stand-in, raise `max_new_tokens`, treat the next `"\n\nHuman:"` as the end of the response, or set the no-EOS score closer to the typical reward-model score, and run for more iterations before judging the trends. The general lesson carries over to real RLHF runs: always look at the fraction of responses that end and at the raw scores, not just the mean reward, because a single hand-set rule can silently overwhelm the reward model.

The fourth lab (Section 6) reruns this loop with several values of $\beta$. Watch the three curves together: reward, KL, and length. A run where reward climbs quickly while KL and length climb with it is usually a run that is learning the reward model's shortcuts rather than what people want.

With the full recipe in place, what did it actually achieve? The next section surveys the models built with PPO-based RLHF, from the first GPT-2 experiments to GPT-4 and Llama 2-Chat.

## Code for this section

These listings reuse `generate` and `response_logprobs` from Code 10.2.1 and `RewardModel` from Code 10.5.1.

### Code 10.7.1: Shaped rewards, GAE, and the PPO losses

The core computations of PPO for language models, all operating on tensors of shape (batch, response length) with a response mask. `shaped_rewards` implements the per-token reward of this section, `gae` computes whitened advantages and value targets with $\gamma = 1$, and `ppo_loss` returns the clipped policy loss, the clipped value loss, and two diagnostics.

```python
import torch
import torch.nn as nn
from transformers import AutoModel

def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()

def masked_whiten(x, mask):
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + 1e-8)

class ValueModel(nn.Module):
    """Critic: a transformer with a scalar head at every position."""
    def __init__(self, name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, seqs, attn, P):
        pos = (attn.cumsum(dim=1) - 1).clamp(min=0)
        h = self.backbone(input_ids=seqs, attention_mask=attn, position_ids=pos).last_hidden_state
        return self.head(h[:, P - 1:-1]).squeeze(-1)         # V(s_t) at the position that predicts y_t

def shaped_rewards(logp_old, ref_logp, score, mask, beta):
    """r_t = -beta * log(pi/pi_ref) at every token, plus the RM score at the last token."""
    log_ratio = (logp_old - ref_logp) * mask
    rewards = -beta * log_ratio
    last = mask.sum(dim=1).long() - 1                        # final response token (the EOS, if any)
    rewards[torch.arange(len(score)), last] += score
    return rewards, log_ratio

def gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Per-token advantages (whitened) and returns; masked positions contribute nothing."""
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(B)
    for t in reversed(range(T)):
        cont = mask[:, t + 1] if t + 1 < T else torch.zeros(B)    # 0 after the last real token
        next_value = values[:, t + 1] * cont if t + 1 < T else torch.zeros(B)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * cont * last_gae
        adv[:, t] = last_gae
    adv = adv * mask
    returns = adv + values
    return masked_whiten(adv, mask), returns

def ppo_loss(logp, logp_old, adv, values, values_old, returns, mask, clip=0.2, vclip=0.2):
    ratio = torch.exp(logp - logp_old)
    pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - clip, 1 + clip))
    pg_loss = masked_mean(pg, mask)
    v_clipped = values_old + (values - values_old).clamp(-vclip, vclip)
    vf = 0.5 * torch.max((values - returns) ** 2, (v_clipped - returns) ** 2)
    vf_loss = masked_mean(vf, mask)
    with torch.no_grad():
        stats = {"approx_kl": masked_mean(0.5 * (logp - logp_old) ** 2, mask).item(),
                 "clipfrac": masked_mean(((ratio - 1).abs() > clip).float(), mask).item()}
    return pg_loss, vf_loss, stats
```

### Code 10.7.2: A minimal PPO loop for a small language model

The third suggested code lab. It loads the reward model saved by Code 10.5.2, initializes the value model from it, and runs PPO on a few HH-RLHF-style prompts. Swap in prompts from the HH-RLHF training set (using `split_prompt` from Code 10.4.1) and increase `ITERS` for a real run; pass `beta` values from 0.01 to 0.2 for the fourth lab.

```python
import copy, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0); random.seed(0)
SFT = "distilgpt2"                                   # stands in for your SFT model
beta, lr, BATCH, MINI, EPOCHS, ITERS = 0.05, 1e-6, 16, 4, 2, 20
NO_EOS_SCORE = -1.0                                  # the "EOS trick"

tok = AutoTokenizer.from_pretrained(SFT)             # policy tokenizer: left padding
tok.pad_token, tok.padding_side = tok.eos_token, "left"
rm_tok = AutoTokenizer.from_pretrained(SFT)          # reward-model tokenizer: right padding
rm_tok.pad_token, rm_tok.padding_side, rm_tok.truncation_side = rm_tok.eos_token, "right", "left"

policy = AutoModelForCausalLM.from_pretrained(SFT)                 # actor (trained)
ref = copy.deepcopy(policy).requires_grad_(False)                  # reference (frozen)
rm = RewardModel(SFT)
rm.load_state_dict(torch.load("reward_model.pt"))                  # from Code 10.5.2
rm.requires_grad_(False)                                           # reward model (frozen)
value = ValueModel(SFT)                                            # critic (trained), initialized from the RM
value.backbone.load_state_dict(rm.backbone.state_dict())
value.head.load_state_dict(rm.head.state_dict())
for m in (policy, ref, rm, value):
    m.eval()                                                       # disable dropout everywhere
opt_pi = torch.optim.AdamW(policy.parameters(), lr=lr)
opt_v = torch.optim.AdamW(value.parameters(), lr=10 * lr)

prompts = ["\n\nHuman: How do I make a cup of tea?\n\nAssistant:",
           "\n\nHuman: Can you suggest a name for my cat?\n\nAssistant:",
           "\n\nHuman: What is a good way to learn to code?\n\nAssistant:",
           "\n\nHuman: How can I sleep better?\n\nAssistant:"]

@torch.no_grad()
def rm_score(prompts, responses, ended):
    texts = [p + r + rm_tok.eos_token for p, r in zip(prompts, responses)]
    enc = rm_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    s = rm(enc["input_ids"], enc["attention_mask"])
    return torch.where(ended, s, torch.full_like(s, NO_EOS_SCORE))

for it in range(ITERS):
    # 1. Rollout: generate, then score with all four models (no gradients).
    batch = random.choices(prompts, k=BATCH)
    seqs, attn, mask, P = generate(policy, tok, batch, max_new_tokens=32)
    ended = (seqs[:, P:] == tok.eos_token_id).any(dim=1)
    responses = tok.batch_decode(seqs[:, P:] * mask + tok.eos_token_id * (1 - mask),
                                 skip_special_tokens=True)
    mask = mask.float()
    with torch.no_grad():
        logp_old = response_logprobs(policy, seqs, attn, P)
        ref_logp = response_logprobs(ref, seqs, attn, P)
        values_old = value(seqs, attn, P)
        score = rm_score(batch, responses, ended)
        # 2. Per-token shaped rewards, then GAE.
        rewards, log_ratio = shaped_rewards(logp_old, ref_logp, score, mask, beta)
        adv, returns = gae(rewards, values_old, mask)
    # 3. A few epochs of clipped updates on minibatches of this rollout.
    for _ in range(EPOCHS):
        for idx in torch.randperm(BATCH).split(MINI):
            logp = response_logprobs(policy, seqs[idx], attn[idx], P)
            v = value(seqs[idx], attn[idx], P)
            pg_loss, vf_loss, stats = ppo_loss(logp, logp_old[idx], adv[idx], v, values_old[idx],
                                               returns[idx], mask[idx])
            opt_pi.zero_grad(); opt_v.zero_grad()
            (pg_loss + 0.1 * vf_loss).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
            opt_pi.step(); opt_v.step()
    print(f"iter {it:2d}  score {score.mean():6.3f}  KL {log_ratio.sum(1).mean():6.3f}  "
          f"len {mask.sum(1).mean():5.1f}  ended {ended.float().mean():.2f}  "
          f"approx_kl {stats['approx_kl']:.4f}  clipfrac {stats['clipfrac']:.3f}")
```

## Key takeaways

- PPO for language models is the PPO of Chapter 9 applied per token; what changes is the setup around it.
- Four models are involved: the trained policy and value model, and the frozen reward and reference models; the value model is usually initialized from the reward model.
- Rollouts are generated responses to a batch of prompts; generation is autoregressive and often dominates training time.
- The reward is shaped per token: $`-\beta \log(\pi_{\theta} / \pi_{\mathrm{ref}})`$ at every token, plus the reward model's score at the last token. GAE with $\gamma = 1$ turns it into per-token advantages.
- The clip bounds each update relative to the rollout policy; the KL penalty bounds the total drift from the reference.
- Stable training depends on reward normalization and advantage whitening, handling responses without EOS, small learning rates, large prompt batches, no dropout, and careful monitoring of reward, KL, and length.
- Four large models and an alternating generate-and-train loop make PPO memory-hungry and complex, which motivates the methods of Chapter 11.

## Further reading

Hu, Jian, et al. "OpenRLHF: An Easy-to-Use, Scalable and High-Performance RLHF Framework." arXiv preprint arXiv:2405.11143, 2024. https://arxiv.org/abs/2405.11143.

Huang, Shengyi, et al. "The 37 Implementation Details of Proximal Policy Optimization." *ICLR Blog Track*, 2022. https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.

Huang, Shengyi, et al. "The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization." In *Conference on Language Modeling*, 2024. https://arxiv.org/abs/2403.17031.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Schulman, John, et al. "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347.

Schulman, John, et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv preprint arXiv:1506.02438, 2015. https://arxiv.org/abs/1506.02438.

Sheng, Guangming, et al. "HybridFlow: A Flexible and Efficient RLHF Framework." In *Proceedings of the Twentieth European Conference on Computer Systems*, 2025. https://arxiv.org/abs/2409.19256.

Stiennon, Nisan, et al. "Learning to Summarize from Human Feedback." In *Advances in Neural Information Processing Systems 33*, 2020. https://arxiv.org/abs/2009.01325.

Touvron, Hugo, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." arXiv preprint arXiv:2307.09288, 2023. https://arxiv.org/abs/2307.09288.

Xu, Shusheng, et al. "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study." In *Proceedings of the 41st International Conference on Machine Learning*, 2024. https://arxiv.org/abs/2404.10719.

Yao, Zhewei, et al. "DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales." arXiv preprint arXiv:2308.01320, 2023. https://arxiv.org/abs/2308.01320.

Ziegler, Daniel M., et al. "Fine-Tuning Language Models from Human Preferences." arXiv preprint arXiv:1909.08593, 2019. https://arxiv.org/abs/1909.08593.
