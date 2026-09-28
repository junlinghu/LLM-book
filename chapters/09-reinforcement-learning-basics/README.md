# Chapter 9: Reinforcement Learning Basics

Pretraining and supervised fine-tuning teach a model to imitate text. Reinforcement learning (RL) teaches it to act so as to earn reward, even when nobody can show it the perfect answer. This chapter covers the core ideas of RL, from Markov decision processes and value functions to Q-learning and policy gradients, and ends by casting language generation as an RL problem. It gives you the vocabulary and intuition needed for RLHF in Chapter 10 and the newer methods in Chapter 11.

## Learning goals

- Describe the agent-environment loop and formalize a problem as a Markov decision process (MDP).
- Define policies, returns, value functions, and Q-functions, and write the Bellman equations.
- Explain the exploration-exploitation trade-off using multi-armed bandits.
- Compare the main families of methods: dynamic programming, Monte Carlo, temporal-difference learning, and policy gradients.
- Derive the REINFORCE estimator and explain why baselines reduce variance.
- Frame text generation as RL and explain why LLM training uses sparse rewards and a KL penalty to a reference model.

## Outline

### 1. What reinforcement learning is
- Learning from reward instead of labeled examples
- The agent-environment loop: observe a state, take an action, receive a reward and the next state
- How RL differs from supervised learning: delayed feedback, data that depends on the agent's own choices, no "correct" action given
- Classic successes: games (TD-Gammon, Atari, Go) and robotics, and now language models

### 2. Markov decision processes
- States, actions, rewards, and transition probabilities
- The Markov property: the future depends only on the current state and action
- Episodes and terminal states vs. continuing tasks
- The discount factor gamma and why we discount future reward
- Running example: a small gridworld

### 3. Policies, returns, and value functions
- Deterministic vs. stochastic policies
- The return: the discounted sum of future rewards

```math
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
```

- The state-value function V and the action-value function Q
- The Bellman expectation equation and the Bellman optimality equation

```math
V^{\pi}(s) = \mathbb{E}_{a \sim \pi,\, s' \sim P}\left[ r(s,a) + \gamma V^{\pi}(s') \right]
```

- Optimal policies: act greedily with respect to the optimal Q-function

### 4. Bandits and exploration
- The multi-armed bandit: RL with a single state
- The exploration-exploitation trade-off
- **Epsilon-greedy**: explore at random with a small probability
- **Upper confidence bound (UCB)**: prefer actions whose value is still uncertain
- Regret as a measure of how much reward exploration costs
- Why bandits matter later: a single prompt-response pair scored once is a contextual bandit

### 5. Dynamic programming (briefly)
- Planning when the model of the environment is known
- **Policy evaluation** by repeatedly applying the Bellman equation
- **Policy iteration**: evaluate, then improve, then repeat
- **Value iteration**: apply the Bellman optimality update until convergence
- Limits: needs the transition model and a table over all states

### 6. Learning from experience: Monte Carlo and TD
- Monte Carlo methods: estimate values by averaging complete returns
- Temporal-difference (TD) learning: update toward a bootstrapped target after each step
- The bias-variance trade-off between Monte Carlo and TD
- **SARSA**: on-policy TD control
- **Q-learning**: off-policy TD control, learning the greedy policy while exploring

```math
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
```

- On-policy vs. off-policy learning, illustrated by the cliff-walking example

### 7. Function approximation and deep Q-networks (briefly)
- Why tables fail for large or continuous state spaces
- Approximating Q with a neural network
- **DQN**: experience replay and a target network to stabilize training
- The "deadly triad": function approximation, bootstrapping, and off-policy learning
- Why value-based methods are awkward for LLMs: the action space is the whole vocabulary at every step

### 8. Policy gradient methods
- Optimizing the policy directly by gradient ascent on expected return
- The policy gradient theorem and the log-derivative trick

```math
\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}}\left[ \sum_t \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\, G_t \right]
```

- **REINFORCE**: a Monte Carlo policy gradient
- High variance and how a **baseline** reduces it without adding bias
- **Actor-critic** methods and the **advantage** function: how much better an action was than expected
- Stability problems with large policy steps, which motivate trust regions and PPO (covered in detail in Chapter 10, with GAE and the clipped objective)

### 9. Language generation as RL
- The token-level MDP: the state is the prompt plus the tokens so far, the action is the next token, and transitions are deterministic
- The language model as a policy: its softmax output is a stochastic policy over the vocabulary
- Sparse, sequence-level reward: one score for the whole response, from a reward model, a human, or a checker
- Two views: the whole response as one action (a contextual bandit) vs. each token as an action (an MDP)
- Credit assignment: which tokens deserve credit for a good or bad reward
- Why we add a KL penalty to a reference model: keep outputs fluent, avoid exploiting the reward, and preserve what pretraining and SFT learned
- Preview: Chapter 10 builds reward models and PPO on this framing; Chapter 11 covers REINFORCE-style methods (RLOO, GRPO), DPO, and RL with verifiable rewards

## Suggested code labs

1. **Bandit exploration.** Implement a 10-armed bandit testbed and compare greedy, epsilon-greedy (several epsilon values), and UCB. Plot average reward and percentage of optimal actions over time.
2. **Gridworld value iteration.** Write a small gridworld MDP from scratch, solve it with value iteration and policy iteration, and visualize the value function and greedy policy. Vary gamma and see how the policy changes.
3. **Q-learning vs. SARSA on CliffWalking.** Implement tabular Q-learning and SARSA on Gymnasium's `CliffWalking-v1` environment. Compare the learned paths and the reward per episode during training, and explain why SARSA learns the safer route.
4. **REINFORCE on CartPole.** Implement REINFORCE with a small policy network on Gymnasium's `CartPole-v1`, with and without a learned value baseline. Compare learning curves and the variance of the gradient estimates across random seeds.
5. **(Optional) A small DQN.** Train a DQN on CartPole with experience replay and a target network. Turn each component off in turn and observe the effect on stability.
6. **A language model as a policy.** Take a tiny pretrained language model (for example a small GPT-2 variant) and fine-tune it with REINFORCE toward a simple, checkable reward, such as including a target word or producing positive sentiment by a small classifier. Add a KL penalty to the original model and compare outputs with and without it to see reward hacking firsthand.

## Key takeaways

- RL learns from reward rather than labels, which lets a model improve beyond its demonstrations.
- MDPs, returns, value functions, and the Bellman equations are the shared language of all RL methods.
- Exploration matters: an agent only learns about actions it tries.
- Value-based methods (Q-learning, DQN) and policy gradient methods (REINFORCE, actor-critic) are the two main families; LLM training relies mostly on policy gradients.
- Text generation is an RL problem with a huge action space, deterministic transitions, and sparse rewards, which is why RLHF adds baselines, advantages, and a KL penalty.

## Further reading

Auer, Peter, et al. "Finite-Time Analysis of the Multiarmed Bandit Problem." *Machine Learning* 47 (2002): 235–256. https://doi.org/10.1023/A:1013689704352.

Mnih, Volodymyr, et al. "Human-Level Control through Deep Reinforcement Learning." *Nature* 518 (2015): 529–533. https://doi.org/10.1038/nature14236.

Schulman, John, et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *arXiv preprint arXiv:1506.02438*, 2015. https://arxiv.org/abs/1506.02438.

Sutton, Richard S., et al. "Policy Gradient Methods for Reinforcement Learning with Function Approximation." In *Advances in Neural Information Processing Systems 12*, 2000. https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html.

Sutton, Richard S., et al. *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press, 2018. http://incompleteideas.net/book/the-book-2nd.html.

Watkins, Christopher J. C. H., et al. "Q-Learning." *Machine Learning* 8 (1992): 279–292. https://doi.org/10.1007/BF00992698.

Williams, Ronald J. "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning* 8 (1992): 229–256. https://doi.org/10.1007/BF00992696.
