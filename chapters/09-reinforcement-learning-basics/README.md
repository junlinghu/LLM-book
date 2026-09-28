# Chapter 9: Reinforcement Learning Basics

Pretraining and supervised fine-tuning teach a model to imitate text. Reinforcement learning (RL) teaches it to act so as to earn reward, even when nobody can show it the perfect answer. This chapter covers the core ideas of RL, from Markov decision processes and value functions to Q-learning, and ends with policy gradients, the family of methods that LLM training builds on. It gives you the vocabulary and intuition needed for Chapter 10, which casts language generation as an RL problem and builds RLHF on top of it, and for the newer methods in Chapter 11.

## Learning goals

- Describe the agent-environment loop and formalize a problem as a Markov decision process (MDP).
- Define policies, returns, value functions, and Q-functions, and write the Bellman equations.
- Explain the exploration-exploitation trade-off using multi-armed bandits.
- Compare the main families of methods: Monte Carlo, temporal-difference learning, and policy gradients.
- Derive the REINFORCE estimator and explain why baselines and advantages reduce variance.

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

### 5. Learning from experience: Monte Carlo and TD
- Monte Carlo methods: estimate values by averaging complete returns
- Temporal-difference (TD) learning: update toward a bootstrapped target after each step
- The bias-variance trade-off between Monte Carlo and TD
- **SARSA**: on-policy TD control
- **Q-learning**: off-policy TD control, learning the greedy policy while exploring

```math
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
```

- On-policy vs. off-policy learning, illustrated by the cliff-walking example

### 6. Function approximation and deep Q-networks (briefly)
- Why tables fail for large or continuous state spaces
- Approximating Q with a neural network
- **DQN**: experience replay and a target network to stabilize training
- The "deadly triad": function approximation, bootstrapping, and off-policy learning
- Why value-based methods are awkward for LLMs: the action space is the whole vocabulary at every step

### 7. Policy gradient methods
- Optimizing the policy directly by gradient ascent on expected return
- The policy gradient theorem and the log-derivative trick

```math
\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}}\left[ \sum_t \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\, G_t \right]
```

- **REINFORCE**: a Monte Carlo policy gradient
- High variance and how a **baseline** reduces it without adding bias
- **Actor-critic** methods and the **advantage** function: how much better an action was than expected
- Stability problems with large policy steps, which motivate trust regions and PPO (covered in detail in Chapter 10, with GAE and the clipped objective)
- Where this leads: Chapter 10 treats a language model as a policy, casts text generation as RL, and applies these policy gradient tools to train LLMs from human feedback

## Suggested code labs

1. **Bandit exploration.** Implement a 10-armed bandit testbed and compare a purely greedy agent with agents that explore at random a small fraction of the time (several exploration rates). Plot average reward and percentage of optimal actions over time.
2. **Gridworld: Monte Carlo vs. TD prediction.** Write a small gridworld MDP from scratch and estimate the value function of a fixed random policy with Monte Carlo and with TD(0). Compare their errors against a reference from a very long Monte Carlo run as the number of episodes grows, and vary gamma to see how the values change.
3. **Q-learning vs. SARSA on CliffWalking.** Implement tabular Q-learning and SARSA on Gymnasium's `CliffWalking-v1` environment. Compare the learned paths and the reward per episode during training, and explain why SARSA learns the safer route.
4. **REINFORCE on CartPole.** Implement REINFORCE with a small policy network on Gymnasium's `CartPole-v1`, with and without a learned value baseline. Compare learning curves and the variance of the gradient estimates across random seeds.
5. **(Optional) A small DQN.** Train a DQN on CartPole with experience replay and a target network. Turn each component off in turn and observe the effect on stability.

## Key takeaways

- RL learns from reward rather than labels, which lets a model improve beyond its demonstrations.
- MDPs, returns, value functions, and the Bellman equations are the shared language of all RL methods.
- Exploration matters: an agent only learns about actions it tries.
- Value-based methods (Q-learning, DQN) and policy gradient methods (REINFORCE, actor-critic) are the two main families; LLM training relies mostly on policy gradients.
- Policy gradients with baselines and advantages are the foundation for Chapter 10, which casts text generation as RL and trains LLMs with RLHF.

## Further reading

Auer, Peter, et al. "Finite-Time Analysis of the Multiarmed Bandit Problem." *Machine Learning* 47 (2002): 235–256. https://doi.org/10.1023/A:1013689704352.

Mnih, Volodymyr, et al. "Human-Level Control through Deep Reinforcement Learning." *Nature* 518 (2015): 529–533. https://doi.org/10.1038/nature14236.

Sutton, Richard S., et al. "Policy Gradient Methods for Reinforcement Learning with Function Approximation." In *Advances in Neural Information Processing Systems 12*, 2000. https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html.

Sutton, Richard S., et al. *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press, 2018. http://incompleteideas.net/book/the-book-2nd.html.

Watkins, Christopher J. C. H., et al. "Q-Learning." *Machine Learning* 8 (1992): 279–292. https://doi.org/10.1007/BF00992698.

Williams, Ronald J. "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning* 8 (1992): 229–256. https://doi.org/10.1007/BF00992696.
