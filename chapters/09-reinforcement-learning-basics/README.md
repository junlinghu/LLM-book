# Chapter 9: Reinforcement Learning Basics

Pretraining and supervised fine-tuning teach a model to imitate text. Reinforcement learning (RL) teaches it to act so as to earn reward, even when nobody can show it the perfect answer. This chapter covers the core RL methods on their own terms, without language models: Markov decision processes and value functions, exploration, Monte Carlo and temporal-difference learning, and then, at greater length, policy gradient methods up to Proximal Policy Optimization (PPO), the algorithm behind classic RLHF. Chapter 10 applies these tools to large language models.

## Learning goals

- Describe the agent-environment loop and formalize a problem as a Markov decision process (MDP).
- Define policies, returns, and value functions, and explain the Bellman equation.
- Explain the exploration-exploitation trade-off using multi-armed bandits.
- Compare the main families of methods: Monte Carlo, temporal-difference learning, and policy gradients.
- Derive the REINFORCE estimator and explain how baselines, advantages, and actor-critic methods reduce variance.
- Explain PPO in detail (the probability ratio, the clipped objective, GAE, the value and entropy terms, and the training loop) and implement it from scratch.

## Outline

### 1. What reinforcement learning is
- Learning from reward instead of labeled examples
- The agent-environment loop: observe a state, take an action, receive a reward and the next state
- How RL differs from supervised learning: delayed feedback, data that depends on the agent's own choices, no "correct" action given
- Classic successes: games (TD-Gammon, Atari, Go) and robotics, and now language models

### 2. MDPs, policies, and value functions
- An MDP: states, actions, rewards, and transition probabilities, where the next state depends only on the current state and action
- A policy maps states to (probabilities of) actions
- The return is the discounted sum of future rewards; the discount factor gamma trades near-term against long-term reward

```math
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
```

- The value function V(s) is the expected return from a state; the action-value function Q(s, a) is the expected return after taking an action
- The Bellman equation: a state's value is the immediate reward plus the discounted value of the next state

```math
V^{\pi}(s) = \mathbb{E}_{a \sim \pi,\, s' \sim P}\left[ r(s,a) + \gamma V^{\pi}(s') \right]
```

- Running example: a small gridworld

### 3. Bandits and exploration
- The multi-armed bandit: RL with a single state
- The exploration-exploitation trade-off

### 4. Learning from experience: Monte Carlo and TD
- Monte Carlo methods: estimate values by averaging complete returns
- Temporal-difference (TD) learning: update toward a bootstrapped target after each step
- The bias-variance trade-off between Monte Carlo and TD
- **SARSA**: on-policy TD control
- **Q-learning**: off-policy TD control, learning the greedy policy while exploring

```math
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
```

- On-policy vs. off-policy learning, illustrated by the cliff-walking example

### 5. Function approximation and deep Q-networks (briefly)
- Why tables fail for large or continuous state spaces
- Approximating Q with a neural network
- **DQN**: experience replay and a target network to stabilize training
- The "deadly triad": function approximation, bootstrapping, and off-policy learning
- Why value-based methods are awkward for LLMs: the action space is the whole vocabulary at every step

### 6. Policy gradient methods
- Optimizing a parameterized policy directly by gradient ascent on expected return; works naturally with stochastic policies and large or continuous action spaces
- The policy gradient theorem and the log-derivative trick

```math
\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}}\left[ \sum_t \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\, G_t \right]
```

- **REINFORCE**: a Monte Carlo policy gradient; increase the log-probability of actions in proportion to the return that followed
- High variance, and how a **baseline** reduces it without adding bias
- The **advantage** function: how much better an action was than expected

```math
A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)
```

- **Actor-critic** methods: learn a value function (the critic) alongside the policy (the actor) and use it to estimate advantages
- On-policy data and its cost: every update needs fresh samples from the current policy
- The step-size problem: a single large update can wreck the policy, and a bad policy then collects bad data

### 7. Proximal Policy Optimization (PPO)
- Trust regions: limit how far each update moves the policy; **TRPO** enforces a KL constraint, but needs second-order optimization
- The probability ratio between the new and old policies, which lets one batch of data be reused for several updates

```math
\rho_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}
```

- The **clipped surrogate objective**: take the pessimistic minimum so there is no gain from moving the ratio outside a small range

```math
L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\left[ \min\left( \rho_t(\theta)\, \hat{A}_t,\ \mathrm{clip}\left(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon\right) \hat{A}_t \right) \right]
```

- **Generalized advantage estimation (GAE)**: an exponentially weighted sum of TD errors, with lambda trading bias against variance

```math
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad \hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l\, \delta_{t+l}
```

- The full loss: clipped policy loss, a value-function (critic) loss, and an entropy bonus that keeps exploring
- The training loop: collect rollouts with the current policy, compute advantages and returns, run several epochs of minibatch updates, repeat
- The KL-penalty variant of PPO, with an adaptive coefficient instead of clipping
- Practical tips: advantage normalization, observation and reward scaling, gradient clipping, learning-rate annealing, early stopping on approximate KL, and monitoring the clip fraction; typical settings (for example epsilon = 0.2, lambda = 0.95)
- Where this leads: Chapter 10 applies PPO to language models, and Chapter 11 covers simpler, critic-free alternatives

## Suggested code labs

1. **Bandit exploration.** Implement a 10-armed bandit testbed and compare a purely greedy agent with agents that explore at random a small fraction of the time (several exploration rates). Plot average reward and percentage of optimal actions over time.
2. **Gridworld: Monte Carlo vs. TD prediction.** Write a small gridworld MDP from scratch and estimate the value function of a fixed random policy with Monte Carlo and with TD(0). Compare their errors against a reference from a very long Monte Carlo run as the number of episodes grows, and vary gamma to see how the values change.
3. **Q-learning vs. SARSA on CliffWalking.** Implement tabular Q-learning and SARSA on Gymnasium's `CliffWalking-v1` environment. Compare the learned paths and the reward per episode during training, and explain why SARSA learns the safer route.
4. **REINFORCE on CartPole.** Implement REINFORCE with a small policy network on Gymnasium's `CartPole-v1`, with and without a learned value baseline. Compare learning curves and the variance of the gradient estimates across random seeds.
5. **PPO from scratch.** Extend your REINFORCE-with-baseline agent from Lab 4 into PPO: add the probability ratio, the clipped objective, GAE, the value loss, the entropy bonus, and minibatch epochs. Solve `CartPole-v1`, then switch to a Gaussian policy for a continuous-control task such as `Pendulum-v1`. Ablate clipping (none vs. epsilon = 0.2), GAE (lambda = 0, 0.95, 1), and the number of epochs, and track reward, approximate KL, and clip fraction.
6. **(Optional) A small DQN.** Train a DQN on CartPole with experience replay and a target network. Turn each component off in turn and observe the effect on stability.

## Key takeaways

- RL learns from reward rather than labels, which lets a model improve beyond its demonstrations.
- MDPs, returns, value functions, and the Bellman equation are the shared language of all RL methods.
- Exploration matters: an agent only learns about actions it tries.
- Value-based methods (Q-learning, DQN) and policy gradient methods (REINFORCE, actor-critic, PPO) are the two main families; LLM training relies mostly on policy gradients.
- Baselines and advantages cut the variance of policy gradients, and a learned critic supplies them in actor-critic methods.
- PPO keeps each update small with a clipped probability ratio, estimates advantages with GAE, and reuses each batch for several epochs; it is the workhorse that Chapter 10 applies to language models.

## Further reading

Auer, Peter, et al. "Finite-Time Analysis of the Multiarmed Bandit Problem." *Machine Learning* 47 (2002): 235–256. https://doi.org/10.1023/A:1013689704352.

Huang, Shengyi, et al. "The 37 Implementation Details of Proximal Policy Optimization." *ICLR Blog Track*, 2022. https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.

Mnih, Volodymyr, et al. "Human-Level Control through Deep Reinforcement Learning." *Nature* 518 (2015): 529–533. https://doi.org/10.1038/nature14236.

Schulman, John, et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *arXiv preprint arXiv:1506.02438*, 2015. https://arxiv.org/abs/1506.02438.

Schulman, John, et al. "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347*, 2017. https://arxiv.org/abs/1707.06347.

Schulman, John, et al. "Trust Region Policy Optimization." In *Proceedings of the 32nd International Conference on Machine Learning*, 2015. https://arxiv.org/abs/1502.05477.

Sutton, Richard S., et al. "Policy Gradient Methods for Reinforcement Learning with Function Approximation." In *Advances in Neural Information Processing Systems 12*, 2000. https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html.

Sutton, Richard S., et al. *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press, 2018. http://incompleteideas.net/book/the-book-2nd.html.

Watkins, Christopher J. C. H., et al. "Q-Learning." *Machine Learning* 8 (1992): 279–292. https://doi.org/10.1007/BF00992698.

Williams, Ronald J. "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning* 8 (1992): 229–256. https://doi.org/10.1007/BF00992696.
