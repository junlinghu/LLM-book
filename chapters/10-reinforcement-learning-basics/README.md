# Chapter 10: Reinforcement Learning Basics

Pretraining and supervised fine-tuning teach a model to imitate text. Reinforcement learning (RL) teaches it to act so as to earn reward, even when nobody can show it the perfect answer. This chapter covers the core RL methods on their own terms, without language models: Markov decision processes and value functions, exploration, Monte Carlo and temporal-difference learning, and then, at greater length, policy gradient methods up to Proximal Policy Optimization (PPO), the algorithm behind classic RLHF. Chapter 11 applies these tools to large language models.

## Sections

1. **[What Reinforcement Learning Is](01-what-reinforcement-learning-is.md)**

   Learning from reward instead of labels, the agent-environment loop run on CartPole with a random and a hand-written policy, three ways RL differs from supervised learning (evaluative and delayed feedback, and data that depend on the agent's own policy, shown with state-visitation maps in a gridworld), the families of RL methods, and a short history from TD-Gammon and DQN to AlphaGo and RLHF.

2. **[MDPs, Policies, and Value Functions](02-mdps-policies-and-value-functions.md)**

   Markov decision processes, policies, discounted returns, the value functions V and Q, and the Bellman expectation and optimality equations, all made concrete in a 5 × 5 gridworld whose values we compute exactly by policy evaluation and value iteration, with a look at how the discount factor changes what the agent values.

3. **[Bandits and Exploration](03-bandits-and-exploration.md)**

   The multi-armed bandit as RL with a single state, incremental value estimates and the "move toward a target" update that every later method reuses, and a 10-armed testbed comparing greedy, ε-greedy, optimistic, and UCB agents (34.7 versus 80 to 86 percent optimal choices after 1,000 steps).

4. **[Learning from Experience: Monte Carlo and TD](04-monte-carlo-and-td.md)**

   Monte Carlo and TD(0) prediction compared on the gridworld, the bias-variance trade-off between full returns and bootstrapped targets, and SARSA versus Q-learning on CliffWalking, where on-policy learning finds the safe path and off-policy learning finds the optimal one.

5. **[Function Approximation and Deep Q-Networks (Briefly)](05-function-approximation-and-dqn.md)**

   Why tables fail in large state spaces, DQN with experience replay and a target network (and what happens on CartPole when each is removed), the deadly triad demonstrated with a two-state divergence example, and why value-based methods are awkward for language models.

6. **[Policy Gradient Methods](06-policy-gradient-methods.md)**

   The log-derivative trick and the policy gradient theorem, REINFORCE on CartPole with and without a learned baseline (with measured gradient variance), the advantage function and actor-critic methods, and the on-policy and step-size problems that make large updates collapse the policy.

7. **[Proximal Policy Optimization (PPO)](07-proximal-policy-optimization.md)**

   Trust regions and TRPO, the probability ratio and clipped surrogate objective, generalized advantage estimation, the full loss and training loop implemented from scratch, PPO solving CartPole in every seed, ablations of clipping and the GAE λ, the KL-penalty variant, and the practical details that make PPO work.

Figures are in [`figures/`](figures/). Every plot is produced by a Python script in [`code/10-reinforcement-learning-basics/`](../../code/10-reinforcement-learning-basics/), which also holds the chapter's from-scratch code (the gridworld, the bandit testbed, DQN, REINFORCE, and PPO). From that folder, the scripts `fig_s1_loop.py` through `fig_s5_triad.py` run in seconds to a minute each; the neural-network figures first need `run_dqn_experiments.py`, `run_reinforce_experiments.py`, and `run_ppo_experiments.py`, which train all the runs in parallel and take several minutes each on a CPU, after which `fig_s5_dqn.py`, `fig_s6_reinforce.py`, and `fig_s7_ppo.py` draw the figures (requires NumPy, Matplotlib, PyTorch, and Gymnasium). The figure scripts write PNGs into `figures/`.

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
- PPO keeps each update small with a clipped probability ratio, estimates advantages with GAE, and reuses each batch for several epochs; it is the workhorse that Chapter 11 applies to language models.

## Further reading

Auer, Peter, et al. "Finite-Time Analysis of the Multiarmed Bandit Problem." *Machine Learning* 47 (2002): 235–256. https://doi.org/10.1023/A:1013689704352.

Barto, Andrew G., et al. "Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems." *IEEE Transactions on Systems, Man, and Cybernetics* SMC-13, no. 5 (1983): 834–846. https://doi.org/10.1109/TSMC.1983.6313077.

Christiano, Paul, et al. "Deep Reinforcement Learning from Human Preferences." *arXiv preprint arXiv:1706.03741*, 2017. https://arxiv.org/abs/1706.03741.

Huang, Shengyi, et al. "The 37 Implementation Details of Proximal Policy Optimization." *ICLR Blog Track*, 2022. https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.

Lai, T. L., et al. "Asymptotically Efficient Adaptive Allocation Rules." *Advances in Applied Mathematics* 6, no. 1 (1985): 4–22. https://doi.org/10.1016/0196-8858(85)90002-8.

Lin, Long-Ji. "Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching." *Machine Learning* 8 (1992): 293–321. https://doi.org/10.1007/BF00992699.

Mnih, Volodymyr, et al. "Asynchronous Methods for Deep Reinforcement Learning." In *Proceedings of the 33rd International Conference on Machine Learning*, 2016. https://arxiv.org/abs/1602.01783.

Mnih, Volodymyr, et al. "Human-Level Control through Deep Reinforcement Learning." *Nature* 518 (2015): 529–533. https://doi.org/10.1038/nature14236.

Mnih, Volodymyr, et al. "Playing Atari with Deep Reinforcement Learning." *arXiv preprint arXiv:1312.5602*, 2013. https://arxiv.org/abs/1312.5602.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." In *Advances in Neural Information Processing Systems 35*, 2022. https://arxiv.org/abs/2203.02155.

Schulman, John, et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *arXiv preprint arXiv:1506.02438*, 2015. https://arxiv.org/abs/1506.02438.

Schulman, John, et al. "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347*, 2017. https://arxiv.org/abs/1707.06347.

Schulman, John, et al. "Trust Region Policy Optimization." In *Proceedings of the 32nd International Conference on Machine Learning*, 2015. https://arxiv.org/abs/1502.05477.

Silver, David, et al. "Mastering the Game of Go with Deep Neural Networks and Tree Search." *Nature* 529 (2016): 484–489. https://doi.org/10.1038/nature16961.

Sutton, Richard S. "Learning to Predict by the Methods of Temporal Differences." *Machine Learning* 3 (1988): 9–44. https://doi.org/10.1007/BF00115009.

Sutton, Richard S., et al. "Policy Gradient Methods for Reinforcement Learning with Function Approximation." In *Advances in Neural Information Processing Systems 12*, 2000. https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html.

Sutton, Richard S., et al. *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press, 2018. http://incompleteideas.net/book/the-book-2nd.html.

Tesauro, Gerald. "Temporal Difference Learning and TD-Gammon." *Communications of the ACM* 38, no. 3 (1995): 58–68. https://doi.org/10.1145/203330.203343.

Thompson, William R. "On the Likelihood That One Unknown Probability Exceeds Another in View of the Evidence of Two Samples." *Biometrika* 25, no. 3–4 (1933): 285–294. https://doi.org/10.1093/biomet/25.3-4.285.

Tsitsiklis, John N., et al. "An Analysis of Temporal-Difference Learning with Function Approximation." *IEEE Transactions on Automatic Control* 42, no. 5 (1997): 674–690. https://doi.org/10.1109/9.580874.

van Hasselt, Hado, et al. "Deep Reinforcement Learning with Double Q-Learning." In *Proceedings of the AAAI Conference on Artificial Intelligence* 30, no. 1 (2016). https://doi.org/10.1609/aaai.v30i1.10295.

Watkins, Christopher J. C. H., et al. "Q-Learning." *Machine Learning* 8 (1992): 279–292. https://doi.org/10.1007/BF00992698.

Williams, Ronald J. "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning* 8 (1992): 229–256. https://doi.org/10.1007/BF00992696.
