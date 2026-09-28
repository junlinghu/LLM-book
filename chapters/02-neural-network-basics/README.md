# Chapter 2: The Basics of Neural Networks

Every large language model is, at its core, a neural network trained by gradient descent. This chapter builds that foundation from scratch: a single neuron, a network with one hidden layer, a loss that measures mistakes, and backpropagation to fix them. Along the way we write a tiny automatic-differentiation engine of our own and then check it against PyTorch, so that later chapters can use PyTorch with a clear picture of what happens underneath.

## Learning goals

- Describe a neuron as a weighted sum plus bias followed by an activation, and relate it to linear and logistic regression.
- Explain why nonlinear activations are needed and compare the common ones (sigmoid, tanh, ReLU, GELU).
- Build and run the forward pass of a multi-layer perceptron with one hidden layer.
- Choose a loss for regression or classification, and connect softmax cross-entropy to next-token prediction in LLMs.
- Explain gradient descent, stochastic gradient descent, minibatches, and the role of the learning rate.
- Derive backpropagation with the chain rule on a computational graph, and implement it both by hand and as a small autograd engine.
- Write a complete training loop with a train/validation split, and recognize underfitting and overfitting.
- Verify from-scratch gradients with finite differences and with PyTorch autograd.

## Sections

| # | Section | Summary |
|---|---|---|
| 1 | [From Biological Inspiration to the Artificial Neuron](01-the-artificial-neuron.md) | McCulloch-Pitts and Rosenblatt, the neuron as a weighted sum plus bias and activation, the perceptron rule and its failure on XOR, decision hyperplanes, and neurons as linear and logistic regression. |
| 2 | [Activation Functions](02-activation-functions.md) | Why stacked linear layers collapse to one, saturation in step, sigmoid, and tanh, ReLU and dead units, and the smooth GELU and SiLU used in transformers, each plotted with its derivative. |
| 3 | [The Multi-Layer Perceptron](03-multi-layer-perceptron.md) | Layers as matrix multiplications with shapes written out, a hand-computed forward pass through a 2-2-1 network, solving XOR and two moons with hidden features, universal approximation, and parameter counts. |
| 4 | [Loss Functions: Measuring Mistakes](04-loss-functions.md) | MSE for regression, stable softmax, cross-entropy and binary cross-entropy, why cross-entropy beats MSE for classification, the p − y gradient, and next-token prediction and perplexity in LLMs. |
| 5 | [Gradient Descent](05-gradient-descent.md) | Minimizing the average loss by stepping against the gradient, learning rates that are too small or too large on 1-D and 2-D surfaces, batch vs. stochastic vs. minibatch gradients, and epochs and iterations. |
| 6 | [Computational Graphs and Backpropagation](06-computational-graphs-and-backpropagation.md) | Computational graphs, the chain rule node by node, forward vs. reverse mode, gradient accumulation, backprop for the MLP by hand, and an original scalar autograd engine that trains an MLP on XOR and two moons. |
| 7 | [Vectorization: From Scalars to Tensors](07-vectorization.md) | Replacing scalar loops with matrix operations, shapes and broadcasting, matrix-form backprop and the transpose pattern, shape checks, gradient checking with finite differences, and why matrix multiplication dominates cost. |
| 8 | [The Training Loop and Generalization](08-training-loop-and-generalization.md) | The training loop, train/validation/test splits, learning curves, underfitting and overfitting as width grows, early stopping, why weights cannot start at zero, and reproducibility with seeds. |
| 9 | [A First Look at PyTorch](09-a-first-look-at-pytorch.md) | Tensors with requires_grad, the dynamic graph and backward(), mapping each from-scratch piece to PyTorch, rebuilding the MLP with torch.nn, and matching our gradients and loss curve to round-off. |

Figures are in [`figures/`](figures/). Every plot is produced by a Python script in [`figures/src/`](figures/src/), which also holds the chapter's from-scratch code (the scalar autograd engine and the NumPy MLP). To regenerate all figures, run `for f in fig_*.py; do python "$f"; done` from inside `figures/src/` (requires NumPy, Matplotlib, and PyTorch).

## Suggested code labs

1. **Perceptron from scratch in NumPy.** Implement the perceptron learning rule on a 2-D linearly separable dataset, plot the decision boundary as it updates, then show that it fails on XOR.
2. **Activation function gallery.** Plot sigmoid, tanh, ReLU, GELU, and SiLU with their derivatives, and mark the regions where each one saturates.
3. **Build a tiny scalar autograd engine from scratch.** Write your own value class that supports addition, multiplication, power, tanh, exp, and log, records the computational graph, and runs reverse-mode backpropagation through a topological sort. Test it on small expressions whose derivatives you can compute by hand. (This is original code written for this book, not a port of an existing library.)
4. **Train an MLP with your autograd engine.** Use the engine from Lab 3 to build a one-hidden-layer network and train it on XOR and on a small two-moons dataset; plot the loss curve and the learned decision boundary.
5. **Vectorized MLP with manual backprop.** Rewrite the same network in NumPy with matrices, derive and code the backward pass by hand, and train with minibatch SGD. Compare its speed with the scalar version and try several learning rates to see slow convergence and divergence.
6. **Gradient checking with finite differences.** Compare the analytic gradients from Labs 3 and 5 against centered finite differences, report the relative error for each parameter, and deliberately introduce a bug to see the check catch it.
7. **Overfitting on purpose.** Train networks of increasing hidden width on a small, noisy dataset with a train/validation split; plot both loss curves and pick a model with early stopping.
8. **The same model in PyTorch.** Rebuild the MLP with `torch.nn`, train it on the same data with the same initial weights, and confirm that the gradients and the loss curve match your from-scratch versions.

## Key takeaways

- A neuron is a weighted sum plus a bias passed through a nonlinearity; without the nonlinearity, depth adds nothing.
- One hidden layer is enough to solve problems like XOR and, in principle, to approximate any continuous function.
- Cross-entropy on a softmax output is the standard classification loss, and it is the same loss that trains LLMs to predict the next token.
- Gradient descent with minibatches is how every model in this book is trained; the learning rate is the most important knob.
- Backpropagation is the chain rule applied in reverse over a computational graph, and a working autograd engine fits in a page of code.
- Always verify gradients (finite differences or a trusted library) and always watch validation loss, not just training loss.

## Further reading

Baydin, Atılım Güneş, et al. "Automatic Differentiation in Machine Learning: A Survey." *Journal of Machine Learning Research* 18, no. 153 (2018): 1–43. https://arxiv.org/abs/1502.05767.

Cybenko, George. "Approximation by Superpositions of a Sigmoidal Function." *Mathematics of Control, Signals, and Systems* 2, no. 4 (1989): 303–314. https://doi.org/10.1007/BF02551274.

Goodfellow, Ian, et al. *Deep Learning*. Cambridge, MA: MIT Press, 2016. https://www.deeplearningbook.org/.

Hornik, Kurt, et al. "Multilayer Feedforward Networks Are Universal Approximators." *Neural Networks* 2, no. 5 (1989): 359–366. https://doi.org/10.1016/0893-6080(89)90020-8.

LeCun, Yann, et al. "Efficient BackProp." In *Neural Networks: Tricks of the Trade*, 9–50. Lecture Notes in Computer Science 1524. Berlin: Springer, 1998. https://doi.org/10.1007/3-540-49430-8_2.

Nielsen, Michael A. *Neural Networks and Deep Learning*. Determination Press, 2015. http://neuralnetworksanddeeplearning.com/.

Rosenblatt, Frank. "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain." *Psychological Review* 65, no. 6 (1958): 386–408. https://doi.org/10.1037/h0042519.

Rumelhart, David E., et al. "Learning Representations by Back-Propagating Errors." *Nature* 323 (1986): 533–536. https://doi.org/10.1038/323533a0.
