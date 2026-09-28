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

## Outline

### 1. From biological inspiration to the artificial neuron
- A short history: McCulloch-Pitts neurons, Rosenblatt's perceptron (1958), the XOR critique, and the comeback with backpropagation (1986)
- The artificial neuron: inputs, weights, bias, weighted sum, activation
- The perceptron learning rule and what it can learn: linearly separable data only
- Geometric view: the weights define a decision boundary (a hyperplane)
- Neurons as linear models: linear regression (no activation) and logistic regression (sigmoid activation)

### 2. Activation functions
- Why nonlinearity matters: stacking linear layers is still one linear layer
- Step and sigmoid: historical choices, and the problem of saturation (flat regions with near-zero gradient)
- Tanh: zero-centered, but still saturates
- ReLU: cheap and non-saturating for positive inputs; "dead" units
- Smooth modern variants used in transformers: GELU and SiLU/Swish (gated variants such as SwiGLU come back in Chapter 6)
- Plotting each function and its derivative side by side

### 3. The multi-layer perceptron (one hidden layer)
- Layers as matrix multiplications: input layer, hidden layer, output layer
- The forward pass step by step, with shapes written out for every tensor
- Solving XOR with one hidden layer: hidden units learn new features
- Universal approximation (Cybenko 1989, Hornik et al. 1989): one wide hidden layer can approximate any continuous function, but says nothing about how to find the weights or how many units are needed
- Counting parameters, and why going deeper instead of wider is the subject of Chapter 3

### 4. Loss functions: measuring mistakes
- The idea of a loss: one number that tells the model how wrong it is
- Mean squared error (MSE) for regression
- From scores to probabilities: the softmax function and its numerically stable form (subtract the maximum)
- Cross-entropy and negative log-likelihood for classification; binary cross-entropy as the two-class case
- Why cross-entropy instead of MSE for classification: better gradients, probabilistic meaning
- The clean gradient of softmax plus cross-entropy (predicted probability minus the one-hot target):

```math
\frac{\partial L}{\partial z_i} = p_i - y_i
```

- **Connection to LLMs**: next-token prediction is classification over the vocabulary; the pretraining loss in Chapter 7 is exactly this cross-entropy, averaged over positions; perplexity as the exponential of the average loss

### 5. Gradient descent
- Training as optimization: find the weights that minimize the average loss
- The gradient as the direction of steepest increase; step in the opposite direction
- The learning rate: too small is slow, too large diverges; visualizing both on a 1-D and 2-D loss surface
- Batch gradient descent vs. stochastic gradient descent (SGD) vs. minibatches
- Why minibatches: noisy but cheap gradient estimates, and good use of vectorized hardware
- Epochs, iterations, and batch size
- Non-convex losses and local minima (brief); momentum, Adam, and learning-rate schedules are covered in Chapter 3

### 6. Computational graphs and backpropagation
- Representing any computation as a graph of simple operations (add, multiply, tanh, exp, log)
- The chain rule, one node at a time: local derivatives times the upstream gradient
- Forward mode vs. reverse mode automatic differentiation, and why reverse mode (backpropagation) wins when there is one scalar loss and many parameters
- Gradients accumulate when a value is used more than once
- Deriving backpropagation for the one-hidden-layer MLP by hand: gradients for both weight matrices and both bias vectors
- Designing a scalar autograd engine from scratch: a value object that stores its data, its gradient, its parent nodes, and a local backward rule; a topological sort to run backward in the right order
- Common pitfalls: forgetting to zero gradients, in-place updates, and numerical overflow in exp and log

### 7. Vectorization: from scalars to tensors
- Why scalar code is too slow: one Python operation per number
- Rewriting the MLP with matrices: a batch of inputs becomes one matrix multiplication per layer
- Tensors, shapes, and broadcasting (adding a bias vector to every row)
- Backpropagation in matrix form: the gradient of a matrix product and the transpose pattern
- Checking shapes as a debugging habit: every gradient has the same shape as its parameter
- A short note on GPUs: why matrix multiplication dominates the cost of neural networks (and later, LLMs)

### 8. The training loop and generalization
- The anatomy of a training loop: sample a minibatch, forward pass, compute loss, backward pass, update, repeat
- Splitting data into training, validation, and test sets, and why the test set is touched only once
- Learning curves: training loss vs. validation loss over time
- Underfitting and overfitting, and model capacity (hidden-layer width)
- Simple remedies previewed: more data, early stopping, smaller models (weight decay, dropout, and normalization are covered in Chapter 3)
- Initialization in brief: why weights must not all start at zero (symmetry); careful initialization schemes for deep networks are in Chapter 3
- Reproducibility: random seeds and logging

### 9. A first look at PyTorch
- Tensors with `requires_grad`, the dynamic computational graph, and `loss.backward()`
- Mapping our from-scratch pieces to PyTorch: value object to tensor, backward rules to autograd, update step to `torch.optim.SGD`
- Building the same MLP with `nn.Linear`, `nn.ReLU`, and `nn.CrossEntropyLoss` (which takes raw logits and applies log-softmax internally)
- Verifying that our hand-derived and autograd gradients match PyTorch's to within floating-point tolerance
- Where the rest of the book goes from here: deeper networks (Chapter 3), learned embeddings (Chapter 4), and the transformer (Chapter 6)

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
