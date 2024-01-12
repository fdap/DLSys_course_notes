# Propagation in Manual Neural Network 

## From linear to nonlinear hypothesis classes.
> the importance of introducing the nonlinear activation function.

When meeting the nonlinear classification boundaries, the linear hypothesis function fails. Then we can apply the linear classifier to some(potentially higher-dimensional) features of the data
$$h_\theta(x) = \theta^T\phi(x), \theta \in \mathbb{R}^{d \times k}, \phi : \mathbb{R}^n \to \mathbb{R}^d$$

> how can we create the feature function $\phi$?
1. Through manual engineering of features relevant to the problem (the "old" way of doing machine learning)
2. In a way that itself is learned from data (the “new” way of doing ML). $$\phi(x) = \sigma(W^Tx), W \in \mathbb{R}^{n \times d}, \sigma: \mathbb{R}^n \to \mathbb{R}^d$$

***
## Neural Network
> the definition of neural network

A neural network refers to a particular type of hypothesis class, consisting of **multiple, parameterized differentiable functions (a.k.a. “layers”)** composed together in any manner to form the output.

> fully-connected deep networks

A more generic form of a L-layer neural network, a.k.a. "Multi-Layer Perception(MLP)"
$$Z_{i+1} = \sigma_i(Z_iW_i), i = 1, ..., L \\
Z_1 = X \\ 
h_\theta(X) = Z_{L+1} \\
Z_i \in \mathbb{R}^{m\times n_i}, W_i \in \mathbb{R}^{n_i \times n_{i+1}}$$
with nonlinearities $\sigma_i: \mathbb{R} \to \mathbb{R}$ applied elementwise, and parameters $\theta = \{W_1, ..., W_L\}$

***
## Backpropagation
> backpropagation in general

The loss function is defined as $l(Z_{L+1}, y)$, for $W_i$, we have
$$\frac{\partial l(Z_{L+1}, y)}{\partial W_i} = \frac{\partial l(Z_{L+1}, y)}{\partial Z_{L+1}} \cdot \frac{\partial Z_{L+1}}{\partial Z_{L}} \cdot ... \cdot \frac{\partial Z_{i+2}}{\partial Z_{i+1}} \cdot \frac{\partial Z_{i+1}}{\partial W_{i}}$$
we note
$$G_{i+1} =  \frac{\partial l(Z_{i+1}, y)}{\partial W_i}$$
then we have
$$G_i = G_{i+1} \cdot \frac{\partial Z_{i+1}}{\partial Z_{i}} = G_{i+1} \cdot \frac{\partial \sigma_i(Z_iW_i)}{\partial Z_iW_i} \cdot \frac{\partial Z_i W_i}{\partial Z_i} = G_{i+1} \cdot \sigma^{'}(Z_iW_i)\cdot W_i.$$

Consider about the shape of $G_i \in \mathbb{R}^{m \times n_i}$, 
$$\frac{\partial l(Z_{L+1}, y)}{\partial W_i} = G_{i+1} \cdot \frac{\partial \sigma_i(Z_iW_i)}{\partial Z_iW_i} \cdot \frac{\partial Z_i W_i}{\partial W_i} = G_{i+1} \cdot \sigma^{'}(Z_iW_i)\cdot Z_i  \\
\Rightarrow \nabla_{ W_i} l(Z_{L+1}, y) = Z_i^T(G_{i+1} \circ \sigma^{'}(Z_iW_i))$$

> Forward and backward passes

- **Forward passes** 
    1. Initialize: $Z_1 = X$
    2. Iterate: $Z_{i+1} = \sigma_i(Z_iW_i), i =1,...,L$

- **Backward passes**
    1. Initialize:  $G_{L+1} = \nabla_{Z_{L+1}}l(Z_{L+1},y) = S - I_y, \ S = \text{softmax}(Z_{L+1})$
    2. Iterate: $G_i = (G_{i+1} \circ \sigma^{'}(Z_iW_i))W_i^T, i=1,...,L$
    3. $\nabla_{ W_i} l(Z_{L+1}, y) = Z_i^T(G_{i+1} \circ \sigma^{'}(Z_iW_i))$

- “Backpropagation” is just chain rule + intelligent caching of intermediate results 