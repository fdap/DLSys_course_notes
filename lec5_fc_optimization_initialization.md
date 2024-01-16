# Fully connected networks, optimization, initialization

***

## Fully Connected Networks
- fully connected network, a.k.a. multi-layer perceptron(MLP), is defined by the iteration
$$z_{i+1} = \sigma_i(W_i^Tz_i + b_i),\quad i = 1,...,L \\ h_{\theta}(x) \equiv z_{L+1} \\ z1 \equiv x$$
with parameter $\theta = \{W_{1:L}, b_{1:L}\}$, and where $\sigma_i(x)$ is the nonlinear activation, usually with $\sigma_{L}(x) = x$.

- Key questions for fc networks
    1. the width and the depth of the network.
    2. how to optimize the objective
    3. the weight initialization.
    4. how ensure the network can continue to be trained easily over multiple optimization iterations.

## Optimization
1. gradient descent
2. Newton's method
3. Momentum
4. Nesterov Momentum
5. Adam

## Weight Initialization
1. 
2. 


