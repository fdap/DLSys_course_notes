# Softmax Regression
***

> Three ingredients of a machine learning algorithm.
1. **The hypothesis class**: the "program structure", parameterized via a set parameters, that describes how we map inputs (e.g., images of digits) to outputs(e.g., class labels, or probabilities of different class labels)
2. **The loss function**: a function that specifies how well a given hypothesis (i.e., a choice of parameters) performs on the task of interest.
3. **An optimization method**: a procedure for determining a set of parameters taht minimize the sum of losses over the training set.

> As for the softmax regression solving the multi-class classification.
- training data: $x^{i} \in \mathbb{R}^n, y^{i} \in \{1,..,k\}$ for $i = 1,...,m$, while the n refers to the dimensionality of the input data, k denotes the number of classes and the m represents the number of points in the training dataset.
- *hypothesis function* maps inputs $x \in \mathbb{R}^n$ to $k$-dimensional vectors
$$h: \mathbb{R}^n \to \mathbb{R}^k.$$

- A linear hypothesis function uses a linear operator (i.e. matrix multiplication) for this transformation 
$$h_\theta(x) = \theta^{T}x, \theta \in \mathbb{R}^{n \times k}$$ 
- *loss function*
    - loss function #1: classification error: just a bool value, bad choice because it is not differentiable.
    - loss function #2: softmax / cross-entropy loss. The $z_i = p(\text{label} = i) = \frac{exp(h_i(x))}{\sum_{k}^{j=1}exp(h_j(x))} \Leftrightarrow z \equiv \text{softmax}(h(x))$. $$l_{ce}(h(x),y) = -\text{log}p(\text{label}=y) = -h_y(x) + \text{log}\sum^{k}_{j=1}\text{exp}((h_j(x)))$$

- *The softmax regression optimization problem* 
$$\mathop{\text{minimize}}\limits_{\theta} \frac{1}{m} \sum^{m}_{i=1}l_{ce}(\theta^Tx^{(i)}, y^{(i)})$$

- gradient descent 
    - stochastic gradient descent: take many gradient steps each based upon a minibatch (small partition of the data)
    - $$\frac{\delta}{\delta \theta}l_{ce}(\theta^Tx, y) = \frac{\delta l_{ce}(\theta^Tx, y)}{\delta\theta^Tx} \frac{\delta\theta^Tx}{\delta\theta} = (z-e_y)(x) ,\text{where} \ z = \text{softmax}(\theta^Tx)$$

- *softmax regression algorithm*
    1. iterate over minibatches $X \in \mathbb{R}^{B\times n}, y \in \{1,...,k\}^{B}$ of training set
    2. "matrix batch" form of the loss, $$\Delta_\theta l_{ce}(X\theta, y) \in \mathbb{R}^{n \times k} = X^T(Z-I_y), \ Z=\text{softmax}(X\theta)$$

