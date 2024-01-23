# Lecture 9. Normalization, Dropout, + Implementation

***

## Normalization
1. Initialization matters a lot for training, may leads to disverge, even when trained successfully, the effects/scales present at initialization persist throughout training

2. **layer normalization**: normalize activations at each layer,$$\hat{z}_{i+1} = \sigma(W_{i}^Tz_i + b_i) \\ z_{i+1} = \frac{\hat{z}_{i+1} - \mathbf{E}[\hat{z}_{i+1}]}{(\mathbf{Var}[\hat{z}_{i+1}] + \epsilon)^{\frac{1}{2}}}$$
Also common to add an additional scalar weight and bias to each term (only changes representation).

3. **batch normalization**:is to compute a running average of mean/variance for all features at each layer $\hat{\mu}_{i+1}, \hat{\sigma}_{i+1}$, and at test time normalize by these quantities. $$(z_{i+1})_j = \frac{(\hat{z}_{i+1})_j - (\hat{\mu_{i+1}})_j}{((\hat{\sigma}_{i+1}^2)_j + \epsilon)^{\frac{1}{2}}}.$$


## Regularization
1. Regularization is the process of “limiting the complexity of the function class” in order to ensure that networks will generalize better to new data; typically occurs in two ways in deep learning:
    1. **Implicit regularization** refers to the manner in which our existing algorithms (namely SGD) or architectures already limit functions considered.
    2. **Explicit regularization** refers to modifications made to the network and training procedure explicitly intended to regularize the network.

2. weight decay ($l_2$ regularization): keeping parameters small by adding a loss: $$\mathop{\text{minimize}}\limits_{W_{1:L}} \frac{1}{m} \sum_{i=1}^{m}\ell_{ce}(h_{W_{1:L}}(x^{(i)}), y^{(i)}) + \frac{\lambda}{2}\sum_{i=1}^{L}\parallel W_i \parallel _2^2$$. Results in the gradient descent updates: $$W_i = W_i - \alpha \nabla_{W_i} \ell(h(X), y) - \alpha \lambda W_i = (1-\alpha \lambda)W_i - \alpha \nabla_{W_i} \ell(h(X), y)$$, i.e., at each iteration we **shrink the weights** by a factor $(1-\alpha \lambda)$ before taking the gradient step.

3. 






## Interaction of optimization, initialization, normalization, regularization







