# Lecture 9. Normalization, Dropout, + Implementation

***

## Normalization
1. Initialization matters a lot for training, may leads to disverge, even when trained successfully, the effects/scales present at initialization persist throughout training

2. **layer normalization**: normalize activations at each layer,$$\hat{z}_{i+1} = \sigma(W_{i}^Tz_i + b_i) \\ z_{i+1} = \frac{\hat{z}_{i+1} - \mathbf{E}[\hat{z}_{i+1}]}{(\mathbf{Var}[\hat{z}_{i+1}] + \epsilon)^{\frac{1}{2}}}$$
Also common to add an additional scalar weight and bias to each term (only changes representation).

3. **batch normalization**:is to compute a running average of mean/variance for all features at each layer $\hat{\mu}_{i+1}, \hat{\sigma}_{i+1}$, and at test time normalize by these quantities. $$(z_{i+1})_j = \frac{(\hat{z}_{i+1})_j - (\hat{\mu_{i+1}})_j}{((\hat{\sigma}_{i+1}^2)_j + \epsilon)^{\frac{1}{2}}}.$$


## Regularization







## Interaction of optimization, initialization, normalization, regularization







