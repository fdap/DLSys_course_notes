# Transformers and Attention
***

## The two approaches to time series modeling
1. The RNN “latent state” approach
    - the RNN approach to time series: maintain “latent state” that summarizes all information up until that point.
    - *Pros*: Potentially “infinite” history, compact representation.
    - *Cons*: Long “compute path” between history and current time ⟹ vanishing / exploding gradients, hard to learn.

2. The “direct prediction” approach
    - In contrast, can also directly predict output $y_t$ ($y_t = f_\theta(x_{1:t})$) (just need a function that can make predictions of differently-sized inputs.
    - *Pros*: Often can map from past to current state with shorter compute path.
    - *Cons*: No compact state representation, finite history in practice.
    - One of the most straightforward ways to specify the function $y_t$: (fully) convolutional networks, a.k.a. temporal convolutional networks (TCNs). Despite their simplicity, CNNs have a notable disadvantage for time series prediction: the receptive field of each convolution is usually relatively small ⟹ need deep networks to actually incorporate past information
    
## Self-attention and Transformers

### “Attention” in deep learning
- “Attention” in deep networks generally refers to any mechanism where individual states are weighted and then combined.
- the self-attention operation.
- Transformer Block.


### Transformers applied to time series
- Pros:
    1. Full receptive field within a single layer (i.e., can immediately use past data)
    2. Mixing over time doesn’t increase parameter count (unlike convolutions).
- Cons:
    1. All outputs depend on all inputs (no good e.g., for autoregressive tasks)
    2. No ordering of data (remember that transformers are equivariant to permutations of the sequence).



### Others
- Masked self-attention
- Positional encodings.