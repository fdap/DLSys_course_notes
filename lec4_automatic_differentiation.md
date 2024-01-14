# Automatic Differentiation
***

## General introduction to different differentiation methods.
- Forward mode automatic differentiation(AD): 
    - define $\dot{v_i} = \frac{\partial v_i}{\partial x_1}$, we can compute the iteratively in the forward topological order of the computational graph.

- Limitation of forward mode AD.
    - For $f:\mathbb{R}^n \to \mathbb{R}^k$, we need $n$ forward AD passes to get the gradient with respect to each input.


## Reverse mode automatic differentiation.
- Reverse mode automatic differentiation(AD):
    - define $\overline{v_i} = \frac{\partial y}{\partial v_i}$, we can compute the $\overline{v_i}$ iteratively in the reverse topological order of the computational graph.
    - for the multiple pathway case: $\overline{v_{i \to j}} = \overline{v_j} \frac{\partial v_j}{\partial v_i}$ for each input output node pair $i$ and $j$. $\overline{v_i} = \sum_{j \in \text{next}(i)} \overline{v_{i \to j}}$

> Reverse mode AD vs Backprop
- Backprop:
    1. Run backward operations the same forward graph.
    2. Used in first generation deep learning.
- Reverse mode AD by extending computational graph:
    1. construct separate graph nodes for adjoints.
    2. Used by modern deep learning frameworks.
