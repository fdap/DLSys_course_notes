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



## Homework 2

### backward computation

The general goal of reverse mode autodifferentiation is to compute the gradient of some downstream function $\ell$ of $f(x,y)$ with respect to $x$ (or $y$).  Written formally, we could write this as trying to compute
$$\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x}.$$
The "incoming backward gradient" is precisely the term $\frac{\partial \ell}{\partial f(x,y)}$, so we want our `gradient()` function to ultimately compute the _product_ between this backward gradient the function's own derivative $\frac{\partial f(x,y)}{\partial x}$.

- `Transpose`: reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, `axes` - tuple)

The derivative of a transposed vector *w.r.t* itself is the identity matrix, but the transpose gets applied to everything after.

$$\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x)} \frac{\partial f(x)}{\partial x} = \text{gradient} \frac{\partial x^T}{\partial x} = \text{gradient}$$


- `Reshape`: reshape an array $x$ to a new shape
$$\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x} = \text{gradient} \frac{\partial f(x,y)}{\partial x}$$

- `BroadcastTo`:
broadcast an array $x$ to a new shape (1 input, `shape` - tuple), $$x\in\mathbb{R}^{...} \to x\in\mathbb{R}^{new \ shape}$$


- `MatMul`: matrix multiplication of the inputs (2 inputs)


- `Summation`: sum of array elements over given axes (1 input, `axes` - tuple)

