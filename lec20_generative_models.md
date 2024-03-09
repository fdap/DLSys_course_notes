# Generative Models
***

## Generative adversarial training(GAN).
- iterative process:
    1. discriminator update:
        1. sample minibatch, get a minibatch.
        2. update $D$ to minimize.
    2. generator update:
        1. sample minibatch.
        2. update $G$.

## Diffusion models
- Diffusion model
    - Learning iterative refinement instead of single step generation.