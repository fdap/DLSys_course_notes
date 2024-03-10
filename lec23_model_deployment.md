# Model Deployment
***

## Model deployment overview
- Model deployment considerations:
    1. Application environment may bring restrictions(model size, no-python).
    2. Leverage local hardware acceleration (mobile GPUs, accelerated CPU instructions, NPUs).
    3. Integration with the applications (data preprocessing, post processing).
- Inference engine internals:
    - Many inference engines are structured as computational graph interpreters, allocate memories for intermediate activations, traverse the graph and execute each of the operators.

## Machine learning compilation
1. Program representation
    - Represent the program/optimization of interest, (e.g. dense tensor linear algebra, data structures).
2. Build search space through a set of transformations
    - Cover common optimizations
    - Find ways for domain experts to provide input
3. Effective search
    - Cost models, transferability