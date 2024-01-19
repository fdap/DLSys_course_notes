# Neural Network Library Abstraction
***

## Programming Abstractions

1. case studies:
    1. **forward and backward layer interface**:
        - defines the forward computation and backward (gradient) operations
        - Example framework: Caffe 1.0 
    2. **computational graph and declarative programming**:
        - First declare the computational graph, Then execute the graph by feeding input value.
        - Example framework: Tensorflow 1.0 
    3. **imperative automatic differentiation**:
        - Executes computation as we construct the computational graph, allow easy mixing of python control flow and construction
        - Example framework: PyTorch (needle:)


## High level modular library components
1. elements of machine learning
    1. the hypothesis class:
    2. the loss function:
    3. an optimization method:
2. Deep learning is modular **in nature**.
