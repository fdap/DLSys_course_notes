# Training Large Models
***

## Techniques for memory saving.
- sources of memory consumption
    1. Model weights.
    2. Optimizer states.
    3. Intermediate activation values.

- techniques:   
    1. We only need O(1) memory for computing the final output of a N layer deep network by cycling through two buffers.
    2. Because the need to keep intermediate value around (checkpoint) for the gradient steps. Training a $N$-layer neural network would require $O(N)$ memory.
    3. For a $N$ layer neural network, if we checkpoint every $K$ layers, then the memory cost becomes sublinear.

## Parallel and distributed training.

1. Data parallel training.
    - let each worker access $\frac{B}{k}$ fraction of the minibatch, and run gradient computation then sum up all gradients together. Every worker runs the same replica of the model.
    - Different ways:
        1. data parallel training via allreduce.
        2. data parallel training via parameter server.
    - Many opportunities to continue computation while sending data over the network.

2. Model parallel training.
    - Definition: maps parts of the computation graph to workers. Partition the graph, put send/recv pairs in the boundary.

3. Tensor Parallelism.
    - Partitions tensor data across devices.
    - Tensor parallel by allgather:
        1. $X \in \mathbb{R}^{b \times n\_dims}$, $W \in \mathbb{R}^{n\_dims \times d}$
        2. $A = XW \in \mathbb{R}^{b \times d}$.
        3. split the $W$ into $k$ $W^{'} \in \mathbb{R}^{n\_dims \times d/k}$, each $W^{'}$ is calculated on each GPU. Then $A^{'} = XW' \in \mathbb{R}^{b \times d/k}$.
        4. allgather the k $W'$ and get the final $W$.
    -Tensor parallel by allreduce:
        1. the $XA$ is splited into k $X'' \in \mathbb{R}^{b \times n\_dims/k}, $the $W$ is splited into $k$ $W'' \in \mathbb{R}^{n\_dims/k \times d}$, each $A'' = X^{''}W^{''} \in \mathbb{R}^{b \times d}$ is calculated on each GPU.
        2. allreduce the $k^2$ $A''$.

## Advanced parallelization methods:
1. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.
2. Beyond Data and Model Parallelism for Deep Neural Networks.
3. GSPMD: General and Scalable Parallelization for ML Computation Graphs.
4. FSDP: Fully Sharded Data Parallel.