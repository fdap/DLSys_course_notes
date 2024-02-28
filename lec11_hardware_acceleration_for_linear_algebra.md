# Hardware Acceleration for Linear Algebra

***

## General acceleration techniques
1. **vectorization**: Adding two arrays of length 256
2. **Data layout and strides**: strides can perform transformation/slicing in zero copy way, but memory access becomes not continuous. 
3. **Parallelization**: Executes the computation on multiple threads

## Case study: matrix multiplication
- **vanilla matrix multiplication**: $O(n^3)$
- **register tiled matrix multiplication**: 
- **cacheline aware tiling**: utilize the L1 cache and pre-fetch the one cacheline data. the register tiling can be applied too. 

the key insight is **memory load reuse**.


