# Hardware Acceleration + GPUs
***

## General Aceleration techniques

### GPU programming mode: SIMT
1. Single instruction multiple threads(SIMT)
2. All threads executes the same code, but can take different path.
3. Threads are grouped into blocks. Thread within the same block have shared memory.
4. Blocks are grouped into a launch grid. 


### GPU memory hierarchy
1. Threads have their own registers.
2. All the threads within one block share the shared memory.
3. All the blocks hold the global memory.


### High level takeaways
1. Launch thread grid and blocks
2. Cooperatively fetch common to shared memory to increase reuse. 


## Case study: matrix multiplication on GPU

1. Thread-level: register tiling.
2. Block-level: shared memory tiling.

> More gpu optimization techniques:
> 
> 1. global memory continuous read
> 2. shared memory bank conflict
> 3. software pipelining
> 4. warp level optimizations
> 5. tensor core
 