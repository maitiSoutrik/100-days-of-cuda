# Day 4: Parallel Reduction - Partial Sum

Today's focus is on implementing a parallel reduction algorithm to compute the sum of an array. This is a fundamental operation in parallel computing and demonstrates important concepts like shared memory usage.

## Partial Sum

A partial sum (also known as a parallel reduction) is an operation where we compute the sum of all elements in an array. In a sequential algorithm, this is trivial - just iterate through the array and add each element to a running sum. However, in parallel computing, we need a different approach to effectively utilize the GPU's massive parallelism.

## Implementation Details

In this implementation, I've created a basic parallel reduction algorithm:

- Each thread loads one element from global memory to shared memory
- Performs a tree-based reduction in shared memory
- Each block produces one partial sum
- The final sum is computed on the CPU by adding all partial sums

## Key CUDA Concepts

- **Shared Memory**: Used to store intermediate results within a thread block, significantly reducing global memory access.
- **Thread Synchronization**: `__syncthreads()` ensures all threads in a block reach the same point before proceeding.
- **Warp Divergence**: Minimized by carefully structuring the reduction algorithm.
- **Tree-Based Reduction**: Efficiently reduces the array by summing pairs of elements in parallel.

## Performance Considerations

- The optimized kernel reduces warp divergence by separating the reduction into two phases:
  1. Block-level reduction with synchronization
  2. Warp-level reduction without synchronization (using volatile memory)
- The algorithm achieves O(log n) time complexity compared to O(n) for the sequential version.
- Shared memory is used to avoid expensive global memory accesses.

## Building and Running

```bash
# Build the project
cmake -B build
cmake --build build

# Run with default array size (1,000,000 elements)
./build/day004/partial_sum

# Run with custom array size
./build/day004/partial_sum 10000000
```

## Results

The implementation compares two approaches:

1. CPU sequential sum
2. GPU parallel reduction

The program outputs the sum, execution time, and speedup for each approach.

## Learning Resources

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- Programming Massively Parallel Processors (PMPP) book, Chapter 3
