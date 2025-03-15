# Day 5: Layer Normalization in CUDA

Today's focus is on implementing Layer Normalization using CUDA. Layer Normalization is a crucial technique in deep learning that helps stabilize the training of neural networks by normalizing the inputs to each layer.

## Layer Normalization

Layer Normalization normalizes the activations of a layer for each sample across all features. For each sample in a batch, it:

1. Computes the mean and variance across all features
2. Normalizes each feature using the formula: (x - mean) / sqrt(variance + epsilon)

This normalization technique is particularly useful in recurrent neural networks and transformers, as it operates on each sample independently, making it batch-size independent.

## Implementation Details

In this implementation, I've created two different CUDA kernels for Layer Normalization:

1. **Basic Kernel**: A straightforward implementation where each thread processes one row of the input matrix.
2. **Shared Memory Kernel**: Uses shared memory to optimize performance by reducing global memory accesses.

The implementation also includes a CPU version for comparison and verification.

## Key CUDA Concepts

- **Shared Memory**: Used to store row data and intermediate results, reducing global memory access.
- **Thread Synchronization**: `__syncthreads()` ensures all threads in a block reach the same point before proceeding.
- **Parallel Reduction**: Efficiently computes mean and variance using a tree-based reduction approach.
- **Memory Coalescing**: Optimized memory access patterns to improve memory bandwidth utilization.
- **Error Handling**: Includes CUDA error checking to detect and report any runtime errors.

## Performance Considerations

- The basic kernel is simple but makes multiple passes through global memory.
- The shared memory kernel reduces global memory accesses but still performs sequential reductions.
- Both kernels are compared against a CPU implementation to measure speedup.

## Building and Running

```bash
# Build the project
cmake -B build
cmake --build build

# Run with default matrix size (1024x256)
./build/day005/layer_norm

# Run with custom matrix size (rows x columns)
./build/day005/layer_norm 2048 512
```

## Results

The implementation compares three approaches:

1. CPU sequential implementation
2. GPU basic kernel
3. GPU shared memory kernel

The program outputs the execution time and verification results for each approach, as well as the speedup achieved by the GPU implementations.

### Performance Analysis

The performance of each implementation depends on the matrix size and hardware:

- For small matrices, the CPU implementation may be faster due to the overhead of data transfer to the GPU.
- For larger matrices, the GPU implementations show significant speedups, with the advanced kernel typically performing the best.
- The shared memory kernel demonstrates the importance of optimizing memory access patterns in CUDA programming.

## Learning Resources

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Layer Normalization Paper](https://arxiv.org/abs/1607.06450)
- [CUDA Memory Optimization](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- Programming Massively Parallel Processors (PMPP) book, Chapter 5 (Memory Optimization)
