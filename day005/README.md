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

### Performance on Nvidia Jetson Nano

```bash
Layer Normalization on matrix of size 1024x256
CPU Time: 3.7992 ms
GPU Basic Kernel Time: 16.2066 ms, Verification: PASSED
GPU Shared Memory Kernel Time: 12.4531 ms, Verification: PASSED

Speedups:
Basic Kernel vs CPU: 0.234424x
Shared Memory Kernel vs CPU: 0.30508x
```

### Performance Analysis

The results from the Jetson Nano reveal several interesting insights:

1. **CPU outperforms GPU**: The CPU implementation is significantly faster than both GPU implementations (about 3-4x faster). This is because:
   - The Layer Normalization operation has a low arithmetic intensity (few calculations per memory access)
   - The overhead of transferring data to and from the GPU memory is significant
   - The Jetson Nano's CPU might be relatively powerful compared to its GPU for this specific workload

2. **Shared Memory Optimization**: The shared memory kernel shows a notable improvement over the basic kernel (about 23% faster), demonstrating that:
   - Reducing global memory accesses through shared memory is effective
   - Having a single thread compute mean and variance reduces redundant calculations
   - Proper thread synchronization is crucial for correct results

3. **Memory Transfer Bottleneck**: Despite the optimizations, both GPU implementations are slower than the CPU version, highlighting that:
   - For operations with low arithmetic intensity, the memory transfer overhead can dominate
   - The computation-to-communication ratio is unfavorable for this particular problem size

## Learning Resources

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Layer Normalization Paper](https://arxiv.org/abs/1607.06450)
- [CUDA Memory Optimization](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- Programming Massively Parallel Processors (PMPP) book, Chapter 5 (Memory Optimization)
