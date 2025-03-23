# Day 13: Optimized BFS with Shared Memory in CUDA

This implementation builds upon the Day 12 BFS algorithm, focusing on optimizing performance through efficient use of shared memory.

## Optimization Approach

The primary optimization in this implementation is the use of shared memory to cache frequently accessed data, reducing global memory access latency. Key improvements include:

1. **Shared Memory Caching**: Frontier vertices and their edge ranges are loaded into shared memory, reducing redundant global memory accesses.

2. **Frontier Management**: A more efficient frontier queue implementation that only processes active vertices.

3. **Two-Phase Kernel Execution**:

   - First kernel processes the current frontier using shared memory
   - Second kernel prepares the next frontier

4. **Increased Problem Size**: Testing with 1 million vertices (up from 10,000) to better utilize GPU parallelism.

## Implementation Details

### Shared Memory Usage

The implementation allocates shared memory for three arrays:

1. `shared_frontier`: Caches the current frontier vertices
2. `shared_edge_start`: Caches the starting edge indices for each frontier vertex
3. `shared_edge_end`: Caches the ending edge indices for each frontier vertex

This approach significantly reduces global memory accesses during the neighbor traversal phase, as each thread can access its vertex's edge range from shared memory instead of global memory.

### Frontier Management

Unlike the previous implementation that scanned all vertices at each level, this version:

1. Maintains an explicit frontier queue of active vertices
2. Only processes vertices in the current frontier
3. Builds the next frontier using a separate kernel

## Files

- `bfs_optimized.cu`: Implementation of the shared memory optimized BFS algorithm
- `CMakeLists.txt`: Build configuration file

## Key CUDA Concepts

- **Shared Memory**: Fast on-chip memory shared by all threads in a block
- **Thread Synchronization**: Using `__syncthreads()` to ensure all threads have loaded data into shared memory before proceeding
- **Atomic Operations**: Using `atomicCAS` and `atomicAdd` for thread-safe updates
- **Kernel Splitting**: Dividing work into multiple specialized kernels

## Performance Expectations

The shared memory optimization is expected to provide significant performance improvements over the Day 12 implementation, especially for larger graphs. The speedup should be more pronounced as the problem size increases, demonstrating the scalability advantages of GPU computing.

## Building and Running

```bash
# Build the project
cmake -B build
cmake --build build

# Run the optimized BFS implementation
./build/day013/bfs_optimized
```

## Performance Results

The implementation was tested on a Jetson Nano with the following results:

```text
Generating random graph with 1000000 vertices...
Graph generated with 7494786 edges

Running optimized GPU BFS with shared memory...
Optimized GPU BFS completed in 0.248296 seconds

Running CPU BFS...
CPU BFS completed in 0.367740 seconds

Verifying results...
Verification successful! GPU and CPU results match.
Speedup: 1.48x

BFS Statistics:
  Maximum BFS level: 10
  Unreachable vertices: 554 (0.06%)
```

## Performance Analysis

1. **GPU Outperformed CPU**: The GPU implementation with shared memory optimization was approximately 1.48x faster than the CPU implementation.

2. **Improvement Over Day 12**: In the previous implementation (Day 12), the CPU was 25x faster than the GPU. Now, our optimized GPU implementation outperforms the CPU.

3. **Factors Contributing to Improved Performance**:

   - **Shared Memory Usage**: Caching frontier vertices and edge ranges in shared memory reduced global memory access latency.
   - **Efficient Frontier Management**: Only processing active vertices rather than scanning all vertices at each level.
   - **Larger Problem Size**: Testing with 1 million vertices (up from 10,000) better utilized GPU parallelism.
   - **Two-Phase Kernel Execution**: Separating frontier processing and preparation improved efficiency.

4. **Scalability**: The results demonstrate that GPU implementations of graph algorithms can outperform CPU implementations when properly optimized and run on sufficiently large datasets.
