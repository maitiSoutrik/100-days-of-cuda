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

## Performance Analysis

After running the implementation, we'll analyze:

1. The speedup compared to the CPU implementation
2. The improvement over the Day 12 GPU implementation
3. Memory access patterns and their impact on performance
4. Scalability with increasing graph size
