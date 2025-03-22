# Day 12: Breadth-First Search (BFS) with CUDA

This implementation demonstrates a parallel Breadth-First Search algorithm using CUDA. BFS is a fundamental graph traversal algorithm that explores all vertices at the current depth before moving to vertices at the next depth level.

## Implementation Details

The BFS algorithm is implemented using the Compressed Sparse Row (CSR) format for efficient graph representation. The implementation includes:

1. **GPU Implementation**: Uses atomic operations to ensure thread-safe updates to vertex labels.
2. **CPU Implementation**: For comparison and verification of results.
3. **Random Graph Generator**: Creates a random graph with a specified number of vertices and edges.

## Files

- `bfs_cpu_gpu.cu`: Combined file containing all implementations (CPU, GPU, kernel, and main function)
- `CMakeLists.txt`: Build configuration file

## Key CUDA Concepts

- **Atomic Operations**: The implementation uses `atomicCAS` (Compare-And-Swap) and `atomicExch` to safely update shared data in parallel.
- **Kernel Invocation**: The BFS kernel is called repeatedly for each level of the traversal.
- **Memory Management**: Efficient allocation and transfer of graph data between host and device.

## Algorithm Overview

1. Initialize all vertex labels to -1 (unvisited).
2. Set the source vertex label to 0.
3. For each level, launch a kernel where each thread processes a vertex at the current level.
4. Each thread explores all neighbors of its assigned vertex and labels unvisited neighbors with the next level.
5. Continue until no new vertices are labeled.

## Performance Results

The implementation was tested on a Jetson Nano with the following results:

```text
Generating random graph with 10000 vertices...
Graph generated with 74398 edges

Running GPU BFS...
GPU BFS completed in 0.060097 seconds

Running CPU BFS...
CPU BFS completed in 0.002299 seconds

Verifying results...
Verification successful! GPU and CPU results match.
Speedup: 0.04x

BFS Statistics:
  Maximum BFS level: 7
  Unreachable vertices: 3 (0.03%)
```

## Learnings and Observations

1. **CPU Outperformed GPU**: Surprisingly, the CPU implementation was approximately 25x faster than the GPU implementation for this problem size.

2. **Reasons for GPU Underperformance**:
   - **Small Problem Size**: 10,000 vertices is relatively small for a GPU workload.
   - **Memory Transfer Overhead**: The time to transfer data between CPU and GPU likely dominated the actual computation time.
   - **Sparse Graph**: With only ~7.4 edges per vertex on average, the graph is relatively sparse, limiting parallelism.
   - **BFS Algorithm Nature**: BFS is inherently sequential in its frontier expansion, which can limit GPU parallelism.

3. **Future Optimization Opportunities**:
   - Increase problem size to better utilize GPU parallelism
   - Optimize memory transfers using pinned memory or CUDA streams
   - Implement a more GPU-friendly algorithm like work-efficient BFS
   - Reduce kernel launch overhead
   - Utilize shared memory for frequently accessed data

4. **Build System Considerations**:
   - Combining all code into a single CUDA file simplified the build process and resolved linking issues
   - This approach works well for educational examples but might not be ideal for larger projects

## Building and Running

```bash
# Build the project
cmake -B build
cmake --build build

# Run the BFS implementation
./build/day012/bfs
```
