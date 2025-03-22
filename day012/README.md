# Day 12: Breadth-First Search (BFS) with CUDA

This implementation demonstrates a parallel Breadth-First Search algorithm using CUDA. BFS is a fundamental graph traversal algorithm that explores all vertices at the current depth before moving to vertices at the next depth level.

## Implementation Details

The BFS algorithm is implemented using the Compressed Sparse Row (CSR) format for efficient graph representation. The implementation includes:

1. **GPU Implementation**: Uses atomic operations to ensure thread-safe updates to vertex labels.
2. **CPU Implementation**: For comparison and verification of results.
3. **Random Graph Generator**: Creates a random graph with a specified number of vertices and edges.

## Files

- `bfs.h`: Header file containing function declarations and constants.
- `bfs_kernel.cu`: CUDA kernel implementation for parallel BFS traversal.
- `bfs_gpu.cu`: Host-side code for GPU BFS implementation.
- `bfs_cpu.c`: CPU implementation for comparison.
- `main.cu`: Main program that runs both implementations and compares results.

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

## Performance Considerations

- The GPU implementation can achieve significant speedup over the CPU version for large graphs.
- The performance depends on the graph structure and connectivity.
- Atomic operations can cause contention, but they're necessary for correctness.

## Building and Running

```bash
# Build the project
cmake -B build
cmake --build build

# Run the BFS implementation
./build/bfs
```


