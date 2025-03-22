#include "bfs.h"

__global__ void bfs_kernel(int level, int num_vertices, int* edges, int* dest, int* labels, int* done) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_vertices && labels[tid] == level) {
        // For each edge of the current vertex
        for (int edge = edges[tid]; edge < edges[tid + 1]; edge++) {
            int neighbor = dest[edge];
            // Try to label the neighbor if it hasn't been visited yet (-1)
            // atomicCAS performs Compare-And-Swap atomically
            // It compares labels[neighbor] with -1, and if equal, sets it to level+1
            // Returns the old value (which should be -1 if we successfully labeled it)
            if (atomicCAS(&labels[neighbor], -1, level + 1) == -1) {
                // If we labeled a new vertex, we need to continue BFS
                atomicExch(done, 0);
            }
        }
    }
}
