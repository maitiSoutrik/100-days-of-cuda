#ifndef BFS_H
#define BFS_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define THREADS_PER_BLOCK 256
#define MAX_FRONTIER_SIZE 100000000
#define AVERAGE_EDGES_PER_VERTEX 8
#define NUM_VERTICES 1000000  // Reduced for faster testing

#ifdef __CUDACC__
#define CHECK_CUDA_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s (code %d) at %s:%d\n", \
                   cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    }
#endif

// Function to generate a random graph in CSR format
void generate_random_graph(int num_vertices, int* num_edges, int** edges, int** dest);

// GPU BFS implementation
void bfs_gpu(int source, int num_vertices, int num_edges, int* h_edges, int* h_dest, int* h_labels);

// CPU BFS implementation for comparison
void bfs_cpu(int source, int num_vertices, int num_edges, int* edges, int* dest, int* labels);

#endif // BFS_H
