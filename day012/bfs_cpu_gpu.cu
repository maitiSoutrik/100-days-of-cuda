#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <cuda_runtime.h>

// Constants
#define THREADS_PER_BLOCK 256
#define MAX_FRONTIER_SIZE 100000000
#define AVERAGE_EDGES_PER_VERTEX 8
#define NUM_VERTICES 1000000  // Reduced for faster testing

// CUDA error checking macro
#define CHECK_CUDA_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s (code %d) at %s:%d\n", \
                   cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    }

// Function to generate a random large graph in CSR format
void generate_random_graph(int num_vertices, int* num_edges, int** edges, int** dest) {
    srand(time(NULL));
    
    // Allocate maximum possible space for destinations
    int max_edges = num_vertices * AVERAGE_EDGES_PER_VERTEX;
    *dest = (int*)malloc(max_edges * sizeof(int));
    *edges = (int*)malloc((num_vertices + 1) * sizeof(int));
    
    int current_edge = 0;
    (*edges)[0] = 0;
    
    // For each vertex
    for (int i = 0; i < num_vertices; i++) {
        // Generate random number of edges for this vertex
        int edges_for_vertex = rand() % (AVERAGE_EDGES_PER_VERTEX * 2);
        
        // Add random edges
        for (int j = 0; j < edges_for_vertex; j++) {
            int dest_vertex = rand() % num_vertices;
            if (dest_vertex != i) {  // Avoid self-loops
                (*dest)[current_edge++] = dest_vertex;
            }
        }
        
        (*edges)[i + 1] = current_edge;
    }
    
    *num_edges = current_edge;
    
    // Reallocate dest array to actual size
    *dest = (int*)realloc(*dest, current_edge * sizeof(int));
}

// CPU implementation of BFS for comparison
void bfs_cpu(int source, int num_vertices, int num_edges, int* edges, int* dest, int* label) {
    // Initialize labels
    for (int i = 0; i < num_vertices; i++) {
        label[i] = -1;
    }
    
    // Allocate frontier arrays
    int* current_frontier = (int*)malloc(num_vertices * sizeof(int));
    int* next_frontier = (int*)malloc(num_vertices * sizeof(int));
    int current_size = 0;
    int next_size = 0;
    
    // Start with source vertex
    label[source] = 0;
    current_frontier[0] = source;
    current_size = 1;
    
    int level = 0;
    
    // BFS traversal
    while (current_size > 0) {
        next_size = 0;
        level++;
        
        // Process current frontier
        for (int i = 0; i < current_size; i++) {
            int vertex = current_frontier[i];
            
            // For each neighbor of the current vertex
            for (int edge = edges[vertex]; edge < edges[vertex + 1]; edge++) {
                int neighbor = dest[edge];
                
                // If neighbor not visited yet
                if (label[neighbor] == -1) {
                    label[neighbor] = level;
                    next_frontier[next_size++] = neighbor;
                }
            }
        }
        
        // Swap frontiers
        int* temp = current_frontier;
        current_frontier = next_frontier;
        next_frontier = temp;
        current_size = next_size;
    }
    
    // Free memory
    free(current_frontier);
    free(next_frontier);
}

// CUDA kernel for BFS
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

// GPU BFS implementation
void bfs_gpu(int source, int num_vertices, int num_edges, int* h_edges, int* h_dest, int* h_labels) {
    int *d_edges, *d_dest, *d_labels, *d_done;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_edges, (num_vertices + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dest, num_edges * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_labels, num_vertices * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_done, sizeof(int)));
    
    // Initialize labels to -1 (unvisited)
    CHECK_CUDA_ERROR(cudaMemset(d_labels, -1, num_vertices * sizeof(int)));
    
    // Copy graph data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_edges, h_edges, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dest, h_dest, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    
    // Set source vertex label to 0 (starting point)
    int initial_level = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels + source, &initial_level, sizeof(int), cudaMemcpyHostToDevice));
    
    int level = 0;
    int h_done;
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (num_vertices + threadsPerBlock - 1) / threadsPerBlock;
    
    // BFS traversal loop
    do {
        h_done = 1;  // Assume we're done unless a new vertex is labeled
        CHECK_CUDA_ERROR(cudaMemcpy(d_done, &h_done, sizeof(int), cudaMemcpyHostToDevice));
        
        // Launch kernel to process current level
        bfs_kernel<<<blocksPerGrid, threadsPerBlock>>>(level, num_vertices, d_edges, d_dest, d_labels, d_done);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // Check if we need to continue
        CHECK_CUDA_ERROR(cudaMemcpy(&h_done, d_done, sizeof(int), cudaMemcpyDeviceToHost));
        level++;
    } while (!h_done && level < num_vertices);
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_labels, d_labels, num_vertices * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_edges));
    CHECK_CUDA_ERROR(cudaFree(d_dest));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_done));
}

// Main function
int main() {
    int num_vertices = 10000;  // Using a smaller size for testing
    int source = 0;           // Source vertex for BFS
    int num_edges;
    int *edges, *dest;
    
    // Generate a random graph
    printf("Generating random graph with %d vertices...\n", num_vertices);
    generate_random_graph(num_vertices, &num_edges, &edges, &dest);
    printf("Graph generated with %d edges\n", num_edges);
    
    // Allocate memory for labels
    int *gpu_labels = (int*)malloc(num_vertices * sizeof(int));
    int *cpu_labels = (int*)malloc(num_vertices * sizeof(int));
    
    // Run GPU BFS
    printf("\nRunning GPU BFS...\n");
    clock_t gpu_start = clock();
    bfs_gpu(source, num_vertices, num_edges, edges, dest, gpu_labels);
    clock_t gpu_end = clock();
    double gpu_time = ((double)(gpu_end - gpu_start)) / CLOCKS_PER_SEC;
    printf("GPU BFS completed in %.6f seconds\n", gpu_time);
    
    // Run CPU BFS for comparison
    printf("\nRunning CPU BFS...\n");
    clock_t cpu_start = clock();
    bfs_cpu(source, num_vertices, num_edges, edges, dest, cpu_labels);
    clock_t cpu_end = clock();
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU BFS completed in %.6f seconds\n", cpu_time);
    
    // Verify results
    printf("\nVerifying results...\n");
    bool correct = true;
    for (int i = 0; i < num_vertices; i++) {
        if (gpu_labels[i] != cpu_labels[i]) {
            printf("Mismatch at vertex %d: GPU=%d, CPU=%d\n", i, gpu_labels[i], cpu_labels[i]);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("Verification successful! GPU and CPU results match.\n");
        printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    }
    
    // Print some statistics
    int max_level = -1;
    int unreachable = 0;
    for (int i = 0; i < num_vertices; i++) {
        if (gpu_labels[i] > max_level) {
            max_level = gpu_labels[i];
        }
        if (gpu_labels[i] == -1) {
            unreachable++;
        }
    }
    
    printf("\nBFS Statistics:\n");
    printf("  Maximum BFS level: %d\n", max_level);
    printf("  Unreachable vertices: %d (%.2f%%)\n", unreachable, (float)unreachable / num_vertices * 100);
    
    // Free memory
    free(edges);
    free(dest);
    free(gpu_labels);
    free(cpu_labels);
    
    return 0;
}
