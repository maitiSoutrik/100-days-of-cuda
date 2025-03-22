#include "bfs.h"
#include "bfs_kernel.cu"

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
