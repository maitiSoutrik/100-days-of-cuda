#include "bfs.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
