#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath> // For fabs

#define BLOCK_SIZE 16

// Error checking macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Matrix multiplication kernel (unchanged from Day 3)
__global__ void matrixMultiply(const float *A, const float *B, float *C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

// CPU implementation (unchanged from Day 3)
void matrixMultiplyCPU(const float *A, const float *B, float *C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

// Initialize matrix (unchanged from Day 3)
void initializeMatrix(float *matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = dis(gen);
    }
}

// Verify results (unchanged from Day 3)
bool verifyResults(const float *cpuResult, const float *gpuResult, int size, float tolerance = 1e-4) { // Slightly increased tolerance
    for (int i = 0; i < size; i++) {
        if (fabs(cpuResult[i] - gpuResult[i]) > tolerance) {
            printf("Verification failed at index %d: CPU = %f, GPU = %f, Diff = %f\n", 
                   i, cpuResult[i], gpuResult[i], fabs(cpuResult[i] - gpuResult[i]));
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    // Matrix dimensions: A(m×n) * B(n×p) = C(m×p)
    int m = 1024; // Rows in A and C
    int n = 1024; // Columns in A, rows in B
    int p = 1024; // Columns in B and C

    if (argc == 3) {
        m = atoi(argv[1]);
        n = m; // Assuming square matrices if only one dimension given for simplicity
        p = atoi(argv[2]);
        printf("Using provided dimensions: m=%d, n=%d, p=%d\n", m, n, p);
    } else if (argc == 2) {
         m = atoi(argv[1]);
         n = m;
         p = m;
         printf("Using provided dimension for square matrices: m=n=p=%d\n", m);
    } else {
        printf("Usage: %s <matrix_dim> OR %s <matrix_rows> <matrix_cols>\n", argv[0], argv[0]);
        printf("Using default square matrix dimensions: m=n=p=%d\n", m);
    }
    
    printf("Matrix dimensions: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", m, n, n, p, m, p);
    size_t sizeA = (size_t)m * n * sizeof(float);
    size_t sizeB = (size_t)n * p * sizeof(float);
    size_t sizeC = (size_t)m * p * sizeof(float);
    printf("Total memory: A=%.2f MB, B=%.2f MB, C=%.2f MB\n\n", 
           sizeA / (1024.0*1024.0), sizeB / (1024.0*1024.0), sizeC / (1024.0*1024.0));

    // --- Host Memory Allocation ---
    // Use pinned memory for asynchronous transfers
    float *h_A, *h_B, *h_C_sync, *h_C_async, *h_C_cpu;
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_A, sizeA));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_B, sizeB));
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_C_sync, sizeC)); // Result buffer for synchronous version
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&h_C_async, sizeC)); // Result buffer for asynchronous version
    h_C_cpu = (float*)malloc(sizeC); // CPU result buffer (can be pageable)
    if (!h_C_cpu) {
        fprintf(stderr, "Failed to allocate host memory for CPU result.\n");
        return EXIT_FAILURE;
    }

    // Initialize matrices
    initializeMatrix(h_A, m, n);
    initializeMatrix(h_B, n, p);

    // --- CPU Execution (for verification) ---
    printf("----- CPU Execution -----\n");
    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, m, n, p);
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end_time - cpu_start_time;
    printf("CPU execution time: %.3f ms\n\n", cpu_duration.count());

    // --- GPU Device Memory Allocation ---
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, sizeA));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, sizeB));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, sizeC));

    // Define block and grid dimensions (same as Day 3)
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, 
                 (m + blockDim.y - 1) / blockDim.y);

    // --- GPU Execution (Synchronous Baseline) ---
    printf("----- GPU Execution (Synchronous) -----\n");
    cudaEvent_t sync_start, sync_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&sync_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&sync_stop));

    CHECK_CUDA_ERROR(cudaEventRecord(sync_start)); // Start timer

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, p);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

    CHECK_CUDA_ERROR(cudaMemcpy(h_C_sync, d_C, sizeC, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaEventRecord(sync_stop)); // Stop timer
    CHECK_CUDA_ERROR(cudaEventSynchronize(sync_stop)); // Wait for completion

    float sync_gpu_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&sync_gpu_time_ms, sync_start, sync_stop));
    printf("Synchronous GPU total time (H2D + Kernel + D2H): %.3f ms\n", sync_gpu_time_ms);

    // Verify synchronous results
    bool sync_correct = verifyResults(h_C_cpu, h_C_sync, m * p);
    printf("Synchronous GPU verification: %s\n\n", sync_correct ? "PASSED" : "FAILED");

    // --- GPU Execution (Asynchronous with Streams) ---
    printf("----- GPU Execution (Asynchronous with Streams) -----\n");

    // Create streams
    cudaStream_t stream_h2d, stream_kernel, stream_d2h;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_h2d));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_kernel));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_d2h));

    // Create events for synchronization
    cudaEvent_t event_h2d_done, event_kernel_done, event_d2h_done;
    CHECK_CUDA_ERROR(cudaEventCreate(&event_h2d_done));
    CHECK_CUDA_ERROR(cudaEventCreate(&event_kernel_done));
    CHECK_CUDA_ERROR(cudaEventCreate(&event_d2h_done));

    // Events for timing the whole async operation
    cudaEvent_t async_start, async_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&async_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&async_stop));

    // Record start time for the entire async sequence (before first operation)
    CHECK_CUDA_ERROR(cudaEventRecord(async_start)); // Can use any stream, or 0

    // 1. Copy H2D on stream_h2d
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, sizeA, cudaMemcpyHostToDevice, stream_h2d));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, sizeB, cudaMemcpyHostToDevice, stream_h2d));
    // Record event after H2D copy is done on stream_h2d
    CHECK_CUDA_ERROR(cudaEventRecord(event_h2d_done, stream_h2d));

    // 2. Launch Kernel on stream_kernel, waiting for H2D to finish
    // Make stream_kernel wait for the event_h2d_done recorded in stream_h2d
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_kernel, event_h2d_done, 0));
    matrixMultiply<<<gridDim, blockDim, 0, stream_kernel>>>(d_A, d_B, d_C, m, n, p);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    // Record event after kernel is done on stream_kernel
    CHECK_CUDA_ERROR(cudaEventRecord(event_kernel_done, stream_kernel));

    // 3. Copy D2H on stream_d2h, waiting for Kernel to finish
    // Make stream_d2h wait for the event_kernel_done recorded in stream_kernel
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_d2h, event_kernel_done, 0));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C_async, d_C, sizeC, cudaMemcpyDeviceToHost, stream_d2h));
    // Record event after D2H copy is done on stream_d2h
    CHECK_CUDA_ERROR(cudaEventRecord(event_d2h_done, stream_d2h));

    // Record stop time for the entire async sequence (after last operation scheduled)
    CHECK_CUDA_ERROR(cudaEventRecord(async_stop, event_d2h_done)); // Record stop after D2H event is complete

    // 4. Synchronize host with the final D2H completion event
    printf("Waiting for asynchronous operations to complete...\n");
    CHECK_CUDA_ERROR(cudaEventSynchronize(event_d2h_done)); // Wait for the last operation
    printf("Asynchronous operations completed.\n");

    float async_gpu_time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&async_gpu_time_ms, async_start, async_stop));
    printf("Asynchronous GPU total time (Overlapped H2D/Kernel/D2H): %.3f ms\n", async_gpu_time_ms);

    // Verify asynchronous results
    bool async_correct = verifyResults(h_C_cpu, h_C_async, m * p);
    printf("Asynchronous GPU verification: %s\n\n", async_correct ? "PASSED" : "FAILED");

    // --- Performance Comparison ---
    printf("----- Performance Summary -----\n");
    printf("CPU Time:                 %.3f ms\n", cpu_duration.count());
    printf("GPU Time (Synchronous):   %.3f ms\n", sync_gpu_time_ms);
    printf("GPU Time (Asynchronous):  %.3f ms\n", async_gpu_time_ms);
    if (async_gpu_time_ms > 0) {
        printf("Async Speedup vs Sync:    %.2fx\n", sync_gpu_time_ms / async_gpu_time_ms);
    }
     if (cpu_duration.count() > 0) {
         printf("Async Speedup vs CPU:     %.2fx\n", cpu_duration.count() / async_gpu_time_ms);
     }


    // --- Cleanup ---
    // Destroy events
    CHECK_CUDA_ERROR(cudaEventDestroy(sync_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(sync_stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(event_h2d_done));
    CHECK_CUDA_ERROR(cudaEventDestroy(event_kernel_done));
    CHECK_CUDA_ERROR(cudaEventDestroy(event_d2h_done));
    CHECK_CUDA_ERROR(cudaEventDestroy(async_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(async_stop));

    // Destroy streams
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_h2d));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_kernel));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_d2h));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    // Free host memory
    CHECK_CUDA_ERROR(cudaFreeHost(h_A));
    CHECK_CUDA_ERROR(cudaFreeHost(h_B));
    CHECK_CUDA_ERROR(cudaFreeHost(h_C_sync));
    CHECK_CUDA_ERROR(cudaFreeHost(h_C_async));
    free(h_C_cpu);

    printf("\nDay 32 Stream Overlap demo finished successfully!\n");

    return EXIT_SUCCESS;
}
