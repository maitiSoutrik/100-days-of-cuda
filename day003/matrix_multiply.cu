#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

#define BLOCK_SIZE 16

// Matrix multiplication kernel
__global__ void matrixMultiply(const float *A, const float *B, float *C, int m, int n, int p) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within matrix bounds
    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

// CPU implementation of matrix multiplication for verification
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

// Initialize matrix with random values
void initializeMatrix(float *matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = dis(gen);
    }
}

// Print matrix (for small matrices)
void printMatrix(const float *matrix, int rows, int cols, const char *name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.3f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Verify results match between CPU and GPU
bool verifyResults(const float *cpuResult, const float *gpuResult, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpuResult[i] - gpuResult[i]) > tolerance) {
            printf("Verification failed at index %d: CPU = %f, GPU = %f\n", 
                   i, cpuResult[i], gpuResult[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Matrix dimensions: A(m×n) * B(n×p) = C(m×p)
    int m = 1024;  // Rows in A and C
    int n = 1024;  // Columns in A, rows in B
    int p = 1024;  // Columns in B and C
    
    // For small matrix testing (uncomment for debugging)
    // m = 4; n = 4; p = 4;
    
    printf("Matrix dimensions: A(%dx%d) * B(%dx%d) = C(%dx%d)\n", m, n, n, p, m, p);
    printf("Total elements: A=%d, B=%d, C=%d\n\n", m*n, n*p, m*p);
    
    // Allocate host memory
    float *h_A = (float*)malloc(m * n * sizeof(float));
    float *h_B = (float*)malloc(n * p * sizeof(float));
    float *h_C_CPU = (float*)malloc(m * p * sizeof(float));
    float *h_C_GPU = (float*)malloc(m * p * sizeof(float));
    
    // Initialize matrices with random values
    initializeMatrix(h_A, m, n);
    initializeMatrix(h_B, n, p);
    
    // For small matrices, print input matrices (uncomment for debugging)
    // if (m <= 8 && n <= 8 && p <= 8) {
    //     printMatrix(h_A, m, n, "Matrix A");
    //     printMatrix(h_B, n, p, "Matrix B");
    // }
    
    // CPU matrix multiplication
    printf("----- CPU Execution -----\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(h_A, h_B, h_C_CPU, m, n, p);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    printf("CPU execution time: %.2f ms\n\n", cpu_duration.count());
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * n * sizeof(float));
    cudaMalloc((void**)&d_B, n * p * sizeof(float));
    cudaMalloc((void**)&d_C, m * p * sizeof(float));
    
    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * p * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, 
                 (m + blockDim.y - 1) / blockDim.y);
    
    printf("----- GPU Implementation -----\n");
    printf("CUDA kernel launch with grid of %d x %d blocks, each with %d x %d threads\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // GPU matrix multiplication
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    
    // Record start time
    cudaEventRecord(gpu_start);
    
    // Launch the kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, p);
    
    // Record end time
    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    
    float gpu_kernel_time;
    cudaEventElapsedTime(&gpu_kernel_time, gpu_start, gpu_end);
    
    // Copy result back to host
    cudaMemcpy(h_C_GPU, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("GPU kernel execution time: %.2f ms\n", gpu_kernel_time);
    
    // Verify implementation results
    bool gpu_correct = verifyResults(h_C_CPU, h_C_GPU, m * p);
    printf("GPU implementation verification: %s\n\n", gpu_correct ? "PASSED" : "FAILED");
    
    // For small matrices, print output matrices (uncomment for debugging)
    // if (m <= 8 && n <= 8 && p <= 8) {
    //     printMatrix(h_C_CPU, m, p, "CPU Result");
    //     printMatrix(h_C_GPU, m, p, "GPU Result");
    // }
    
    printf("----- Performance Comparison -----\n");
    printf("CPU execution time: %.2f ms\n", cpu_duration.count());
    printf("GPU kernel execution time: %.2f ms\n", gpu_kernel_time);
    printf("Speedup (GPU vs CPU): %.2fx\n\n", cpu_duration.count() / gpu_kernel_time);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_CPU);
    free(h_C_GPU);
    
    printf("Matrix multiplication completed successfully!\n");
    
    return 0;
}
