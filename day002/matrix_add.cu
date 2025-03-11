#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Function to get current time in microseconds
double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000000 + (double)tv.tv_usec;
}

// CUDA kernel for matrix addition
__global__ void matrixAdd(const float *d_A, const float *d_B, float *d_C, int rows, int cols) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within matrix bounds
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

// Function to initialize a matrix with random values
void initializeMatrix(float *h_matrix, int size) {
    for (int i = 0; i < size; i++) {
        h_matrix[i] = (float)(rand() % 100) / 10.0f;
    }
}

// CPU implementation of matrix addition
void matrixAddCPU(const float *h_A, const float *h_B, float *h_C, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int idx = row * cols + col;
            h_C[idx] = h_A[idx] + h_B[idx];
        }
    }
}

// Function to print a matrix
void printMatrix(const float *h_matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f\t", h_matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Print device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Set matrix dimensions - using larger matrices for better timing comparison
    int rows = 1024;
    int cols = 1024;
    int matrixSize = rows * cols;
    size_t matrixBytes = matrixSize * sizeof(float);
    
    printf("Matrix dimensions: %d x %d (Total elements: %d)\n", rows, cols, matrixSize);
    
    // Seed the random number generator
    srand(42);
    
    // Allocate host memory
    float *h_A = (float*)malloc(matrixBytes);
    float *h_B = (float*)malloc(matrixBytes);
    float *h_C = (float*)malloc(matrixBytes);
    float *h_C_CPU = (float*)malloc(matrixBytes);  // For CPU results
    
    if (h_A == NULL || h_B == NULL || h_C == NULL || h_C_CPU == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize matrices with random values
    initializeMatrix(h_A, matrixSize);
    initializeMatrix(h_B, matrixSize);
    
    // For small matrices, print input and output
    if (rows <= 8 && cols <= 8) {
        printf("Matrix A:\n");
        printMatrix(h_A, rows, cols);
        printf("Matrix B:\n");
        printMatrix(h_B, rows, cols);
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, matrixBytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, matrixBytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, matrixBytes));
    
    // Run and time CPU version
    printf("\n----- CPU Execution -----\n");
    double cpuStart = getCurrentTime();
    matrixAddCPU(h_A, h_B, h_C_CPU, rows, cols);
    double cpuEnd = getCurrentTime();
    double cpuElapsed = (cpuEnd - cpuStart) / 1000.0; // Convert to milliseconds
    printf("CPU execution time: %.2f ms\n", cpuElapsed);
    
    // Run and time GPU version
    printf("\n----- GPU Execution -----\n");
    
    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, 
                 (rows + blockDim.y - 1) / blockDim.y);
    
    printf("CUDA kernel launch with grid of %d x %d blocks, each with %d x %d threads\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Start timing GPU operations (including memory transfers)
    double gpuStartWithTransfer = getCurrentTime();
    
    // Copy matrices from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, matrixBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, matrixBytes, cudaMemcpyHostToDevice));
    
    // Start timing kernel execution only
    double gpuStartKernelOnly = getCurrentTime();
    
    // Launch the kernel
    matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Wait for kernel to finish
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // End timing kernel execution only
    double gpuEndKernelOnly = getCurrentTime();
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, matrixBytes, cudaMemcpyDeviceToHost));
    
    // End timing GPU operations (including memory transfers)
    double gpuEndWithTransfer = getCurrentTime();
    
    double gpuElapsedKernelOnly = (gpuEndKernelOnly - gpuStartKernelOnly) / 1000.0; // Convert to milliseconds
    double gpuElapsedWithTransfer = (gpuEndWithTransfer - gpuStartWithTransfer) / 1000.0; // Convert to milliseconds
    
    printf("GPU kernel execution time: %.2f ms\n", gpuElapsedKernelOnly);
    printf("GPU total time (with memory transfers): %.2f ms\n", gpuElapsedWithTransfer);
    
    // For small matrices, print result
    if (rows <= 8 && cols <= 8) {
        printf("\nResult Matrix (A + B) from GPU:\n");
        printMatrix(h_C, rows, cols);
        
        printf("Result Matrix (A + B) from CPU:\n");
        printMatrix(h_C_CPU, rows, cols);
    }
    
    // Verify the results
    bool resultsMatch = true;
    for (int i = 0; i < matrixSize; ++i) {
        // Verify GPU result against expected value
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "GPU result verification failed at element %d!\n", i);
            resultsMatch = false;
            break;
        }
        
        // Verify CPU result against expected value
        if (fabs(h_A[i] + h_B[i] - h_C_CPU[i]) > 1e-5) {
            fprintf(stderr, "CPU result verification failed at element %d!\n", i);
            resultsMatch = false;
            break;
        }
        
        // Verify that CPU and GPU results match
        if (fabs(h_C[i] - h_C_CPU[i]) > 1e-5) {
            fprintf(stderr, "CPU and GPU results don't match at element %d!\n", i);
            resultsMatch = false;
            break;
        }
    }
    
    if (resultsMatch) {
        printf("\nAll tests PASSED\n");
    }
    
    // Print performance comparison
    printf("\n----- Performance Comparison -----\n");
    printf("CPU execution time: %.2f ms\n", cpuElapsed);
    printf("GPU kernel execution time: %.2f ms\n", gpuElapsedKernelOnly);
    printf("GPU total time (with memory transfers): %.2f ms\n", gpuElapsedWithTransfer);
    printf("Speedup (kernel only): %.2fx\n", cpuElapsed / gpuElapsedKernelOnly);
    printf("Speedup (with transfers): %.2fx\n", cpuElapsed / gpuElapsedWithTransfer);
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_CPU);
    
    printf("\nMatrix addition completed successfully!\n");
    
    return 0;
}
