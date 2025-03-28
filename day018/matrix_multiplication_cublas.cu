#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(error), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS_ERROR(call) \
{ \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "CUBLAS error: %d at line %d\n", status, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// CPU matrix multiplication implementation for comparison
void matrix_multiply_cpu(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Print a matrix
void print_matrix(const float* matrix, int rows, int cols, const char* name) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Matrix dimensions: A(m x k) * B(k x n) = C(m x n)
    const int m = 2;  // rows of A and C
    const int n = 2;  // cols of B and C
    const int k = 3;  // cols of A and rows of B

    // Allocate and initialize host matrices
    float* h_A = (float*)malloc(m * k * sizeof(float));
    float* h_B = (float*)malloc(k * n * sizeof(float));
    float* h_C = (float*)malloc(m * n * sizeof(float));
    float* h_C_cpu = (float*)malloc(m * n * sizeof(float));

    // Initialize matrices with some values
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            h_A[i * k + j] = i * k + j + 1;  // 1-based indexing for easier verification
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            h_B[i * n + j] = i * n + j + 7;  // Start from 7 for easier verification
        }
    }

    // Print input matrices
    print_matrix(h_A, m, k, "Matrix A");
    print_matrix(h_B, k, n, "Matrix B");

    // Perform CPU matrix multiplication and measure time
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrix_multiply_cpu(h_A, h_B, h_C_cpu, m, n, k);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, m * k * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, k * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, m * n * sizeof(float)));

    // Copy matrices from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    // Create CUBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    // Perform matrix multiplication using CUBLAS: C = A * B
    // CUBLAS uses column-major order, but our matrices are in row-major order
    // We can use the formula: C^T = B^T * A^T to compute the correct result
    // This is equivalent to computing: C = A * B in row-major order
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Measure GPU execution time
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    // C^T = B^T * A^T
    // In CUBLAS: cublasSgemm(handle, transB, transA, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                 n,    // Number of rows of matrix C
                                 m,    // Number of columns of matrix C
                                 k,    // Number of columns of matrix A
                                 &alpha,
                                 d_B,  // B matrix
                                 n,    // Leading dimension of B
                                 d_A,  // A matrix
                                 k,    // Leading dimension of A
                                 &beta,
                                 d_C,  // C matrix
                                 n     // Leading dimension of C
    ));
    
    // Synchronize to ensure completion
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> gpu_duration = gpu_end - gpu_start;

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Print result matrix
    print_matrix(h_C, m, n, "Matrix C = A × B");

    // Print timing information
    printf("CPU Execution Time: %.3f ms\n", cpu_duration.count());
    printf("GPU Execution Time: %.3f ms\n", gpu_duration.count());
    printf("Speedup: %.2f\n", cpu_duration.count() / gpu_duration.count());

    // Verify results
    bool correct = true;
    for (int i = 0; i < m * n; i++) {
        if (fabs(h_C[i] - h_C_cpu[i]) > 1e-5) {
            printf("Error: h_C[%d] = %.2f, h_C_cpu[%d] = %.2f\n", i, h_C[i], i, h_C_cpu[i]);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("\nResults verified: GPU and CPU calculations match!\n");
    } else {
        printf("\nResults do not match!\n");
    }

    // Clean up
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);

    return 0;
}
