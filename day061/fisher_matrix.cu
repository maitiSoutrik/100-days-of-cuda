#include "fisher_matrix.cuh"
#include <vector>
#include <numeric> 
#include <cmath>   
#include <iostream> 
#include <cassert> // For assert, if used directly from example

// CUDA kernel for Fisher Information Matrix
__global__ void fisher_kernel(const float* log_probs, float* fisher, 
                             int n_samples, int n_params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Parameter index 1 (row)
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Parameter index 2 (col)
    
    if (i < n_params && j < n_params) {
        float sum = 0.0f;
        // Sum over samples
        for (int k = 0; k < n_samples; k++) {
            // log_probs[k * n_params + i] is the i-th score component for k-th sample
            // log_probs[k * n_params + j] is the j-th score component for k-th sample
            sum += log_probs[k * n_params + i] * log_probs[k * n_params + j];
        }
        // The Fisher Information is the expectation of this outer product.
        // If log_probs are already individual sample scores, then averaging is appropriate.
        fisher[i * n_params + j] = sum / static_cast<float>(n_samples);
    }
}

// Wrapper function for GPU computation
void compute_fisher_matrix_gpu(const float* h_log_probs, float* h_fisher_matrix, 
                               int n_samples, int n_params) {
    float *d_log_probs, *d_fisher_matrix;
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_log_probs, static_cast<size_t>(n_samples) * n_params * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fisher_matrix, static_cast<size_t>(n_params) * n_params * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_log_probs, h_log_probs, 
                                static_cast<size_t>(n_samples) * n_params * sizeof(float),
                                cudaMemcpyHostToDevice));
    
    // Launch kernel
    // Each thread computes one element of the n_params x n_params Fisher matrix
    dim3 blockDim(16, 16); // 256 threads per block
    dim3 gridDim((n_params + blockDim.x - 1) / blockDim.x,
                 (n_params + blockDim.y - 1) / blockDim.y);
    
    fisher_kernel<<<gridDim, blockDim>>>(d_log_probs, d_fisher_matrix, n_samples, n_params);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_fisher_matrix, d_fisher_matrix, 
                                static_cast<size_t>(n_params) * n_params * sizeof(float),
                                cudaMemcpyDeviceToHost));
    
    // Cleanup device memory
    CHECK_CUDA_ERROR(cudaFree(d_log_probs));
    CHECK_CUDA_ERROR(cudaFree(d_fisher_matrix));
}

// CPU reference implementation
void compute_fisher_matrix_cpu(const float* log_probs, float* fisher_matrix, 
                               int n_samples, int n_params) {
    for (int i = 0; i < n_params; i++) { // Row of Fisher matrix
        for (int j = 0; j < n_params; j++) { // Column of Fisher matrix
            float sum = 0.0f;
            for (int k = 0; k < n_samples; k++) { // Sum over samples
                // log_probs[k * n_params + i] is score_i for sample k
                // log_probs[k * n_params + j] is score_j for sample k
                sum += log_probs[k * n_params + i] * log_probs[k * n_params + j];
            }
            fisher_matrix[i * n_params + j] = sum / static_cast<float>(n_samples);
        }
    }
}
