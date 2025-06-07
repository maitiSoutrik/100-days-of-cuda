#include "frobenius_norm.cuh"

// Kernel to compute sum of squares of matrix elements
__global__ void sumOfSquaresKernel(const float* d_matrix, int rows, int cols, float* d_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;

    if (idx < total_elements) {
        atomicAdd(d_sum, d_matrix[idx] * d_matrix[idx]);
    }
}

// Kernel to compute sum of squares of matrix elements using shared memory for reduction
__global__ void sumOfSquaresReductionKernel(const float* d_matrix, int rows, int cols, float* d_partial_sums) {
    extern __shared__ float sdata[]; // Shared memory for partial sums within a block

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;

    // Each thread loads one element from global to shared memory
    if (i < total_elements) {
        sdata[tid] = d_matrix[i] * d_matrix[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads(); // Ensure all elements are loaded

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // Ensure all additions are complete before next iteration
    }

    // Write result for this block to global memory
    if (tid == 0) {
        d_partial_sums[blockIdx.x] = sdata[0];
    }
}


float frobeniusNormGPU(const float* d_matrix, int rows, int cols) {
    int total_elements = rows * cols;
    if (total_elements == 0) return 0.0f;

    float* d_partial_sums;
    float h_final_sum = 0.0f;

    // Determine grid and block sizes
    // Using a reduction kernel, so blockDim.x should be a power of 2, max 1024
    int threads_per_block = 256; // Can be tuned
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_partial_sums, num_blocks * sizeof(float)));

    // First kernel launch: each block computes a partial sum of squares
    sumOfSquaresReductionKernel<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
        d_matrix, rows, cols, d_partial_sums
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // If num_blocks is large, we might need a second reduction step.
    // For simplicity here, we'll copy partial sums to host and sum them.
    // A more robust solution would involve a recursive reduction on the GPU.
    if (num_blocks > 1) {
        float* h_partial_sums = (float*)malloc(num_blocks * sizeof(float));
        if (h_partial_sums == NULL) {
            fprintf(stderr, "Failed to allocate host memory for partial sums.\n");
            cudaFree(d_partial_sums);
            exit(EXIT_FAILURE);
        }
        CHECK_CUDA_ERROR(cudaMemcpy(h_partial_sums, d_partial_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < num_blocks; ++i) {
            h_final_sum += h_partial_sums[i];
        }
        free(h_partial_sums);
    } else if (num_blocks == 1) { // Only one block, its sum is the final sum
        CHECK_CUDA_ERROR(cudaMemcpy(&h_final_sum, d_partial_sums, sizeof(float), cudaMemcpyDeviceToHost));
    }
    // If total_elements was 0, num_blocks would be 0, h_final_sum remains 0.0f

    CHECK_CUDA_ERROR(cudaFree(d_partial_sums));

    return sqrtf(h_final_sum);
}

float frobeniusNormCPU(const float* h_matrix, int rows, int cols) {
    double sum_sq = 0.0; // Use double for intermediate sum to maintain precision
    int total_elements = rows * cols;
    for (int i = 0; i < total_elements; ++i) {
        sum_sq += (double)h_matrix[i] * h_matrix[i];
    }
    return sqrtf((float)sum_sq);
}
