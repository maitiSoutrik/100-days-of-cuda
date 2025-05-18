#include "mse.cuh"
#include <cmath>     // For std::pow
#include <numeric>   // For std::accumulate (though not used in final kernel for sum)
#include <vector>    // For intermediate sum storage if needed

// CUDA Kernel for calculating squared errors and summing them up
__global__ void mse_kernel(const float* predictions, const float* targets, float* squared_errors, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float diff = predictions[idx] - targets[idx];
        squared_errors[idx] = diff * diff;
    }
}

// Kernel for sum reduction (a simple version, can be optimized further)
// This version uses a single block for summing, which is not optimal for very large N
// but simpler for demonstration. For larger N, a multi-block reduction is needed.
__global__ void sum_reduction_kernel(const float* data, float* out_sum, int N) {
    extern __shared__ float sdata[]; // Shared memory for partial sums

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // Global index

    // Load data into shared memory
    if (i < N) {
        sdata[tid] = data[i];
    } else {
        sdata[tid] = 0;
    }
    if (i + blockDim.x < N) {
        sdata[tid] += data[i + blockDim.x];
    }

    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        out_sum[blockIdx.x] = sdata[0];
    }
}


// CPU implementation of MSE
float mse_cpu(const float* predictions, const float* targets, int N) {
    if (N == 0) return 0.0f;
    double sum_sq_error = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff = static_cast<double>(predictions[i]) - static_cast<double>(targets[i]);
        sum_sq_error += diff * diff;
    }
    return static_cast<float>(sum_sq_error / N);
}

// GPU implementation of MSE
void mse_gpu(const float* h_predictions, const float* h_targets, int N, float* mse_result) {
    if (N == 0) {
        *mse_result = 0.0f;
        return;
    }

    float *d_predictions, *d_targets, *d_squared_errors;
    float *d_partial_sums, *d_total_sum;

    // Allocate memory on the device
    CHECK_CUDA_ERROR(cudaMalloc(&d_predictions, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_targets, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_squared_errors, N * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_predictions, h_predictions, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_targets, h_targets, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel to calculate squared errors
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    mse_kernel<<<blocks_per_grid, threads_per_block>>>(d_predictions, d_targets, d_squared_errors, N);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    // --- Sum reduction part ---
    // For simplicity, if N is small enough for one block reduction, we do that.
    // Otherwise, a more complex multi-block reduction is needed.
    // This example will use a simpler approach: sum on GPU then finish on CPU if many partial sums,
    // or a single reduction kernel if N is small.

    // Let's refine the reduction. We'll have sum_reduction_kernel produce partial sums.
    // If blocks_per_grid (from mse_kernel) is 1, then d_squared_errors already has all values,
    // and sum_reduction_kernel can sum them into a single value.
    // If blocks_per_grid > 1, then sum_reduction_kernel needs to be called multiple times or a multi-level reduction.

    // Number of blocks for the sum reduction kernel.
    // Each block will reduce a portion of d_squared_errors.
    // The size of shared memory for sum_reduction_kernel is threads_per_block * sizeof(float).
    int reduction_threads = 256; // Can be tuned
    int reduction_blocks = (N + (reduction_threads * 2) - 1) / (reduction_threads * 2);
    if (reduction_blocks == 0) reduction_blocks = 1; // Ensure at least one block

    CHECK_CUDA_ERROR(cudaMalloc(&d_partial_sums, reduction_blocks * sizeof(float)));

    // Launch sum reduction kernel
    // Shared memory size: reduction_threads * sizeof(float)
    sum_reduction_kernel<<<reduction_blocks, reduction_threads, reduction_threads * sizeof(float)>>>(
        d_squared_errors, d_partial_sums, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    float total_sum_gpu;
    if (reduction_blocks == 1) {
        // If only one block was used for reduction, the result is in d_partial_sums[0]
        CHECK_CUDA_ERROR(cudaMemcpy(&total_sum_gpu, d_partial_sums, sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        // If multiple blocks produced partial sums, we need to sum these partial sums.
        // For simplicity, copy partial sums to host and sum them there.
        // A more robust solution would do a second-level reduction on GPU.
        std::vector<float> h_partial_sums(reduction_blocks);
        CHECK_CUDA_ERROR(cudaMemcpy(h_partial_sums.data(), d_partial_sums, reduction_blocks * sizeof(float), cudaMemcpyDeviceToHost));
        
        double final_sum_host = 0.0;
        for(int i = 0; i < reduction_blocks; ++i) {
            final_sum_host += h_partial_sums[i];
        }
        total_sum_gpu = static_cast<float>(final_sum_host);
    }

    *mse_result = total_sum_gpu / N;

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    CHECK_CUDA_ERROR(cudaFree(d_targets));
    CHECK_CUDA_ERROR(cudaFree(d_squared_errors));
    CHECK_CUDA_ERROR(cudaFree(d_partial_sums));
    // d_total_sum was not used if reduction_blocks > 1, or was a host variable
}
