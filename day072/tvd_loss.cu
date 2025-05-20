#include "tvd_loss.cuh"
#include <cmath>      // For fabs
#include <numeric>    // For std::accumulate (for CPU reduction, if needed)
#include <cuda_runtime.h>
#include <device_launch_parameters.h> // For blockIdx, threadIdx etc.

// CUDA Kernel to calculate sum of absolute differences
__global__ void sum_abs_diff_kernel(const float* p, const float* q, int n, float* partial_sums) {
    extern __shared__ float sdata[]; // Shared memory for partial sums within a block

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    if (tid < blockDim.x) { // Ensure tid is within bounds for sdata
        sdata[tid] = 0.0f;
    }

    // Each thread computes absolute difference for its assigned elements and accumulates
    if (i < n) {
        sdata[tid] = fabsf(p[i] - q[i]);
    }
    __syncthreads(); // Synchronize to ensure all sdata is loaded

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Simpler kernel if not using shared memory reduction, or for a second pass sum
__global__ void sum_kernel(const float* data, int n, float* out_sum) {
    // This kernel assumes 'n' is small enough for a single block, or it's summing partial sums
    // For simplicity in this example, we'll assume it's summing a small number of partial_sums
    // A more robust version would use multiple blocks and another reduction pass.
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? data[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // If multiple blocks are summing, this should be atomic or a final CPU sum
        // For this example, assuming one block for summing partial_sums
        *out_sum = sdata[0]; 
    }
}


void calculate_tvd_gpu(const float* d_p, const float* d_q, int n, float* d_tvd) {
    // Configuration
    const int threads_per_block = 256;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    // Allocate memory for partial sums (one per block)
    float* d_partial_sums;
    CHECK_CUDA_ERROR(cudaMalloc(&d_partial_sums, num_blocks * sizeof(float)));

    // Launch kernel to compute sum of absolute differences per block
    // Shared memory size: threads_per_block * sizeof(float)
    sum_abs_diff_kernel<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(d_p, d_q, n, d_partial_sums);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Sum the partial sums
    // If num_blocks is large, this would need another kernel launch or a more sophisticated reduction.
    // For simplicity, if num_blocks is small enough for one block to handle:
    if (num_blocks == 1) {
        // If only one block, its sum is the total sum
        // We still need to copy it to d_tvd and multiply by 0.5
        // Or, modify sum_abs_diff_kernel to write directly to d_tvd if blockIdx.x == 0 and num_blocks == 1
        // For now, let's assume a second kernel call for summing, even if it's just one element.
    }
    
    // Allocate memory for the final sum on device (if not already d_tvd)
    float* d_total_sum_abs_diff;
    CHECK_CUDA_ERROR(cudaMalloc(&d_total_sum_abs_diff, sizeof(float)));

    // Sum the partial sums. For simplicity, using a single block if num_blocks <= threads_per_block.
    // A robust implementation would handle larger num_blocks with multiple reduction steps.
    if (num_blocks <= threads_per_block) {
        sum_kernel<<<1, threads_per_block, threads_per_block * sizeof(float)>>>(d_partial_sums, num_blocks, d_total_sum_abs_diff);
    } else {
        // Fallback for larger num_blocks: copy to host and sum, or implement multi-stage reduction
        // This is a simplification for the example.
        std::vector<float> h_partial_sums(num_blocks);
        CHECK_CUDA_ERROR(cudaMemcpy(h_partial_sums.data(), d_partial_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
        float total_sum_cpu = 0.0f;
        for(int i=0; i < num_blocks; ++i) {
            total_sum_cpu += h_partial_sums[i];
        }
        CHECK_CUDA_ERROR(cudaMemcpy(d_total_sum_abs_diff, &total_sum_cpu, sizeof(float), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy the total sum to d_tvd and multiply by 0.5 (or do it in a kernel)
    // For simplicity, copy to host, calculate, then copy back if d_tvd must be on device.
    // Or, if d_tvd is the final destination for the sum:
    // (Assuming d_tvd can be used as the output for sum_kernel if num_blocks <= threads_per_block)
    // If d_tvd is distinct:
    // cudaMemcpy(d_tvd, d_total_sum_abs_diff, sizeof(float), cudaMemcpyDeviceToDevice);
    // Then a small kernel to multiply by 0.5:
    // multiply_by_scalar_kernel<<<1,1>>>(d_tvd, 0.5f);

    // Simpler: copy sum to host, calculate 0.5 * sum, copy back to d_tvd
    float h_total_sum_abs_diff;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_total_sum_abs_diff, d_total_sum_abs_diff, sizeof(float), cudaMemcpyDeviceToHost));
    
    float tvd_result = 0.5f * h_total_sum_abs_diff;
    CHECK_CUDA_ERROR(cudaMemcpy(d_tvd, &tvd_result, sizeof(float), cudaMemcpyHostToDevice));

    // Free intermediate device memory
    CHECK_CUDA_ERROR(cudaFree(d_partial_sums));
    CHECK_CUDA_ERROR(cudaFree(d_total_sum_abs_diff));
}

float calculate_tvd_cpu(const std::vector<float>& h_p, const std::vector<float>& h_q) {
    if (h_p.size() != h_q.size()) {
        fprintf(stderr, "Error: Input vectors must have the same size.\n");
        return -1.0f; // Indicate error
    }
    if (h_p.empty()) {
        return 0.0f; // Or handle as an error
    }

    double sum_abs_diff = 0.0; // Use double for precision in sum
    for (size_t i = 0; i < h_p.size(); ++i) {
        sum_abs_diff += std::fabs(h_p[i] - h_q[i]);
    }

    return 0.5f * static_cast<float>(sum_abs_diff);
}
