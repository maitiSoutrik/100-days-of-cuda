#include "hinge_loss.cuh"
#include <cooperative_groups.h> // For cooperative group reduction if needed, or use simpler reduction

namespace cg = cooperative_groups;

// Kernel to compute Hinge Loss for each element
__global__ void hinge_loss_kernel(const int* true_labels, const float* pred_scores, float* loss, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float t_y = (float)true_labels[idx] * pred_scores[idx];
        loss[idx] = max(0.0f, 1.0f - t_y);
    }
}

// Host function to launch the Hinge Loss kernel
void hinge_loss_cuda(const int* d_true_labels, const float* d_pred_scores, float* d_loss, int num_elements) {
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    hinge_loss_kernel<<<blocks_per_grid, threads_per_block>>>(d_true_labels, d_pred_scores, d_loss, num_elements);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure kernel completion
}


// Simple sum reduction kernel (can be optimized further with shared memory / warp shuffle)
__global__ void sum_kernel(const float* data, float* out_sum, int n) {
    // This is a very basic sum reduction, not optimized for performance.
    // For production, a more sophisticated reduction (e.g., using shared memory, warp shuffles, or CUB/Thrust) would be used.
    // This kernel assumes 'out_sum' is initialized to 0 on the host before calling if it's a partial sum.
    // Or, if it's a single-block reduction, the block can write its sum.
    // For simplicity here, we'll assume a single block reduction for small N or a multi-stage reduction for large N.
    // This example will be a naive global sum, which is inefficient.
    // A proper reduction would be multi-stage.
    
    // For this example, let's assume n is small enough that a single block can sum it,
    // or this is one stage of a larger reduction.
    // A more robust implementation would handle arbitrary N.
    // This kernel is illustrative and not a high-performance reduction.
    
    // Let's make this a single-threaded sum for simplicity of the example,
    // acknowledging it's not how a real GPU sum is done.
    // The `sum_hinge_loss_cuda` will use a temporary array for individual losses first.
    if (threadIdx.x == 0 && blockIdx.x == 0) { // Only one thread does the sum
        float sum_val = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum_val += data[i];
        }
        *out_sum = sum_val;
    }
}


// Host function to compute and sum Hinge Loss
void sum_hinge_loss_cuda(const int* d_true_labels, const float* d_pred_scores, float* d_total_loss, int num_elements, float* d_temp_storage_for_individual_losses) {
    // Step 1: Compute individual hinge losses and store them in d_temp_storage_for_individual_losses
    int threads_per_block = 256;
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    hinge_loss_kernel<<<blocks_per_grid, threads_per_block>>>(d_true_labels, d_pred_scores, d_temp_storage_for_individual_losses, num_elements);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Step 2: Sum the individual losses from d_temp_storage_for_individual_losses
    // For simplicity, this example uses a naive sum kernel.
    // In a real application, use a library like CUB or Thrust for efficient reduction,
    // or implement a more optimized parallel reduction kernel.
    
    // The sum_kernel provided is extremely naive. A better approach for summing on GPU:
    // 1. Each block sums a portion of the array into shared memory.
    // 2. One thread per block writes the block's partial sum to global memory.
    // 3. Repeat if necessary, or copy the partial sums to CPU and sum there.
    // Or, use a single kernel with atomicAdd for small arrays (contention issues) or a library.

    // Given the constraints and to keep it runnable, we'll perform the sum by copying
    // the individual losses back to the host and summing there. This is inefficient for large
    // num_elements but simple to implement for this example.
    // A proper GPU reduction is complex and beyond a quick example if not using libraries.

    float* h_individual_losses = (float*)malloc(num_elements * sizeof(float));
    if (h_individual_losses == nullptr) {
        fprintf(stderr, "Failed to allocate host memory for individual losses\n");
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_ERROR(cudaMemcpy(h_individual_losses, d_temp_storage_for_individual_losses, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    double total_loss_cpu = 0.0; // Use double for host sum to maintain precision
    for (int i = 0; i < num_elements; ++i) {
        total_loss_cpu += h_individual_losses[i];
    }
    free(h_individual_losses);

    // Copy the final sum back to the device pointer d_total_loss
    float final_sum_float = (float)total_loss_cpu;
    CHECK_CUDA_ERROR(cudaMemcpy(d_total_loss, &final_sum_float, sizeof(float), cudaMemcpyHostToDevice));

    // Note: The sum_kernel is not used in this simplified sum path to ensure correctness
    // without implementing a full parallel reduction. The d_temp_storage parameter in the
    // function signature was intended for a GPU-side reduction's intermediate results,
    // but here it's used to store individual losses before summing on CPU.
}
