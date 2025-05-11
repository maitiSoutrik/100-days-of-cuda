#include "batched_l2_norm.cuh"
#include <cmath> // For sqrtf, fabsf
#include <vector>
#include <numeric> // For std::iota
#include <algorithm> // For std::transform, std::accumulate
#include <iostream> // For std::cout, std::endl for debugging in host code

// Kernel implementation
__global__ void batched_l2_norm_kernel(const float* d_vectors,
                                       float* d_norms,
                                       int num_batches,
                                       int vector_dim) {
    // Each block processes one vector from the batch
    int batch_idx = blockIdx.x;
    if (batch_idx >= num_batches) {
        return;
    }

    // Shared memory for reduction within the block
    // Size should be blockDim.x, but ensure it's large enough for typical block sizes
    // Max typical block size is 1024.
    extern __shared__ float s_data[];

    // Calculate offset for the current vector in the batch
    const float* current_vector_start = d_vectors + batch_idx * vector_dim;

    // Each thread calculates sum of squares for a portion of the vector
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < vector_dim; i += blockDim.x) {
        float val = current_vector_start[i];
        sum_sq += val * val;
    }
    s_data[threadIdx.x] = sum_sq;
    __syncthreads();

    // Perform reduction in shared memory
    // This is a common parallel reduction pattern
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result (sqrt of sum of squares) for this batch
    if (threadIdx.x == 0) {
        d_norms[batch_idx] = sqrtf(s_data[0]);
    }
}

// Wrapper function to launch the kernel
void compute_batched_l2_norm_gpu(const float* h_vectors,
                                 float* h_norms,
                                 int num_batches,
                                 int vector_dim) {
    if (num_batches == 0) {
        // No work to do, h_norms should remain empty or as is.
        // The test expects h_gpu_norms to be size 0 if num_batches is 0.
        // The caller should ensure h_norms is appropriately sized or handled.
        return;
    }

    size_t vector_data_size = num_batches * vector_dim * sizeof(float);
    size_t norm_data_size = num_batches * sizeof(float);

    float* d_vectors;
    float* d_norms;

    CHECK_CUDA_ERROR(cudaMalloc(&d_vectors, vector_data_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_norms, norm_data_size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_vectors, h_vectors, vector_data_size, cudaMemcpyHostToDevice));

    // Determine block size and grid size
    // Let's use a block size that's a power of 2, e.g., 256.
    // This is a common choice and good for shared memory reductions.
    // The block size should ideally be tuned based on vector_dim and GPU architecture.
    // If vector_dim is small, a smaller block size might be better.
    // If vector_dim is large, a larger block size helps process more elements per thread.
    int block_size = 256;
    if (vector_dim < 256 && vector_dim > 0) { // Adjust if vector_dim is small
        // Find next power of 2 for block_size if vector_dim is smaller than 256
        // or simply use vector_dim if it's small enough and a power of 2.
        // For simplicity, if vector_dim is very small, we might over-provision threads
        // or use a smaller block_size. Let's cap it at vector_dim if vector_dim < block_size.
        // However, for shared memory reduction, it's good to have block_size as power of 2.
        // Let's keep block_size = 256, threads will handle elements in a strided loop.
        // If vector_dim is very small (e.g. < 32), this might not be optimal.
        // For this example, we'll stick to 256.
    }


    // Grid size is the number of batches, as each block handles one batch.
    int grid_size = num_batches;

    // Shared memory size: block_size * sizeof(float)
    size_t shared_mem_size = block_size * sizeof(float);

    batched_l2_norm_kernel<<<grid_size, block_size, shared_mem_size>>>(
        d_vectors, d_norms, num_batches, vector_dim
    );
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    CHECK_CUDA_ERROR(cudaMemcpy(h_norms, d_norms, norm_data_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_vectors));
    CHECK_CUDA_ERROR(cudaFree(d_norms));
}

// CPU implementation for verification
void compute_batched_l2_norm_cpu(const float* h_vectors,
                                 float* h_norms_cpu,
                                 int num_batches,
                                 int vector_dim) {
    for (int i = 0; i < num_batches; ++i) {
        double sum_sq = 0.0; // Use double for intermediate sum for precision
        const float* current_vector_start = h_vectors + i * vector_dim;
        for (int j = 0; j < vector_dim; ++j) {
            sum_sq += static_cast<double>(current_vector_start[j]) * current_vector_start[j];
        }
        h_norms_cpu[i] = static_cast<float>(sqrt(sum_sq));
    }
}
