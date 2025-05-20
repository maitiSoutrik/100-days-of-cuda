#include "group_norm_forward.cuh"
#include <cmath> // For sqrtf, fabsf
#include <vector> // For CPU implementation
#include <numeric> // For std::accumulate
#include <algorithm> // For std::fill

// Kernel to compute mean and variance per group, then normalize
__global__ void groupNormForwardKernel(
    float* output,
    const float* input,
    int N, int C, int H, int W, int G,
    const float* gamma,
    const float* beta,
    float epsilon) {

    int num_pixels = H * W;
    int channels_per_group = C / G;

    // Each thread block processes one group for one sample in the batch
    // blockIdx.x maps to sample index (N)
    // blockIdx.y maps to group index (G)
    int n = blockIdx.x;
    int group_idx = blockIdx.y;

    if (n >= N || group_idx >= G) {
        return;
    }

    // Calculate sum and sum of squares for the current group
    // Using shared memory for reduction within a block
    extern __shared__ float sdata[]; // sdata[0] for sum, sdata[1] for sum_sq

    // Each thread in the block processes a subset of elements in the group
    // A group consists of `channels_per_group * H * W` elements
    int group_size = channels_per_group * num_pixels;

    // Initialize shared memory
    if (threadIdx.x == 0) {
        sdata[0] = 0.0f; // sum
        sdata[1] = 0.0f; // sum_sq
    }
    __syncthreads();

    // Parallel reduction for sum and sum_sq
    // Iterate over all elements in the current group (n, group_idx)
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_offset = i / num_pixels; // Channel index within the group
        int pixel_offset = i % num_pixels; // Pixel index (h*W + w)

        int current_channel = group_idx * channels_per_group + c_offset;
        int data_idx = n * C * num_pixels + current_channel * num_pixels + pixel_offset;
        
        float val = input[data_idx];
        atomicAdd(&sdata[0], val);
        atomicAdd(&sdata[1], val * val);
    }
    __syncthreads();

    // Finalize mean and variance (thread 0 of the block)
    float mean = 0.0f;
    float variance = 0.0f;
    if (threadIdx.x == 0) {
        mean = sdata[0] / group_size;
        variance = (sdata[1] / group_size) - (mean * mean);
    }
    __syncthreads(); // Ensure all threads see the calculated mean and variance

    // Broadcast mean and variance (can be done by reading from sdata or by __syncthreads and then reading)
    // For simplicity, let's re-read from global or pass via registers if blockDim is small enough
    // Or, better, keep them in shared memory if accessible by all threads after calculation.
    // Here, we assume thread 0 has computed them and they are implicitly broadcasted by __syncthreads
    // or all threads can re-calculate them if needed, but that's inefficient.
    // A common pattern is for thread 0 to write to shared memory, then all threads read.
    // Let's assume mean and variance are now known to all threads in the block.
    // For this simple kernel, we'll have thread 0 write them back to sdata and others read.
    if (threadIdx.x == 0) {
        sdata[0] = mean;
        sdata[1] = variance;
    }
    __syncthreads();
    
    mean = sdata[0];
    variance = sdata[1];
    // Using 1.0f / sqrtf for potentially better precision matching with CPU
    float inv_stddev = 1.0f / sqrtf(variance + epsilon); 

    // Apply normalization and scale/shift
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_offset = i / num_pixels; // Channel index within the group
        int pixel_offset = i % num_pixels; // Pixel index (h*W + w)

        int current_channel = group_idx * channels_per_group + c_offset;
        int data_idx = n * C * num_pixels + current_channel * num_pixels + pixel_offset;
        
        float val = input[data_idx];
        float normalized_val = (val - mean) * inv_stddev;
        output[data_idx] = normalized_val * gamma[current_channel] + beta[current_channel];
    }
}


void groupNormForward(
    float* output_d,
    const float* input_d,
    int N, int C, int H, int W, int G,
    const float* gamma_d,
    const float* beta_d,
    float epsilon) {

    if (C % G != 0) {
        fprintf(stderr, "Error: Number of channels C (%d) must be divisible by number of groups G (%d).\n", C, G);
        exit(EXIT_FAILURE);
    }

    // Configure kernel launch parameters
    // One block per (sample, group) pair
    dim3 num_blocks(N, G); 
    
    // Threads per block: ideally, enough to cover elements in a group or a fraction
    // Max threads per block is 1024.
    // Let's use a fixed number of threads, e.g., 256 or 512.
    // The loop inside the kernel will handle cases where group_size > blockDim.x
    int threads_per_block = 256; 
    dim3 threads_per_block_dim(threads_per_block);

    // Shared memory: 2 floats for sum and sum_sq
    size_t shared_mem_size = 2 * sizeof(float); 

    groupNormForwardKernel<<<num_blocks, threads_per_block_dim, shared_mem_size>>>(
        output_d, input_d, N, C, H, W, G, gamma_d, beta_d, epsilon
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// CPU implementation for verification
void groupNormForwardCPU(
    float* output,
    const float* input,
    int N, int C, int H, int W, int G,
    const float* gamma,
    const float* beta,
    float epsilon) {

    if (C % G != 0) {
        fprintf(stderr, "CPU Error: Number of channels C (%d) must be divisible by number of groups G (%d).\n", C, G);
        return; // Or exit, depending on desired behavior
    }

    int num_pixels = H * W;
    int channels_per_group = C / G;
    int group_size = channels_per_group * num_pixels;

    for (int n = 0; n < N; ++n) { // Iterate over batch
        for (int g = 0; g < G; ++g) { // Iterate over groups
            // Calculate mean and variance for the current group (n, g)
            float sum = 0.0f;
            float sum_sq = 0.0f;

            for (int c_group = 0; c_group < channels_per_group; ++c_group) { // Iterate channels in group
                int current_channel_abs = g * channels_per_group + c_group;
                for (int p = 0; p < num_pixels; ++p) { // Iterate pixels
                    int data_idx = n * C * num_pixels + current_channel_abs * num_pixels + p;
                    float val = input[data_idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }

            float mean = sum / group_size;
            float variance = (sum_sq / group_size) - (mean * mean);
            float inv_stddev = 1.0f / sqrtf(variance + epsilon);

            // Apply normalization, scale, and shift
            for (int c_group = 0; c_group < channels_per_group; ++c_group) {
                int current_channel_abs = g * channels_per_group + c_group;
                for (int p = 0; p < num_pixels; ++p) {
                    int data_idx = n * C * num_pixels + current_channel_abs * num_pixels + p;
                    float val = input[data_idx];
                    float normalized_val = (val - mean) * inv_stddev;
                    output[data_idx] = normalized_val * gamma[current_channel_abs] + beta[current_channel_abs];
                }
            }
        }
    }
}
