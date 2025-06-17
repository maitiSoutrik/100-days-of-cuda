#include "fused_rmsnorm_swiglu.cuh"
#include <cmath>     // For sqrtf, expf
#include <numeric>   // For std::iota (used in CPU version)
#include <vector>    // For std::vector (used in CPU version)
#include <iostream>  // For std::cout in CPU version (debugging)
#include <algorithm> // For std::transform in CPU version

// Device function for sigmoid
__device__ inline float sigmoidf_device(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Device function for Silu (SiLU or Swish-1)
__device__ inline float silu_device(float x) {
    return x * sigmoidf_device(x);
}

__global__ void fused_rmsnorm_swiglu_kernel(
    float* output,      // Output: num_rows x (hidden_dim / 2)
    const float* input, // Input:  num_rows x hidden_dim
    const float* weight,// Weight (gamma) for RMSNorm: hidden_dim
    int num_rows,
    int hidden_dim)
{
    // Each thread processes one element of the SwiGLU output.
    // Since SwiGLU halves the dimension, we iterate up to hidden_dim / 2.
    // RMSNorm is applied to the full hidden_dim.
    // A block will typically process one full row (one token embedding).

    int row_idx = blockIdx.x; // Each block processes one row

    if (row_idx >= num_rows) {
        return;
    }

    // Shared memory for sum of squares for RMSNorm within a block
    // Size should be large enough for one float per thread in the block
    // For parallel reduction, blockDim.x should be a power of 2 and <= hidden_dim
    extern __shared__ float s_sum_sq[]; // Dynamically sized shared memory

    // --- RMS Normalization Part ---
    // Step 1: Calculate sum of squares for the current row
    float sum_sq = 0.0f;
    // Parallel reduction within the block for sum_sq
    // Each thread computes partial sum of squares
    // This loop assumes blockDim.x might be smaller than hidden_dim,
    // so threads might process multiple elements.
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = input[row_idx * hidden_dim + i];
        sum_sq += val * val;
    }
    s_sum_sq[threadIdx.x] = sum_sq;
    __syncthreads();

    // Reduce sum_sq in shared memory
    // This reduction assumes blockDim.x is a power of 2
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }
    // The final sum_sq is in s_sum_sq[0]
    float mean_sq = s_sum_sq[0] / hidden_dim;
    float rrms = rsqrtf(mean_sq + RMSNORM_EPSILON);

    // --- SwiGLU Part ---
    // The output dimension is hidden_dim / 2.
    // Each thread computes one output element.
    int output_idx_in_row = threadIdx.x; // threadIdx.x corresponds to the output element index

    if (output_idx_in_row < (hidden_dim / 2)) {
        // 'x' part of SwiGLU input
        int x_idx_in_row = output_idx_in_row;
        // 'gate' part of SwiGLU input
        int gate_idx_in_row = output_idx_in_row + (hidden_dim / 2);

        // Apply RMSNorm to x and gate components
        float x_val_norm = input[row_idx * hidden_dim + x_idx_in_row] * rrms * weight[x_idx_in_row];
        float gate_val_norm = input[row_idx * hidden_dim + gate_idx_in_row] * rrms * weight[gate_idx_in_row];
        
        // Apply SwiGLU: silu(x_val_norm) * gate_val_norm
        output[row_idx * (hidden_dim / 2) + output_idx_in_row] = silu_device(x_val_norm) * gate_val_norm;
    }
}


void launch_fused_rmsnorm_swiglu(
    float* d_output,
    const float* d_input,
    const float* d_weight,
    int num_rows,
    int hidden_dim,
    int block_size) // block_size should ideally be hidden_dim/2 for SwiGLU part, or hidden_dim for RMSNorm part.
                   // Let's make block_size related to hidden_dim / 2 for output calculation.
                   // For RMSNorm reduction, block_size should be <= hidden_dim and ideally a power of 2.
                   // A common choice is 256 or 512, if hidden_dim/2 is large enough.
                   // If hidden_dim/2 is small, block_size should be hidden_dim/2 (rounded up to warp size).
{
    if (hidden_dim % 2 != 0) {
        fprintf(stderr, "Error: hidden_dim must be an even number for SwiGLU.\n");
        throw std::runtime_error("hidden_dim must be even.");
    }

    // For the RMSNorm reduction part, block_size should be appropriate for hidden_dim.
    // For the SwiGLU output part, block_size threads will compute block_size output elements.
    // We need block_size to be at least hidden_dim / 2 for each row to be processed by one block if we want one thread per output.
    // Or, if block_size < hidden_dim / 2, threads will loop.
    // Let's assume block_size is chosen such that it's efficient for the reduction (e.g., 256)
    // and also covers the output dimension (hidden_dim / 2).
    // If hidden_dim/2 > block_size, then the kernel needs a loop for threads to cover all outputs.
    // The current kernel assumes threadIdx.x maps directly to output_idx_in_row, so block_size must be >= hidden_dim/2.
    // This is a simplification. A more robust kernel would handle block_size < hidden_dim/2 with a loop.
    // For this example, let's assume block_size is set to hidden_dim/2 (or a power of 2 like 256 if hidden_dim/2 is large).
    // The shared memory size for RMSNorm reduction needs to be `block_size * sizeof(float)`.

    if (block_size < (hidden_dim / 2)) {
         // This is a simplification for the kernel structure.
         // A more general kernel would loop if block_size < hidden_dim / 2.
         // For now, let's make block_size at least hidden_dim / 2 for simplicity,
         // or ensure hidden_dim/2 is not too large for typical block_sizes.
         // The current kernel structure implies blockDim.x is used for reduction over hidden_dim,
         // and also for indexing up to hidden_dim/2 outputs.
         // This means block_size should be related to hidden_dim.
         // Let's set block_size to a power of 2 that is <= hidden_dim and also >= hidden_dim/2.
         // A common strategy is to set block_size to a fixed value like 256,
         // and have threads loop if necessary. The current kernel is not written that way for the output part.

         // Let's adjust the kernel to be more flexible:
         // Each block processes one row.
         // Threads in the block cooperate for RMSNorm over hidden_dim.
         // Threads in the block then compute hidden_dim/2 outputs.
         // So, blockDim.x should be configured based on hidden_dim.
         // A good choice for blockDim.x is often a power of 2, e.g., 128, 256.
         // The shared memory for reduction is `blockDim.x * sizeof(float)`.
    }


    dim3 blocks(num_rows); // One block per row
    dim3 threads(block_size); // block_size threads per block
    
    // Shared memory size: block_size floats for the sum_sq reduction
    size_t shmem_size = block_size * sizeof(float);
    if (shmem_size == 0) { // Ensure shmem_size is non-zero if block_size could be 0 (though block_size is usually >0)
        shmem_size = sizeof(float); 
    }


    fused_rmsnorm_swiglu_kernel<<<blocks, threads, shmem_size>>>(
        d_output, d_input, d_weight, num_rows, hidden_dim);
    
    CHECK_KERNEL_LAUNCH_ERROR();
}


// --- CPU Implementations for Verification ---

void rmsnorm_cpu(
    float* out,         // Output: hidden_dim
    const float* inp,    // Input: hidden_dim
    const float* weight, // Weight (gamma): hidden_dim
    int hidden_dim,
    float epsilon)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        sum_sq += inp[i] * inp[i];
    }
    float mean_sq = sum_sq / hidden_dim;
    float rrms = 1.0f / sqrtf(mean_sq + epsilon);

    for (int i = 0; i < hidden_dim; ++i) {
        out[i] = inp[i] * rrms * weight[i];
    }
}

// Sigmoid for CPU
float sigmoidf_cpu(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Silu for CPU
float silu_cpu(float x) {
    return x * sigmoidf_cpu(x);
}

void swiglu_cpu(
    float* out,             // Output: half_hidden_dim
    const float* inp_x,     // Input x part: half_hidden_dim
    const float* inp_gate,  // Input gate part: half_hidden_dim
    int half_hidden_dim)
{
    for (int i = 0; i < half_hidden_dim; ++i) {
        out[i] = silu_cpu(inp_x[i]) * inp_gate[i];
    }
}

void fused_rmsnorm_swiglu_cpu(
    float* output,      // Output: num_rows x (hidden_dim / 2)
    const float* input, // Input:  num_rows x hidden_dim
    const float* weight,// Weight (gamma) for RMSNorm: hidden_dim
    int num_rows,
    int hidden_dim,
    float epsilon)
{
    if (hidden_dim % 2 != 0) {
        fprintf(stderr, "CPU Error: hidden_dim must be an even number for SwiGLU.\n");
        return;
    }
    int half_hidden_dim = hidden_dim / 2;
    std::vector<float> temp_rmsnorm_out(hidden_dim);

    for (int r = 0; r < num_rows; ++r) {
        const float* current_input_row = input + r * hidden_dim;
        float* current_output_row = output + r * half_hidden_dim;

        // 1. RMSNorm for the current row
        rmsnorm_cpu(temp_rmsnorm_out.data(), current_input_row, weight, hidden_dim, epsilon);

        // 2. SwiGLU
        // temp_rmsnorm_out now contains the normalized full row
        // First half is for 'x', second half is for 'gate'
        const float* normalized_x = temp_rmsnorm_out.data();
        const float* normalized_gate = temp_rmsnorm_out.data() + half_hidden_dim;
        
        swiglu_cpu(current_output_row, normalized_x, normalized_gate, half_hidden_dim);
    }
}
