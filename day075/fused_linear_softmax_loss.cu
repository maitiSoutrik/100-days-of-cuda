#include "fused_linear_softmax_loss.cuh"
#include <cmath>      // For logf, expf
#include <cfloat>     // For FLT_MAX
#include <numeric>    // For std::accumulate (for CPU calculation if any)
#include <vector>     // For std::vector (for CPU calculation if any)
#include <iostream>   // For printf in wrapper

// Helper device function for block-wide sum reduction
__device__ float blockReduceSum(float val, float* s_reduction_scratch) {
    int tid = threadIdx.x;
    int bid = blockIdx.x; // Not used here, but good to have context
    int warpSize = 32; // Standard warp size

    // Each thread puts its value into shared memory
    s_reduction_scratch[tid] = val;
    __syncthreads();

    // Iterative reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduction_scratch[tid] += s_reduction_scratch[tid + s];
        }
        __syncthreads();
    }
    return s_reduction_scratch[0];
}

// Helper device function for block-wide max reduction
__device__ float blockReduceMax(float val, float* s_reduction_scratch) {
    int tid = threadIdx.x;
    s_reduction_scratch[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduction_scratch[tid] = fmaxf(s_reduction_scratch[tid], s_reduction_scratch[tid + s]);
        }
        __syncthreads();
    }
    return s_reduction_scratch[0];
}


__global__ void fused_linear_softmax_loss_kernel(
    const float* d_input_features,  // M x K
    const float* d_weights,         // N x K
    const float* d_bias,            // N
    const int* d_true_labels,       // M
    float* d_output_loss_per_sample, // M
    int M,
    int K,
    int N
) {
    extern __shared__ float s_mem[];
    float* s_logits_dyn = s_mem; // First N floats for logits
    float* s_reduction_scratch_dyn = &s_mem[N]; // Next blockDim.x floats for reduction scratch space

    int m = blockIdx.x; // Current sample index (0 to M-1)
    if (m >= M) return;

    // Step 1: Compute N logits for sample m and store in s_logits_dyn
    // Each thread in the block computes a portion of the N logits.
    // s_logits_dyn is of size N.
    for (int n_idx = threadIdx.x; n_idx < N; n_idx += blockDim.x) {
        float current_logit = 0.0f;
        // MatMul: input_row[m] (1xK) * weights_col[n_idx] (Kx1)
        // weights are N x K, so weights[n_idx * K + k_idx]
        for (int k_idx = 0; k_idx < K; ++k_idx) {
            current_logit += d_input_features[m * K + k_idx] * d_weights[n_idx * K + k_idx];
        }
        current_logit += d_bias[n_idx];
        s_logits_dyn[n_idx] = current_logit;
    }
    __syncthreads(); // Ensure all N logits for sample m are computed and in s_logits_dyn

    // Step 2: Softmax and Loss calculation using s_logits_dyn (N elements)

    // a. Find max_logit in s_logits_dyn[0...N-1]
    float thread_max_logit = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        thread_max_logit = fmaxf(thread_max_logit, s_logits_dyn[i]);
    }
    float max_logit = blockReduceMax(thread_max_logit, s_reduction_scratch_dyn);
    // After blockReduceMax, max_logit is consistent for all threads in the block (it's s_reduction_scratch_dyn[0])
    // No __syncthreads() needed here if blockReduceMax includes it before returning s_reduction_scratch_dyn[0]
    // and all threads read s_reduction_scratch_dyn[0]. Our impl does.

    // b. Get logit_true_class (original value from s_logits_dyn)
    int true_class_idx = d_true_labels[m];
    // Ensure true_class_idx is valid
    // This read must happen *before* s_logits_dyn might be overwritten if it were used for sum_exp calc
    float logit_tc = (true_class_idx >= 0 && true_class_idx < N) ? s_logits_dyn[true_class_idx] : 0.0f;


    // c. Compute sum_exp_shifted_logits = sum(exp(s_logits_dyn[j] - max_logit)) for j in [0, N-1]
    float thread_sum_exp_shifted = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        thread_sum_exp_shifted += expf(s_logits_dyn[i] - max_logit);
    }
    float sum_exp_shifted = blockReduceSum(thread_sum_exp_shifted, s_reduction_scratch_dyn);
    // sum_exp_shifted is now s_reduction_scratch_dyn[0]

    // d. Compute loss for sample m: loss = log(sum_exp_shifted) - (logit_true_class - max_logit)
    float loss_m = 0.0f;
    if (sum_exp_shifted > 0) { // Avoid log(0) or log(negative)
         loss_m = logf(sum_exp_shifted) - (logit_tc - max_logit);
    } else {
        // This case should ideally not happen with proper inputs if N > 0.
        // Could indicate numerical issues or all logits being extremely small.
        // Assign a large loss or handle as an error.
        loss_m = FLT_MAX; // Or some other indicator of error/problem
    }
   

    // e. Store loss (only one thread per block writes the result for sample m)
    if (threadIdx.x == 0) {
        d_output_loss_per_sample[m] = loss_m;
    }
}


float compute_fused_linear_softmax_loss_gpu(
    const float* h_input_features,
    const float* h_weights,
    const float* h_bias,
    const int* h_true_labels,
    float* h_output_loss_per_sample,
    int M,
    int K,
    int N
) {
    float *d_input_features, *d_weights, *d_bias, *d_output_loss_per_sample_device;
    int *d_true_labels_device;

    // Allocate memory on device
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input_features, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_weights, N * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_bias, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_true_labels_device, M * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output_loss_per_sample_device, M * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_features, h_input_features, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, h_weights, N * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias, h_bias, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_true_labels_device, h_true_labels, M * sizeof(int), cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int THREADS_PER_BLOCK = 256; // Common choice for block size
    dim3 gridDim(M); // M blocks, one for each input sample
    dim3 blockDim(THREADS_PER_BLOCK);

    // Dynamic shared memory size:
    // N floats for s_logits_dyn (logits for one sample)
    // THREADS_PER_BLOCK floats for s_reduction_scratch_dyn (scratch space for reductions)
    size_t dynamic_shared_mem_size = (N + THREADS_PER_BLOCK) * sizeof(float);

    // Launch the kernel
    fused_linear_softmax_loss_kernel<<<gridDim, blockDim, dynamic_shared_mem_size>>>(
        d_input_features, d_weights, d_bias, d_true_labels_device, d_output_loss_per_sample_device, M, K, N
    );

    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for errors during kernel launch
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    // Copy results from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_loss_per_sample, d_output_loss_per_sample_device, M * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_input_features));
    CHECK_CUDA_ERROR(cudaFree(d_weights));
    CHECK_CUDA_ERROR(cudaFree(d_bias));
    CHECK_CUDA_ERROR(cudaFree(d_true_labels_device));
    CHECK_CUDA_ERROR(cudaFree(d_output_loss_per_sample_device));

    // Calculate average loss on CPU
    double total_loss = 0.0;
    for (int i = 0; i < M; ++i) {
        total_loss += h_output_loss_per_sample[i];
    }
    
    if (M == 0) return 0.0f;
    return static_cast<float>(total_loss / M);
}

// The kernel uses `extern __shared__ float s_mem[];` and calculates pointers
// (s_logits_dyn, s_reduction_scratch_dyn) to partitions of this shared memory.
