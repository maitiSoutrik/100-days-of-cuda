#include "jsd_loss.cuh"
#include <cmath> // For logf, fabsf
#include <cstdio> // For printf in kernels (debug)
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Epsilon for numerical stability
const float DEFAULT_EPSILON = 1e-8f;

// Helper device function for KL divergence component
// D_KL(p || m) = p * log(p/m)
__device__ float kl_divergence_element(float val_dist, float val_m, float epsilon) {
    // Ensure val_dist and val_m are positive and non-zero to avoid log(0) or division by zero
    val_dist = fmaxf(val_dist, epsilon);
    val_m = fmaxf(val_m, epsilon);
    if (val_dist <= epsilon) return 0.0f; // if p is zero, contribution is zero
    return val_dist * logf(val_dist / val_m);
}

// Kernel for JSD forward and backward pass per distribution (row)
// Each block processes one distribution (one row of P and Q)
__global__ void jsd_loss_kernel(const float* P, const float* Q,
                                float* d_per_row_loss,
                                float* d_grad_P, float* d_grad_Q,
                                int num_distributions, int num_elements,
                                float beta, float epsilon) {
    int row_idx = blockIdx.x;
    if (row_idx >= num_distributions) return;

    extern __shared__ float s_data[]; // Shared memory for reduction

    float row_loss_sum = 0.0f;

    // Phase 1: Compute M and per-element KL contributions, sum them up for the row
    // Each thread handles multiple elements if num_elements > blockDim.x
    for (int elem_idx = threadIdx.x; elem_idx < num_elements; elem_idx += blockDim.x) {
        int flat_idx = row_idx * num_elements + elem_idx;
        float p_val = P[flat_idx];
        float q_val = Q[flat_idx];

        // M = beta * P + (1 - beta) * Q (element-wise for the current row)
        // However, the JSD definition is JSD(P||Q) = pi_1 * D_KL(P||M) + pi_2 * D_KL(Q||M)
        // where M = pi_1*P + pi_2*Q.
        // For generalized JSD with parameter beta:
        // M_val = beta * p_val + (1-beta) * q_val is NOT the standard mixture for JSD.
        // The standard JSD is JSD(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M) with M = 0.5*(P+Q)
        // The problem asks for a "generalized" JSD with beta:
        // beta = 1.0 -> D_KL(P || M_pq) where M_pq = 0.5 * (P+Q) (This seems to be a specific interpretation)
        // beta = 0.0 -> D_KL(Q || M_pq)
        // beta = 0.5 -> 0.5 * D_KL(P || M_pq) + 0.5 * D_KL(Q || M_pq)
        // Let's assume M is always the average M = 0.5 * (P+Q)
        // And beta controls which KL term is active or how they are weighted.

        float m_val = 0.5f * (p_val + q_val);
        m_val = fmaxf(m_val, epsilon); // Ensure M is not zero

        float kl_p_m = kl_divergence_element(p_val, m_val, epsilon);
        float kl_q_m = kl_divergence_element(q_val, m_val, epsilon);

        float current_jsd_contrib = 0.0f;
        if (fabsf(beta - 0.5f) < epsilon) { // Symmetric JSD
            current_jsd_contrib = 0.5f * kl_p_m + 0.5f * kl_q_m;
        } else if (fabsf(beta - 1.0f) < epsilon) { // Forward KL-like (P || M)
            current_jsd_contrib = kl_p_m;
        } else if (fabsf(beta - 0.0f) < epsilon) { // Reverse KL-like (Q || M)
            current_jsd_contrib = kl_q_m;
        } else { // Weighted combination (custom interpretation if needed)
             // For this implementation, stick to the three cases or define a clear general form.
             // Defaulting to symmetric if beta is not 0, 0.5, or 1.
             // Or, more generally: beta * D_KL(P||M) + (1-beta) * D_KL(Q||M)
             // Let's use this general form:
            current_jsd_contrib = beta * kl_p_m + (1.0f - beta) * kl_q_m;
        }
        row_loss_sum += current_jsd_contrib;

        // Backward pass: Gradients
        // d(JSD)/dP_i = beta * d(D_KL(P||M))/dP_i + (1-beta) * d(D_KL(Q||M))/dP_i
        // d(D_KL(P||M))/dP_i = log(P_i/M_i) + 1 - P_i/M_i * 0.5 (since dM/dP_i = 0.5)
        // d(D_KL(Q||M))/dP_i = - Q_i/M_i * 0.5
        // d(JSD)/dQ_i = beta * d(D_KL(P||M))/dQ_i + (1-beta) * d(D_KL(Q||M))/dQ_i
        // d(D_KL(P||M))/dQ_i = - P_i/M_i * 0.5
        // d(D_KL(Q||M))/dQ_i = log(Q_i/M_i) + 1 - Q_i/M_i * 0.5

        // Ensure p_val, q_val, m_val are safe for log
        float safe_p_val = fmaxf(p_val, epsilon);
        float safe_q_val = fmaxf(q_val, epsilon);
        // m_val is already fmaxf(m_val, epsilon)

        float log_p_m = logf(safe_p_val / m_val);
        float log_q_m = logf(safe_q_val / m_val);

        float grad_kl_p_m_dp = log_p_m + 1.0f - 0.5f * (safe_p_val / m_val); // d(P log(P/M)) / dP
        float grad_kl_q_m_dp = -0.5f * (safe_q_val / m_val);                 // d(Q log(Q/M)) / dP

        float grad_kl_p_m_dq = -0.5f * (safe_p_val / m_val);                 // d(P log(P/M)) / dQ
        float grad_kl_q_m_dq = log_q_m + 1.0f - 0.5f * (safe_q_val / m_val); // d(Q log(Q/M)) / dQ
        
        float grad_p = 0.0f;
        float grad_q = 0.0f;

        if (fabsf(beta - 0.5f) < epsilon) { // Symmetric JSD
            grad_p = 0.5f * grad_kl_p_m_dp + 0.5f * grad_kl_q_m_dp;
            grad_q = 0.5f * grad_kl_p_m_dq + 0.5f * grad_kl_q_m_dq;
        } else if (fabsf(beta - 1.0f) < epsilon) { // Forward KL (P || M)
            grad_p = grad_kl_p_m_dp;
            grad_q = grad_kl_p_m_dq;
        } else if (fabsf(beta - 0.0f) < epsilon) { // Reverse KL (Q || M)
            grad_p = grad_kl_q_m_dp;
            grad_q = grad_kl_q_m_dq;
        } else { // General weighted case: beta * D_KL(P||M) + (1-beta) * D_KL(Q||M)
            grad_p = beta * grad_kl_p_m_dp + (1.0f - beta) * grad_kl_q_m_dp;
            grad_q = beta * grad_kl_p_m_dq + (1.0f - beta) * grad_kl_q_m_dq;
        }

        // Normalize gradients by num_elements because the final loss is an average per distribution
        // Or, if the final loss is a sum over distributions, then this normalization is not needed here,
        // but rather when combining d_per_row_loss.
        // Assuming d_per_row_loss is the sum for that row, and final loss is sum of d_per_row_loss.
        d_grad_P[flat_idx] = grad_p;
        d_grad_Q[flat_idx] = grad_q;
    }

    // Reduction within the block for row_loss_sum
    s_data[threadIdx.x] = row_loss_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_per_row_loss[row_idx] = s_data[0];
    }
}


// Kernel for sum reduction to get the final scalar loss
// This is a generic sum reduction. For large N, a multi-pass reduction is better.
__global__ void sum_reduction_kernel(const float* d_input, float* d_output, int N) {
    extern __shared__ float s_cache[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < N) {
        s_cache[tid] = d_input[i];
    } else {
        s_cache[tid] = 0.0f;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_cache[tid] += s_cache[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        if (gridDim.x == 1) { // Only one block in the grid, this is the final sum.
            d_output[0] = s_cache[0];
        } else { // Multiple blocks in the grid, this block writes its partial sum to d_output[blockIdx.x].
            d_output[blockIdx.x] = s_cache[0];
        }
    }
}


void jsd_loss_gpu(const float* P, const float* Q, float* d_loss,
                  float* d_grad_P, float* d_grad_Q,
                  int num_distributions, int num_elements,
                  float beta, float epsilon) {

    if (epsilon <= 0) epsilon = DEFAULT_EPSILON;

    // Allocate memory for per-row losses
    float* d_per_row_loss;
    CHECK_CUDA_ERROR(cudaMalloc(&d_per_row_loss, num_distributions * sizeof(float)));

    // Kernel launch parameters for jsd_loss_kernel
    // One block per distribution (row). Threads per block can be tuned.
    int threads_per_block_jsd = 256; // Example, can be tuned
    if (num_elements < threads_per_block_jsd) {
        threads_per_block_jsd = num_elements; // Ensure not more threads than elements
    }
     // Ensure threads_per_block_jsd is power of 2 for reduction, or handle non-power-of-2 in reduction
    if ((threads_per_block_jsd & (threads_per_block_jsd - 1)) != 0 && threads_per_block_jsd > 1) {
        // Find next power of 2 if not already, or adjust reduction logic
        // For simplicity, let's assume it's a power of 2 or handled by reduction logic.
        // Or, more simply, ensure it's <= 1024 and a multiple of warp size (32)
        threads_per_block_jsd = (threads_per_block_jsd > 0) ? 1 << static_cast<int>(floor(log2(static_cast<float>(threads_per_block_jsd)))) : 1;
         if (threads_per_block_jsd == 0) threads_per_block_jsd = 1; // Handle num_elements = 0 or 1
    }


    dim3 blocks_jsd(num_distributions);
    dim3 threads_jsd(threads_per_block_jsd);
    size_t shared_mem_size_jsd = threads_per_block_jsd * sizeof(float); // For reduction in each block

    jsd_loss_kernel<<<blocks_jsd, threads_jsd, shared_mem_size_jsd>>>(
        P, Q, d_per_row_loss, d_grad_P, d_grad_Q,
        num_distributions, num_elements, beta, epsilon
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    // Reduction of d_per_row_loss to a single scalar d_loss
    // This part needs a robust reduction.
    // If num_distributions is large, a multi-stage reduction is needed.
    // For simplicity, if num_distributions is small (e.g., <= 1024), one kernel call can do it.
    int threads_per_block_reduce = 256; // Example
    if (num_distributions < threads_per_block_reduce) {
        threads_per_block_reduce = num_distributions;
    }
    if ((threads_per_block_reduce & (threads_per_block_reduce - 1)) != 0 && threads_per_block_reduce > 1) {
         threads_per_block_reduce = (threads_per_block_reduce > 0) ? 1 << static_cast<int>(floor(log2(static_cast<float>(threads_per_block_reduce)))) : 1;
         if (threads_per_block_reduce == 0) threads_per_block_reduce = 1;
    }


    int num_blocks_reduce = (num_distributions + threads_per_block_reduce - 1) / threads_per_block_reduce;
    size_t shared_mem_size_reduce = threads_per_block_reduce * sizeof(float);

    if (num_blocks_reduce == 1) {
        sum_reduction_kernel<<<1, threads_per_block_reduce, shared_mem_size_reduce>>>(
            d_per_row_loss, d_loss, num_distributions
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
    } else {
        // Multi-stage reduction needed
        float* d_partial_sums;
        CHECK_CUDA_ERROR(cudaMalloc(&d_partial_sums, num_blocks_reduce * sizeof(float)));
        sum_reduction_kernel<<<num_blocks_reduce, threads_per_block_reduce, shared_mem_size_reduce>>>(
            d_per_row_loss, d_partial_sums, num_distributions
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Second stage: reduce d_partial_sums to d_loss
        // For simplicity, if num_blocks_reduce is small, can copy to host and sum, or launch another kernel.
        // Let's assume for now num_blocks_reduce will be small enough for a second kernel pass.
        int threads_for_final_reduction = 256;
        if (num_blocks_reduce < threads_for_final_reduction) {
            threads_for_final_reduction = num_blocks_reduce;
        }
        if ((threads_for_final_reduction & (threads_for_final_reduction - 1)) != 0 && threads_for_final_reduction > 1) {
            threads_for_final_reduction = (threads_for_final_reduction > 0) ? 1 << static_cast<int>(floor(log2(static_cast<float>(threads_for_final_reduction)))) : 1;
            if (threads_for_final_reduction == 0) threads_for_final_reduction = 1;
        }

        sum_reduction_kernel<<<1, threads_for_final_reduction, threads_for_final_reduction * sizeof(float)>>>(
            d_partial_sums, d_loss, num_blocks_reduce
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaFree(d_partial_sums));
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Free intermediate memory
    CHECK_CUDA_ERROR(cudaFree(d_per_row_loss));
}

// Helper host function for KL divergence component (CPU version)
static float kl_divergence_element_cpu(float val_dist, float val_m, float epsilon) {
    // Ensure val_dist and val_m are positive and non-zero to avoid log(0) or division by zero
    val_dist = fmaxf(val_dist, epsilon);
    val_m = fmaxf(val_m, epsilon);
    if (val_dist <= epsilon) return 0.0f; // if p is zero, contribution is zero
    return val_dist * logf(val_dist / val_m);
}

// CPU implementation for forward pass (for benchmarking/verification)
float jsd_loss_forward_cpu(const std::vector<float>& h_P, const std::vector<float>& h_Q,
                           int num_distributions, int num_elements,
                           float beta, float epsilon) {
    if (epsilon <= 0) epsilon = DEFAULT_EPSILON;
    double total_loss = 0.0;

    for (int i = 0; i < num_distributions; ++i) {
        double row_loss = 0.0;
        for (int j = 0; j < num_elements; ++j) {
            int flat_idx = i * num_elements + j;
            float p_val = h_P[flat_idx];
            float q_val = h_Q[flat_idx];

            float m_val = 0.5f * (p_val + q_val);
            m_val = fmaxf(m_val, epsilon);

            float kl_p_m = kl_divergence_element_cpu(p_val, m_val, epsilon);
            float kl_q_m = kl_divergence_element_cpu(q_val, m_val, epsilon);
            
            float current_jsd_contrib = 0.0f;
            if (fabsf(beta - 0.5f) < epsilon) {
                current_jsd_contrib = 0.5f * kl_p_m + 0.5f * kl_q_m;
            } else if (fabsf(beta - 1.0f) < epsilon) {
                current_jsd_contrib = kl_p_m;
            } else if (fabsf(beta - 0.0f) < epsilon) {
                current_jsd_contrib = kl_q_m;
            } else {
                current_jsd_contrib = beta * kl_p_m + (1.0f - beta) * kl_q_m;
            }
            row_loss += current_jsd_contrib;
        }
        total_loss += row_loss;
    }
    // The problem asks for a scalar output, so this is the sum of losses over all distributions.
    // If an average is needed, divide by num_distributions.
    return static_cast<float>(total_loss);
}
