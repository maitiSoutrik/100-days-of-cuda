#ifndef JSD_LOSS_CUH
#define JSD_LOSS_CUH

#include <cuda_runtime.h>
#include <vector>

// Error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// Function declarations

/**
 * @brief Computes the generalized Jensen-Shannon Divergence (JSD) loss on the GPU.
 *
 * This function orchestrates the JSD computation, including forward and backward passes.
 *
 * @param P Pointer to the first input probability distribution matrix (device memory).
 * @param Q Pointer to the second input probability distribution matrix (device memory).
 * @param d_loss Pointer to store the computed JSD loss (scalar, device memory).
 * @param d_grad_P Pointer to store the gradient with respect to P (device memory).
 * @param d_grad_Q Pointer to store the gradient with respect to Q (device memory).
 * @param num_distributions Number of distributions (rows in P and Q).
 * @param num_elements Number of elements per distribution (columns in P and Q).
 * @param beta Generalization parameter for JSD.
 *             beta = 0.0 for D_KL(Q || M)
 *             beta = 1.0 for D_KL(P || M)
 *             beta = 0.5 for symmetric JSD: 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
 *             Other values for weighted combinations.
 * @param epsilon Small value to prevent log(0) and division by zero.
 */
void jsd_loss_gpu(const float* P, const float* Q, float* d_loss,
                  float* d_grad_P, float* d_grad_Q,
                  int num_distributions, int num_elements,
                  float beta, float epsilon = 1e-8f);

/**
 * @brief Computes the forward pass of the generalized JSD loss on the CPU.
 *
 * @param h_P Pointer to the first input probability distribution matrix (host memory).
 * @param h_Q Pointer to the second input probability distribution matrix (host memory).
 * @param num_distributions Number of distributions (rows in P and Q).
 * @param num_elements Number of elements per distribution (columns in P and Q).
 * @param beta Generalization parameter for JSD.
 * @param epsilon Small value to prevent log(0) and division by zero.
 * @return The computed JSD loss (scalar).
 */
float jsd_loss_forward_cpu(const std::vector<float>& h_P, const std::vector<float>& h_Q,
                           int num_distributions, int num_elements,
                           float beta, float epsilon = 1e-8f);

// Helper for KL divergence part of JSD
__device__ float kl_divergence_element(float p, float q, float m, float epsilon);

// Kernel for JSD forward and backward pass per distribution (row)
__global__ void jsd_loss_kernel(const float* P, const float* Q,
                                float* d_per_row_loss,
                                float* d_grad_P, float* d_grad_Q,
                                int num_distributions, int num_elements,
                                float beta, float epsilon);

// Kernel for sum reduction to get the final scalar loss
__global__ void sum_reduction_kernel(const float* d_input, float* d_output, int N);


#endif // JSD_LOSS_CUH
