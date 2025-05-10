#ifndef FISHER_MATRIX_CUH
#define FISHER_MATRIX_CUH

#include <cstdio> // For printf in error macro
#include <cuda_runtime.h>

// CUDA error checking macro (already defined, but good to ensure it's here)
#ifndef CHECK_CUDA_ERROR // Ensure it's not redefined if included multiple times indirectly
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }
#endif

/**
 * @brief CUDA kernel to compute elements of the Fisher Information Matrix.
 * Each thread computes one element F_ij of the Fisher matrix.
 * F_ij = (1/n_samples) * sum_k (log_probs[k*n_params + i] * log_probs[k*n_params + j])
 * where log_probs[k*n_params + i] is the i-th component of the score vector for sample k.
 * 
 * @param log_probs Device pointer to the log probabilities (scores). 
 *                  Layout: [sample0_param0, sample0_param1, ..., sample1_param0, ...]
 *                  Total size: n_samples * n_params.
 * @param fisher_matrix Device pointer to the output Fisher matrix (flattened, row-major).
 *                      Total size: n_params * n_params.
 * @param n_samples Number of samples.
 * @param n_params Number of parameters.
 */
__global__ void fisher_kernel(const float* log_probs, float* fisher_matrix, 
                             int n_samples, int n_params);

/**
 * @brief Computes the Fisher Information Matrix using CUDA.
 * This function is a wrapper that handles memory management and kernel launch.
 * 
 * @param h_log_probs Host pointer to the log probabilities (scores).
 * @param h_fisher_matrix Host pointer to store the resulting Fisher matrix.
 * @param n_samples Number of samples.
 * @param n_params Number of parameters.
 */
void compute_fisher_matrix_gpu(const float* h_log_probs, float* h_fisher_matrix, 
                               int n_samples, int n_params);

/**
 * @brief CPU reference implementation for Fisher Information Matrix.
 * 
 * @param log_probs Host pointer to the log probabilities (scores).
 * @param fisher_matrix Host pointer to store the resulting Fisher matrix.
 * @param n_samples Number of samples.
 * @param n_params Number of parameters.
 */
void compute_fisher_matrix_cpu(const float* log_probs, float* fisher_matrix, 
                               int n_samples, int n_params);

#endif // FISHER_MATRIX_CUH
