#ifndef SPECTRAL_NORM_CUH
#define SPECTRAL_NORM_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <stdexcept> // For std::runtime_error

// Error checking macros
#define CHECK_CUDA_ERROR(call)                                                    \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            throw std::runtime_error(cudaGetErrorString(err));                    \
        }                                                                         \
    } while (0)

#define CHECK_CUBLAS_ERROR(call)                                                  \
    do {                                                                          \
        cublasStatus_t status = call;                                             \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuBLAS Error at %s:%d - %d\n", __FILE__, __LINE__,   \
                    status);                                                      \
            /* You might want a more descriptive error message for cuBLAS */      \
            char error_msg[256];                                                  \
            sprintf(error_msg, "cuBLAS error code %d", status);                   \
            throw std::runtime_error(error_msg);                                  \
        }                                                                         \
    } while (0)


/**
 * @brief Estimates the spectral norm (largest singular value) of a matrix W (m x n)
 *        using the power iteration method.
 *        W_op * v = W * v if transpose_W is false
 *        W_op * u = W^T * u if transpose_W is true
 *
 * @param handle cuBLAS handle.
 * @param W Device pointer to the matrix W (m rows, n columns, column-major).
 * @param m Number of rows of W.
 * @param n Number of columns of W.
 * @param u Device pointer to a vector of size m (workspace and result for W*v).
 * @param v Device pointer to a vector of size n (workspace and result for W^T*u).
 * @param iterations Number of power iterations to perform.
 * @return The estimated spectral norm of W.
 */
float estimate_spectral_norm(cublasHandle_t handle, const float* d_W, int m, int n,
                             float* d_u, float* d_v, int iterations = 10);

/**
 * @brief Normalizes a matrix W by its spectral norm. W_norm = W / sigma(W).
 *
 * @param handle cuBLAS handle.
 * @param d_W_in_out Device pointer to the matrix W (m rows, n columns, column-major).
 *                   This matrix will be modified in-place.
 * @param m Number of rows of W.
 * @param n Number of columns of W.
 * @param d_u Device pointer to a workspace vector of size m.
 * @param d_v Device pointer to a workspace vector of size n.
 * @param iterations Number of power iterations for spectral norm estimation.
 */
void spectral_normalize_matrix(cublasHandle_t handle, float* d_W_in_out, int m, int n,
                               float* d_u, float* d_v, int iterations = 10);

/**
 * @brief Kernel to normalize a matrix by a scalar value (element-wise division).
 *        matrix = matrix / scalar
 */
__global__ void scale_matrix_kernel(float* matrix, int num_elements, float scalar);

#endif // SPECTRAL_NORM_CUH
