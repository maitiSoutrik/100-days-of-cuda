#ifndef CGM_CUBLAS_CUH
#define CGM_CUBLAS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio> // For printf in error checking
#include <cstdlib> // For exit in error checking

// Error checking macros (consistent with project rules)
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS_ERROR(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d - Status code: %d\n", __FILE__, __LINE__, status); \
        /* You might want a function to convert cublasStatus_t to string if available/needed */ \
        exit(EXIT_FAILURE); \
    } \
} while (0)


/**
 * @brief Solves the linear system Ax = b using the Conjugate Gradient Method with cuBLAS.
 *
 * Assumes A is a symmetric positive-definite matrix.
 * All pointers (d_A, d_b, d_x) point to device memory.
 *
 * @param handle cuBLAS handle.
 * @param n Dimension of the square matrix A and vectors x, b.
 * @param d_A Pointer to the matrix A in device memory (column-major format expected by cuBLAS).
 * @param d_b Pointer to the vector b in device memory.
 * @param d_x Pointer to the initial guess for x, will contain the solution upon return (device memory).
 * @param max_iters Maximum number of iterations.
 * @param tolerance Convergence tolerance based on the residual norm.
 * @return int Number of iterations performed, or -1 if convergence not reached within max_iters.
 */
int conjugateGradientMethodCuBLAS(cublasHandle_t handle, int n, const double *d_A, const double *d_b, double *d_x,
                                  int max_iters, double tolerance);

#endif // CGM_CUBLAS_CUH
