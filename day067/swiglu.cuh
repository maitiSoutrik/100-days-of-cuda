#ifndef SWIGLU_CUH
#define SWIGLU_CUH

#include <cuda_runtime.h>
#include <cstdio> // For printf in error macro

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s at %s:%d\n", \
                    cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * @brief Computes the SwiGLU activation function forward pass.
 * c = silu(a) * b, where silu(a) = a * sigmoid(a)
 *
 * @param d_a Device pointer to input matrix a (rows x cols).
 * @param d_b Device pointer to input matrix b (rows x cols).
 * @param d_c Device pointer to output matrix c (rows x cols).
 * @param rows Number of rows in the matrices.
 * @param cols Number of columns in the matrices.
 */
__global__ void swiglu_forward_kernel(const float* d_a,
                                      const float* d_b,
                                      float* d_c,
                                      int rows,
                                      int cols);

/**
 * @brief Computes the SwiGLU activation function backward pass.
 * Calculates gradients da and db given dc.
 *
 * @param d_a Device pointer to input matrix a from forward pass (rows x cols).
 * @param d_b Device pointer to input matrix b from forward pass (rows x cols).
 * @param d_dc Device pointer to incoming gradient dc (gradient of loss w.r.t c) (rows x cols).
 * @param d_da Device pointer to output gradient da (gradient of loss w.r.t a) (rows x cols).
 * @param d_db Device pointer to output gradient db (gradient of loss w.r.t b) (rows x cols).
 * @param rows Number of rows in the matrices.
 * @param cols Number of columns in the matrices.
 */
__global__ void swiglu_backward_kernel(const float* d_a,
                                       const float* d_b,
                                       const float* d_dc,
                                       float* d_da,
                                       float* d_db,
                                       int rows,
                                       int cols);

// Wrapper functions for launching kernels (optional, but good practice)
void launch_swiglu_forward(const float* d_a, const float* d_b, float* d_c, int rows, int cols, cudaStream_t stream = 0);
void launch_swiglu_backward(const float* d_a, const float* d_b, const float* d_dc, float* d_da, float* d_db, int rows, int cols, cudaStream_t stream = 0);

#endif // SWIGLU_CUH
