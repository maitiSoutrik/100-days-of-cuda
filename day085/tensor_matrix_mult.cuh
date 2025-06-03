#ifndef TENSOR_MATRIX_MULT_CUH
#define TENSOR_MATRIX_MULT_CUH

#include <cuda_runtime.h>
#include <iostream> // For std::cerr

// CUDA Error Checking Macro
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
template <typename T>
inline void checkCuda(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(err) << " (" << func << ")" << std::endl;
        // Depending on the project's error handling strategy,
        // you might throw an exception or exit.
        // For this project, we'll print the error.
        // exit(EXIT_FAILURE); // Or handle more gracefully
    }
}

// Macro to check for errors after kernel launch
#define CHECK_KERNEL_LAUNCH()                                           \
    do {                                                                \
        cudaError_t err = cudaPeekAtLastError();                        \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA kernel launch error at " << __FILE__     \
                      << ":" << __LINE__ << ": "                        \
                      << cudaGetErrorString(err) << std::endl;          \
        }                                                               \
        err = cudaDeviceSynchronize();                                  \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA device synchronize error at " << __FILE__ \
                      << ":" << __LINE__ << ": "                        \
                      << cudaGetErrorString(err) << std::endl;          \
        }                                                               \
    } while (0)


/**
 * @brief Performs tensor-matrix multiplication C = A * B on the GPU.
 *
 * A is a tensor of shape (B_dim, I_dim, J_dim, L_dim).
 * B is a matrix of shape (L_dim, K_dim).
 * C is the resulting tensor of shape (B_dim, I_dim, J_dim, K_dim).
 *
 * The operation is C[b,i,j,k] = sum_l (A[b,i,j,l] * B[l,k]).
 *
 * @param A_dev Pointer to the input tensor A on the device.
 * @param B_dev Pointer to the input matrix B on the device.
 * @param C_dev Pointer to the output tensor C on the device.
 * @param B_dim Dimension b of tensor A and C.
 * @param I_dim Dimension i of tensor A and C.
 * @param J_dim Dimension j of tensor A and C.
 * @param L_dim Shared dimension l of tensor A and matrix B (contraction dimension).
 * @param K_dim Dimension k of matrix B and tensor C.
 */
extern "C" void tensor_matrix_multiply(
    const float* A_dev,
    const float* B_dev,
    float* C_dev,
    size_t B_dim,
    size_t I_dim,
    size_t J_dim,
    size_t L_dim,
    size_t K_dim
);

#endif // TENSOR_MATRIX_MULT_CUH