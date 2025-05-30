#ifndef SWISH_MATRIX_OPS_CUH
#define SWISH_MATRIX_OPS_CUH

#include <cuda_runtime.h>
#include <cmath> // For exp

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Swish activation kernel
__global__ void swish_activation_kernel(float* input, float* output, int size, float beta = 1.0f);

// Matrix multiplication with Swish activation and scaling kernel
__global__ void matrix_mul_swish_scale_kernel(
    const float* A, const float* B,
    float* C,
    int M, int N, int K,
    float scale = 1.0f,
    float beta = 1.0f
);

// Host function to launch matrix multiplication with Swish and scaling
cudaError_t matrix_mul_swish_scale(
    const float* A, const float* B,
    float* C,
    int M, int N, int K,
    float scale = 1.0f,
    float beta = 1.0f,
    dim3 threadsPerBlock = dim3(16, 16) // Default block size
);

#endif // SWISH_MATRIX_OPS_CUH
