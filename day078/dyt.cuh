#ifndef DYT_CUH
#define DYT_CUH

#include <cuda_runtime.h>

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// Forward pass kernel declaration
__global__ void dyt_forward_kernel(const float* x, float* y, float alpha, float beta, int n);

// Backward pass kernel declaration
__global__ void dyt_backward_kernel(const float* upstream_grad, const float* x,
                                   float* x_grad, float* alpha_grad_atomic, float* beta_grad_atomic,
                                   float alpha, float beta, int n);

// Wrapper functions (optional, but good practice for calling from C++ host code if needed elsewhere)
// For this standalone example, kernels might be called directly from dyt_main.cu

#endif // DYT_CUH
