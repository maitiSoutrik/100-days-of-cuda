#include "dyt.cuh"
#include <cmath> // For tanhf (used by device code)
#include <cstdio> // For fprintf in CHECK_CUDA_ERROR
#include <cstdlib> // For exit in CHECK_CUDA_ERROR


// Forward pass kernel
__global__ void dyt_forward_kernel(const float* x, float* y, float alpha, float beta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * tanhf(beta * x[idx]);
    }
}

// Backward pass kernel
__global__ void dyt_backward_kernel(const float* upstream_grad, const float* x,
                                   float* x_grad, float* alpha_grad_atomic, float* beta_grad_atomic,
                                   float alpha, float beta, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float beta_x = beta * x[idx];
        float tanh_beta_x = tanhf(beta_x);
        float one_minus_tanh_sq = 1.0f - tanh_beta_x * tanh_beta_x;

        // Gradient w.r.t. x
        x_grad[idx] = upstream_grad[idx] * alpha * beta * one_minus_tanh_sq;

        // Contribution to gradient w.r.t. alpha
        atomicAdd(alpha_grad_atomic, upstream_grad[idx] * tanh_beta_x);

        // Contribution to gradient w.r.t. beta
        atomicAdd(beta_grad_atomic, upstream_grad[idx] * alpha * x[idx] * one_minus_tanh_sq);
    }
}
