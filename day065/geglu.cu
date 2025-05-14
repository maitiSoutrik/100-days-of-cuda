#include "geglu.cuh"
#include <cmath> // For M_PI, tanhf, sqrtf

// Define M_PI_F if not available, or use M_PI
#ifndef M_PI_F
#define M_PI_F ((float)M_PI)
#endif

// GELU device function (approximation)
// GELU(x) ≈ 0.5x * (1 + tanh(sqrt(2/π) * (x + 0.044715x^3)))
__device__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI_F) * (x + 0.044715f * x * x * x)));
}

// GEGLU kernel
// output[i] = gelu_approx(input_a[i]) * input_b[i]
__global__ void geglu_kernel(const float* input_a, const float* input_b, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a_val = input_a[idx];
        float b_val = input_b[idx];
        output[idx] = gelu_approx(a_val) * b_val;
    }
}

// Wrapper function to launch the GEGLU kernel
void launch_geglu_kernel(const float* d_input_a, const float* d_input_b, float* d_output, int n, int threads_per_block) {
    if (n <= 0) return; // No work to do

    // Ensure threads_per_block is positive, default to 256 if not sensible
    if (threads_per_block <= 0) {
        threads_per_block = 256;
    }
    
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    geglu_kernel<<<num_blocks, threads_per_block>>>(d_input_a, d_input_b, d_output, n);
    
    // It's good practice to check for kernel launch errors, especially during development.
    // cudaGetLastError() should be checked after kernel launch.
    // The CHECK_CUDA_ERROR macro can be used here if it's adapted for runtime errors,
    // or a specific check can be added. For now, relying on subsequent CUDA calls to fail.
    // A more robust way:
    // CHECK_CUDA_ERROR(cudaGetLastError()); 
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // To be sure kernel finished and catch errors
}
