#include "softplus_activation.cuh"
#include <cmath> // For std::log, std::exp

// CUDA Kernel for Softplus Activation
__global__ void softplusKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Softplus(x) = log(1 + exp(x))
        // To avoid overflow for large positive x, and underflow for large negative x:
        // if x > threshold_pos: log(exp(x)*(exp(-x) + 1)) = x + log(exp(-x) + 1) approx x
        // if x < threshold_neg: log(1 + exp(x)) approx exp(x) for very small exp(x)
        // However, a more direct approach is often stable enough for typical float ranges.
        // For higher precision or wider ranges, more sophisticated approximations might be needed.
        output[idx] = logf(1.0f + expf(input[idx]));
    }
}

// Wrapper function to launch the Softplus kernel
void softplusActivation(const float* d_input, float* d_output, int N) {
    // Define block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    softplusKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for the kernel to complete
}

// CPU implementation for verification
void softplusActivationCPU(const float* h_input, float* h_output, int N) {
    for (int i = 0; i < N; ++i) {
        h_output[i] = std::log(1.0f + std::exp(h_input[i]));
    }
}