#include "elu_activation.cuh"

// CUDA Kernel for ELU Activation
__global__ void elu_kernel(float* input, float* output, int n, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        if (val > 0.0f) {
            output[idx] = val;
        } else {
            output[idx] = alpha * (expf(val) - 1.0f);
        }
    }
}

// Wrapper function to launch the ELU kernel
void elu_activation_kernel_wrapper(float* d_input, float* d_output, int n, float alpha) {
    // Define block and grid dimensions
    // Common choice for threadsPerBlock, can be tuned based on GPU architecture and problem size
    int threadsPerBlock = 256; 
    // Calculate the number of blocks needed in the grid
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    elu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n, alpha);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for errors during kernel launch
}

// Host function to manage ELU activation on the GPU
void elu_activation_gpu(float* h_input, float* h_output, int n, float alpha) {
    float* d_input = nullptr;
    float* d_output = nullptr;
    size_t size = n * sizeof(float);

    // Allocate memory on the device
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));

    // Copy input data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Launch the kernel
    elu_activation_kernel_wrapper(d_input, d_output, n, alpha);

    // Synchronize to ensure kernel completion before copying results
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy results from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

// CPU implementation of the ELU activation function
void elu_activation_cpu(const float* input, float* output, int n, float alpha) {
    for (int i = 0; i < n; ++i) {
        float val = input[i];
        if (val > 0.0f) {
            output[i] = val;
        } else {
            output[i] = alpha * (expf(val) - 1.0f);
        }
    }
}
