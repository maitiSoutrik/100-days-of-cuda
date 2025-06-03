// day086/hard_sigmoid.cu
#include "hard_sigmoid.cuh"
#include "common_utils.h" // For error checking macros
#include <cstdio> // For printf if any debugging, though not in kernel

__global__ void hard_sigmoid_kernel(const float* input, float* output, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return; 

    float x = input[idx];
    if (x <= -3.0f) {
        output[idx] = 0.0f;
    } else if (x >= 3.0f) {
        output[idx] = 1.0f;
    } else {
        output[idx] = (x + 3.0f) / 6.0f;
    }
}

extern "C" void hard_sigmoid_solution(const float* input, float* output, size_t n, size_t m) {
    size_t total_elements = n * m;
    if (total_elements == 0) return;

    float *d_input = nullptr;
    float *d_output = nullptr;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, total_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, total_elements * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input, total_elements * sizeof(float), cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    
hard_sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, total_elements);
    CHECK_LAST_CUDA_ERROR(); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    CHECK_CUDA_ERROR(cudaMemcpy(output, d_output, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}
