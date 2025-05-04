// day056/mish_activation_lib.cu
#include "mish_activation.cuh" // Include the header

#include <vector>
#include <cmath>
#include <iostream> // For error messages

// CPU implementation of Mish activation (definition)
void mish_cpu(const std::vector<float>& input, std::vector<float>& output) {
    if (input.size() != output.size()) {
         std::cerr << "Error: Input and output vector sizes must match for mish_cpu." << std::endl;
         // Consider throwing an exception in a real application
         return;
    }
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = mish(input[i]); // Uses the inline function from the header
    }
}

// GPU kernel for Mish activation (definition)
// (Declaration is in the header)
__global__ void mish_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = mish(input[idx]); // Uses the inline function from the header
    }
}

// GPU wrapper function (definition)
void mish_gpu_wrapper(const float* d_input, float* d_output, int n, cudaEvent_t start, cudaEvent_t stop) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    if (start) CHECK_CUDA_ERROR(cudaEventRecord(start));

    mish_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    // It's good practice to check for kernel launch errors immediately
    CHECK_CUDA_ERROR(cudaGetLastError());

    if (stop) CHECK_CUDA_ERROR(cudaEventRecord(stop));
}
