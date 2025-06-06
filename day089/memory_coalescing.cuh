#ifndef MEMORY_COALESCING_CUH
#define MEMORY_COALESCING_CUH

#include <cuda_runtime.h>
#include <iostream>

// CUDA error checking macro
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " code=" << err << " \"" << func << "\" " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Kernel declarations
__global__ void coalesced_access_kernel(const float* input, float* output, int n, float scalar);
__global__ void uncoalesced_access_kernel(const float* input, float* output, int n, float scalar, int stride_factor);

#endif // MEMORY_COALESCING_CUH

