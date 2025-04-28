#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Function to check for CUDA errors
inline void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed with error: %s\n", msg, cudaGetErrorString(err));
        // In a test environment, we might throw an exception or use ASSERT
        // For simplicity here, we'll still exit, but a real test suite
        // might integrate this check with the test framework's assertions.
        exit(EXIT_FAILURE); 
    }
}

// CUDA kernel for vector addition
__global__ void vectorAddKernel(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}
