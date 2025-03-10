#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// Function to check for CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed with error: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    // Print device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Vector size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("Vector size: %d elements\n", numElements);
    
    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize host vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    
    checkCudaError(cudaMalloc((void **)&d_A, size), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void **)&d_B, size), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void **)&d_C, size), "cudaMalloc d_C");
    
    // Copy vectors from host to device
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "cudaMemcpy h_A to d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "cudaMemcpy h_B to d_B");
    
    // Launch the CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    checkCudaError(cudaGetLastError(), "kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "kernel execution");
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_C to h_C");
    
    // Verify the result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    
    printf("Test PASSED\n");
    
    // Print a few results for verification
    printf("Sample results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%.6f + %.6f = %.6f\n", h_A[i], h_B[i], h_C[i]);
    }
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Done\n");
    return 0;
}
