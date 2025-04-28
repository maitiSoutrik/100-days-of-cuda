#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>
#include <cmath> // For fabs

// Forward declarations for functions defined in vector_add_kernel.cu
// These are declared with C linkage in the other file.
extern "C" {
    void checkCudaError(cudaError_t err, const char *msg);
    __global__ void vectorAddKernel(const float *A, const float *B, float *C, int numElements);
}

// CPU implementation of vector addition (for verification)
void vectorAddCPU(const float *A, const float *B, float *C, int numElements) {
    for (int i = 0; i < numElements; i++) {
        C[i] = A[i] + B[i];
    }
}

// Function to get current time in microseconds
double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000000 + (double)tv.tv_usec;
}

int main(void) {
    // Print device properties
    cudaDeviceProp prop;
    int deviceId = 0; // Assuming device 0
    checkCudaError(cudaGetDeviceProperties(&prop, deviceId), "cudaGetDeviceProperties");
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Vector size
    int numElements = 10000000; // Use a large size for timing
    size_t size = numElements * sizeof(float);
    printf("Vector size: %d elements\n", numElements);
    
    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_GPU = (float *)malloc(size); // Result from GPU
    float *h_C_CPU = (float *)malloc(size); // Result from CPU for verification
    
    if (h_A == NULL || h_B == NULL || h_C_GPU == NULL || h_C_CPU == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        return EXIT_FAILURE;
    }
    
    // Initialize host vectors
    srand(time(NULL)); // Seed random number generator
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
    
    // --- Time CPU Version ---
    printf("\n----- CPU Execution -----\n");
    double cpuStart = getCurrentTime();
    vectorAddCPU(h_A, h_B, h_C_CPU, numElements);
    double cpuEnd = getCurrentTime();
    double cpuElapsed = (cpuEnd - cpuStart) / 1000.0; // ms
    printf("CPU execution time: %.2f ms\n", cpuElapsed);

    // --- Time GPU Version ---
    printf("\n----- GPU Execution -----\n");
    
    // Start timing GPU operations (including memory transfers)
    double gpuStartWithTransfer = getCurrentTime();
    
    // Copy vectors from host to device
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "cudaMemcpy h_A to d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "cudaMemcpy h_B to d_B");
    
    // Start timing kernel execution only
    double gpuStartKernelOnly = getCurrentTime();
    
    // Launch the CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    checkCudaError(cudaGetLastError(), "kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "kernel execution");
    
    // End timing kernel execution only
    double gpuEndKernelOnly = getCurrentTime();
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C_GPU, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_C to h_C_GPU");
    
    // End timing GPU operations (including memory transfers)
    double gpuEndWithTransfer = getCurrentTime();
    
    double gpuElapsedKernelOnly = (gpuEndKernelOnly - gpuStartKernelOnly) / 1000.0; // ms
    double gpuElapsedWithTransfer = (gpuEndWithTransfer - gpuStartWithTransfer) / 1000.0; // ms
    
    printf("GPU kernel execution time: %.2f ms\n", gpuElapsedKernelOnly);
    printf("GPU total time (with memory transfers): %.2f ms\n", gpuElapsedWithTransfer);
    
    // Verify the GPU result against the CPU result
    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_C_CPU[i] - h_C_GPU[i]) > 1e-5) {
            fprintf(stderr, "Verification failed at element %d! CPU=%.6f, GPU=%.6f\n", i, h_C_CPU[i], h_C_GPU[i]);
            success = false;
            break; // Exit loop on first failure
        }
    }
    
    if (success) {
        printf("\nVerification PASSED\n");
    } else {
        printf("\nVerification FAILED\n");
    }
    
    // Print performance comparison
    printf("\n----- Performance Comparison -----\n");
    printf("CPU execution time: %.2f ms\n", cpuElapsed);
    printf("GPU kernel execution time: %.2f ms\n", gpuElapsedKernelOnly);
    printf("GPU total time (with memory transfers): %.2f ms\n", gpuElapsedWithTransfer);
    if (gpuElapsedKernelOnly > 0) {
        printf("Speedup (kernel only vs CPU): %.2fx\n", cpuElapsed / gpuElapsedKernelOnly);
    }
    if (gpuElapsedWithTransfer > 0) {
        printf("Speedup (total GPU vs CPU): %.2fx\n", cpuElapsed / gpuElapsedWithTransfer);
    }
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_GPU);
    free(h_C_CPU);
    
    printf("\nDone\n");
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
