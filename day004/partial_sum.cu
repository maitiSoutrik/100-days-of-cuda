#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel for partial sum using parallel reduction
__global__ void partialSumKernel(float *input, float *output, int n) {
    extern __shared__ float sharedData[];
    
    // Each thread loads one element from global memory to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sharedData[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// CPU implementation of sum for verification
float sumArray(float *arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// Function to compute partial sum on GPU
float partialSumGPU(float *h_input, int n) {
    float *d_input, *d_output;
    float h_sum = 0.0f;
    
    // Determine block size and number of blocks
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_output, numBlocks * sizeof(float)));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate host memory for output
    float *h_output = (float*)malloc(numBlocks * sizeof(float));
    
    // Launch kernel
    partialSumKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy output data back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Sum the partial sums on CPU
    for (int i = 0; i < numBlocks; i++) {
        h_sum += h_output[i];
    }
    
    // Free memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_output);
    
    return h_sum;
}

int main(int argc, char *argv[]) {
    // Default array size
    int n = 1000000;
    
    // Parse command line arguments
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    
    printf("Computing partial sum of %d elements\n", n);
    
    // Allocate and initialize host array
    float *h_input = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // Initialize with 1.0 for easy verification
    }
    
    // Compute sum on CPU for verification
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_sum = sumArray(h_input, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = cpu_end - cpu_start;
    
    // Compute sum on GPU
    auto gpu_start = std::chrono::high_resolution_clock::now();
    float gpu_sum = partialSumGPU(h_input, n);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = gpu_end - gpu_start;
    
    // Print results
    printf("CPU Sum: %.1f, Time: %.6f seconds\n", cpu_sum, cpu_time.count());
    printf("GPU Sum: %.1f, Time: %.6f seconds\n", gpu_sum, gpu_time.count());
    
    // Verify results
    float tolerance = 1e-5;
    if (fabs(cpu_sum - gpu_sum) > tolerance) {
        printf("Verification FAILED!\n");
    } else {
        printf("Verification PASSED!\n");
    }
    
    // Calculate speedup
    printf("GPU Speedup: %.2fx\n", cpu_time.count() / gpu_time.count());
    
    // Free memory
    free(h_input);
    
    return 0;
}
