#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define cudaCheckError() {\
    cudaError_t e = cudaGetLastError();\
    if (e != cudaSuccess) {\
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));\
        exit(EXIT_FAILURE);\
    }\
}

/**
 * Brent-Kung Prefix Sum Algorithm Implementation
 * 
 * This algorithm computes prefix sums in parallel using a tree-based approach:
 * 1. Up-sweep (reduction) phase: builds a sum tree from the bottom up
 * 2. Down-sweep phase: propagates values back down the tree
 */
__global__ void brentKungScan(float* input, float* output, int n) {
    extern __shared__ float temp[];
    
    int thid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory
    // Each thread loads one element
    if (thid < n) {
        temp[thid] = input[thid + blockIdx.x * blockDim.x];
    } else {
        temp[thid] = 0.0f;
    }
    __syncthreads();
    
    // Up-sweep (Reduction) phase
    // Build sum tree from bottom up
    for (int d = n >> 1; d > 0; d >>= 1) {
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        __syncthreads();
    }
    
    // Clear the last element (preparation for down-sweep)
    if (thid == 0) {
        temp[n - 1] = 0.0f; // Clear the last element
    }
    __syncthreads();
    
    // Down-sweep phase
    // Traverse back down the tree to build scan
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }
    
    // Write results to output array
    if (thid < n) {
        output[thid + blockIdx.x * blockDim.x] = temp[thid];
    }
}

/**
 * Host function to perform exclusive prefix sum using Brent-Kung algorithm
 */
void exclusiveScan(float* h_input, float* h_output, int n) {
    float *d_input, *d_output;
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));
    cudaCheckError();
    
    // Copy input data to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Calculate grid and block dimensions
    int blockSize = 256; // This should be a power of 2
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Calculate shared memory size
    int sharedMemSize = blockSize * sizeof(float);
    
    // Launch kernel
    brentKungScan<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, blockSize);
    cudaCheckError();
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaCheckError();
}

/**
 * Visual demonstration of the Brent-Kung algorithm steps
 */
void demonstrateBrentKung(float* input, int n) {
    printf("\nDemonstrating Brent-Kung Algorithm Steps:\n");
    printf("Input array: ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", input[i]);
    }
    printf("\n\n");
    
    // Simulate the up-sweep phase
    printf("Up-sweep (Reduction) Phase:\n");
    float temp[16]; // Assuming n <= 16 for demonstration
    for (int i = 0; i < n; i++) {
        temp[i] = input[i];
    }
    
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        printf("Step (d=%d): ", d);
        for (int i = 0; i < n; i++) {
            printf("%.1f ", temp[i]);
        }
        printf("\n");
        
        for (int i = 0; i < d; i++) {
            int ai = offset * (2 * i + 1) - 1;
            int bi = offset * (2 * i + 2) - 1;
            if (bi < n) {
                temp[bi] += temp[ai];
            }
        }
        offset *= 2;
    }
    
    // Simulate the down-sweep phase
    printf("\nDown-sweep Phase:\n");
    temp[n - 1] = 0.0f; // Clear the last element
    
    printf("Initial: ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", temp[i]);
    }
    printf("\n");
    
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        
        for (int i = 0; i < d; i++) {
            int ai = offset * (2 * i + 1) - 1;
            int bi = offset * (2 * i + 2) - 1;
            if (bi < n) {
                float t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        
        printf("Step (d=%d): ", d);
        for (int i = 0; i < n; i++) {
            printf("%.1f ", temp[i]);
        }
        printf("\n");
    }
}

int main() {
    // Example array for demonstration
    const int n = 8;
    float h_input[n] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float h_output[n];
    
    // Demonstrate the algorithm steps
    demonstrateBrentKung(h_input, n);
    
    // Perform the actual scan
    exclusiveScan(h_input, h_output, n);
    
    // Print results
    printf("\nFinal Result (Exclusive Scan):\n");
    printf("Input:  ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_input[i]);
    }
    printf("\nOutput: ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n");
    
    return 0;
}
