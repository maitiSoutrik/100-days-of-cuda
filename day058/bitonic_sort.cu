#include "bitonic_sort.cuh"
#include <stdio.h>
#include <stdlib.h> // For rand()
#include <cuda_runtime.h>

// Define the fixed size for this specific Bitonic Sort implementation
// This version is designed to sort an array that fits entirely within
// one thread block's shared memory and thread limits.
const int N_CONST = 1024;

__global__ void bitonic_sort_kernel(float* d_array) {
    __shared__ float s_data[N_CONST];
    int tid = threadIdx.x;
    
    // Load data from global memory to shared memory
    // Assumes blockDim.x is N_CONST
    if (tid < N_CONST) {
        s_data[tid] = d_array[tid];
    }
    __syncthreads(); // Ensure all data is loaded before starting sort
    
    // Bitonic sort stages
    // Outer loop for the size of the bitonic sequence to merge (k)
    for (int k = 2; k <= N_CONST; k *= 2) {
        // Inner loop for the comparison distance (j)
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j; // Calculate the index of the element to compare with

            // Ensure the paired index is within the current sub-array bounds
            // and that we are not comparing an element with itself (ixj > tid)
            if (ixj > tid) {
                // Determine sort direction (ascending or descending)
                // (tid & k) == 0 means we are in the "lower" part of the k-element sequence,
                // which should be sorted ascendingly to form an overall ascending sequence.
                // Otherwise, we are in the "upper" part, sorted descendingly.
                bool ascending_direction = ((tid & k) == 0);
                
                if (ascending_direction) {
                    if (s_data[tid] > s_data[ixj]) {
                        // Swap
                        float temp = s_data[tid];
                        s_data[tid] = s_data[ixj];
                        s_data[ixj] = temp;
                    }
                } else { // Descending direction
                    if (s_data[tid] < s_data[ixj]) {
                        // Swap
                        float temp = s_data[tid];
                        s_data[tid] = s_data[ixj];
                        s_data[ixj] = temp;
                    }
                }
            }
            __syncthreads(); // Synchronize after each comparison pass
        }
    }
    
    // Write sorted data from shared memory back to global memory
    if (tid < N_CONST) {
        d_array[tid] = s_data[tid];
    }
}

void bitonic_sort_gpu(float* h_array, int array_size) {
    if (array_size != N_CONST) {
        fprintf(stderr, "Error: This Bitonic Sort implementation is fixed for N_CONST = %d elements.\n", N_CONST);
        fprintf(stderr, "Requested array size: %d\n", array_size);
        // For a real application, you might pad or handle this differently.
        // For this example, we'll exit if the size doesn't match.
        exit(EXIT_FAILURE); 
    }

    float* d_array;
    size_t size_bytes = N_CONST * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_array, size_bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_array, h_array, size_bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel with 1 block and N_CONST threads
    // This assumes N_CONST is <= max threads per block (typically 1024)
    bitonic_sort_kernel<<<1, N_CONST>>>(d_array);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_array, d_array, size_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_array));
}

void print_array_host(float* array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ", array[i]);
        if ((i + 1) % 10 == 0 && i > 0) printf("\n"); // Print 10 numbers per line
    }
    printf("\n");
}
