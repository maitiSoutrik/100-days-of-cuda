#ifndef BITONIC_SORT_CUH
#define BITONIC_SORT_CUH

#include <stdio.h> // For fprintf, stderr
#include <stdlib.h> // For exit, EXIT_FAILURE
#include <cuda_runtime.h> // For cudaError_t, cudaGetErrorString, cudaSuccess

// Error checking macro
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief Sorts an array of floats on the GPU using a Bitonic Sort algorithm
 *        optimized with shared memory. This implementation is specifically for
 *        an array of size N_CONST (defined in bitonic_sort.cu, typically 1024)
 *        that fits within a single thread block's shared memory and thread limits.
 *
 * @param h_array Pointer to the host array to be sorted. The array is sorted in-place.
 * @param array_size The number of elements in the array. Must be equal to N_CONST.
 */
void bitonic_sort_gpu(float* h_array, int array_size);

/**
 * @brief Prints the elements of a float array.
 *
 * @param array Pointer to the array.
 * @param size The number of elements in the array.
 */
void print_array_host(float* array, int size); // Renamed to avoid conflict if a device print_array is ever made

#endif // BITONIC_SORT_CUH
