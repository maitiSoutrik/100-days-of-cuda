#ifndef CUMULATIVE_PRODUCT_CUH
#define CUMULATIVE_PRODUCT_CUH

#include <vector>
#include <cuda_runtime.h>
#include <stdexcept> // For std::runtime_error
#include <iostream>   // For std::cerr

// Error checking macro
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        throw std::runtime_error("CUDA Error");
    }
}


/**
 * @brief Computes the inclusive cumulative product of an array on the GPU.
 * 
 * This function takes an array of floats, computes its inclusive cumulative product
 * in place on the GPU using a parallel scan algorithm.
 * 
 * @param d_data Pointer to the device memory containing the input array. 
 *               The array will be modified in place to store the cumulative product.
 * @param n Number of elements in the array.
 */
void inclusive_scan_gpu(float* d_data, int n);

/**
 * @brief Computes the inclusive cumulative product of an array on the CPU.
 * 
 * This function serves as a reference implementation.
 * 
 * @param h_data Pointer to the host memory containing the input array.
 *               The array will be modified in place.
 * @param n Number of elements in the array.
 */
void inclusive_scan_cpu(float* h_data, int n);

#endif // CUMULATIVE_PRODUCT_CUH
