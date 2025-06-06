#ifndef WARP_LEVEL_REDUCTION_CUH
#define WARP_LEVEL_REDUCTION_CUH

#include <cstdio>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
inline void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, static_cast<unsigned int>(result),
                cudaGetErrorName(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Performs a sum reduction within each warp of a block.
 *
 * Each thread contributes a value. The sum of these values within each warp
 * is computed using __shfl_down_sync. The thread with lane ID 0 in each warp
 * stores the result for that warp in the output array.
 *
 * @param input_data Pointer to the input array on the device. Each thread reads one element.
 * @param output_data Pointer to the output array on the device. Each warp writes one sum.
 *                    The size should be (num_threads / warpSize).
 * @param num_elements Total number of elements in input_data (should match total threads launched).
 */
__global__ void warpSumReductionKernel(const int *input_data, int *output_data, int num_elements);

#endif // WARP_LEVEL_REDUCTION_CUH
