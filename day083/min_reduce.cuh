#ifndef MIN_REDUCE_CUH
#define MIN_REDUCE_CUH

#include <cuda_runtime.h>
#include <cstddef> // For size_t

/**
 * @brief Performs a minimum reduction on a multi-dimensional array along a specified dimension.
 *
 * This function calculates the minimum value along the 'dim' axis of the input array
 * and stores the result in the output array. The output array will have one less
 * dimension than the input array (or the specified dimension will have size 1).
 *
 * @param input Pointer to the input multi-dimensional array (flattened, row-major).
 * @param dim The dimension along which to perform the reduction (0-indexed).
 * @param output Pointer to the output array where the reduced values will be stored.
 * @param shape An array representing the shape (dimensions) of the input array.
 * @param ndim The number of dimensions in the input array.
 */
extern "C" void min_reduction_dimension_cuda(
    const float* input,
    int dim,
    float* output,
    const size_t* shape,
    size_t ndim
);

#endif // MIN_REDUCE_CUH
