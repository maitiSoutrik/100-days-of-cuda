#ifndef PRODUCT_REDUCTION_CUH
#define PRODUCT_REDUCTION_CUH

#include <cuda_runtime.h>
#include <cstddef> // For size_t

/**
 * @brief Performs a product reduction on a multi-dimensional array along a specified dimension.
 *
 * Given an `ndim`-dimensional tensor `input` (flattened in row-major order), this routine computes the
 * multiplicative product along the axis `dim` and stores the results into `output`.  The resulting
 * tensor therefore has the same shape as the input tensor except that the length of the reduced
 * dimension is 1 (and is typically omitted by higher-level frameworks).
 *
 * All pointers must refer to device (GPU) memory.  Shapes are provided as a host array of length
 * `ndim`.
 *
 * @param input  Pointer to input tensor data (device).
 * @param dim    Dimension along which to compute the product (0-indexed).
 * @param output Pointer to output tensor data (device).
 * @param shape  Host pointer to array of dimension sizes.
 * @param ndim   Number of dimensions of the input tensor.
 */
extern "C" void product_reduction_dimension_cuda(
    const float* input,
    int          dim,
    float*       output,
    const size_t* shape,
    size_t       ndim);

#endif // PRODUCT_REDUCTION_CUH
