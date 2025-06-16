#include "product_reduction.cuh"
#include <cstdio>

// Error checking macro (project-wide convention)
#define CHECK_CUDA_ERROR(err)                                                       \
    if (err != cudaSuccess) {                                                      \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                          \
        exit(EXIT_FAILURE);                                                       \
    }

/**
 * Kernel computes the product along a selected dimension.
 * Tensor is conceptually split into three contiguous regions: before, dim, after.
 * Each output element corresponds to the product of `dim_size` contiguous elements
 * in the input separated by the after_size stride.
 */
__global__ void productReduceKernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    // const size_t* __restrict__ shape, // Unused
                                    // size_t ndim, // Unused
                                    // int dim_to_reduce, // Unused
                                    size_t before_size,
                                    size_t dim_size,
                                    size_t after_size) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_elements = before_size * after_size;
    if (out_idx >= out_elements) return;

    size_t before_idx = out_idx / after_size;
    size_t after_idx  = out_idx % after_size;

    float prod = 1.0f;
    for (size_t d = 0; d < dim_size; ++d) {
        size_t in_idx = (before_idx * dim_size + d) * after_size + after_idx;
        prod *= input[in_idx];
    }
    output[out_idx] = prod;
}

extern "C" void product_reduction_dimension_cuda(const float* input,
                                                  int dim,
                                                  float* output,
                                                  const size_t* host_shape, // Changed: shape is a host pointer
                                                  size_t ndim) {
    if (dim < 0 || dim >= static_cast<int>(ndim)) {
        fprintf(stderr, "product_reduction_dimension_cuda: dim %d out of bounds (ndim=%zu)\n", dim, ndim);
        return;
    }
    if (!input || !output || !host_shape) { // Changed: check host_shape
        fprintf(stderr, "product_reduction_dimension_cuda: null pointer argument\n");
        return;
    }

    size_t before_size = 1;
    for (int i = 0; i < dim; ++i) before_size *= host_shape[i]; // Changed: use host_shape
    size_t dim_size = host_shape[dim]; // Changed: use host_shape
    if (dim_size == 0) {
        fprintf(stderr, "product_reduction_dimension_cuda: size of reduction dimension is 0\n");
        return;
    }
    size_t after_size = 1;
    for (size_t i = dim + 1; i < ndim; ++i) after_size *= host_shape[i]; // Changed: use host_shape

    size_t out_size = before_size * after_size;
    if (out_size == 0) return; // nothing to compute

    constexpr int blockSize = 256;
    int gridSize = (out_size + blockSize - 1) / blockSize;

    productReduceKernel<<<gridSize, blockSize>>>(input, output, /*shape, ndim, dim,*/ before_size, dim_size, after_size); // Removed unused parameters
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
