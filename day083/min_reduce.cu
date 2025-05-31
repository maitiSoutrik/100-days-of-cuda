#include "min_reduce.cuh"
#include <cfloat>  // For FLT_MAX
#include <cstdio>  // For printf, if needed for debugging, though not in final code

// Error checking macro (as per .clinerules)
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

__global__ void minReduceKernel(const float* input, float* output,
                              const size_t* shape, size_t ndim, int dim_to_reduce, // Renamed 'dim' to 'dim_to_reduce' to avoid conflict
                              size_t before_size, size_t dim_size, size_t after_size) {
    // Calculate the global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the total number of elements in the output array
    size_t output_elements = before_size * after_size;

    // Boundary check: ensure the thread is within the bounds of the output array
    if (idx >= output_elements) {
        return;
    }

    // Decompose the 1D output index 'idx' into multi-dimensional indices
    // for the 'before' and 'after' parts of the original tensor dimensions
    size_t before_idx = idx / after_size; // Index in the dimensions before the reduced dimension
    size_t after_idx = idx % after_size;  // Index in the dimensions after the reduced dimension

    float min_val = FLT_MAX;

    // Iterate over the dimension being reduced
    for (size_t d = 0; d < dim_size; ++d) {
        // Calculate the 1D index in the input array
        // This corresponds to:
        // (current_before_idx * size_of_reduced_dimension + current_element_in_reduced_dimension) * size_of_after_dimensions + current_after_idx
        size_t input_idx = (before_idx * dim_size + d) * after_size + after_idx;
        min_val = min(min_val, input[input_idx]);
    }

    // Write the minimum value found to the output array
    output[idx] = min_val;
}

extern "C" void min_reduction_dimension_cuda(
    const float* input,
    int dim, // This is the dimension to reduce
    float* output,
    const size_t* shape,
    size_t ndim
) {
    if (dim < 0 || dim >= ndim) {
        fprintf(stderr, "Error: Reduction dimension %d is out of bounds for ndim %zu.\n", dim, ndim);
        return;
    }
    if (input == nullptr || output == nullptr || shape == nullptr) {
        fprintf(stderr, "Error: Null pointer provided for input, output, or shape.\n");
        return;
    }
    if (ndim == 0) {
        fprintf(stderr, "Error: Number of dimensions (ndim) cannot be zero.\n");
        return;
    }


    size_t before_size = 1;
    for (int i = 0; i < dim; ++i) {
        before_size *= shape[i];
    }

    size_t dim_size = shape[dim];
    if (dim_size == 0) {
        fprintf(stderr, "Error: Size of dimension to reduce (shape[%d]) cannot be zero.\n", dim);
        // Depending on desired behavior, could return or produce an empty output.
        // For now, let's return to prevent division by zero or empty loops.
        return;
    }


    size_t after_size = 1;
    for (size_t i = dim + 1; i < ndim; ++i) {
        after_size *= shape[i];
    }

    size_t output_size = before_size * after_size;
    if (output_size == 0 && !(before_size == 0 || after_size == 0)) {
         // This case implies dim_size might have been the only non-zero dimension,
         // leading to a scalar output if before_size and after_size are 1.
         // If before_size or after_size is 0, output_size is legitimately 0.
         // If output_size is 0 but input was not, it's an issue unless it's a reduction to scalar from a 1D array.
         // Let's assume if output_size is 0, no work is needed unless it's a specific case like reducing a 1D array to a scalar.
         // For simplicity, if output_size is 0, we might not need to launch the kernel.
         // However, the kernel itself has a check `if (idx >= output_elements) return;`
         // If output_size is 0, numBlocks might be 0 or 1.
         // If numBlocks is 0, kernel won't launch. If 1, threads will exit due to idx >= 0.
    }


    // Configure kernel launch parameters
    int blockSize = 256; // Typical block size
    if (output_size == 0) { // No work to do if output size is 0
        // Potentially fill output with a default value if necessary, or simply return.
        // For now, we assume if output_size is 0, no kernel launch is needed.
        // This can happen if, for example, one of the non-reduced dimensions is 0.
        return;
    }
    int numBlocks = (output_size + blockSize - 1) / blockSize;

    // Launch the kernel
    minReduceKernel<<<numBlocks, blockSize>>>(
        input, output, shape, ndim, dim,
        before_size, dim_size, after_size
    );
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for errors after kernel launch
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure kernel completion before host proceeds
}
