#ifndef NEGATIVE_COSINE_SIMILARITY_CUH
#define NEGATIVE_COSINE_SIMILARITY_CUH

#include <cuda_runtime.h>
#include <cstddef> // For size_t
#include <cstdio>  // For printf in CHECK_CUDA_ERROR
#include <cstdlib> // For exit in CHECK_CUDA_ERROR


// Error checking macro (standard practice)
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

__global__ void cosine_similarity_kernel(const float* predictions, const float* targets, float* output, size_t n, size_t d);

extern "C" void launch_cosine_similarity_kernel(const float* predictions, const float* targets, float* output, size_t n, size_t d);

#endif // NEGATIVE_COSINE_SIMILARITY_CUH
