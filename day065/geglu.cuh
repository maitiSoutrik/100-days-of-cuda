#ifndef GEGLU_CUH
#define GEGLU_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <cstdio> // For printf in error macros if needed

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// GELU device function (approximation)
__device__ float gelu_approx(float x);

// GEGLU kernel
__global__ void geglu_kernel(const float* input_a, const float* input_b, float* output, int n);

// Wrapper function to launch the GEGLU kernel (optional, but good practice)
void launch_geglu_kernel(const float* d_input_a, const float* d_input_b, float* d_output, int n, int threads_per_block = 256);

#endif // GEGLU_CUH
