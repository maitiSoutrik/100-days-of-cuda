#ifndef BARNSLEY_FERN_CUH
#define BARNSLEY_FERN_CUH

#include <cuda_runtime.h>
#include <cstdio> // For printf in macro

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Forward declarations for CUDA kernels
__global__ void setup_kernel(curandState *state, unsigned long long seed, int num_threads);

__global__ void generate_fern_kernel(
    unsigned int* image_buffer,
    int image_width,
    int image_height,
    int num_iterations_per_thread,
    curandState* rand_states,
    float fern_x_min, float fern_x_max, float fern_y_min, float fern_y_max,
    int warmup_iterations);

#endif // BARNSLEY_FERN_CUH
