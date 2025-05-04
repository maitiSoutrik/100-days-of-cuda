// day056/mish_activation.cuh
#ifndef MISH_ACTIVATION_CUH
#define MISH_ACTIVATION_CUH

#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream> // For error checking output

// Simple CUDA error checking macro/function definition (inline in header)
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
inline void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") "
                  << " at " << file << ":" << line << " '" << func << "' \\n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Mish activation function: f(x) = x * tanh(softplus(x))
// softplus(x) = log(1 + exp(x))
__host__ __device__ inline float mish(float x) {
    // Using logf(1 + expf(x)) for softplus
    return x * tanhf(logf(1.0f + expf(x)));
}

// CPU implementation declaration
void mish_cpu(const std::vector<float>& input, std::vector<float>& output);

// GPU kernel declaration
__global__ void mish_kernel(const float* input, float* output, int n);

// GPU wrapper function declaration (takes events for timing)
void mish_gpu_wrapper(const float* d_input, float* d_output, int n, cudaEvent_t start = nullptr, cudaEvent_t stop = nullptr);

#endif // MISH_ACTIVATION_CUH
