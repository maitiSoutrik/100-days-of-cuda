#ifndef QUANTIZATION_KERNELS_H
#define QUANTIZATION_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

// Error checking macro (can be defined here or included from a common utility header if one exists)
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
inline void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        // cudaDeviceReset(); // Avoid resetting device in header/library code
        exit(EXIT_FAILURE);
    }
}

#define TILE_DIM 16 // Threads per block dimension
#define FP8_MIN_VAL -448.0f // Example range for FP8 E5M2
#define FP8_MAX_VAL  448.0f

// --- FP8 Simulation ---
__device__ __forceinline__ uint8_t quantize_fp8_e5m2_sim(float x) {
    x = fmaxf(FP8_MIN_VAL, fminf(FP8_MAX_VAL, x));
    float scale = 255.0f / (FP8_MAX_VAL - FP8_MIN_VAL);
    return (uint8_t)roundf((x - FP8_MIN_VAL) * scale);
}

__device__ __forceinline__ float dequantize_fp8_e5m2_sim(uint8_t x) {
    float scale = (FP8_MAX_VAL - FP8_MIN_VAL) / 255.0f;
    return FP8_MIN_VAL + ((float)x * scale);
}

// --- Matrix Multiplication Kernels ---
__global__ void matmul_fp32_kernel(const float *A, const float *B, float *C, int N);
__global__ void matmul_fp16_kernel(const __half *A, const __half *B, __half *C, int N);
__global__ void matmul_fp8_sim_kernel(const uint8_t *A_fp8, const uint8_t *B_fp8, float *C, int N);

// --- Conversion Kernels ---
__global__ void fp32_to_fp16_kernel(const float* input, __half* output, int N);
__global__ void fp16_to_fp32_kernel(const __half* input, float* output, int N);
__global__ void fp32_to_fp8_sim_kernel(const float* input, uint8_t* output, int N);

// --- CPU Reference ---
void matmul_cpu(const float *A, const float *B, float *C, int N);

// --- Verification Function ---
// Returns max relative error
float verify_results(const float *ref, const float *res, int N, const char* type_name);


#endif // QUANTIZATION_KERNELS_H
