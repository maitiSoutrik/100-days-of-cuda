#include "quantization_kernels.h"

// --- Matrix Multiplication Kernels ---

// Baseline FP32 Matrix Multiplication
__global__ void matmul_fp32_kernel(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized FP16 Matrix Multiplication using half-precision intrinsics
__global__ void matmul_fp16_kernel(const __half *A, const __half *B, __half *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum_f = 0.0f; // Use FP32 for accumulation
        for (int k = 0; k < N; k++) {
            // Multiply in FP16, but add to FP32 accumulator
            sum_f += __half2float(__hmul(A[row * N + k], B[k * N + col]));
        }
        // Convert final result back to FP16 for storage
        C[row * N + col] = __float2half(sum_f);
    }
}

// Simulated FP8 Matrix Multiplication (Reads uint8_t, computes in FP32)
__global__ void matmul_fp8_sim_kernel(const uint8_t *A_fp8, const uint8_t *B_fp8, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            // Dequantize on the fly for computation
            float a_val = dequantize_fp8_e5m2_sim(A_fp8[row * N + k]);
            float b_val = dequantize_fp8_e5m2_sim(B_fp8[k * N + col]);
            sum += a_val * b_val;
        }
        C[row * N + col] = sum;
    }
}

// --- Conversion Kernels ---

// Kernel to convert FP32 matrix to FP16
__global__ void fp32_to_fp16_kernel(const float* input, __half* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        output[idx] = __float2half(input[idx]);
    }
}

// Kernel to convert FP16 matrix to FP32
__global__ void fp16_to_fp32_kernel(const __half* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        output[idx] = __half2float(input[idx]);
    }
}

// Kernel to quantize FP32 matrix to simulated FP8 (uint8_t)
__global__ void fp32_to_fp8_sim_kernel(const float* input, uint8_t* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        output[idx] = quantize_fp8_e5m2_sim(input[idx]);
    }
}

// --- CPU Reference ---
void matmul_cpu(const float *A, const float *B, float *C, int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// --- Verification Function ---
float verify_results(const float *ref, const float *res, int N, const char* type_name) {
    float max_rel_error = 0.0f;
    double total_abs_error = 0.0;
    int n_elements = N * N;

    for (int i = 0; i < n_elements; ++i) {
        float abs_error = fabsf(ref[i] - res[i]);
        total_abs_error += abs_error;
        float rel_error = 0.0f;
        if (fabsf(ref[i]) > 1e-7) { // Avoid division by zero or near-zero, use a slightly larger epsilon
             rel_error = abs_error / fabsf(ref[i]);
        } else if (abs_error > 1e-7) { // If ref is near zero, consider absolute error
             rel_error = abs_error; // Or handle as a special case if needed
        }
        max_rel_error = fmaxf(max_rel_error, rel_error);
    }
    printf("  Verifying %s GPU result:\n", type_name);
    printf("    Average Absolute Error: %e\n", total_abs_error / n_elements);
    printf("    Maximum Relative Error: %e\n", max_rel_error);
    return max_rel_error;
}
