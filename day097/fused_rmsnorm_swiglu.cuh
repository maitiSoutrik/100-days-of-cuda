#ifndef FUSED_RMSNORM_SWIGLU_CUH
#define FUSED_RMSNORM_SWIGLU_CUH

#include "cuda_utils.h"
#include <cuda_fp16.h> // For half precision if needed, can be float for simplicity

// Define a small epsilon for RMSNorm to prevent division by zero
#define RMSNORM_EPSILON 1e-5f

/**
 * @brief Fused RMS Normalization and SwiGLU activation kernel.
 *
 * This kernel performs RMS normalization on the input, then applies
 * the SwiGLU activation function. SwiGLU(x, gate) = Silu(x) * gate,
 * where Silu(x) = x * sigmoid(x).
 * The input is assumed to be of size (batch_size, seq_len, hidden_dim).
 * The kernel processes each row (token embedding) independently.
 * For SwiGLU, the input hidden_dim is typically split into two halves,
 * one for 'x' and one for 'gate'. So, if input hidden_dim is D,
 * 'x' is input[0...D/2-1] and 'gate' is input[D/2...D-1].
 * The output dimension will be D/2.
 *
 * @param output Pointer to the output tensor on the device. Shape: (num_rows, hidden_dim / 2).
 * @param input Pointer to the input tensor on the device. Shape: (num_rows, hidden_dim).
 * @param weight Pointer to the weight tensor for RMSNorm (gamma). Shape: (hidden_dim).
 * @param num_rows Total number of rows (e.g., batch_size * seq_len) to process.
 * @param hidden_dim Dimension of each input feature vector. Must be an even number.
 */
__global__ void fused_rmsnorm_swiglu_kernel(
    float* output,
    const float* input,
    const float* weight,
    int num_rows,
    int hidden_dim);

/**
 * @brief Wrapper function to launch the fused_rmsnorm_swiglu_kernel.
 *
 * @param d_output Pointer to the output tensor on the device.
 * @param d_input Pointer to the input tensor on the device.
 * @param d_weight Pointer to the RMSNorm weight tensor (gamma) on the device.
 * @param num_rows Total number of rows (e.g., batch_size * seq_len).
 * @param hidden_dim Dimension of each input feature vector.
 * @param block_size CUDA block size for launching the kernel.
 */
void launch_fused_rmsnorm_swiglu(
    float* d_output,
    const float* d_input,
    const float* d_weight,
    int num_rows,
    int hidden_dim,
    int block_size = 256);

// CPU implementations for verification (optional, but good for testing)
void rmsnorm_cpu(
    float* out,
    const float* inp,
    const float* weight,
    int hidden_dim,
    float epsilon = RMSNORM_EPSILON);

void swiglu_cpu(
    float* out,
    const float* inp_x, // First half of original input
    const float* inp_gate, // Second half of original input
    int half_hidden_dim);

void fused_rmsnorm_swiglu_cpu(
    float* output,
    const float* input,
    const float* weight,
    int num_rows,
    int hidden_dim,
    float epsilon = RMSNORM_EPSILON);

#endif // FUSED_RMSNORM_SWIGLU_CUH
