#ifndef FUSED_LINEAR_SOFTMAX_LOSS_CUH
#define FUSED_LINEAR_SOFTMAX_LOSS_CUH

#include <cuda_runtime.h>
#include <cstdio> // For printf
#include <cstdlib> // For exit, EXIT_FAILURE

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief Computes fused linear transformation (matrix multiplication + bias)
 *        followed by Softmax activation and Cross-Entropy loss.
 *
 * This function is a C++ wrapper that manages memory and launches the CUDA kernel.
 *
 * @param h_input_features Pointer to host input features (M x K matrix, row-major).
 * @param h_weights Pointer to host weights (N x K matrix, row-major).
 * @param h_bias Pointer to host bias (N x 1 vector).
 * @param h_true_labels Pointer to host true labels (M x 1 vector, integer class indices).
 * @param h_output_loss_per_sample Pointer to host array to store computed loss for each sample (M x 1 vector).
 * @param M Batch size (number of input samples).
 * @param K Number of input features per sample.
 * @param N Number of output classes.
 * @return float The average cross-entropy loss over the batch.
 */
float compute_fused_linear_softmax_loss_gpu(
    const float* h_input_features,
    const float* h_weights,
    const float* h_bias,
    const int* h_true_labels,
    float* h_output_loss_per_sample,
    int M,
    int K,
    int N
);

/**
 * @brief CUDA kernel for fused linear transformation, Softmax, and Cross-Entropy loss.
 *
 * Each block typically processes one or more rows (samples).
 * Each thread within a block computes parts of the logits for a row,
 * then participates in Softmax and loss calculation for that row.
 *
 * @param d_input_features Device pointer to input features (M x K).
 * @param d_weights Device pointer to weights (N x K).
 * @param d_bias Device pointer to bias (N).
 * @param d_true_labels Device pointer to true labels (M).
 * @param d_output_loss_per_sample Device pointer to store per-sample loss (M).
 * @param M Batch size.
 * @param K Input feature dimension.
 * @param N Output class dimension.
 */
__global__ void fused_linear_softmax_loss_kernel(
    const float* d_input_features,
    const float* d_weights,
    const float* d_bias,
    const int* d_true_labels,
    float* d_output_loss_per_sample,
    int M,
    int K,
    int N
);

#endif // FUSED_LINEAR_SOFTMAX_LOSS_CUH
