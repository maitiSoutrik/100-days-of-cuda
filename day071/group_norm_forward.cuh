#ifndef GROUP_NORM_FORWARD_CUH
#define GROUP_NORM_FORWARD_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h> // For printf in kernels if needed for debugging

// Error checking macro
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief Performs Group Normalization forward pass on the input tensor.
 *
 * @param output Pointer to the output tensor on the device.
 * @param input Pointer to the input tensor on the device.
 * @param N Batch size.
 * @param C Number of channels.
 * @param H Height of the feature maps.
 * @param W Width of the feature maps.
 * @param G Number of groups.
 * @param gamma Pointer to the scale parameters (gamma) on the device (size C).
 * @param beta Pointer to the shift parameters (beta) on the device (size C).
 * @param epsilon Small value to prevent division by zero.
 */
void groupNormForward(
    float* output,
    const float* input,
    int N, int C, int H, int W, int G,
    const float* gamma,
    const float* beta,
    float epsilon
);

/**
 * @brief CPU implementation of Group Normalization forward pass for verification.
 *
 * @param output Pointer to the output tensor on the host.
 * @param input Pointer to the input tensor on the host.
 * @param N Batch size.
 * @param C Number of channels.
 * @param H Height of the feature maps.
 * @param W Width of the feature maps.
 * @param G Number of groups.
 * @param gamma Pointer to the scale parameters (gamma) on the host (size C).
 * @param beta Pointer to the shift parameters (beta) on the host (size C).
 * @param epsilon Small value to prevent division by zero.
 */
void groupNormForwardCPU(
    float* output,
    const float* input,
    int N, int C, int H, int W, int G,
    const float* gamma,
    const float* beta,
    float epsilon
);

#endif // GROUP_NORM_FORWARD_CUH
