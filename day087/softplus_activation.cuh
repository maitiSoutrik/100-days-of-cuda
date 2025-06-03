#ifndef SOFTPLUS_ACTIVATION_CUH
#define SOFTPLUS_ACTIVATION_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath> // For std::log and std::exp

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

/**
 * @brief Computes the Softplus activation function on the GPU.
 *
 * Softplus(x) = log(1 + exp(x))
 *
 * @param d_input Pointer to the input tensor on the device.
 * @param d_output Pointer to the output tensor on the device.
 * @param N Number of elements in the tensors.
 */
void softplusActivation(const float* d_input, float* d_output, int N);

/**
 * @brief Computes the Softplus activation function on the CPU for verification.
 *
 * @param h_input Pointer to the input tensor on the host.
 * @param h_output Pointer to the output tensor on the host.
 * @param N Number of elements in the tensors.
 */
void softplusActivationCPU(const float* h_input, float* h_output, int N);

#endif // SOFTPLUS_ACTIVATION_CUH