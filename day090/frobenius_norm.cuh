#ifndef FROBENIUS_NORM_CUH
#define FROBENIUS_NORM_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // For sqrtf

// Error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

/**
 * @brief Calculates the Frobenius norm of a matrix on the GPU.
 *
 * @param d_matrix Pointer to the matrix data on the device.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The Frobenius norm of the matrix.
 */
float frobeniusNormGPU(const float* d_matrix, int rows, int cols);

/**
 * @brief Calculates the Frobenius norm of a matrix on the CPU.
 *
 * @param h_matrix Pointer to the matrix data on the host.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The Frobenius norm of the matrix.
 */
float frobeniusNormCPU(const float* h_matrix, int rows, int cols);

#endif // FROBENIUS_NORM_CUH
