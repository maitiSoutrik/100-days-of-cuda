#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cublas_v2.h> // For cuBLAS errors if needed
#include <curand.h>   // For cuRAND errors if needed
#include <cufft.h>    // For cuFFT errors if needed
#include <iostream>
#include <stdexcept> // For std::runtime_error

// Macro to check CUDA API call errors
#define CHECK_CUDA_ERROR(call)                                                 \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__,     \
                    __LINE__, cudaGetErrorString(err));                        \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

// Macro to check cuBLAS API call errors
#define CHECK_CUBLAS_ERROR(call)                                               \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS Error in %s at line %d: Error code %d\n",  \
                    __FILE__, __LINE__, status);                               \
            /* You might want a more descriptive error message here */         \
            throw std::runtime_error("cuBLAS error");                          \
        }                                                                      \
    } while (0)

// Macro to check cuRAND API call errors
#define CHECK_CURAND_ERROR(call)                                               \
    do {                                                                       \
        curandStatus_t status = call;                                          \
        if (status != CURAND_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuRAND Error in %s at line %d: Error code %d\n",  \
                    __FILE__, __LINE__, status);                               \
            throw std::runtime_error("cuRAND error");                          \
        }                                                                      \
    } while (0)

// Macro to check cuFFT API call errors
#define CHECK_CUFFT_ERROR(call)                                                \
    do {                                                                       \
        cufftResult status = call;                                             \
        if (status != CUFFT_SUCCESS) {                                         \
            fprintf(stderr, "cuFFT Error in %s at line %d: Error code %d\n",   \
                    __FILE__, __LINE__, status);                               \
            throw std::runtime_error("cuFFT error");                           \
        }                                                                      \
    } while (0)

// Macro to check for kernel launch errors
#define CHECK_KERNEL_LAUNCH_ERROR()                                            \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Kernel Launch Error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
        err = cudaDeviceSynchronize(); /* Ensure kernel completion */          \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Device Sync Error in %s at line %d: %s\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

#endif // CUDA_UTILS_H
