#ifndef ADAHESSIAN_H
#define ADAHESSIAN_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> // For exit, EXIT_FAILURE

// Simple CUDA error checking macro
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %d: %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// AdaHessian update kernel Declaration
// Parameters:
//   theta:            current model parameters (to be updated)
//   grad:             gradient computed at theta
//   gradPerturbed:    gradient computed at theta + delta
//   m:                first moment estimate (accumulator for gradients)
//   v:                second moment estimate (accumulator for Hessian diag squared)
//   lr:               learning rate
//   beta1:            exponential decay rate for first moment
//   beta2:            exponential decay rate for second moment
//   epsilon:          small constant for numerical stability
//   delta:            finite difference perturbation value
//   N:                total number of parameters
__global__ void adaHessianUpdateKernel(
    float* theta,
    const float* grad,
    const float* gradPerturbed,
    float* m,
    float* v,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float delta,
    int N
);

#endif // ADAHESSIAN_H
