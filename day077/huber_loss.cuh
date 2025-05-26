#ifndef HUBER_LOSS_CUH
#define HUBER_LOSS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath> // For fabsf
#include <cstdio> // For printf in error checks

// Error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

/**
 * @brief Computes the Huber loss.
 *
 * The Huber loss function is defined as:
 * L_delta(a) = 0.5 * a^2                   if |a| <= delta
 * L_delta(a) = delta * (|a| - 0.5 * delta)  if |a| > delta
 *
 * @param predictions Pointer to the array of predicted values on the device.
 * @param targets Pointer to the array of target values on the device.
 * @param loss Pointer to the array where the computed loss for each element will be stored on the device.
 * @param n Number of elements.
 * @param delta The threshold parameter for Huber loss.
 */
__global__ void huber_loss_kernel(const float* predictions,
                                  const float* targets,
                                  float* loss,
                                  int n,
                                  float delta);

/**
 * @brief Computes the derivative of the Huber loss.
 *
 * The derivative of the Huber loss function is:
 * dL/da = a              if |a| <= delta
 * dL/da = delta * sign(a) if |a| > delta
 *
 * @param predictions Pointer to the array of predicted values on the device.
 * @param targets Pointer to the array of target values on the device.
 * @param gradients Pointer to the array where the computed gradients for each element will be stored on the device.
 * @param n Number of elements.
 * @param delta The threshold parameter for Huber loss.
 */
__global__ void huber_loss_derivative_kernel(const float* predictions,
                                             const float* targets,
                                             float* gradients,
                                             int n,
                                             float delta);

// Host functions to launch kernels
void compute_huber_loss_gpu(const float* h_predictions,
                            const float* h_targets,
                            float* h_loss,
                            int n,
                            float delta);

void compute_huber_loss_derivative_gpu(const float* h_predictions,
                                       const float* h_targets,
                                       float* h_gradients,
                                       int n,
                                       float delta);

// CPU implementations for comparison/verification
void huber_loss_cpu(const float* predictions,
                    const float* targets,
                    float* loss,
                    int n,
                    float delta);

void huber_loss_derivative_cpu(const float* predictions,
                               const float* targets,
                               float* gradients,
                               int n,
                               float delta);

#endif // HUBER_LOSS_CUH
