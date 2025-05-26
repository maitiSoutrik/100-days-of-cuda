#include "huber_loss.cuh"
#include <cooperative_groups.h> // Not strictly needed for this simple kernel, but good practice
#include <cstdio> // For printf in CPU functions if needed

// Kernel to compute Huber loss
__global__ void huber_loss_kernel(const float* predictions,
                                  const float* targets,
                                  float* loss,
                                  int n,
                                  float delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float error = predictions[idx] - targets[idx];
        float abs_error = fabsf(error);
        if (abs_error <= delta) {
            loss[idx] = 0.5f * error * error;
        } else {
            loss[idx] = delta * (abs_error - 0.5f * delta);
        }
    }
}

// Kernel to compute the derivative of Huber loss
__global__ void huber_loss_derivative_kernel(const float* predictions,
                                             const float* targets,
                                             float* gradients,
                                             int n,
                                             float delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float error = predictions[idx] - targets[idx];
        float abs_error = fabsf(error);
        if (abs_error <= delta) {
            gradients[idx] = error;
        } else {
            gradients[idx] = delta * ((error > 0.0f) ? 1.0f : -1.0f);
        }
    }
}

// Host function to launch Huber loss kernel
void compute_huber_loss_gpu(const float* h_predictions,
                            const float* h_targets,
                            float* h_loss,
                            int n,
                            float delta) {
    float *d_predictions, *d_targets, *d_loss;
    size_t size = n * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc(&d_predictions, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_targets, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_loss, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_predictions, h_predictions, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_targets, h_targets, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    huber_loss_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_targets, d_loss, n, delta);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    CHECK_CUDA_ERROR(cudaMemcpy(h_loss, d_loss, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    CHECK_CUDA_ERROR(cudaFree(d_targets));
    CHECK_CUDA_ERROR(cudaFree(d_loss));
}

// Host function to launch Huber loss derivative kernel
void compute_huber_loss_derivative_gpu(const float* h_predictions,
                                       const float* h_targets,
                                       float* h_gradients,
                                       int n,
                                       float delta) {
    float *d_predictions, *d_targets, *d_gradients;
    size_t size = n * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc(&d_predictions, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_targets, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_gradients, size));

    CHECK_CUDA_ERROR(cudaMemcpy(d_predictions, h_predictions, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_targets, h_targets, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    huber_loss_derivative_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_targets, d_gradients, n, delta);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    CHECK_CUDA_ERROR(cudaMemcpy(h_gradients, d_gradients, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    CHECK_CUDA_ERROR(cudaFree(d_targets));
    CHECK_CUDA_ERROR(cudaFree(d_gradients));
}

// CPU implementation for Huber loss
void huber_loss_cpu(const float* predictions,
                    const float* targets,
                    float* loss,
                    int n,
                    float delta) {
    for (int i = 0; i < n; ++i) {
        float error = predictions[i] - targets[i];
        float abs_error = std::fabs(error); // Use std::fabs for float
        if (abs_error <= delta) {
            loss[i] = 0.5f * error * error;
        } else {
            loss[i] = delta * (abs_error - 0.5f * delta);
        }
    }
}

// CPU implementation for Huber loss derivative
void huber_loss_derivative_cpu(const float* predictions,
                               const float* targets,
                               float* gradients,
                               int n,
                               float delta) {
    for (int i = 0; i < n; ++i) {
        float error = predictions[i] - targets[i];
        float abs_error = std::fabs(error); // Use std::fabs for float
        if (abs_error <= delta) {
            gradients[i] = error;
        } else {
            gradients[i] = delta * ((error > 0.0f) ? 1.0f : -1.0f);
        }
    }
}
