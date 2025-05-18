#include "mse.cuh"
#include <cmath>     // For std::pow
#include <numeric>   // For std::accumulate (though not used in final kernel for sum)
#include <vector>    // For intermediate sum storage if needed

/*
// --- Custom Kernels (Replaced by cuBLAS version) ---
// CUDA Kernel for calculating squared errors and summing them up
__global__ void mse_kernel(const float* predictions, const float* targets, float* squared_errors, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float diff = predictions[idx] - targets[idx];
        squared_errors[idx] = diff * diff;
    }
}

// Kernel for sum reduction (a simple version, can be optimized further)
__global__ void sum_reduction_kernel(const float* data, float* out_sum, int N) {
    extern __shared__ float sdata[]; // Shared memory for partial sums

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x; // Global index

    // Load data into shared memory
    if (i < N) {
        sdata[tid] = data[i];
    } else {
        sdata[tid] = 0;
    }
    if (i + blockDim.x < N) {
        sdata[tid] += data[i + blockDim.x];
    }

    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        out_sum[blockIdx.x] = sdata[0];
    }
}
*/

// CPU implementation of MSE
float mse_cpu(const float* predictions, const float* targets, int N) {
    if (N == 0) return 0.0f;
    double sum_sq_error = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff = static_cast<double>(predictions[i]) - static_cast<double>(targets[i]);
        sum_sq_error += diff * diff;
    }
    return static_cast<float>(sum_sq_error / N);
}

// GPU implementation of MSE using cuBLAS
void mse_gpu(const float* h_predictions, const float* h_targets, int N, float* mse_result) {
    if (N == 0) {
        *mse_result = 0.0f;
        return;
    }

    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    float *d_predictions, *d_targets, *d_diff;

    // Allocate memory on the device
    CHECK_CUDA_ERROR(cudaMalloc(&d_predictions, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_targets, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_diff, N * sizeof(float))); // To store (predictions - targets)

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_predictions, h_predictions, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_targets, h_targets, N * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate D = P - T
    // 1. Copy P to D: d_diff = d_predictions
    CHECK_CUBLAS_ERROR(cublasScopy(handle, N, d_predictions, 1, d_diff, 1));
    
    // 2. D = -1*T + D: d_diff = -1.0f * d_targets + d_diff
    float alpha = -1.0f;
    CHECK_CUBLAS_ERROR(cublasSaxpy(handle, N, &alpha, d_targets, 1, d_diff, 1));

    // Calculate sum_sq_error = D . D (dot product of d_diff with itself)
    float sum_sq_error_host; // cublasSdot result is written to a host pointer
    // Ensure cuBLAS pointer mode is host for the result of sdot
    // (This is the default, but can be set explicitly if needed: cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST))
    CHECK_CUBLAS_ERROR(cublasSdot(handle, N, d_diff, 1, d_diff, 1, &sum_sq_error_host));
    
    *mse_result = sum_sq_error_host / static_cast<float>(N);

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    CHECK_CUDA_ERROR(cudaFree(d_targets));
    CHECK_CUDA_ERROR(cudaFree(d_diff));
    
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
}
