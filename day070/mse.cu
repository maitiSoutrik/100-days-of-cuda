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

// Functor for x*x used by Thrust's second step
struct square_functor {
    __host__ __device__
    float operator()(const float& x) const {
        return x * x;
    }
};

// GPU implementation of MSE using a two-step Thrust approach
void mse_gpu(const float* h_predictions, const float* h_targets, int N, float* mse_result) {
    if (N == 0) {
        *mse_result = 0.0f;
        return;
    }

    float *d_predictions, *d_targets, *d_diff;

    // Allocate memory on the device
    CHECK_CUDA_ERROR(cudaMalloc(&d_predictions, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_targets, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_diff, N * sizeof(float))); // For storing P - T

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_predictions, h_predictions, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_targets, h_targets, N * sizeof(float), cudaMemcpyHostToDevice));

    // Wrap raw device pointers with thrust::device_ptr
    thrust::device_ptr<const float> d_predictions_ptr(d_predictions);
    thrust::device_ptr<const float> d_targets_ptr(d_targets);
    thrust::device_ptr<float> d_diff_ptr(d_diff);

    // Step 1: Compute differences D = P - T
    // D[i] = P[i] - T[i]
    thrust::transform(
        d_predictions_ptr,       // Start of first input range (P)
        d_predictions_ptr + N,   // End of first input range (P)
        d_targets_ptr,           // Start of second input range (T)
        d_diff_ptr,              // Start of output range (D)
        thrust::minus<float>()   // Binary operation (P[i] - T[i])
    );

    // Step 2: Compute sum of squares of differences
    // sum_sq_error = sum(D[i] * D[i])
    float sum_sq_error = thrust::transform_reduce(
        d_diff_ptr,              // Start of input range (D)
        d_diff_ptr + N,          // End of input range (D)
        square_functor(),        // Unary transform operation (D[i]*D[i])
        0.0f,                    // Initial value for the reduction
        thrust::plus<float>()    // Reduction operation (summation)
    );
    
    *mse_result = sum_sq_error / static_cast<float>(N);

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_predictions));
    CHECK_CUDA_ERROR(cudaFree(d_targets));
    CHECK_CUDA_ERROR(cudaFree(d_diff));
}
