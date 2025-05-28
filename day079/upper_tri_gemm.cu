#include "upper_tri_gemm.cuh"

template <typename T, size_t BLOCK_SIZE>
__global__ void gemm_upper_tri(const T* A, const T* B, T* C, int n) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ T A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T B_tile[BLOCK_SIZE][BLOCK_SIZE];

    T sum = 0;

    for (int t = 0; t < n; t += BLOCK_SIZE) {
        if (i < n && t + threadIdx.x < n) {
            A_tile[threadIdx.y][threadIdx.x] = A[i * n + t + threadIdx.x];
        }
        if (t + threadIdx.y < n && j < n) {
            B_tile[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * n + j];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            if (i < j) break;
            if (t + k < n) {
                sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (i < n && j < n && i <= j) {
        C[i * n + j] = sum;
    }
}

// Explicit instantiations
template __global__ void gemm_upper_tri<float, 16>(const float*, const float*, float*, int);
template __global__ void gemm_upper_tri<double, 16>(const double*, const double*, double*, int);
template __global__ void gemm_upper_tri<float, 2>(const float*, const float*, float*, int);
template __global__ void gemm_upper_tri<double, 2>(const double*, const double*, double*, int);
