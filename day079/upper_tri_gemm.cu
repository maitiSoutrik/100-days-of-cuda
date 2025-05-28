#include "upper_tri_gemm.cuh"

template <typename T, size_t BLOCK_SIZE>
__global__ void gemm_upper_tri(const T* A, const T* B, T* C, int n) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ T A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T B_tile[BLOCK_SIZE][BLOCK_SIZE];

    T sum = 0;

    for (int t = 0; t < n; t += BLOCK_SIZE) {
        // Load A_tile elements. A_tile[threadIdx.y][threadIdx.x] corresponds to A[i][t + threadIdx.x].
        // If the access is out of bounds for A, or if A[i][t + threadIdx.x] is in the lower triangle (i > t + threadIdx.x),
        // it should be treated as zero. Since input A is already upper triangular, we only need to check bounds.
        if (i < n && (t + threadIdx.x) < n) {
            A_tile[threadIdx.y][threadIdx.x] = A[i * n + (t + threadIdx.x)];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0; // Out of bounds for A
        }

        // Load B_tile elements. B_tile[threadIdx.y][threadIdx.x] corresponds to B[t + threadIdx.y][j].
        // If the access is out of bounds for B, or if B[t + threadIdx.y][j] is in the lower triangle (t + threadIdx.y > j),
        // it should be treated as zero. Since input B is already upper triangular, we only need to check bounds.
        if ((t + threadIdx.y) < n && j < n) {
            B_tile[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * n + j];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0; // Out of bounds for B
        }
        __syncthreads();

        for (int k_idx = 0; k_idx < BLOCK_SIZE; ++k_idx) {
            int current_k = t + k_idx;
            if (current_k < n && i <= current_k && current_k <= j) { // Only multiply if elements are in upper triangle
                sum += A_tile[threadIdx.y][k_idx] * B_tile[k_idx][threadIdx.x];
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
