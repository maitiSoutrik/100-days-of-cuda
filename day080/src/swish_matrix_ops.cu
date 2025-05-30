#include "../include/swish_matrix_ops.cuh"
#include <device_launch_parameters.h> // For __syncthreads, etc.

// Swish activation: f(x) = x * sigmoid(beta * x)
__global__ void swish_activation_kernel(float* input, float* output, int size, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = val * (1.0f / (1.0f + expf(-beta * val))); // Use expf for float
    }
}

// Matrix multiplication (C = A * B) with Swish activation and scaling
// C(M, N) = swish(scale * (A(M, K) * B(K, N)))
__global__ void matrix_mul_swish_scale_kernel(
    const float* A, const float* B,
    float* C,
    int M, int N, int K,
    float scale,
    float beta
) {
    // Using 16x16 tile size
    const int TILE_DIM = 16;

    // Identify the row and column of the C element to work on
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Shared memory for tiles of A and B
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    float C_value = 0.0f;

    // Loop over the tiles of A and B required to compute the C element
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load sA tile
        if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + (t * TILE_DIM + threadIdx.x)];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load sB tile
        // Note: B is accessed column-major for sB to be row-major for multiplication
        // Or, more simply, B is accessed such that sB[threadIdx.y][threadIdx.x] corresponds to B's elements
        // that multiply with A's elements in sA.
        // B[(t * TILE_DIM + threadIdx.y) * N + col]
        if ((t * TILE_DIM + threadIdx.y) < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles
        for (int i = 0; i < TILE_DIM; ++i) {
            C_value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the C_value to global memory, apply scaling and Swish
    if (row < M && col < N) {
        float scaled_value = C_value * scale;
        C[row * N + col] = scaled_value * (1.0f / (1.0f + expf(-beta * scaled_value)));
    }
}

// Host function to launch matrix multiplication with Swish and scaling
cudaError_t matrix_mul_swish_scale(
    const float* A, const float* B,
    float* C,
    int M, int N, int K,
    float scale,
    float beta,
    dim3 threadsPerBlock // Now passed as an argument
) {
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    matrix_mul_swish_scale_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A, B, C, M, N, K, scale, beta
    );

    // It's good practice to check for kernel launch errors immediately
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch matrix_mul_swish_scale_kernel: %s\n", cudaGetErrorString(err));
    }
    return err; // Return the error status
}
