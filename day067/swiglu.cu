#include "swiglu.cuh"
#include <cmath> // For expf

// Device function for sigmoid (can be in .cuh if used by multiple .cu, or static here)
__device__ inline float sigmoidf_device(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void swiglu_forward_kernel(const float* d_a,
                                      const float* d_b,
                                      float* d_c,
                                      int rows,
                                      int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        float val_a = d_a[idx];
        float val_b = d_b[idx];

        float s_a = sigmoidf_device(val_a);
        float silu_a = val_a * s_a;
        
        d_c[idx] = silu_a * val_b;
    }
}

__global__ void swiglu_backward_kernel(const float* d_a,
                                       const float* d_b,
                                       const float* d_dc,
                                       float* d_da,
                                       float* d_db,
                                       int rows,
                                       int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        float val_a = d_a[idx];
        float val_b = d_b[idx];
        float val_dc = d_dc[idx];

        // Recompute sigmoid(a)
        float s_a = sigmoidf_device(val_a);

        // Gradient for a: da = dc * b * s_a * (1 + a * (1 - s_a))
        d_da[idx] = val_dc * val_b * s_a * (1.0f + val_a * (1.0f - s_a));
        
        // Gradient for b: db = dc * a * s_a
        d_db[idx] = val_dc * val_a * s_a;
    }
}

void launch_swiglu_forward(const float* d_a, const float* d_b, float* d_c, int rows, int cols, cudaStream_t stream) {
    // As per problem: one block per row, one thread per column.
    // This assumes 'cols' is not greater than maxThreadsPerBlock.x (typically 1024).
    // If 'cols' can be larger, a different launch configuration or a grid-stride loop inside the kernel would be needed.
    if (cols == 0 || rows == 0) return; // Avoid launching empty grids

    dim3 threadsPerBlock(cols);
    dim3 numBlocks(rows);
    
    swiglu_forward_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_a, d_b, d_c, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
}

void launch_swiglu_backward(const float* d_a, const float* d_b, const float* d_dc, float* d_da, float* d_db, int rows, int cols, cudaStream_t stream) {
    if (cols == 0 || rows == 0) return;

    dim3 threadsPerBlock(cols);
    dim3 numBlocks(rows);

    swiglu_backward_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_a, d_b, d_dc, d_da, d_db, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
}
