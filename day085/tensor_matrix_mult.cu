#include "tensor_matrix_mult.cuh"

__global__ void tensorMatrixMultKernel(
    const float* A,
    const float* B,
    float* C,
    size_t B_dim,
    size_t I_dim,
    size_t J_dim,
    size_t L_dim,
    size_t K_dim
) {
    // Calculate the global thread ID
    // Each thread computes one element of the output tensor C
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total elements in C: B_dim * I_dim * J_dim * K_dim
    size_t total_elements_C = B_dim * I_dim * J_dim * K_dim;

    if (idx < total_elements_C) {
        // Decompose idx into (b, i, j, k) indices for tensor C
        size_t k = idx % K_dim;
        size_t temp_idx = idx / K_dim;
        size_t j = temp_idx % J_dim;
        temp_idx /= J_dim;
        size_t i = temp_idx % I_dim;
        size_t b = temp_idx / I_dim;

        float sum = 0.0f;

        // Index for A: A[b, i, j, l]
        // A is stored as B_dim * I_dim * J_dim * L_dim
        // Offset for A[b,i,j,:] is (b * I_dim * J_dim * L_dim) + (i * J_dim * L_dim) + (j * L_dim)
        // Simplified: ((b * I_dim + i) * J_dim + j) * L_dim
        size_t a_base_offset = ((b * I_dim + i) * J_dim + j) * L_dim;

        // Index for B: B[l, k]
        // B is stored as L_dim * K_dim (row-major)

        for (size_t l = 0; l < L_dim; ++l) {
            sum += A[a_base_offset + l] * B[l * K_dim + k];
        }

        // C[idx] is C[b,i,j,k]
        C[idx] = sum;
    }
}

extern "C" void tensor_matrix_multiply(
    const float* A_dev,
    const float* B_dev,
    float* C_dev,
    size_t B_dim,
    size_t I_dim,
    size_t J_dim,
    size_t L_dim,
    size_t K_dim
) {
    size_t total_output_elements = B_dim * I_dim * J_dim * K_dim;
    if (total_output_elements == 0) {
        // Or handle error appropriately
        return;
    }

    // Define block and grid dimensions
    // Max threads per block is typically 1024 for compute capability 5.3 (Jetson Nano)
    // Let's use 256 as a common default.
    int threadsPerBlock = 256;
    // Ensure threadsPerBlock does not exceed device capabilities if necessary,
    // but 256 is safe for sm_53.
    
    // Calculate number of blocks needed
    // (total_elements + threads_per_block - 1) / threads_per_block ensures enough blocks
    int blocksPerGrid = (total_output_elements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    tensorMatrixMultKernel<<<blocksPerGrid, threadsPerBlock>>>(
        A_dev, B_dev, C_dev,
        B_dim, I_dim, J_dim, L_dim, K_dim
    );

    // It's good practice for the calling code (e.g., main or test) to check for errors
    // after calling this function using cudaGetLastError() and cudaDeviceSynchronize().
    // The CHECK_KERNEL_LAUNCH() macro in the .cuh can be used for this.
}
