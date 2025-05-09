#ifndef MUON_OPTIMIZATION_CUH
#define MUON_OPTIMIZATION_CUH

#include <cuda_runtime.h>
#include <cstdio> // For printf in kernels (debug) and host

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// Helper function to print matrix from device (for debugging)
void print_matrix_device(const float* d_matrix, int rows, int cols, const char* label);
void print_matrix_host(const float* h_matrix, int rows, int cols, const char* label);

// Kernel declarations for matrix operations
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols);
__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int A_rows, int A_cols, int B_cols);
__global__ void matrix_scalar_multiply_kernel(float* matrix, float scalar, int rows, int cols);
__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int rows, int cols); // C = A + B
__global__ void matrix_elementwise_subtract_kernel(const float* A, const float* B, float* C, int rows, int cols); // C = A - B
__global__ void matrix_copy_kernel(const float* source, float* destination, int rows, int cols);
__global__ void frobenius_norm_squared_kernel(const float* matrix, float* d_result, int N); // N = rows * cols
__global__ void matrix_scalar_divide_kernel(float* matrix, float scalar, int rows, int cols);
__global__ void initialize_matrix_kernel(float* matrix, int rows, int cols, float val); // Initialize with a specific value
__global__ void initialize_identity_matrix_kernel(float* matrix, int N); // Initialize NxN identity matrix


// Host function to orchestrate Newton-Schulz iteration on the device
// G_in: input matrix (e.g., gradient update), rows x cols
// G_out: output matrix (orthogonalized), rows x cols
// temp_matrices: array of pointers to pre-allocated temporary device matrices
// num_iterations: number of NS iterations
void newton_schulz_iteration_device(
    const float* d_G_in,
    float* d_G_out,
    int rows,
    int cols,
    int num_ns_iterations,
    float* d_temp_O,      // For O_k
    float* d_temp_O_T,    // For O_k^T
    float* d_temp_prod1,  // For O_k * O_k^T or O_k^T * O_k
    float* d_temp_prod2,  // For O_k * O_k^T * O_k or O_k^T * O_k * O_k
    float* d_partial_sums // For Frobenius norm reduction
);

#endif // MUON_OPTIMIZATION_CUH
