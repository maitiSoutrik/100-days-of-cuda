#ifndef BATCHED_L2_NORM_CUH
#define BATCHED_L2_NORM_CUH

#include <cuda_runtime.h>
#include <cstdio> // For printf in kernel if needed for debugging

// Error checking macro
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief Computes the L2 norm for a batch of vectors.
 *
 * Each thread block processes one vector. Threads within the block cooperatively
 * compute the L2 norm (sum of squares, then sqrt) using shared memory reduction.
 *
 * @param d_vectors Pointer to the input vectors on the device.
 *                  Data is assumed to be laid out contiguously:
 *                  [vec1_val1, vec1_val2, ..., vec1_val_dim,
 *                   vec2_val1, vec2_val2, ..., vec2_val_dim,
 *                   ...]
 * @param d_norms Pointer to the output norms on the device.
 * @param num_batches The number of vectors in the batch.
 * @param vector_dim The dimension of each vector.
 */
__global__ void batched_l2_norm_kernel(const float* d_vectors,
                                       float* d_norms,
                                       int num_batches,
                                       int vector_dim);

/**
 * @brief Wrapper function to launch the batched L2 norm kernel.
 *
 * @param h_vectors Pointer to the input vectors on the host.
 * @param h_norms Pointer to the output norms on the host (will be populated).
 * @param num_batches The number of vectors in the batch.
 * @param vector_dim The dimension of each vector.
 */
void compute_batched_l2_norm_gpu(const float* h_vectors,
                                 float* h_norms,
                                 int num_batches,
                                 int vector_dim);

/**
 * @brief Computes the L2 norm for a batch of vectors on the CPU.
 *
 * Used for verification.
 *
 * @param h_vectors Pointer to the input vectors on the host.
 * @param h_norms_cpu Pointer to the output norms on the host (will be populated by CPU computation).
 * @param num_batches The number of vectors in the batch.
 * @param vector_dim The dimension of each vector.
 */
void compute_batched_l2_norm_cpu(const float* h_vectors,
                                 float* h_norms_cpu,
                                 int num_batches,
                                 int vector_dim);

#endif // BATCHED_L2_NORM_CUH
