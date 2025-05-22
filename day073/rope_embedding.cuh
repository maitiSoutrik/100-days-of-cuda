#ifndef ROPE_EMBEDDING_CUH
#define ROPE_EMBEDDING_CUH

#include <cuda_runtime.h>
#include <vector>
#include <cmath> // For std::cos, std::sin, std::pow in potential CPU reference

// CUDA error checking macro (as per .clinerules)
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

/**
 * @brief Applies 1D Rotary Positional Embedding (RoPE) to a batch of token embeddings.
 *
 * @param d_output_embeddings Pointer to the device memory for output embeddings (N x D).
 * @param d_input_embeddings Pointer to the device memory for input embeddings (N x D).
 * @param d_positions Pointer to the device memory for 1D positions of tokens (N).
 * @param num_tokens The number of tokens (N).
 * @param embedding_dim The dimension of each embedding (D). Must be even.
 * @param base_period The base period for frequency calculation (e.g., 10000.0f).
 * @param stream CUDA stream for asynchronous execution.
 */
void apply_rope_1d_embedding_cuda(
    float* d_output_embeddings,
    const float* d_input_embeddings,
    const int* d_positions,
    int num_tokens,
    int embedding_dim,
    float base_period,
    cudaStream_t stream = 0
);

// Optional: A CPU reference implementation for testing/verification
void apply_rope_1d_embedding_cpu(
    std::vector<float>& output_embeddings,
    const std::vector<float>& input_embeddings,
    const std::vector<int>& positions,
    int num_tokens,
    int embedding_dim,
    float base_period
);

#endif // ROPE_EMBEDDING_CUH
