#ifndef ROPE_2D_CUH
#define ROPE_2D_CUH

#include <cuda_runtime.h>
#include <cstdio> // For printf in error macro
#include <cstdlib> // For exit in error macro

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief Applies 2D Rotary Positional Embeddings (RoPE) to a batch of 2D data.
 *
 * The embedding dimension (`embedding_dim`) is split into two halves.
 * The first `embedding_dim / 2` dimensions are rotated based on the height position (`h`).
 * The second `embedding_dim / 2` dimensions are rotated based on the width position (`w`).
 * Each half is treated as a sequence of 2D vectors (pairs of features) that are rotated.
 *
 * For a token at position (h, w):
 * - For the first half of features (indices 0 to embedding_dim/2 - 1):
 *   Each pair of features (x_2i, x_{2i+1}) is rotated by an angle m_h * theta_i,
 *   where m_h = h, and theta_i is derived from theta_base.
 * - For the second half of features (indices embedding_dim/2 to embedding_dim - 1):
 *   Each pair of features (x_2j, x_{2j+1}) is rotated by an angle m_w * theta_j,
 *   where m_w = w, and theta_j is derived from theta_base.
 *
 * @param d_embeddings Device pointer to the embeddings. Data is assumed to be in
 *                     row-major order for tokens (pixels), then feature-major for embeddings.
 *                     Layout: (height * width, embedding_dim).
 * @param height The height of the 2D grid of tokens.
 * @param width The width of the 2D grid of tokens.
 * @param embedding_dim The dimension of each embedding vector.
 *                      This dimension MUST be a multiple of 4, ensuring that
 *                      embedding_dim / 2 is an even number, allowing for proper pairing
 *                      of features in each half.
 * @param theta_base The base value for calculating theta frequencies (typically 10000.0f).
 *                   The frequency for the k-th pair in a half is calculated as:
 *                   theta_k = theta_base ^ (-2k / (embedding_dim / 2)).
 */
void apply_rope_2d_embeddings_gpu(
    float* d_embeddings,
    int height,
    int width,
    int embedding_dim,
    float theta_base = 10000.0f
);

#endif // ROPE_2D_CUH
