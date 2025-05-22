#include "rope_2d.cuh"
#include <cmath> // For powf, cosf, sinf

__global__ void rope_2d_kernel(
    float* embeddings,
    int height,
    int width,
    int embedding_dim,
    float theta_base) {

    // Calculate the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int num_tokens = height * width;
    int pairs_per_token = embedding_dim / 2; // Total pairs for one token's embedding vector
    int total_pairs_to_process = num_tokens * pairs_per_token;

    if (tid >= total_pairs_to_process) {
        return; // Out of bounds
    }

    // Determine which token and which pair within that token this thread handles
    int token_idx = tid / pairs_per_token;
    int pair_idx_in_token = tid % pairs_per_token; // Ranges from 0 to (embedding_dim/2 - 1)

    // Determine 2D coordinates (h, w) of the token
    int h = token_idx / width;
    int w = token_idx % width;

    // Determine which half of the embedding this pair belongs to
    int embedding_dim_half = embedding_dim / 2; // Features in one half
    int num_pairs_per_half = embedding_dim / 4; // Pairs in one half

    float m; // The position value (h or w)
    int feature_pair_offset_in_half; // Index of this pair within its half (0 to num_pairs_per_half - 1)
    int base_feature_idx; // Starting index for this pair in the full embedding vector

    if (pair_idx_in_token < num_pairs_per_half) {
        // First half: rotate based on height (h)
        m = static_cast<float>(h);
        feature_pair_offset_in_half = pair_idx_in_token;
        base_feature_idx = feature_pair_offset_in_half * 2;
    } else {
        // Second half: rotate based on width (w)
        m = static_cast<float>(w);
        feature_pair_offset_in_half = pair_idx_in_token - num_pairs_per_half;
        base_feature_idx = embedding_dim_half + feature_pair_offset_in_half * 2;
    }

    // Calculate theta_k for this pair
    // theta_k = theta_base ^ (-2k / d_half), where d_half = embedding_dim_half
    float theta_k = powf(theta_base, -2.0f * feature_pair_offset_in_half / static_cast<float>(embedding_dim_half));

    float m_theta = m * theta_k;
    float cos_m_theta = cosf(m_theta);
    float sin_m_theta = sinf(m_theta);

    // Get the indices for the pair of features
    int idx0 = token_idx * embedding_dim + base_feature_idx;
    int idx1 = idx0 + 1;

    // Load the original feature values
    float x0 = embeddings[idx0];
    float x1 = embeddings[idx1];

    // Apply rotation
    embeddings[idx0] = x0 * cos_m_theta - x1 * sin_m_theta;
    embeddings[idx1] = x0 * sin_m_theta + x1 * cos_m_theta;
}

void apply_rope_2d_embeddings_gpu(
    float* d_embeddings,
    int height,
    int width,
    int embedding_dim,
    float theta_base) {

    // Input validation
    if (embedding_dim % 4 != 0) {
        fprintf(stderr, "Error: embedding_dim must be a multiple of 4 for 2D RoPE. Got %d\n", embedding_dim);
        exit(EXIT_FAILURE);
    }
    if (embedding_dim == 0) {
        fprintf(stderr, "Error: embedding_dim cannot be zero.\n");
        exit(EXIT_FAILURE);
    }
     if (height <= 0 || width <= 0) {
        fprintf(stderr, "Error: height and width must be positive. Got height=%d, width=%d\n", height, width);
        exit(EXIT_FAILURE);
    }


    int num_tokens = height * width;
    int pairs_per_token = embedding_dim / 2;
    int total_pairs_to_process = num_tokens * pairs_per_token;

    if (total_pairs_to_process == 0) {
        // Nothing to process, e.g. if height or width is 0 (though validated above) or embedding_dim is 0 (validated above)
        return;
    }

    // Kernel launch configuration
    int threads_per_block = 256;
    int blocks_per_grid = (total_pairs_to_process + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    rope_2d_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_embeddings,
        height,
        width,
        embedding_dim,
        theta_base
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    // No explicit cudaDeviceSynchronize here, caller can synchronize if needed.
    // For library functions, it's often better to let the caller manage synchronization.
}
