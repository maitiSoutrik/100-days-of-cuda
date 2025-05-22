#include "rope_embedding.cuh"
#include <cstdio> // For printf in kernel if needed for debugging (remove for final)
#include <iostream> // For std::cout in CPU version

// CUDA Kernel to apply 1D RoPE
__global__ void rope_1d_embedding_kernel(
    float* output_embeddings,
    const float* input_embeddings,
    const int* positions,
    int num_tokens,
    int embedding_dim,
    float base_period) {

    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (token_idx < num_tokens) {
        int pos = positions[token_idx];
        const float* current_input_embedding = input_embeddings + token_idx * embedding_dim;
        float* current_output_embedding = output_embeddings + token_idx * embedding_dim;

        for (int i = 0; i < embedding_dim / 2; ++i) {
            float theta_i = 1.0f / powf(base_period, (2.0f * i) / static_cast<float>(embedding_dim));
            float freq = static_cast<float>(pos) * theta_i;

            float cos_freq = cosf(freq);
            float sin_freq = sinf(freq);

            float val1 = current_input_embedding[2 * i];
            float val2 = current_input_embedding[2 * i + 1];

            current_output_embedding[2 * i]     = val1 * cos_freq - val2 * sin_freq;
            current_output_embedding[2 * i + 1] = val1 * sin_freq + val2 * cos_freq;
        }
    }
}

// Host wrapper function to launch the CUDA kernel
void apply_rope_1d_embedding_cuda(
    float* d_output_embeddings,
    const float* d_input_embeddings,
    const int* d_positions,
    int num_tokens,
    int embedding_dim,
    float base_period,
    cudaStream_t stream) {

    if (embedding_dim % 2 != 0) {
        fprintf(stderr, "Error: embedding_dim must be even for RoPE.\n");
        return;
    }

    dim3 threads_per_block(256);
    dim3 num_blocks((num_tokens + threads_per_block.x - 1) / threads_per_block.x);

    rope_1d_embedding_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_output_embeddings,
        d_input_embeddings,
        d_positions,
        num_tokens,
        embedding_dim,
        base_period
    );
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
}

// CPU reference implementation
void apply_rope_1d_embedding_cpu(
    std::vector<float>& output_embeddings,
    const std::vector<float>& input_embeddings,
    const std::vector<int>& positions,
    int num_tokens,
    int embedding_dim,
    float base_period) {

    if (embedding_dim % 2 != 0) {
        std::cerr << "Error: embedding_dim must be even for RoPE." << std::endl;
        return;
    }
    if (input_embeddings.size() != static_cast<size_t>(num_tokens * embedding_dim)) {
        std::cerr << "Error: input_embeddings size mismatch." << std::endl;
        return;
    }
    if (positions.size() != static_cast<size_t>(num_tokens)) {
        std::cerr << "Error: positions size mismatch." << std::endl;
        return;
    }

    output_embeddings.resize(num_tokens * embedding_dim);

    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
        int pos = positions[token_idx];
        const float* current_input_embedding = input_embeddings.data() + token_idx * embedding_dim;
        float* current_output_embedding = output_embeddings.data() + token_idx * embedding_dim;

        for (int i = 0; i < embedding_dim / 2; ++i) {
            float theta_i = 1.0f / std::pow(base_period, (2.0f * i) / static_cast<float>(embedding_dim));
            float freq = static_cast<float>(pos) * theta_i;

            float cos_freq = std::cos(freq);
            float sin_freq = std::sin(freq);

            float val1 = current_input_embedding[2 * i];
            float val2 = current_input_embedding[2 * i + 1];

            current_output_embedding[2 * i]     = val1 * cos_freq - val2 * sin_freq;
            current_output_embedding[2 * i + 1] = val1 * sin_freq + val2 * cos_freq;
        }
    }
}
