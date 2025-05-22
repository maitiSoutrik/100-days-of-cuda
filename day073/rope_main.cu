#include "rope_embedding.cuh"
#include <iostream>
#include <vector>
#include <iomanip> // For std::fixed and std::setprecision
#include <numeric> // For std::iota
#include <algorithm> // For std::generate
#include <random>    // For std::mt19937 and std::uniform_real_distribution

// Helper function to print embeddings
void print_embeddings(const std::string& title, const std::vector<float>& embeddings, int num_tokens, int embedding_dim, int tokens_to_print = 2, int dims_to_print = 4) {
    std::cout << title << std::endl;
    dims_to_print = std::min(dims_to_print, embedding_dim);
    tokens_to_print = std::min(tokens_to_print, num_tokens);

    for (int i = 0; i < tokens_to_print; ++i) {
        std::cout << "  Token " << i << ": [";
        for (int j = 0; j < dims_to_print; ++j) {
            std::cout << std::fixed << std::setprecision(4) << embeddings[i * embedding_dim + j] << (j == dims_to_print - 1 ? "" : ", ");
        }
        if (dims_to_print < embedding_dim) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
    if (tokens_to_print < num_tokens) std::cout << "  ..." << std::endl;
    std::cout << std::endl;
}

int main() {
    // Parameters
    int num_tokens = 10;       // Example: 10 tokens
    int embedding_dim = 8;    // Example: 8 dimensions (must be even)
    float base_period = 10000.0f;

    std::cout << "1D Rotary Positional Embedding (RoPE) Demonstration" << std::endl;
    std::cout << "Number of tokens: " << num_tokens << std::endl;
    std::cout << "Embedding dimension: " << embedding_dim << std::endl;
    std::cout << "Base period: " << base_period << std::endl << std::endl;

    // Initialize host data
    std::vector<float> h_input_embeddings(num_tokens * embedding_dim);
    std::vector<int> h_positions(num_tokens);
    std::vector<float> h_output_embeddings_cuda(num_tokens * embedding_dim);
    std::vector<float> h_output_embeddings_cpu(num_tokens * embedding_dim);

    // Fill input embeddings with some values (e.g., 0.1, 0.2, ...)
    // And positions with 0, 1, 2, ...
    std::mt19937 gen(0); // Seed for reproducibility
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);
    for (int i = 0; i < num_tokens; ++i) {
        h_positions[i] = i;
        for (int j = 0; j < embedding_dim; ++j) {
            h_input_embeddings[i * embedding_dim + j] = distrib(gen);
        }
    }

    print_embeddings("Host Input Embeddings (First few):", h_input_embeddings, num_tokens, embedding_dim);

    // Allocate device memory
    float* d_input_embeddings;
    float* d_output_embeddings;
    int* d_positions;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input_embeddings, num_tokens * embedding_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_embeddings, num_tokens * embedding_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_positions, num_tokens * sizeof(int)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_embeddings, h_input_embeddings.data(), num_tokens * embedding_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_positions, h_positions.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice));

    // Launch CUDA kernel
    apply_rope_1d_embedding_cuda(d_output_embeddings, d_input_embeddings, d_positions, num_tokens, embedding_dim, base_period);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    // Copy results from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_embeddings_cuda.data(), d_output_embeddings, num_tokens * embedding_dim * sizeof(float), cudaMemcpyDeviceToHost));

    print_embeddings("CUDA Output Embeddings (First few):", h_output_embeddings_cuda, num_tokens, embedding_dim);

    // Run CPU reference implementation
    apply_rope_1d_embedding_cpu(h_output_embeddings_cpu, h_input_embeddings, h_positions, num_tokens, embedding_dim, base_period);
    print_embeddings("CPU Output Embeddings (First few):", h_output_embeddings_cpu, num_tokens, embedding_dim);

    // Compare CUDA and CPU results (simple check for a few values)
    bool match = true;
    float tolerance = 1e-5f;
    for (size_t i = 0; i < h_output_embeddings_cuda.size(); ++i) {
        if (std::abs(h_output_embeddings_cuda[i] - h_output_embeddings_cpu[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": CUDA=" << h_output_embeddings_cuda[i]
                      << ", CPU=" << h_output_embeddings_cpu[i] << std::endl;
            match = false;
            break;
        }
    }
    if (match) {
        std::cout << "CUDA and CPU results match within tolerance." << std::endl;
    } else {
        std::cout << "CUDA and CPU results DO NOT match." << std::endl;
    }

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_input_embeddings));
    CHECK_CUDA_ERROR(cudaFree(d_output_embeddings));
    CHECK_CUDA_ERROR(cudaFree(d_positions));

    std::cout << "\nDemonstration complete." << std::endl;

    return 0;
}
