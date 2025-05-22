#include "rope_2d.cuh"
#include <iostream>
#include <vector>
#include <iomanip> // For std::fixed and std::setprecision
#include <random>   // For std::mt19937 and std::uniform_real_distribution
#include <stdio.h>

// Helper function to print embeddings for a specific token
void print_token_embeddings(const std::string& title, const float* embeddings, int token_idx, int embedding_dim, int grid_width) {
    int h = token_idx / grid_width;
    int w = token_idx % grid_width;
    std::cout << title << " for token at (" << h << ", " << w << ") [index " << token_idx << "]:\n";
    for (int i = 0; i < embedding_dim; ++i) {
        std::cout << std::fixed << std::setprecision(4) << embeddings[token_idx * embedding_dim + i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Parameters
    const int height = 4;          // Example height of the 2D grid
    const int width = 4;           // Example width of the 2D grid
    const int embedding_dim = 8;   // Example embedding dimension (must be multiple of 4)
    const float theta_base = 10000.0f;

    if (embedding_dim % 4 != 0) {
        std::cerr << "Error: embedding_dim in main must be a multiple of 4. Got " << embedding_dim << std::endl;
        return 1;
    }
    if (embedding_dim == 0) {
         std::cerr << "Error: embedding_dim in main cannot be zero." << std::endl;
        return 1;
    }


    const int num_tokens = height * width;
    const size_t data_size = num_tokens * embedding_dim * sizeof(float);

    // Initialize host embeddings with some patterned data for easier visual inspection
    std::vector<float> h_embeddings(num_tokens * embedding_dim);
    std::mt19937 gen(0); // Standard mersenne_twister_engine seeded with 0
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

    for (int i = 0; i < num_tokens; ++i) {
        for (int j = 0; j < embedding_dim; ++j) {
            // Simple pattern: token_index.feature_index
            // h_embeddings[i * embedding_dim + j] = static_cast<float>(i + (j * 0.1f));
            h_embeddings[i * embedding_dim + j] = distrib(gen);
        }
    }

    // Allocate device memory
    float* d_embeddings;
    CHECK_CUDA_ERROR(cudaMalloc(&d_embeddings, data_size));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_embeddings, h_embeddings.data(), data_size, cudaMemcpyHostToDevice));

    // Print original embeddings for a few tokens
    std::cout << "Original Embeddings (Sample):\n";
    print_token_embeddings("Original", h_embeddings.data(), 0, embedding_dim, width); // Token (0,0)
    if (num_tokens > 1) {
         print_token_embeddings("Original", h_embeddings.data(), width + 1, embedding_dim, width); // Token (1,1) if exists
    }


    // Apply 2D RoPE
    std::cout << "\nApplying 2D RoPE...\n";
    apply_rope_2d_embeddings_gpu(d_embeddings, height, width, embedding_dim, theta_base);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure kernel execution is complete

    // Copy results back to host
    std::vector<float> h_transformed_embeddings(num_tokens * embedding_dim);
    CHECK_CUDA_ERROR(cudaMemcpy(h_transformed_embeddings.data(), d_embeddings, data_size, cudaMemcpyDeviceToHost));

    // Print transformed embeddings for a few tokens
    std::cout << "\nTransformed Embeddings (Sample):\n";
    print_token_embeddings("Transformed", h_transformed_embeddings.data(), 0, embedding_dim, width); // Token (0,0)
     if (num_tokens > 1) {
        print_token_embeddings("Transformed", h_transformed_embeddings.data(), width + 1, embedding_dim, width); // Token (1,1) if exists
    }

    // --- Verification (Optional - Simple check for NaN/Inf) ---
    bool nan_inf_found = false;
    for (int i = 0; i < num_tokens * embedding_dim; ++i) {
        if (std::isnan(h_transformed_embeddings[i]) || std::isinf(h_transformed_embeddings[i])) {
            std::cerr << "Error: NaN or Inf found in transformed_embeddings at index " << i << std::endl;
            nan_inf_found = true;
            break;
        }
    }
    if (!nan_inf_found) {
        std::cout << "\nVerification: No NaN or Inf values found in transformed embeddings." << std::endl;
    }


    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_embeddings));

    std::cout << "\n2D RoPE demonstration finished.\n";
    return 0;
}
