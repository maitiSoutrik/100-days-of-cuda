#include "gtest/gtest.h"
#include "rope_embedding.cuh"
#include <vector>
#include <random>
#include <algorithm> // For std::equal, std::iota
#include <cmath> // For std::abs

// Helper to compare two float vectors with tolerance
::testing::AssertionResult AreVectorsNear(const std::vector<float>& vec1, const std::vector<float>& vec2, float tolerance) {
    if (vec1.size() != vec2.size()) {
        return ::testing::AssertionFailure() << "Vector sizes differ: " << vec1.size() << " vs " << vec2.size();
    }
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::abs(vec1[i] - vec2[i]) > tolerance) {
            return ::testing::AssertionFailure() << "Mismatch at index " << i << ": " << vec1[i] << " vs " << vec2[i]
                                             << ", diff " << std::abs(vec1[i] - vec2[i]);
        }
    }
    return ::testing::AssertionSuccess();
}

class RoPEEmbeddingTest : public ::testing::Test {
protected:
    int num_tokens;
    int embedding_dim;
    float base_period;

    std::vector<float> h_input_embeddings;
    std::vector<int> h_positions;
    std::vector<float> h_output_embeddings_cuda;
    std::vector<float> h_output_embeddings_cpu;

    float* d_input_embeddings;
    float* d_output_embeddings;
    int* d_positions;

    std::mt19937 gen;
    std::uniform_real_distribution<float> distrib;

    RoPEEmbeddingTest() : gen(123), distrib(0.0f, 1.0f) {} // Seed for reproducibility in tests

    void SetUp(int n_tokens, int emb_dim, float b_period) {
        num_tokens = n_tokens;
        embedding_dim = emb_dim;
        base_period = b_period;

        h_input_embeddings.resize(num_tokens * embedding_dim);
        h_positions.resize(num_tokens);
        h_output_embeddings_cuda.resize(num_tokens * embedding_dim);
        h_output_embeddings_cpu.resize(num_tokens * embedding_dim);

        for (int i = 0; i < num_tokens; ++i) {
            h_positions[i] = i; // Simple sequential positions for testing
            for (int j = 0; j < embedding_dim; ++j) {
                h_input_embeddings[i * embedding_dim + j] = distrib(gen);
            }
        }

        CHECK_CUDA_ERROR(cudaMalloc(&d_input_embeddings, num_tokens * embedding_dim * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output_embeddings, num_tokens * embedding_dim * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_positions, num_tokens * sizeof(int)));

        CHECK_CUDA_ERROR(cudaMemcpy(d_input_embeddings, h_input_embeddings.data(), num_tokens * embedding_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_positions, h_positions.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice));
    }

    void TearDown() override {
        CHECK_CUDA_ERROR(cudaFree(d_input_embeddings));
        CHECK_CUDA_ERROR(cudaFree(d_output_embeddings));
        CHECK_CUDA_ERROR(cudaFree(d_positions));
    }

    void RunTest() {
        apply_rope_1d_embedding_cuda(d_output_embeddings, d_input_embeddings, d_positions, num_tokens, embedding_dim, base_period);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(h_output_embeddings_cuda.data(), d_output_embeddings, num_tokens * embedding_dim * sizeof(float), cudaMemcpyDeviceToHost));

        apply_rope_1d_embedding_cpu(h_output_embeddings_cpu, h_input_embeddings, h_positions, num_tokens, embedding_dim, base_period);

        EXPECT_TRUE(AreVectorsNear(h_output_embeddings_cuda, h_output_embeddings_cpu, 1e-5f));
    }
};

TEST_F(RoPEEmbeddingTest, SmallInput) {
    SetUp(5, 4, 10000.0f); // 5 tokens, 4 dimensions
    RunTest();
}

TEST_F(RoPEEmbeddingTest, MediumInput) {
    SetUp(128, 64, 10000.0f); // 128 tokens, 64 dimensions
    RunTest();
}

TEST_F(RoPEEmbeddingTest, LargerEmbeddingDim) {
    SetUp(32, 256, 10000.0f); // 32 tokens, 256 dimensions
    RunTest();
}

TEST_F(RoPEEmbeddingTest, DifferentBasePeriod) {
    SetUp(10, 8, 5000.0f); // 10 tokens, 8 dimensions, different base period
    RunTest();
}

TEST_F(RoPEEmbeddingTest, NonSequentialPositions) {
    SetUp(4, 8, 10000.0f);
    h_positions = {0, 5, 2, 10}; // Custom positions
    // Re-copy positions to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_positions, h_positions.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice));
    RunTest();
}

// Main function to run tests (needed if not linking with gtest_main)
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
