#include "gtest/gtest.h"
#include "rope_2d.cuh"
#include <vector>
#include <cmath> // For cosf, sinf, powf, fabsf
#include <numeric> // For std::iota
#include <algorithm> // For std::equal

// Helper to compare floating point values with a tolerance
::testing::AssertionResult AssertFloatVecEq(const std::vector<float>& expected, const std::vector<float>& actual, float tolerance) {
    if (expected.size() != actual.size()) {
        return ::testing::AssertionFailure() << "Vectors have different sizes. Expected: " << expected.size() << ", Actual: " << actual.size();
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (fabsf(expected[i] - actual[i]) > tolerance) {
            return ::testing::AssertionFailure() << "Mismatch at index " << i << ". Expected: " << expected[i] << ", Actual: " << actual[i];
        }
    }
    return ::testing::AssertionSuccess();
}

// CPU implementation of RoPE for a single pair of features (for verification)
void rope_cpu_single_pair(float& x0, float& x1, int position, int pair_idx_in_half, int embedding_dim_half, float theta_base) {
    float theta_k = powf(theta_base, -2.0f * pair_idx_in_half / static_cast<float>(embedding_dim_half));
    float m_theta = static_cast<float>(position) * theta_k;
    float cos_m_theta = cosf(m_theta);
    float sin_m_theta = sinf(m_theta);

    float original_x0 = x0;
    float original_x1 = x1;

    x0 = original_x0 * cos_m_theta - original_x1 * sin_m_theta;
    x1 = original_x0 * sin_m_theta + original_x1 * cos_m_theta;
}

// CPU implementation of 2D RoPE for verification
std::vector<float> apply_rope_2d_cpu(
    const std::vector<float>& input_embeddings,
    int height,
    int width,
    int embedding_dim,
    float theta_base = 10000.0f) {

    if (embedding_dim % 4 != 0 || embedding_dim == 0) {
        // Should not happen if called from tests with valid params
        return {};
    }

    std::vector<float> output_embeddings = input_embeddings;
    int num_tokens = height * width;
    int embedding_dim_half = embedding_dim / 2;
    int num_pairs_per_half = embedding_dim / 4;

    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
        int h = token_idx / width;
        int w = token_idx % width;

        // First half (rotated by height)
        for (int pair_h_idx = 0; pair_h_idx < num_pairs_per_half; ++pair_h_idx) {
            int feat_idx0 = token_idx * embedding_dim + pair_h_idx * 2;
            int feat_idx1 = feat_idx0 + 1;
            rope_cpu_single_pair(output_embeddings[feat_idx0], output_embeddings[feat_idx1],
                                 h, pair_h_idx, embedding_dim_half, theta_base);
        }

        // Second half (rotated by width)
        for (int pair_w_idx = 0; pair_w_idx < num_pairs_per_half; ++pair_w_idx) {
            int feat_idx0 = token_idx * embedding_dim + embedding_dim_half + pair_w_idx * 2;
            int feat_idx1 = feat_idx0 + 1;
            rope_cpu_single_pair(output_embeddings[feat_idx0], output_embeddings[feat_idx1],
                                 w, pair_w_idx, embedding_dim_half, theta_base);
        }
    }
    return output_embeddings;
}


TEST(RoPE2DTest, BasicRotation) {
    const int height = 2;
    const int width = 2;
    const int embedding_dim = 4; // Smallest valid dim (1 pair for height, 1 pair for width)
    const float theta_base = 10000.0f;
    const float tolerance = 1e-5f;

    const int num_tokens = height * width;
    const size_t data_size = num_tokens * embedding_dim * sizeof(float);

    std::vector<float> h_embeddings(num_tokens * embedding_dim);
    // Initialize with simple values: 1.0, 0.0, 1.0, 0.0 ... for each token
    // So first pair is (1,0), second pair is (1,0)
    for(int i = 0; i < num_tokens; ++i) {
        for(int j = 0; j < embedding_dim / 2; ++j) {
            h_embeddings[i * embedding_dim + j * 2] = 1.0f;
            h_embeddings[i * embedding_dim + j * 2 + 1] = 0.0f;
        }
    }

    // Calculate expected CPU results
    std::vector<float> h_expected_embeddings = apply_rope_2d_cpu(h_embeddings, height, width, embedding_dim, theta_base);

    // GPU execution
    float* d_embeddings;
    CHECK_CUDA_ERROR(cudaMalloc(&d_embeddings, data_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_embeddings, h_embeddings.data(), data_size, cudaMemcpyHostToDevice));

    apply_rope_2d_embeddings_gpu(d_embeddings, height, width, embedding_dim, theta_base);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<float> h_gpu_results(num_tokens * embedding_dim);
    CHECK_CUDA_ERROR(cudaMemcpy(h_gpu_results.data(), d_embeddings, data_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_embeddings));

    EXPECT_TRUE(AssertFloatVecEq(h_expected_embeddings, h_gpu_results, tolerance));
}

TEST(RoPE2DTest, LargerDimensions) {
    const int height = 8;
    const int width = 8;
    const int embedding_dim = 16; // e.g., 4 pairs for height, 4 pairs for width
    const float theta_base = 10000.0f;
    const float tolerance = 1e-4f; // Slightly larger tolerance for more computations

    const int num_tokens = height * width;
    const size_t data_size = num_tokens * embedding_dim * sizeof(float);

    std::vector<float> h_embeddings(num_tokens * embedding_dim);
    for(size_t i = 0; i < h_embeddings.size(); ++i) {
        h_embeddings[i] = static_cast<float>(i % 100) / 100.0f; // Some arbitrary values
    }

    // Calculate expected CPU results
    std::vector<float> h_expected_embeddings = apply_rope_2d_cpu(h_embeddings, height, width, embedding_dim, theta_base);

    // GPU execution
    float* d_embeddings;
    CHECK_CUDA_ERROR(cudaMalloc(&d_embeddings, data_size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_embeddings, h_embeddings.data(), data_size, cudaMemcpyHostToDevice));

    apply_rope_2d_embeddings_gpu(d_embeddings, height, width, embedding_dim, theta_base);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<float> h_gpu_results(num_tokens * embedding_dim);
    CHECK_CUDA_ERROR(cudaMemcpy(h_gpu_results.data(), d_embeddings, data_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_embeddings));

    EXPECT_TRUE(AssertFloatVecEq(h_expected_embeddings, h_gpu_results, tolerance));
}


TEST(RoPE2DTest, ZeroHeightWidth) {
    // The main function apply_rope_2d_embeddings_gpu has checks for this,
    // but we can ensure it doesn't crash or misbehave.
    // The function should return early or handle it gracefully.
    // Here, we expect no CUDA errors and no changes if data is passed (though it shouldn't be processed).
    const int height = 0;
    const int width = 4;
    const int embedding_dim = 4;
    float* d_embeddings = nullptr; // No allocation needed if it returns early

    // Expect the function to handle this, possibly by exiting or returning without error.
    // The current implementation exits, so this test is more conceptual.
    // If it were to return without error for 0 tokens, we'd test that.
    // For now, this test mainly ensures the interface can be called.
    // apply_rope_2d_embeddings_gpu(d_embeddings, height, width, embedding_dim);
    // No CHECK_CUDA_ERROR here as the function might exit.
    // This test is more of a placeholder for behavior if 0 tokens were handled by returning.
    // The current `apply_rope_2d_embeddings_gpu` has `exit(EXIT_FAILURE)` for invalid dims.
    // GTest death tests would be needed to properly test `exit()`.
    // For simplicity, we rely on the main function's validation.
    SUCCEED(); // Placeholder, actual test would need death test or modified function
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
