#include "gtest/gtest.h"
#include "batched_l2_norm.cuh"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm> // For std::generate

// Helper to compare float values with a tolerance
void EXPECT_FLOAT_VECTORS_NEAR(const std::vector<float>& expected,
                               const std::vector<float>& actual,
                               float tolerance) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(expected[i], actual[i], tolerance) << "Mismatch at index " << i;
    }
}

TEST(BatchedL2NormTest, HandlesEmptyInput) {
    const int num_batches = 0;
    const int vector_dim = 10;
    std::vector<float> h_vectors; // Empty
    std::vector<float> h_gpu_norms; // Empty
    std::vector<float> h_cpu_norms; // Empty

    // GPU
    compute_batched_l2_norm_gpu(h_vectors.data(), h_gpu_norms.data(), num_batches, vector_dim);
    // CPU
    compute_batched_l2_norm_cpu(h_vectors.data(), h_cpu_norms.data(), num_batches, vector_dim);
    
    EXPECT_EQ(h_gpu_norms.size(), 0);
    EXPECT_EQ(h_cpu_norms.size(), 0);
}

TEST(BatchedL2NormTest, HandlesZeroDimension) {
    const int num_batches = 5;
    const int vector_dim = 0; // Zero dimension
    std::vector<float> h_vectors; // Effectively empty vectors
    std::vector<float> h_gpu_norms(num_batches);
    std::vector<float> h_cpu_norms(num_batches);
    std::vector<float> expected_norms(num_batches, 0.0f); // Norm of zero-dim vector is 0

    // GPU
    compute_batched_l2_norm_gpu(nullptr, h_gpu_norms.data(), num_batches, vector_dim); // Pass nullptr for data if dim is 0
    // CPU
    compute_batched_l2_norm_cpu(nullptr, h_cpu_norms.data(), num_batches, vector_dim);

    EXPECT_FLOAT_VECTORS_NEAR(expected_norms, h_gpu_norms, 1e-6f);
    EXPECT_FLOAT_VECTORS_NEAR(expected_norms, h_cpu_norms, 1e-6f);
}


TEST(BatchedL2NormTest, SingleBatchSingleDimension) {
    const int num_batches = 1;
    const int vector_dim = 1;
    std::vector<float> h_vectors = {5.0f};
    std::vector<float> h_gpu_norms(num_batches);
    std::vector<float> h_cpu_norms(num_batches);
    std::vector<float> expected_norms = {5.0f}; // sqrt(5^2) = 5

    compute_batched_l2_norm_gpu(h_vectors.data(), h_gpu_norms.data(), num_batches, vector_dim);
    compute_batched_l2_norm_cpu(h_vectors.data(), h_cpu_norms.data(), num_batches, vector_dim);

    EXPECT_FLOAT_VECTORS_NEAR(expected_norms, h_gpu_norms, 1e-6f);
    EXPECT_FLOAT_VECTORS_NEAR(expected_norms, h_cpu_norms, 1e-6f);
}

TEST(BatchedL2NormTest, MultipleBatchesSmallDimension) {
    const int num_batches = 3;
    const int vector_dim = 2;
    std::vector<float> h_vectors = {
        3.0f, 4.0f,  // Batch 0: sqrt(9+16) = 5
        1.0f, 0.0f,  // Batch 1: sqrt(1+0) = 1
        0.0f, -2.0f  // Batch 2: sqrt(0+4) = 2
    };
    std::vector<float> h_gpu_norms(num_batches);
    std::vector<float> h_cpu_norms(num_batches);
    std::vector<float> expected_norms = {5.0f, 1.0f, 2.0f};

    compute_batched_l2_norm_gpu(h_vectors.data(), h_gpu_norms.data(), num_batches, vector_dim);
    compute_batched_l2_norm_cpu(h_vectors.data(), h_cpu_norms.data(), num_batches, vector_dim);

    EXPECT_FLOAT_VECTORS_NEAR(expected_norms, h_gpu_norms, 1e-6f);
    EXPECT_FLOAT_VECTORS_NEAR(expected_norms, h_cpu_norms, 1e-6f);
}

TEST(BatchedL2NormTest, LargerBatchesAndDimensions) {
    const int num_batches = 64;
    const int vector_dim = 128; // Dimension less than typical block size (256)
    
    std::vector<float> h_vectors(num_batches * vector_dim);
    std::vector<float> h_gpu_norms(num_batches);
    std::vector<float> h_cpu_norms(num_batches);

    // Initialize with some patterned data for predictability if needed, or random
    std::mt19937 rng(67890);
    std::uniform_real_distribution<float> dist(0.1f, 1.5f); // Small positive values
    std::generate(h_vectors.begin(), h_vectors.end(), [&]() { return dist(rng); });

    compute_batched_l2_norm_gpu(h_vectors.data(), h_gpu_norms.data(), num_batches, vector_dim);
    compute_batched_l2_norm_cpu(h_vectors.data(), h_cpu_norms.data(), num_batches, vector_dim);

    // Compare GPU against CPU as source of truth
    EXPECT_FLOAT_VECTORS_NEAR(h_cpu_norms, h_gpu_norms, 1e-4f); // Might need slightly larger tolerance for many ops
}

TEST(BatchedL2NormTest, DimensionLargerThanBlockSize) {
    const int num_batches = 32;
    const int vector_dim = 512; // Dimension larger than typical block size (256)
    
    std::vector<float> h_vectors(num_batches * vector_dim);
    std::vector<float> h_gpu_norms(num_batches);
    std::vector<float> h_cpu_norms(num_batches);

    std::mt19937 rng(13579);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::generate(h_vectors.begin(), h_vectors.end(), [&]() { return dist(rng); });

    compute_batched_l2_norm_gpu(h_vectors.data(), h_gpu_norms.data(), num_batches, vector_dim);
    compute_batched_l2_norm_cpu(h_vectors.data(), h_cpu_norms.data(), num_batches, vector_dim);

    EXPECT_FLOAT_VECTORS_NEAR(h_cpu_norms, h_gpu_norms, 1e-4f);
}

TEST(BatchedL2NormTest, AllZeroVectors) {
    const int num_batches = 4;
    const int vector_dim = 3;
    std::vector<float> h_vectors = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };
    std::vector<float> h_gpu_norms(num_batches);
    std::vector<float> h_cpu_norms(num_batches);
    std::vector<float> expected_norms = {0.0f, 0.0f, 0.0f, 0.0f};

    compute_batched_l2_norm_gpu(h_vectors.data(), h_gpu_norms.data(), num_batches, vector_dim);
    compute_batched_l2_norm_cpu(h_vectors.data(), h_cpu_norms.data(), num_batches, vector_dim);

    EXPECT_FLOAT_VECTORS_NEAR(expected_norms, h_gpu_norms, 1e-6f);
    EXPECT_FLOAT_VECTORS_NEAR(expected_norms, h_cpu_norms, 1e-6f);
}

// Entry point for running the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
