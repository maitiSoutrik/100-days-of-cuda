#include "fisher_matrix.cuh"
#include "gtest/gtest.h"
#include <vector>
#include <numeric> // For std::accumulate
#include <cstdlib> // For rand, RAND_MAX
#include <cmath>   // For fabs
#include <vector>

// Test fixture for Fisher Matrix tests
class FisherMatrixTest : public ::testing::Test {
protected:
    // Using smaller values for unit tests to keep them fast
    const int num_samples_ = 100; 
    const int num_params_ = 8;   // Keep n_params a multiple of blockDim for simplicity in tests if needed
                                 // but the kernel handles non-multiples too.
    
    std::vector<float> h_log_probs_;
    std::vector<float> h_fisher_cpu_;
    std::vector<float> h_fisher_gpu_;

    void SetUp() override {
        h_log_probs_.resize(static_cast<size_t>(num_samples_) * num_params_);
        h_fisher_cpu_.resize(static_cast<size_t>(num_params_) * num_params_);
        h_fisher_gpu_.resize(static_cast<size_t>(num_params_) * num_params_);

        // Initialize random data for log_probs (scores)
        srand(42); // Seed for reproducibility in tests
        for (size_t i = 0; i < static_cast<size_t>(num_samples_) * num_params_; ++i) {
            h_log_probs_[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Scores between -1 and 1
        }
    }
};

// Test the GPU implementation against the CPU implementation
TEST_F(FisherMatrixTest, GpuVsCpuComparison) {
    compute_fisher_matrix_cpu(h_log_probs_.data(), h_fisher_cpu_.data(), 
                              num_samples_, num_params_);
    
    compute_fisher_matrix_gpu(h_log_probs_.data(), h_fisher_gpu_.data(), 
                              num_samples_, num_params_);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    for (size_t i = 0; i < static_cast<size_t>(num_params_) * num_params_; ++i) {
        EXPECT_NEAR(h_fisher_cpu_[i], h_fisher_gpu_[i], 1e-5f)
            << "Mismatch at index " << i << " (row " << i / num_params_ << ", col " << i % num_params_ << ")";
    }
}

// Test with different dimensions
TEST_F(FisherMatrixTest, GpuVsCpuComparisonLargerParams) {
    const int n_s = 50;
    const int n_p = 16; // Larger than blockDim.x to test gridDim calculation
    
    std::vector<float> log_probs(static_cast<size_t>(n_s) * n_p);
    std::vector<float> fisher_cpu(static_cast<size_t>(n_p) * n_p);
    std::vector<float> fisher_gpu(static_cast<size_t>(n_p) * n_p);

    for (size_t i = 0; i < static_cast<size_t>(n_s) * n_p; ++i) {
        log_probs[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    compute_fisher_matrix_cpu(log_probs.data(), fisher_cpu.data(), n_s, n_p);
    compute_fisher_matrix_gpu(log_probs.data(), fisher_gpu.data(), n_s, n_p);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    for (size_t i = 0; i < static_cast<size_t>(n_p) * n_p; ++i) {
        EXPECT_NEAR(fisher_cpu[i], fisher_gpu[i], 1e-5f)
            << "Mismatch at index " << i << " (row " << i / n_p << ", col " << i % n_p << ")";
    }
}


// Entry point for running tests
// The main function for tests is usually not needed if CMake's gtest_discover_tests is used,
// as it links against GTest::gtest_main which provides one.
// However, including it ensures it can be compiled as a standalone test runner if needed.
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
