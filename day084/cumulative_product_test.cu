#include "cumulative_product.cuh"
#include "gtest/gtest.h"
#include <vector>
#include <random>
#include <algorithm> // For std::generate, std::copy, std::equal
#include <cmath>   // For std::fabs
#include <limits>  // For std::numeric_limits

// Helper to compare float arrays with a tolerance
::testing::AssertionResult AssertFloatArraysEqual(const float* expected, const float* actual, int size, float epsilon = 1e-5f) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(expected[i] - actual[i]) > epsilon * std::max(1.0f, std::fabs(expected[i]))) {
            if (std::fabs(expected[i] - actual[i]) > epsilon) { // Fallback for very small numbers
                return ::testing::AssertionFailure() << "Mismatch at index " << i
                                                   << ": expected " << expected[i]
                                                   << ", actual " << actual[i];
            }
        }
    }
    return ::testing::AssertionSuccess();
}

TEST(CumulativeProductTest, CPU_EmptyArray) {
    std::vector<float> data;
    inclusive_scan_cpu(data.data(), data.size());
    ASSERT_TRUE(data.empty());
}

TEST(CumulativeProductTest, GPU_EmptyArray) {
    std::vector<float> data_vec;
    float* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, 0 * sizeof(float))); // Allocate 0 bytes
    inclusive_scan_gpu(d_data, 0);
    // No data to copy back or check, just ensure no crash
    CHECK_CUDA_ERROR(cudaFree(d_data));
    SUCCEED(); // If it didn't crash, it's a pass for empty.
}

TEST(CumulativeProductTest, CPU_SingleElement) {
    std::vector<float> data = {5.0f};
    std::vector<float> expected = {5.0f};
    inclusive_scan_cpu(data.data(), data.size());
    EXPECT_EQ(data.size(), expected.size());
    EXPECT_TRUE(AssertFloatArraysEqual(expected.data(), data.data(), data.size()));
}

TEST(CumulativeProductTest, GPU_SingleElement) {
    std::vector<float> h_data = {5.0f};
    std::vector<float> h_expected = {5.0f};
    std::vector<float> h_result(h_data.size());

    float* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, h_data.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    inclusive_scan_gpu(d_data, h_data.size());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_result.data(), d_data, h_result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_data));

    EXPECT_EQ(h_result.size(), h_expected.size());
    EXPECT_TRUE(AssertFloatArraysEqual(h_expected.data(), h_result.data(), h_result.size()));
}


TEST(CumulativeProductTest, CPU_SimpleProduct) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expected = {1.0f, 2.0f, 6.0f, 24.0f};
    inclusive_scan_cpu(data.data(), data.size());
    EXPECT_TRUE(AssertFloatArraysEqual(expected.data(), data.data(), data.size()));
}

TEST(CumulativeProductTest, GPU_SimpleProduct) {
    std::vector<float> h_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_expected = {1.0f, 2.0f, 6.0f, 24.0f};
    std::vector<float> h_result(h_data.size());

    float* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, h_data.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    inclusive_scan_gpu(d_data, h_data.size());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_result.data(), d_data, h_result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_data));

    EXPECT_TRUE(AssertFloatArraysEqual(h_expected.data(), h_result.data(), h_result.size()));
}

TEST(CumulativeProductTest, CPU_WithZeros) {
    std::vector<float> data = {1.0f, 2.0f, 0.0f, 4.0f, 5.0f};
    std::vector<float> expected = {1.0f, 2.0f, 0.0f, 0.0f, 0.0f};
    inclusive_scan_cpu(data.data(), data.size());
    EXPECT_TRUE(AssertFloatArraysEqual(expected.data(), data.data(), data.size()));
}

TEST(CumulativeProductTest, GPU_WithZeros) {
    std::vector<float> h_data = {1.0f, 2.0f, 0.0f, 4.0f, 5.0f};
    std::vector<float> h_expected = {1.0f, 2.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> h_result(h_data.size());

    float* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, h_data.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    inclusive_scan_gpu(d_data, h_data.size());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_result.data(), d_data, h_result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_data));

    EXPECT_TRUE(AssertFloatArraysEqual(h_expected.data(), h_result.data(), h_result.size()));
}

TEST(CumulativeProductTest, CPU_WithNegativeNumbers) {
    std::vector<float> data = {1.0f, -2.0f, 3.0f, -1.0f, 2.0f};
    std::vector<float> expected = {1.0f, -2.0f, -6.0f, 6.0f, 12.0f};
    inclusive_scan_cpu(data.data(), data.size());
    EXPECT_TRUE(AssertFloatArraysEqual(expected.data(), data.data(), data.size()));
}

TEST(CumulativeProductTest, GPU_WithNegativeNumbers) {
    std::vector<float> h_data = {1.0f, -2.0f, 3.0f, -1.0f, 2.0f};
    std::vector<float> h_expected = {1.0f, -2.0f, -6.0f, 6.0f, 12.0f};
    std::vector<float> h_result(h_data.size());

    float* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, h_data.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    inclusive_scan_gpu(d_data, h_data.size());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_result.data(), d_data, h_result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_data));

    EXPECT_TRUE(AssertFloatArraysEqual(h_expected.data(), h_result.data(), h_result.size()));
}


// Test with a larger array that fits within single block limits
TEST(CumulativeProductTest, GPU_LargerArraySingleBlock) {
    int n = 256; // Should be <= 2 * threads_per_block (e.g. 512 for 256 threads)
    std::vector<float> h_input(n);
    std::vector<float> h_cpu_expected(n);
    std::vector<float> h_gpu_result(n);

    std::mt19937 rng(123); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.8f, 1.2f); // Values around 1 to avoid rapid under/overflow
    std::generate(h_input.begin(), h_input.end(), [&]() { return dist(rng); });

    std::copy(h_input.begin(), h_input.end(), h_cpu_expected.begin());
    inclusive_scan_cpu(h_cpu_expected.data(), n); // Compute expected result with CPU

    float* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    inclusive_scan_gpu(d_data, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_gpu_result.data(), d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    
    EXPECT_TRUE(AssertFloatArraysEqual(h_cpu_expected.data(), h_gpu_result.data(), n, 1e-4f)); // Slightly higher epsilon for many products
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
