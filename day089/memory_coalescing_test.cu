#include "gtest/gtest.h"
#include "memory_coalescing.cuh" // For kernel declarations and CHECK_CUDA_ERROR
#include <vector>
#include <numeric>
#include <algorithm>

// Helper to initialize data and expected results for tests
void setup_test_data(std::vector<float>& h_input, 
                     std::vector<float>& h_expected_coalesced,
                     std::vector<float>& h_expected_uncoalesced_permuted,
                     int n, float scalar, int stride_factor) {
    h_input.resize(n);
    h_expected_coalesced.resize(n);
    h_expected_uncoalesced_permuted.resize(n);

    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i % 50 + i * 0.1f); // Slightly more varied pattern
        h_expected_coalesced[i] = h_input[i] * scalar;
    }

    int elements_per_stride_group = n / stride_factor;
    if (elements_per_stride_group == 0 && n > 0) elements_per_stride_group = 1;

    // Initialize with a distinct value to catch unwritten elements
    std::fill(h_expected_uncoalesced_permuted.begin(), h_expected_uncoalesced_permuted.end(), -9999.0f); 

    for (int global_thread_idx = 0; global_thread_idx < n; ++global_thread_idx) {
        if (elements_per_stride_group > 0) {
            int group_idx = global_thread_idx / elements_per_stride_group;
            int idx_in_group = global_thread_idx % elements_per_stride_group;
            if (group_idx < stride_factor) {
                int uncoalesced_idx = idx_in_group * stride_factor + group_idx;
                if (uncoalesced_idx < n) {
                    h_expected_uncoalesced_permuted[uncoalesced_idx] = h_input[uncoalesced_idx] * scalar;
                }
            }
        }
    }
}

// Test fixture for memory coalescing tests
class MemoryCoalescingTest : public ::testing::Test {
protected:
    const int n_small = 1024 * 8; // 8K elements for tests
    const float scalar = 3.0f;
    const int stride_factor_test = 16; // Smaller stride for tests, but still >1

    std::vector<float> h_input;
    std::vector<float> h_output;
    std::vector<float> h_expected_coalesced;
    std::vector<float> h_expected_uncoalesced_permuted;

    float *d_input, *d_output;

    void SetUp() override {
        setup_test_data(h_input, h_expected_coalesced, h_expected_uncoalesced_permuted, 
                        n_small, scalar, stride_factor_test);
        h_output.resize(n_small);

        CHECK_CUDA_ERROR(cudaMalloc(&d_input, n_small * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, n_small * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), n_small * sizeof(float), cudaMemcpyHostToDevice));
    }

    void TearDown() override {
        CHECK_CUDA_ERROR(cudaFree(d_input));
        CHECK_CUDA_ERROR(cudaFree(d_output));
    }

    void verify(const std::vector<float>& expected) {
        const float epsilon = 1e-4f; // Adjusted epsilon for float comparisons
        for (int i = 0; i < n_small; ++i) {
            ASSERT_NEAR(h_output[i], expected[i], epsilon) << "Mismatch at index " << i;
        }
    }
};

TEST_F(MemoryCoalescingTest, CoalescedKernelCorrectness) {
    int threads_per_block = 128;
    int blocks_per_grid = (n_small + threads_per_block - 1) / threads_per_block;

    coalesced_access_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, n_small, scalar);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure kernel completion
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, n_small * sizeof(float), cudaMemcpyDeviceToHost));
    
    verify(h_expected_coalesced);
}

TEST_F(MemoryCoalescingTest, UncoalescedKernelCorrectness) {
    int threads_per_block = 128;
    int blocks_per_grid = (n_small + threads_per_block - 1) / threads_per_block;

    CHECK_CUDA_ERROR(cudaMemset(d_output, 0, n_small * sizeof(float))); // Clear output buffer
    uncoalesced_access_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, n_small, scalar, stride_factor_test);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure kernel completion
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, n_small * sizeof(float), cudaMemcpyDeviceToHost));

    verify(h_expected_uncoalesced_permuted);
}

// Main function to run tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
