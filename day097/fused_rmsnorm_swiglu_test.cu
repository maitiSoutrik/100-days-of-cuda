#include "fused_rmsnorm_swiglu.cuh"
#include "cuda_utils.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm> // For std::generate, std::min
#include <cmath>     // For fabs

// Helper to initialize data for tests
void initialize_test_data(std::vector<float>& h_data, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    std::generate(h_data.begin(), h_data.end(), [&]() { return dis(gen); });
}

// Test fixture for Fused RMSNorm SwiGLU tests
class FusedRMSNormSwiGLUTest : public ::testing::Test {
protected:
    // Test parameters
    int num_rows = 2;
    int hidden_dim = 128; // Must be even
    int output_dim;

    // Host data
    std::vector<float> h_input;
    std::vector<float> h_weight;
    std::vector<float> h_output_gpu;
    std::vector<float> h_output_cpu;

    // Device data
    float *d_input = nullptr, *d_weight = nullptr, *d_output = nullptr;

    void SetUp() override {
        output_dim = hidden_dim / 2;

        h_input.resize(num_rows * hidden_dim);
        h_weight.resize(hidden_dim);
        h_output_gpu.resize(num_rows * output_dim);
        h_output_cpu.resize(num_rows * output_dim);

        initialize_test_data(h_input);
        initialize_test_data(h_weight, 0.8f, 1.2f); // Weights typically around 1

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_weight, h_weight.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, h_output_gpu.size() * sizeof(float)));

        // Copy to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void TearDown() override {
        if (d_input) CHECK_CUDA_ERROR(cudaFree(d_input));
        if (d_weight) CHECK_CUDA_ERROR(cudaFree(d_weight));
        if (d_output) CHECK_CUDA_ERROR(cudaFree(d_output));
    }

    void RunTest(int test_num_rows, int test_hidden_dim) {
        num_rows = test_num_rows;
        hidden_dim = test_hidden_dim;
        output_dim = hidden_dim / 2;

        h_input.assign(num_rows * hidden_dim, 0.0f);
        h_weight.assign(hidden_dim, 0.0f);
        h_output_gpu.assign(num_rows * output_dim, 0.0f);
        h_output_cpu.assign(num_rows * output_dim, 0.0f);
        
        initialize_test_data(h_input);
        initialize_test_data(h_weight, 0.8f, 1.2f);

        if (d_input) CHECK_CUDA_ERROR(cudaFree(d_input));
        if (d_weight) CHECK_CUDA_ERROR(cudaFree(d_weight));
        if (d_output) CHECK_CUDA_ERROR(cudaFree(d_output));

        CHECK_CUDA_ERROR(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_weight, h_weight.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, h_output_gpu.size() * sizeof(float)));

        CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice));
        
        // Determine block_size for the kernel launch
        int block_size = 256; // Default
        int temp_block_size_calc = 1;
        while(temp_block_size_calc < (hidden_dim / 2) && temp_block_size_calc < 512) temp_block_size_calc *= 2;
        block_size = temp_block_size_calc;
        if (block_size < 32) block_size = 32; // Min warp size
        if (block_size > hidden_dim) { // For reduction part
             int temp_bs = 1;
             while(temp_bs * 2 <= hidden_dim && temp_bs < 512) temp_bs *=2;
             block_size = std::max(32, temp_bs); // Ensure it's at least warp size
        }
        block_size = std::min(block_size, 512); // Cap block size

        // Launch GPU kernel
        launch_fused_rmsnorm_swiglu(d_output, d_input, d_weight, num_rows, hidden_dim, block_size);
        CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, h_output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // CPU computation
        fused_rmsnorm_swiglu_cpu(h_output_cpu.data(), h_input.data(), h_weight.data(), num_rows, hidden_dim);

        // Compare results
        ASSERT_EQ(h_output_gpu.size(), h_output_cpu.size());
        for (size_t i = 0; i < h_output_gpu.size(); ++i) {
            EXPECT_NEAR(h_output_gpu[i], h_output_cpu[i], 1e-3f)
                << "Mismatch at index " << i
                << "; Row: " << (i / output_dim)
                << ", Col: " << (i % output_dim);
        }
    }
};

TEST_F(FusedRMSNormSwiGLUTest, SmallInput) {
    RunTest(2, 64); // num_rows, hidden_dim
}

TEST_F(FusedRMSNormSwiGLUTest, MediumInput) {
    RunTest(4, 128);
}

TEST_F(FusedRMSNormSwiGLUTest, LargerHiddenDim) {
    RunTest(2, 256);
}

TEST_F(FusedRMSNormSwiGLUTest, HiddenDimEqualsBlockSize) {
    // Test case where hidden_dim/2 might be equal to a typical block size like 128 or 256
    RunTest(1, 512); // hidden_dim = 512, so hidden_dim/2 = 256
}

TEST_F(FusedRMSNormSwiGLUTest, MinimalHiddenDim) {
    RunTest(10, 4); // Smallest practical hidden_dim (e.g. 2 for x, 2 for gate)
}

TEST_F(FusedRMSNormSwiGLUTest, OddNumRows) {
    RunTest(3, 128);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
