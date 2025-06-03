#include "gtest/gtest.h"
#include "softplus_activation.cuh" // Contains CHECK_CUDA_ERROR and function declarations
#include <vector>
#include <cmath>     // For std::abs, std::log, std::exp
#include <random>    // For std::random_device, std::mt19937, std::uniform_real_distribution
#include <algorithm> // For std::generate

// Helper to initialize data
void initializeTestData(std::vector<float>& data, int N, float min_val = -5.0f, float max_val = 5.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(min_val, max_val);
    data.resize(N);
    std::generate(data.begin(), data.end(), [&]() { return distrib(gen); });
}

// Test fixture for Softplus Activation tests
class SoftplusActivationTest : public ::testing::Test {
protected:
    const float epsilon = 1e-5f; // Tolerance for floating point comparisons
    const int N = 1024;          // Number of elements for testing
    size_t bytes;

    std::vector<float> h_input;
    std::vector<float> h_output_gpu;
    std::vector<float> h_output_cpu;

    float *d_input = nullptr;
    float *d_output = nullptr;

    void SetUp() override {
        bytes = N * sizeof(float);
        initializeTestData(h_input, N);
        h_output_gpu.resize(N);
        h_output_cpu.resize(N);

        CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, bytes));
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    }

    void TearDown() override {
        if (d_input) CHECK_CUDA_ERROR(cudaFree(d_input));
        if (d_output) CHECK_CUDA_ERROR(cudaFree(d_output));
        d_input = nullptr;
        d_output = nullptr;
    }

    void runGpuKernel() {
        softplusActivation(d_input, d_output, N);
        CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    }

    void runCpuReference() {
        softplusActivationCPU(h_input.data(), h_output_cpu.data(), N);
    }
};

// Test case: Basic functionality and comparison with CPU
TEST_F(SoftplusActivationTest, HandlesMixedValues) {
    runGpuKernel();
    runCpuReference();

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(h_output_cpu[i], h_output_gpu[i], epsilon)
            << "Mismatch at index " << i << " for input " << h_input[i];
    }
}

// Test case: Edge case - all zeros
TEST_F(SoftplusActivationTest, HandlesAllZeros) {
    std::fill(h_input.begin(), h_input.end(), 0.0f);
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    runGpuKernel();
    runCpuReference(); // log(1 + exp(0)) = log(2)

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(std::log(2.0f), h_output_gpu[i], epsilon)
            << "Mismatch for zero input at index " << i;
        EXPECT_NEAR(h_output_cpu[i], h_output_gpu[i], epsilon)
            << "Mismatch between CPU and GPU for zero input at index " << i;
    }
}

// Test case: Edge case - all positive values
TEST_F(SoftplusActivationTest, HandlesAllPositive) {
    initializeTestData(h_input, N, 0.1f, 10.0f); // Strictly positive
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    runGpuKernel();
    runCpuReference();

    for (int i = 0; i < N; ++i) {
        // Softplus(x) is always > 0. For positive x, Softplus(x) > x if x is small, and Softplus(x) approaches x if x is large.
        // A more robust check is simply that it's positive and matches CPU.
        EXPECT_GT(h_output_gpu[i], 0.0f);
        EXPECT_NEAR(h_output_cpu[i], h_output_gpu[i], epsilon)
            << "Mismatch at index " << i << " for positive input " << h_input[i];
    }
}

// Test case: Edge case - all negative values
TEST_F(SoftplusActivationTest, HandlesAllNegative) {
    initializeTestData(h_input, N, -10.0f, -0.1f); // Strictly negative
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    runGpuKernel();
    runCpuReference();

    for (int i = 0; i < N; ++i) {
        EXPECT_GT(h_output_gpu[i], 0.0f)
            << "Softplus output should be greater than 0 for negative inputs";
        EXPECT_NEAR(h_output_cpu[i], h_output_gpu[i], epsilon)
            << "Mismatch at index " << i << " for negative input " << h_input[i];
    }
}

// Test with a single element
TEST_F(SoftplusActivationTest, HandlesSingleElement) {
    const int singleN = 1;
    size_t singleBytes = singleN * sizeof(float);
    std::vector<float> single_h_input = {2.5f};
    std::vector<float> single_h_output_gpu(singleN);
    std::vector<float> single_h_output_cpu(singleN);

    float* single_d_input = nullptr;
    float* single_d_output = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&single_d_input, singleBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&single_d_output, singleBytes));
    CHECK_CUDA_ERROR(cudaMemcpy(single_d_input, single_h_input.data(), singleBytes, cudaMemcpyHostToDevice));

    softplusActivation(single_d_input, single_d_output, singleN);
    CHECK_CUDA_ERROR(cudaMemcpy(single_h_output_gpu.data(), single_d_output, singleBytes, cudaMemcpyDeviceToHost));

    softplusActivationCPU(single_h_input.data(), single_h_output_cpu.data(), singleN);

    EXPECT_NEAR(single_h_output_cpu[0], single_h_output_gpu[0], epsilon);

    CHECK_CUDA_ERROR(cudaFree(single_d_input));
    CHECK_CUDA_ERROR(cudaFree(single_d_output));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
