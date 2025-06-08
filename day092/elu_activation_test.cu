#include "gtest/gtest.h"
#include "elu_activation.cuh" // Includes CHECK_CUDA_ERROR, CPU and GPU function declarations
#include <vector>
#include <cmath>     // For expf, std::abs
#include <algorithm> // For std::generate
#include <random>    // For std::mt19937, std::uniform_real_distribution
#include <chrono>    // For std::chrono::steady_clock

const float DEFAULT_ALPHA = 1.0f;
const float TEST_TOLERANCE = 1e-5f;

// Helper function to compare two float vectors with a tolerance
::testing::AssertionResult AreVectorsNear(const std::vector<float>& vec1, const std::vector<float>& vec2, float tolerance) {
    if (vec1.size() != vec2.size()) {
        return ::testing::AssertionFailure() << "Vector sizes differ. Expected: " << vec1.size()
                                             << ", Actual: " << vec2.size();
    }
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::abs(vec1[i] - vec2[i]) > tolerance) {
            return ::testing::AssertionFailure() << "Mismatch at index " << i << ". Expected: " << vec1[i]
                                                 << ", Actual: " << vec2[i] << ", Diff: " << std::abs(vec1[i] - vec2[i]);
        }
    }
    return ::testing::AssertionSuccess();
}

// Test fixture for ELU tests
class ELUActivationTest : public ::testing::Test {
protected:
    std::vector<float> input_data;
    std::vector<float> output_cpu;
    std::vector<float> output_gpu;
    float alpha = DEFAULT_ALPHA;

    void SetUpData(const std::vector<float>& data, float test_alpha = DEFAULT_ALPHA) {
        input_data = data;
        alpha = test_alpha;
        output_cpu.resize(input_data.size());
        output_gpu.resize(input_data.size());
    }
};

TEST_F(ELUActivationTest, CPU_PositiveValues) {
    SetUpData({1.0f, 0.5f, 10.0f, 100.0f});
    elu_activation_cpu(input_data.data(), output_cpu.data(), input_data.size(), alpha);
    EXPECT_TRUE(AreVectorsNear(input_data, output_cpu, TEST_TOLERANCE)); // For positive x, ELU(x) = x
}

TEST_F(ELUActivationTest, CPU_NegativeValues) {
    SetUpData({-1.0f, -0.5f, -2.0f}, 1.0f);
    std::vector<float> expected_output = {
        1.0f * (expf(-1.0f) - 1.0f),
        1.0f * (expf(-0.5f) - 1.0f),
        1.0f * (expf(-2.0f) - 1.0f)
    };
    elu_activation_cpu(input_data.data(), output_cpu.data(), input_data.size(), alpha);
    EXPECT_TRUE(AreVectorsNear(expected_output, output_cpu, TEST_TOLERANCE));
}

TEST_F(ELUActivationTest, CPU_ZeroValue) {
    SetUpData({0.0f}, 1.0f);
    // ELU(0) = alpha * (exp(0) - 1) = alpha * (1 - 1) = 0
    std::vector<float> expected_output = {0.0f};
    elu_activation_cpu(input_data.data(), output_cpu.data(), input_data.size(), alpha);
    EXPECT_TRUE(AreVectorsNear(expected_output, output_cpu, TEST_TOLERANCE));
}

TEST_F(ELUActivationTest, CPU_MixedValues_AlphaOne) {
    SetUpData({1.5f, -2.0f, 0.0f, -0.8f, 3.0f}, 1.0f);
    std::vector<float> expected_output = {
        1.5f,
        1.0f * (expf(-2.0f) - 1.0f),
        0.0f, // alpha * (expf(0.0f) - 1.0f) = 0
        1.0f * (expf(-0.8f) - 1.0f),
        3.0f
    };
    elu_activation_cpu(input_data.data(), output_cpu.data(), input_data.size(), alpha);
    EXPECT_TRUE(AreVectorsNear(expected_output, output_cpu, TEST_TOLERANCE));
}

TEST_F(ELUActivationTest, CPU_MixedValues_AlphaHalf) {
    SetUpData({1.5f, -2.0f, 0.0f, -0.8f, 3.0f}, 0.5f);
    std::vector<float> expected_output = {
        1.5f,
        0.5f * (expf(-2.0f) - 1.0f),
        0.0f, // alpha * (expf(0.0f) - 1.0f) = 0
        0.5f * (expf(-0.8f) - 1.0f),
        3.0f
    };
    elu_activation_cpu(input_data.data(), output_cpu.data(), input_data.size(), alpha);
    EXPECT_TRUE(AreVectorsNear(expected_output, output_cpu, TEST_TOLERANCE));
}

TEST_F(ELUActivationTest, GPU_vs_CPU_SmallRandomData) {
    const int N = 256;
    std::vector<float> random_data(N);
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::generate(random_data.begin(), random_data.end(), [&]() { return dist(rng); });

    SetUpData(random_data, 1.0f);
    elu_activation_cpu(input_data.data(), output_cpu.data(), N, alpha);
    elu_activation_gpu(input_data.data(), output_gpu.data(), N, alpha);
    EXPECT_TRUE(AreVectorsNear(output_cpu, output_gpu, TEST_TOLERANCE));
}

TEST_F(ELUActivationTest, GPU_vs_CPU_LargeRandomData_AlphaVaries) {
    const int N = 1024 * 16;
    std::vector<float> random_data(N);
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::generate(random_data.begin(), random_data.end(), [&]() { return dist(rng); });
    
    float test_alpha = 0.75f;
    SetUpData(random_data, test_alpha);
    elu_activation_cpu(input_data.data(), output_cpu.data(), N, alpha);
    elu_activation_gpu(input_data.data(), output_gpu.data(), N, alpha);
    EXPECT_TRUE(AreVectorsNear(output_cpu, output_gpu, TEST_TOLERANCE));
}

// Entry point for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
