#include "gtest/gtest.h"
#include "mse.cuh"
#include <vector>
#include <cmath>   // For std::fabs
#include <numeric> // For std::iota (if needed for predictable data)
#include <random>  // For generating test data

// Helper to generate predictable or random data for tests
void generate_test_data(std::vector<float>& predictions, std::vector<float>& targets, int N, bool random = false) {
    predictions.resize(N);
    targets.resize(N);

    if (random) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distrib(0.0f, 10.0f);
        for (int i = 0; i < N; ++i) {
            predictions[i] = distrib(gen);
            targets[i] = distrib(gen);
        }
    } else {
        for (int i = 0; i < N; ++i) {
            predictions[i] = static_cast<float>(i + 1);
            targets[i] = static_cast<float>(i + 2); // e.g., target is prediction + 1
        }
    }
}

TEST(MSETest, HandlesZeroElements) {
    std::vector<float> predictions_empty;
    std::vector<float> targets_empty;
    float mse_gpu_res;

    EXPECT_FLOAT_EQ(mse_cpu(predictions_empty.data(), targets_empty.data(), 0), 0.0f);
    
    mse_gpu(predictions_empty.data(), targets_empty.data(), 0, &mse_gpu_res);
    EXPECT_FLOAT_EQ(mse_gpu_res, 0.0f);
}

TEST(MSETest, BasicCalculationCPU) {
    std::vector<float> predictions = {1.0f, 2.0f, 3.0f};
    std::vector<float> targets = {1.0f, 3.0f, 5.0f};
    // Errors: (1-1)^2=0, (2-3)^2=1, (3-5)^2=4
    // Sum of squared errors: 0 + 1 + 4 = 5
    // MSE: 5 / 3 = 1.666666...
    float expected_mse = (0.0f*0.0f + (-1.0f)*(-1.0f) + (-2.0f)*(-2.0f)) / 3.0f;
    EXPECT_FLOAT_EQ(mse_cpu(predictions.data(), targets.data(), 3), expected_mse);
}

TEST(MSETest, BasicCalculationGPU) {
    std::vector<float> predictions = {1.0f, 2.0f, 3.0f};
    std::vector<float> targets = {1.0f, 3.0f, 5.0f};
    float expected_mse = (0.0f*0.0f + (-1.0f)*(-1.0f) + (-2.0f)*(-2.0f)) / 3.0f;
    float mse_gpu_res;
    mse_gpu(predictions.data(), targets.data(), 3, &mse_gpu_res);
    EXPECT_FLOAT_EQ(mse_gpu_res, expected_mse);
}

TEST(MSETest, CPUvsGPUMediumSize) {
    const int N = 1 << 10; // 1024 elements
    std::vector<float> h_predictions;
    std::vector<float> h_targets;
    generate_test_data(h_predictions, h_targets, N, true); // Use random data

    float mse_cpu_result = mse_cpu(h_predictions.data(), h_targets.data(), N);
    float mse_gpu_result;
    mse_gpu(h_predictions.data(), h_targets.data(), N, &mse_gpu_result);

    EXPECT_NEAR(mse_cpu_result, mse_gpu_result, 1e-5f);
}

TEST(MSETest, LargerRandomData) {
    const int N = 1 << 16; // 65536 elements
    std::vector<float> h_predictions;
    std::vector<float> h_targets;
    generate_test_data(h_predictions, h_targets, N, true);

    float mse_cpu_result = mse_cpu(h_predictions.data(), h_targets.data(), N);
    float mse_gpu_result;
    mse_gpu(h_predictions.data(), h_targets.data(), N, &mse_gpu_result);
    
    // It's possible for larger datasets and more complex reductions that
    // floating point precision differences might be slightly larger.
    // Adjust tolerance if necessary, but aim for high precision.
    EXPECT_NEAR(mse_cpu_result, mse_gpu_result, 1e-4f); 
}


// It might be useful to add a test for non-zero but small N, e.g., N=1
TEST(MSETest, SingleElement) {
    std::vector<float> predictions = {5.0f};
    std::vector<float> targets = {2.0f};
    // Error: (5-2)^2 = 3^2 = 9
    // MSE: 9 / 1 = 9
    float expected_mse = ( (5.0f - 2.0f) * (5.0f - 2.0f) ) / 1.0f;
    
    float mse_cpu_res = mse_cpu(predictions.data(), targets.data(), 1);
    EXPECT_FLOAT_EQ(mse_cpu_res, expected_mse);

    float mse_gpu_res;
    mse_gpu(predictions.data(), targets.data(), 1, &mse_gpu_res);
    EXPECT_FLOAT_EQ(mse_gpu_res, expected_mse);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
