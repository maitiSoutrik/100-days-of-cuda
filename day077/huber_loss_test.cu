#include "huber_loss.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath> // For std::fabs

// Helper to compare two float vectors with a tolerance
void EXPECT_VECTORS_NEAR(const std::vector<float>& vec1, const std::vector<float>& vec2, float tolerance) {
    ASSERT_EQ(vec1.size(), vec2.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        EXPECT_NEAR(vec1[i], vec2[i], tolerance) << "Mismatch at index " << i;
    }
}

class HuberLossTest : public ::testing::Test {
protected:
    std::vector<float> predictions;
    std::vector<float> targets;
    std::vector<float> loss_cpu;
    std::vector<float> loss_gpu;
    std::vector<float> gradients_cpu;
    std::vector<float> gradients_gpu;
    int n;
    float delta;

    void SetUp() override {
        n = 1024; // Test with a smaller size for unit tests
        delta = 1.0f;

        predictions.resize(n);
        targets.resize(n);
        loss_cpu.resize(n);
        loss_gpu.resize(n);
        gradients_cpu.resize(n);
        gradients_gpu.resize(n);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distrib_val(-5.0f, 5.0f);
        
        for (int i = 0; i < n; ++i) {
            predictions[i] = distrib_val(gen);
            targets[i] = distrib_val(gen);
        }
        // Add specific test cases for edges
        if (n >= 4) {
            // Case 1: error == delta
            predictions[0] = 2.0f; targets[0] = 1.0f; // error = 1.0f
            // Case 2: error == -delta
            predictions[1] = 1.0f; targets[1] = 2.0f; // error = -1.0f
            // Case 3: error < delta
            predictions[2] = 1.5f; targets[2] = 1.0f; // error = 0.5f
            // Case 4: error > delta
            predictions[3] = 3.0f; targets[3] = 1.0f; // error = 2.0f
        }
    }
};

TEST_F(HuberLossTest, CPULossCalculation) {
    huber_loss_cpu(predictions.data(), targets.data(), loss_cpu.data(), n, delta);
    // Test specific values based on setup
    if (n >= 4) {
        // Case 1: error = 1.0f, |error| = delta. Loss = 0.5 * 1.0^2 = 0.5
        EXPECT_NEAR(loss_cpu[0], 0.5f * 1.0f * 1.0f, 1e-5);
        // Case 2: error = -1.0f, |error| = delta. Loss = 0.5 * (-1.0)^2 = 0.5
        EXPECT_NEAR(loss_cpu[1], 0.5f * (-1.0f) * (-1.0f), 1e-5);
        // Case 3: error = 0.5f, |error| < delta. Loss = 0.5 * 0.5^2 = 0.125
        EXPECT_NEAR(loss_cpu[2], 0.5f * 0.5f * 0.5f, 1e-5);
        // Case 4: error = 2.0f, |error| > delta. Loss = delta * (|2.0| - 0.5*delta) = 1.0 * (2.0 - 0.5*1.0) = 1.5
        EXPECT_NEAR(loss_cpu[3], delta * (2.0f - 0.5f * delta), 1e-5);
    }
}

TEST_F(HuberLossTest, CPUDerivativeCalculation) {
    huber_loss_derivative_cpu(predictions.data(), targets.data(), gradients_cpu.data(), n, delta);
    if (n >= 4) {
        // Case 1: error = 1.0f, |error| = delta. Grad = error = 1.0
        EXPECT_NEAR(gradients_cpu[0], 1.0f, 1e-5);
        // Case 2: error = -1.0f, |error| = delta. Grad = error = -1.0
        EXPECT_NEAR(gradients_cpu[1], -1.0f, 1e-5);
        // Case 3: error = 0.5f, |error| < delta. Grad = error = 0.5
        EXPECT_NEAR(gradients_cpu[2], 0.5f, 1e-5);
        // Case 4: error = 2.0f, |error| > delta. Grad = delta * sign(error) = 1.0 * 1 = 1.0
        EXPECT_NEAR(gradients_cpu[3], delta * 1.0f, 1e-5);
    }
}

TEST_F(HuberLossTest, GPULossMatchesCPU) {
    huber_loss_cpu(predictions.data(), targets.data(), loss_cpu.data(), n, delta);
    compute_huber_loss_gpu(predictions.data(), targets.data(), loss_gpu.data(), n, delta);
    EXPECT_VECTORS_NEAR(loss_cpu, loss_gpu, 1e-5f);
}

TEST_F(HuberLossTest, GPUDerivativeMatchesCPU) {
    huber_loss_derivative_cpu(predictions.data(), targets.data(), gradients_cpu.data(), n, delta);
    compute_huber_loss_derivative_gpu(predictions.data(), targets.data(), gradients_gpu.data(), n, delta);
    EXPECT_VECTORS_NEAR(gradients_cpu, gradients_gpu, 1e-5f);
}

// Test with a different delta
class HuberLossTestDifferentDelta : public HuberLossTest {
protected:
    void SetUp() override {
        HuberLossTest::SetUp(); // Call base SetUp
        delta = 0.5f; // Change delta for this test suite
    }
};

TEST_F(HuberLossTestDifferentDelta, CPULossCalculation) {
    huber_loss_cpu(predictions.data(), targets.data(), loss_cpu.data(), n, delta);
    if (n >= 4) {
        // Case 1: error = 1.0f, |error| > delta (0.5). Loss = 0.5 * (1.0 - 0.5*0.5) = 0.5 * 0.75 = 0.375
        EXPECT_NEAR(loss_cpu[0], delta * (1.0f - 0.5f * delta), 1e-5);
        // Case 2: error = -1.0f, |error| > delta (0.5). Loss = 0.5 * (1.0 - 0.5*0.5) = 0.375
        EXPECT_NEAR(loss_cpu[1], delta * (1.0f - 0.5f * delta), 1e-5);
        // Case 3: error = 0.5f, |error| = delta (0.5). Loss = 0.5 * 0.5^2 = 0.125
        EXPECT_NEAR(loss_cpu[2], 0.5f * 0.5f * 0.5f, 1e-5);
        // Case 4: error = 2.0f, |error| > delta (0.5). Loss = 0.5 * (2.0 - 0.5*0.5) = 0.5 * 1.75 = 0.875
        EXPECT_NEAR(loss_cpu[3], delta * (2.0f - 0.5f * delta), 1e-5);
    }
}

TEST_F(HuberLossTestDifferentDelta, CPUDerivativeCalculation) {
    huber_loss_derivative_cpu(predictions.data(), targets.data(), gradients_cpu.data(), n, delta);
     if (n >= 4) {
        // Case 1: error = 1.0f, |error| > delta. Grad = delta * sign(error) = 0.5 * 1 = 0.5
        EXPECT_NEAR(gradients_cpu[0], delta * 1.0f, 1e-5);
        // Case 2: error = -1.0f, |error| > delta. Grad = delta * sign(error) = 0.5 * -1 = -0.5
        EXPECT_NEAR(gradients_cpu[1], delta * -1.0f, 1e-5);
        // Case 3: error = 0.5f, |error| = delta. Grad = error = 0.5
        EXPECT_NEAR(gradients_cpu[2], 0.5f, 1e-5);
        // Case 4: error = 2.0f, |error| > delta. Grad = delta * sign(error) = 0.5 * 1 = 0.5
        EXPECT_NEAR(gradients_cpu[3], delta * 1.0f, 1e-5);
    }
}

TEST_F(HuberLossTestDifferentDelta, GPULossMatchesCPU) {
    huber_loss_cpu(predictions.data(), targets.data(), loss_cpu.data(), n, delta);
    compute_huber_loss_gpu(predictions.data(), targets.data(), loss_gpu.data(), n, delta);
    EXPECT_VECTORS_NEAR(loss_cpu, loss_gpu, 1e-5f);
}

TEST_F(HuberLossTestDifferentDelta, GPUDerivativeMatchesCPU) {
    huber_loss_derivative_cpu(predictions.data(), targets.data(), gradients_cpu.data(), n, delta);
    compute_huber_loss_derivative_gpu(predictions.data(), targets.data(), gradients_gpu.data(), n, delta);
    EXPECT_VECTORS_NEAR(gradients_cpu, gradients_gpu, 1e-5f);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
