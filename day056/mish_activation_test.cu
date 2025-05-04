// day056/mish_activation_test.cu
#include <gtest/gtest.h>
#include "mish_activation.cuh" // Include the header with function declarations
#include <vector>
#include <cmath>

// Define a tolerance for floating-point comparisons
const float FLOAT_TOLERANCE = 1e-5;

// Test fixture for GPU tests (optional, but good practice for setup/teardown)
class MishGpuTest : public ::testing::Test {
protected:
    int n = 1024; // Small size for testing
    size_t bytes;
    std::vector<float> h_input;
    std::vector<float> h_output_gpu;
    std::vector<float> h_expected_output;
    float *d_input = nullptr, *d_output = nullptr;

    void SetUp() override {
        bytes = n * sizeof(float);
        h_input.resize(n);
        h_output_gpu.resize(n);
        h_expected_output.resize(n);

        // Initialize input data (e.g., simple range for predictability)
        for (int i = 0; i < n; ++i) {
            h_input[i] = static_cast<float>(i - n / 2) * 0.1f; // Example range
        }

        // Calculate expected output using CPU logic
        mish_cpu(h_input, h_expected_output);

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, bytes));

        // Copy input to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    }

    void TearDown() override {
        // Free device memory
        if (d_input) CHECK_CUDA_ERROR(cudaFree(d_input));
        if (d_output) CHECK_CUDA_ERROR(cudaFree(d_output));
        // Note: checkCuda in TearDown might abort; consider logging instead if needed
    }
};

// Test the basic mish function (host/device)
TEST(MishActivationTest, BasicValues) {
    // Test some known or expected values
    EXPECT_NEAR(mish(0.0f), 0.0f, FLOAT_TOLERANCE);
    // Mish minimum is around x=-1.1924, value ≈-0.30884
    EXPECT_NEAR(mish(-1.1924f), -0.30884f, 1e-4); // Use slightly larger tolerance due to approximation
    // For large positive x, mish(x) approaches x
    EXPECT_NEAR(mish(10.0f), 10.0f, FLOAT_TOLERANCE);
    // For large negative x, mish(x) approaches 0
    EXPECT_NEAR(mish(-10.0f), -0.0004539f, FLOAT_TOLERANCE); // x * tanh(log(1+exp(-10))) approx -10 * tanh(exp(-10)) approx -10 * exp(-10)
}

// Test the GPU kernel wrapper using the fixture
TEST_F(MishGpuTest, KernelExecution) {
    // Run the GPU kernel
    mish_gpu_wrapper(d_input, d_output, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure kernel finishes

    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    for (int i = 0; i < n; ++i) {
        ASSERT_NEAR(h_output_gpu[i], h_expected_output[i], FLOAT_TOLERANCE)
            << "Mismatch at index " << i;
    }
}

// Main function to run the tests (needed when not linking gtest_main)
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
// Note: We will link against gtest_main via CMake, so this main is usually not needed.
