// day086/hard_sigmoid_test.cu
#include "hard_sigmoid.cuh"
#include "common_utils.h" // For error checking macros
#include <gtest/gtest.h>
#include <vector>
#include <cmath> // For std::abs

// Helper function to compute expected Hard Sigmoid on CPU
float hard_sigmoid_cpu_scalar(float x) {
    if (x <= -3.0f) {
        return 0.0f;
    } else if (x >= 3.0f) {
        return 1.0f;
    } else {
        return (x + 3.0f) / 6.0f;
    }
}

TEST(HardSigmoidTest, HandlesNegativeValues) {
    const size_t n = 1, m = 5;
    const size_t total_elements = n * m;
    std::vector<float> h_input = {-10.0f, -5.0f, -3.0f, -2.0f, -0.5f};
    std::vector<float> h_output(total_elements);
    std::vector<float> expected_output(total_elements);

    for(size_t i = 0; i < total_elements; ++i) {
        expected_output[i] = hard_sigmoid_cpu_scalar(h_input[i]);
    }

    hard_sigmoid_solution(h_input.data(), h_output.data(), n, m);
    CHECK_LAST_CUDA_ERROR();

    for (size_t i = 0; i < total_elements; ++i) {
        ASSERT_NEAR(h_output[i], expected_output[i], 1e-5f)
            << "Input: " << h_input[i];
    }
}

TEST(HardSigmoidTest, HandlesPositiveValues) {
    const size_t n = 1, m = 5;
    const size_t total_elements = n * m;
    std::vector<float> h_input = {0.0f, 1.5f, 3.0f, 5.0f, 10.0f};
    std::vector<float> h_output(total_elements);
    std::vector<float> expected_output(total_elements);

    for(size_t i = 0; i < total_elements; ++i) {
        expected_output[i] = hard_sigmoid_cpu_scalar(h_input[i]);
    }

    hard_sigmoid_solution(h_input.data(), h_output.data(), n, m);
    CHECK_LAST_CUDA_ERROR();

    for (size_t i = 0; i < total_elements; ++i) {
        ASSERT_NEAR(h_output[i], expected_output[i], 1e-5f)
            << "Input: " << h_input[i];
    }
}

TEST(HardSigmoidTest, HandlesMixedValues) {
    const size_t n = 2, m = 3;
    const size_t total_elements = n * m;
    std::vector<float> h_input = {-4.0f, -3.0f, 0.0f, 2.0f, 3.0f, 5.0f};
    std::vector<float> h_output(total_elements);
    std::vector<float> expected_output(total_elements);

    for(size_t i = 0; i < total_elements; ++i) {
        expected_output[i] = hard_sigmoid_cpu_scalar(h_input[i]);
    }

    hard_sigmoid_solution(h_input.data(), h_output.data(), n, m);
    CHECK_LAST_CUDA_ERROR();

    for (size_t i = 0; i < total_elements; ++i) {
        ASSERT_NEAR(h_output[i], expected_output[i], 1e-5f)
            << "Input: " << h_input[i];
    }
}

TEST(HardSigmoidTest, HandlesZeroElements) {
    const size_t n = 0, m = 0;
    const size_t total_elements = n * m;
    std::vector<float> h_input(total_elements); // Empty
    std::vector<float> h_output(total_elements); // Empty

    // Should not crash or cause errors
    hard_sigmoid_solution(h_input.data(), h_output.data(), n, m);
    CHECK_LAST_CUDA_ERROR();
    ASSERT_TRUE(h_output.empty());
}

TEST(HardSigmoidTest, HandlesSingleElement) {
    const size_t n = 1, m = 1;
    const size_t total_elements = n * m;
    std::vector<float> h_input = {-1.0f};
    std::vector<float> h_output(total_elements);
    std::vector<float> expected_output = {(-1.0f + 3.0f) / 6.0f};

    hard_sigmoid_solution(h_input.data(), h_output.data(), n, m);
    CHECK_LAST_CUDA_ERROR();

    ASSERT_NEAR(h_output[0], expected_output[0], 1e-5f)
        << "Input: " << h_input[0];
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
