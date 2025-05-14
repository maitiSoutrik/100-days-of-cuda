#include "gtest/gtest.h"
#include "geglu.cuh" // Should provide CHECK_CUDA_ERROR, launch_geglu_kernel
#include <vector>
#include <cmath>   // For M_PI, fabsf, tanhf, sqrtf
#include <cstdlib> // For rand
#include <iomanip> // For std::fixed, std::setprecision

// Define M_PI_F for CPU side if not already (should match .cu file)
#ifndef M_PI_F
#define M_PI_F ((float)M_PI)
#endif

// CPU GELU approximation (for verification within tests)
float test_cpu_gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI_F) * (x + 0.044715f * x * x * x)));
}

// CPU GEGLU implementation (for verification within tests)
void test_cpu_geglu(const std::vector<float>& input_a, const std::vector<float>& input_b, std::vector<float>& output) {
    ASSERT_EQ(input_a.size(), input_b.size());
    ASSERT_EQ(input_a.size(), output.size());
    for (size_t i = 0; i < input_a.size(); ++i) {
        output[i] = test_cpu_gelu_approx(input_a[i]) * input_b[i];
    }
}

// Helper to compare two float vectors
void compare_vectors(const std::vector<float>& vec1, const std::vector<float>& vec2, float epsilon) {
    ASSERT_EQ(vec1.size(), vec2.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        EXPECT_NEAR(vec1[i], vec2[i], epsilon) << "Mismatch at index " << i;
    }
}

class GegluTest : public ::testing::Test {
protected:
    void run_geglu_test(int n, float epsilon = 1e-5f) {
        if (n == 0) { // Special case for n=0
            std::vector<float> h_input_a_empty, h_input_b_empty, h_output_gpu_empty, h_output_cpu_empty;
            
            float *d_input_a_empty = nullptr, *d_input_b_empty = nullptr, *d_output_empty = nullptr;
            // No allocation or kernel launch needed for n=0
            // launch_geglu_kernel should handle n=0 gracefully.
            launch_geglu_kernel(d_input_a_empty, d_input_b_empty, d_output_empty, 0);
            // No copy back needed.
            // CPU version should also handle empty.
            test_cpu_geglu(h_input_a_empty, h_input_b_empty, h_output_cpu_empty);
            EXPECT_TRUE(h_output_gpu_empty.empty()); // Should remain empty
            EXPECT_TRUE(h_output_cpu_empty.empty()); // Should also be empty
            return;
        }

        std::vector<float> h_input_a(n);
        std::vector<float> h_input_b(n);
        std::vector<float> h_output_gpu(n);
        std::vector<float> h_output_cpu(n);

        for (int i = 0; i < n; ++i) {
            h_input_a[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f)) - 1.0f; // Random between -1 and 1
            h_input_b[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f)) - 1.0f; // Random between -1 and 1
        }

        float *d_input_a, *d_input_b, *d_output;
        CHECK_CUDA_ERROR(cudaMalloc(&d_input_a, n * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_input_b, n * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));

        CHECK_CUDA_ERROR(cudaMemcpy(d_input_a, h_input_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_input_b, h_input_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

        launch_geglu_kernel(d_input_a, d_input_b, d_output, n);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));

        test_cpu_geglu(h_input_a, h_input_b, h_output_cpu);
        compare_vectors(h_output_gpu, h_output_cpu, epsilon);

        CHECK_CUDA_ERROR(cudaFree(d_input_a));
        CHECK_CUDA_ERROR(cudaFree(d_input_b));
        CHECK_CUDA_ERROR(cudaFree(d_output));
    }
};

TEST_F(GegluTest, HandlesZeroSize) {
    run_geglu_test(0);
}

TEST_F(GegluTest, SmallInput) {
    const int n = 5;
    std::vector<float> h_input_a = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f};
    std::vector<float> h_input_b = {1.0f, 0.5f,  2.0f, -1.0f, 0.0f};
    std::vector<float> h_output_gpu(n);
    std::vector<float> h_output_cpu(n);

    float *d_input_a, *d_input_b, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_a, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_b, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input_a, h_input_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_b, h_input_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    launch_geglu_kernel(d_input_a, d_input_b, d_output, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    test_cpu_geglu(h_input_a, h_input_b, h_output_cpu);
    compare_vectors(h_output_gpu, h_output_cpu, 1e-5f);

    // Print for manual check if needed
    // for(int i=0; i<n; ++i) {
    //     std::cout << "A: " << h_input_a[i] << ", B: " << h_input_b[i] 
    //               << ", GELU(A): " << test_cpu_gelu_approx(h_input_a[i])
    //               << ", GPU: " << h_output_gpu[i] << ", CPU: " << h_output_cpu[i] << std::endl;
    // }

    CHECK_CUDA_ERROR(cudaFree(d_input_a));
    CHECK_CUDA_ERROR(cudaFree(d_input_b));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

TEST_F(GegluTest, RandomMediumSizeInput) {
    run_geglu_test(1024);
}

TEST_F(GegluTest, RandomLargeSizeInput) {
    run_geglu_test(65536); // 2^16
}

// It might be useful to test GELU approximation directly if it were a public host function
// For now, it's implicitly tested via GEGLU.

// main function for gtest
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
// The above main is usually not needed if GTest::gtest_main is linked.
