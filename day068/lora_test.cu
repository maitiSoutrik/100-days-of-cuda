#include "gtest/gtest.h"
#include "lora.cuh"
#include <vector>
#include <cmath>     // For std::abs
#include <numeric>   // For std::iota
#include <algorithm> // For std::generate

// Helper to compare float vectors with tolerance
void ASSERT_VECTORS_NEAR(const std::vector<float>& vec1, const std::vector<float>& vec2, float tolerance) {
    ASSERT_EQ(vec1.size(), vec2.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        ASSERT_NEAR(vec1[i], vec2[i], tolerance) << "Mismatch at index " << i;
    }
}

// Test fixture for LoRA tests
class LoRATest : public ::testing::Test {
protected:
    LoRAParameters params;
    const int d_model = 64; // Smaller dimension for tests
    const int rank = 8;     // Smaller rank for tests
    const float alpha = 0.5f;
    const float tolerance = 1e-4f; // Tolerance for float comparisons

    std::vector<float> h_input;
    float* d_input = nullptr;
    
    std::vector<float> h_output_cpu;
    std::vector<float> h_output_gpu_host; // Host copy of GPU output
    float* d_output_gpu = nullptr;


    void SetUp() override {
        try {
            initializeLoRAParameters(params, d_model, rank, alpha);
        } catch (const std::runtime_error& e) {
            FAIL() << "LoRA Initialization failed in SetUp: " << e.what();
        }

        h_input.resize(d_model);
        // Generate some deterministic input data
        std::iota(h_input.begin(), h_input.end(), 0.1f); // 0.1, 1.1, 2.1, ...
        std::generate(h_input.begin(), h_input.end(), [n = 0.0f]() mutable { n += 0.1f; return n; });


        h_output_cpu.resize(d_model);
        h_output_gpu_host.resize(d_model);

        ASSERT_EQ(cudaMalloc(&d_input, d_model * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_output_gpu, d_model * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_input, h_input.data(), d_model * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    }

    void TearDown() override {
        freeLoRAParameters(params);
        if (d_input) cudaFree(d_input);
        if (d_output_gpu) cudaFree(d_output_gpu);
    }
};

// Test case: Compare CPU and GPU LoRA forward pass results
TEST_F(LoRATest, ForwardPassComparison) {
    // Run CPU version
    ASSERT_NO_THROW(loraForwardCPU(h_input.data(), h_output_cpu.data(), params));

    // Run GPU version
    ASSERT_NO_THROW(loraForwardGPU(d_input, d_output_gpu, params));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess); // Ensure GPU computation is complete
    
    // Copy GPU output back to host
    ASSERT_EQ(cudaMemcpy(h_output_gpu_host.data(), d_output_gpu, d_model * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    // Compare results
    ASSERT_VECTORS_NEAR(h_output_cpu, h_output_gpu_host, tolerance);
}

// Test case: Check if LoRA parameters are initialized (basic check)
TEST_F(LoRATest, ParameterInitialization) {
    // Check if host pointers are not null after initialization
    ASSERT_NE(params.h_A, nullptr);
    ASSERT_NE(params.h_B, nullptr);
    // Check if device pointers are not null after initialization
    ASSERT_NE(params.d_A, nullptr);
    ASSERT_NE(params.d_B, nullptr);

    // Check dimensions
    ASSERT_EQ(params.d_model, d_model);
    ASSERT_EQ(params.rank, rank);
    ASSERT_EQ(params.alpha, alpha);

    // Simple check: copy a small part of A and B from device to host and see if it's not all zeros (A shouldn't be, B should be)
    std::vector<float> test_A_host(rank); // first row of A
    std::vector<float> test_B_host(rank); // first row of B

    if (rank > 0 && d_model > 0) {
         ASSERT_EQ(cudaMemcpy(test_A_host.data(), params.d_A, rank * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
         ASSERT_EQ(cudaMemcpy(test_B_host.data(), params.d_B, rank * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

        bool a_is_nonzero = false;
        for(float val : test_A_host) {
            if (val != 0.0f) {
                a_is_nonzero = true;
                break;
            }
        }
        ASSERT_TRUE(a_is_nonzero) << "Matrix A (d_A) was all zeros after initialization, which is unlikely for random init.";

        bool b_is_zero = true;
        for(float val : test_B_host) {
            if (val != 0.0f) {
                b_is_zero = false;
                break;
            }
        }
        ASSERT_TRUE(b_is_zero) << "Matrix B (d_B) was not all zeros after initialization.";
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
