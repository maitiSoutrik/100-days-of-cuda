#include "rms_norm.cuh"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>

class RMSNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random seed for reproducible tests
        srand(42);
    }
    
    void TearDown() override {
        // Clean up any CUDA memory if needed
        cudaDeviceReset();
    }
    
    // Helper function to create test data
    void createTestData(std::vector<float>& input, std::vector<float>& gamma, 
                       int batch_size, int seq_len, int hidden_dim) {
        int total_elements = batch_size * seq_len * hidden_dim;
        input.resize(total_elements);
        gamma.resize(hidden_dim);
        
        // Initialize with known values for predictable testing
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (int i = 0; i < total_elements; i++) {
            input[i] = dist(gen);
        }
        
        for (int i = 0; i < hidden_dim; i++) {
            gamma[i] = 1.0f + 0.1f * dist(gen);
        }
    }
    
    // Helper function to compare floating point arrays
    bool compareArrays(const float* a, const float* b, int size, float tolerance = 1e-4f) {
        for (int i = 0; i < size; i++) {
            if (std::abs(a[i] - b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

// Test basic RMS normalization functionality
TEST_F(RMSNormTest, BasicFunctionality) {
    const int batch_size = 2;
    const int seq_len = 3;
    const int hidden_dim = 4;
    const int total_elements = batch_size * seq_len * hidden_dim;
    
    std::vector<float> input, gamma;
    createTestData(input, gamma, batch_size, seq_len, hidden_dim);
    
    std::vector<float> output_cpu(total_elements);
    std::vector<float> output_gpu(total_elements);
    
    // Run CPU version
    rms_norm_cpu(input.data(), output_cpu.data(), gamma.data(), 
                 batch_size, seq_len, hidden_dim);
    
    // Run GPU version
    float *d_input, *d_gamma, *d_output;
    ASSERT_EQ(cudaMalloc(&d_input, total_elements * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_gamma, hidden_dim * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, total_elements * sizeof(float)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(d_input, input.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gamma, gamma.data(), hidden_dim * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    
    rms_norm_gpu(d_input, d_output, d_gamma, batch_size, seq_len, hidden_dim);
    
    ASSERT_EQ(cudaMemcpy(output_gpu.data(), d_output, total_elements * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // Compare results
    EXPECT_TRUE(compareArrays(output_cpu.data(), output_gpu.data(), total_elements));
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_output);
}

// Test with simple known values
TEST_F(RMSNormTest, KnownValues) {
    const int batch_size = 1;
    const int seq_len = 1;
    const int hidden_dim = 4;
    
    // Simple test case: [1, 2, 3, 4]
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> gamma = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> output(hidden_dim);
    
    rms_norm_cpu(input.data(), output.data(), gamma.data(), 
                 batch_size, seq_len, hidden_dim);
    
    // Calculate expected values manually
    float sum_squares = 1.0f + 4.0f + 9.0f + 16.0f; // 30
    float mean_square = sum_squares / 4.0f; // 7.5
    float rms_norm_factor = 1.0f / sqrtf(mean_square + EPSILON);
    
    std::vector<float> expected = {
        1.0f * rms_norm_factor,
        2.0f * rms_norm_factor,
        3.0f * rms_norm_factor,
        4.0f * rms_norm_factor
    };
    
    for (int i = 0; i < hidden_dim; i++) {
        EXPECT_NEAR(output[i], expected[i], 1e-6f);
    }
}

// Test edge cases
TEST_F(RMSNormTest, EdgeCases) {
    const int batch_size = 1;
    const int seq_len = 1;
    const int hidden_dim = 3;
    
    // Test with zeros
    std::vector<float> input_zeros = {0.0f, 0.0f, 0.0f};
    std::vector<float> gamma = {1.0f, 1.0f, 1.0f};
    std::vector<float> output(hidden_dim);
    
    rms_norm_cpu(input_zeros.data(), output.data(), gamma.data(), 
                 batch_size, seq_len, hidden_dim);
    
    // With all zeros, output should be zeros (due to epsilon preventing division by zero)
    for (int i = 0; i < hidden_dim; i++) {
        EXPECT_NEAR(output[i], 0.0f, 1e-6f);
    }
    
    // Test with very small values
    std::vector<float> input_small = {1e-8f, 2e-8f, 3e-8f};
    rms_norm_cpu(input_small.data(), output.data(), gamma.data(), 
                 batch_size, seq_len, hidden_dim);
    
    // Should not produce NaN or infinity
    for (int i = 0; i < hidden_dim; i++) {
        EXPECT_TRUE(std::isfinite(output[i]));
    }
}

// Test different tensor shapes
TEST_F(RMSNormTest, DifferentShapes) {
    struct TestShape {
        int batch_size;
        int seq_len;
        int hidden_dim;
    };
    
    std::vector<TestShape> shapes = {
        {1, 1, 8},      // Single element
        {1, 10, 16},    // Single batch, multiple sequences
        {4, 5, 32},     // Multiple batches
        {8, 128, 512},  // Typical transformer size
        {1, 1, 1024}    // Large hidden dimension
    };
    
    for (const auto& shape : shapes) {
        std::vector<float> input, gamma;
        createTestData(input, gamma, shape.batch_size, shape.seq_len, shape.hidden_dim);
        
        int total_elements = shape.batch_size * shape.seq_len * shape.hidden_dim;
        std::vector<float> output_cpu(total_elements);
        std::vector<float> output_gpu(total_elements);
        
        // CPU version
        rms_norm_cpu(input.data(), output_cpu.data(), gamma.data(), 
                     shape.batch_size, shape.seq_len, shape.hidden_dim);
        
        // GPU version
        float *d_input, *d_gamma, *d_output;
        ASSERT_EQ(cudaMalloc(&d_input, total_elements * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_gamma, shape.hidden_dim * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_output, total_elements * sizeof(float)), cudaSuccess);
        
        ASSERT_EQ(cudaMemcpy(d_input, input.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_gamma, gamma.data(), shape.hidden_dim * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        
        rms_norm_gpu(d_input, d_output, d_gamma, shape.batch_size, shape.seq_len, shape.hidden_dim);
        
        ASSERT_EQ(cudaMemcpy(output_gpu.data(), d_output, total_elements * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        
        // Compare results
        EXPECT_TRUE(compareArrays(output_cpu.data(), output_gpu.data(), total_elements))
            << "Failed for shape: " << shape.batch_size << "x" << shape.seq_len << "x" << shape.hidden_dim;
        
        // Cleanup
        cudaFree(d_input);
        cudaFree(d_gamma);
        cudaFree(d_output);
    }
}

// Test gamma scaling parameter
TEST_F(RMSNormTest, GammaScaling) {
    const int batch_size = 1;
    const int seq_len = 1;
    const int hidden_dim = 4;
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> gamma1 = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> gamma2 = {2.0f, 0.5f, 3.0f, 0.1f};
    
    std::vector<float> output1(hidden_dim);
    std::vector<float> output2(hidden_dim);
    
    rms_norm_cpu(input.data(), output1.data(), gamma1.data(), 
                 batch_size, seq_len, hidden_dim);
    rms_norm_cpu(input.data(), output2.data(), gamma2.data(), 
                 batch_size, seq_len, hidden_dim);
    
    // Check that gamma scaling works correctly
    for (int i = 0; i < hidden_dim; i++) {
        float expected = output1[i] * gamma2[i] / gamma1[i];
        EXPECT_NEAR(output2[i], expected, 1e-6f);
    }
}

// Test numerical stability
TEST_F(RMSNormTest, NumericalStability) {
    const int batch_size = 1;
    const int seq_len = 1;
    const int hidden_dim = 4;
    
    // Test with very large values
    std::vector<float> input_large = {1e6f, 2e6f, 3e6f, 4e6f};
    std::vector<float> gamma = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> output(hidden_dim);
    
    rms_norm_cpu(input_large.data(), output.data(), gamma.data(), 
                 batch_size, seq_len, hidden_dim);
    
    // Should not produce NaN or infinity
    for (int i = 0; i < hidden_dim; i++) {
        EXPECT_TRUE(std::isfinite(output[i]));
        EXPECT_FALSE(std::isnan(output[i]));
    }
    
    // Test with very small values
    std::vector<float> input_small = {1e-6f, 2e-6f, 3e-6f, 4e-6f};
    rms_norm_cpu(input_small.data(), output.data(), gamma.data(), 
                 batch_size, seq_len, hidden_dim);
    
    for (int i = 0; i < hidden_dim; i++) {
        EXPECT_TRUE(std::isfinite(output[i]));
        EXPECT_FALSE(std::isnan(output[i]));
    }
}

// Test RMS vs Layer Norm properties
TEST_F(RMSNormTest, CompareWithLayerNorm) {
    const int batch_size = 2;
    const int seq_len = 3;
    const int hidden_dim = 8;
    const int total_elements = batch_size * seq_len * hidden_dim;
    
    std::vector<float> input, gamma;
    createTestData(input, gamma, batch_size, seq_len, hidden_dim);
    
    std::vector<float> beta(hidden_dim, 0.0f); // Zero bias for fair comparison
    std::vector<float> output_rms(total_elements);
    std::vector<float> output_layer(total_elements);
    
    rms_norm_cpu(input.data(), output_rms.data(), gamma.data(), 
                 batch_size, seq_len, hidden_dim);
    layer_norm_cpu(input.data(), output_layer.data(), gamma.data(), beta.data(),
                   batch_size, seq_len, hidden_dim);
    
    // RMS norm and layer norm should be different (RMS doesn't center)
    bool are_different = false;
    for (int i = 0; i < total_elements; i++) {
        if (std::abs(output_rms[i] - output_layer[i]) > 1e-6f) {
            are_different = true;
            break;
        }
    }
    EXPECT_TRUE(are_different);
    
    // But both should produce finite, reasonable values
    for (int i = 0; i < total_elements; i++) {
        EXPECT_TRUE(std::isfinite(output_rms[i]));
        EXPECT_TRUE(std::isfinite(output_layer[i]));
        EXPECT_LT(std::abs(output_rms[i]), 100.0f); // Reasonable magnitude
        EXPECT_LT(std::abs(output_layer[i]), 100.0f);
    }
}

// Performance test (not a strict unit test, but useful for validation)
TEST_F(RMSNormTest, PerformanceComparison) {
    const int batch_size = 8;
    const int seq_len = 128;
    const int hidden_dim = 768;
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int num_iterations = 10;
    
    std::vector<float> input, gamma;
    createTestData(input, gamma, batch_size, seq_len, hidden_dim);
    
    std::vector<float> output_cpu(total_elements);
    std::vector<float> output_gpu(total_elements);
    
    // Allocate GPU memory
    float *d_input, *d_gamma, *d_output;
    ASSERT_EQ(cudaMalloc(&d_input, total_elements * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_gamma, hidden_dim * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, total_elements * sizeof(float)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(d_input, input.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gamma, gamma.data(), hidden_dim * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    
    // Warm up
    rms_norm_gpu(d_input, d_output, d_gamma, batch_size, seq_len, hidden_dim);
    cudaDeviceSynchronize();
    
    // Time GPU execution
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        rms_norm_gpu(d_input, d_output, d_gamma, batch_size, seq_len, hidden_dim);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto gpu_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    // Time CPU execution
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        rms_norm_cpu(input.data(), output_cpu.data(), gamma.data(), 
                     batch_size, seq_len, hidden_dim);
    }
    end = std::chrono::high_resolution_clock::now();
    
    auto cpu_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    // Verify correctness
    ASSERT_EQ(cudaMemcpy(output_gpu.data(), d_output, total_elements * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(compareArrays(output_cpu.data(), output_gpu.data(), total_elements));
    
    // GPU should be faster (this is more of an informational test)
    std::cout << "CPU time: " << cpu_time << " ms, GPU time: " << gpu_time 
              << " ms, Speedup: " << (cpu_time / gpu_time) << "x" << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_output);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
