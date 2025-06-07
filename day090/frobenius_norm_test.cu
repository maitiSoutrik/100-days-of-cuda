#include "frobenius_norm.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <cmath> // For fabs
#include <random>

// Helper function to initialize a matrix with random values for testing
void initializeTestMatrix(std::vector<float>& matrix_vec, int rows, int cols) {
    matrix_vec.resize(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-5.0, 5.0); // Smaller range for easier debugging if needed
    for (int i = 0; i < rows * cols; ++i) {
        matrix_vec[i] = static_cast<float>(distrib(gen));
    }
}

TEST(FrobeniusNormTest, HandlesEmptyMatrix) {
    int rows = 0, cols = 0;
    std::vector<float> h_matrix_vec; // Empty
    float* d_matrix = nullptr; // No device allocation for empty matrix

    EXPECT_FLOAT_EQ(frobeniusNormCPU(h_matrix_vec.data(), rows, cols), 0.0f);
    EXPECT_FLOAT_EQ(frobeniusNormGPU(d_matrix, rows, cols), 0.0f);

    rows = 5; cols = 0; // Still empty
    EXPECT_FLOAT_EQ(frobeniusNormCPU(h_matrix_vec.data(), rows, cols), 0.0f);
    EXPECT_FLOAT_EQ(frobeniusNormGPU(d_matrix, rows, cols), 0.0f);
}

TEST(FrobeniusNormTest, HandlesSingleElementMatrix) {
    int rows = 1, cols = 1;
    std::vector<float> h_matrix_vec = { -3.5f };
    float* d_matrix;
    size_t matrix_size_bytes = h_matrix_vec.size() * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix, matrix_size_bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix_vec.data(), matrix_size_bytes, cudaMemcpyHostToDevice));

    float expected_norm = 3.5f;
    EXPECT_FLOAT_EQ(frobeniusNormCPU(h_matrix_vec.data(), rows, cols), expected_norm);
    EXPECT_FLOAT_EQ(frobeniusNormGPU(d_matrix, rows, cols), expected_norm);

    CHECK_CUDA_ERROR(cudaFree(d_matrix));
}

TEST(FrobeniusNormTest, SmallMatrixVerification) {
    int rows = 2, cols = 2;
    std::vector<float> h_matrix_vec = {1.0f, 2.0f, 3.0f, 4.0f}; // sqrt(1^2+2^2+3^2+4^2) = sqrt(1+4+9+16) = sqrt(30)
    float* d_matrix;
    size_t matrix_size_bytes = h_matrix_vec.size() * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix, matrix_size_bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix_vec.data(), matrix_size_bytes, cudaMemcpyHostToDevice));

    float norm_cpu = frobeniusNormCPU(h_matrix_vec.data(), rows, cols);
    float norm_gpu = frobeniusNormGPU(d_matrix, rows, cols);

    float expected_norm = sqrtf(1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f + 4.0f*4.0f);
    EXPECT_FLOAT_EQ(norm_cpu, expected_norm);
    EXPECT_FLOAT_EQ(norm_gpu, expected_norm);
    EXPECT_NEAR(norm_gpu, norm_cpu, 1e-5); // GPU and CPU results should be very close

    CHECK_CUDA_ERROR(cudaFree(d_matrix));
}

TEST(FrobeniusNormTest, LargerMatrixVerification) {
    int rows = 64, cols = 128; // Larger, but not excessively large for a test
    std::vector<float> h_matrix_vec;
    initializeTestMatrix(h_matrix_vec, rows, cols);
    
    float* d_matrix;
    size_t matrix_size_bytes = h_matrix_vec.size() * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix, matrix_size_bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix_vec.data(), matrix_size_bytes, cudaMemcpyHostToDevice));

    float norm_cpu = frobeniusNormCPU(h_matrix_vec.data(), rows, cols);
    float norm_gpu = frobeniusNormGPU(d_matrix, rows, cols);

    // For larger matrices, allow a slightly larger tolerance due to potential floating point differences
    EXPECT_NEAR(norm_gpu, norm_cpu, 1e-3); 

    CHECK_CUDA_ERROR(cudaFree(d_matrix));
}

TEST(FrobeniusNormTest, RowVector) {
    int rows = 1, cols = 5;
    std::vector<float> h_matrix_vec = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f}; // sqrt(1+1+4+4+9) = sqrt(19)
    float* d_matrix;
    size_t matrix_size_bytes = h_matrix_vec.size() * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix, matrix_size_bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix_vec.data(), matrix_size_bytes, cudaMemcpyHostToDevice));
    
    float expected_norm = sqrtf(1.0f + 1.0f + 4.0f + 4.0f + 9.0f);
    EXPECT_FLOAT_EQ(frobeniusNormCPU(h_matrix_vec.data(), rows, cols), expected_norm);
    EXPECT_FLOAT_EQ(frobeniusNormGPU(d_matrix, rows, cols), expected_norm);

    CHECK_CUDA_ERROR(cudaFree(d_matrix));
}

TEST(FrobeniusNormTest, ColumnVector) {
    int rows = 4, cols = 1;
    std::vector<float> h_matrix_vec = {0.5f, 1.5f, 2.5f, 3.5f}; // sqrt(0.25 + 2.25 + 6.25 + 12.25) = sqrt(21)
    float* d_matrix;
    size_t matrix_size_bytes = h_matrix_vec.size() * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix, matrix_size_bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix_vec.data(), matrix_size_bytes, cudaMemcpyHostToDevice));

    float expected_norm = sqrtf(0.25f + 2.25f + 6.25f + 12.25f);
    EXPECT_FLOAT_EQ(frobeniusNormCPU(h_matrix_vec.data(), rows, cols), expected_norm);
    EXPECT_FLOAT_EQ(frobeniusNormGPU(d_matrix, rows, cols), expected_norm);
    
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
}

// Entry point for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
