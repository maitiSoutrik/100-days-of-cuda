#include "swiglu.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iomanip> // For std::fixed, std::setprecision in potential debug prints
#include <numeric>   // For std::iota
#include <algorithm> // For std::generate

// --- CPU Reference Implementations (copied from main for test self-containment) ---
// CPU implementation for sigmoid
static float sigmoid_cpu_test(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// CPU implementation for SwiGLU forward
static void swiglu_forward_cpu_test(const std::vector<float>& h_a,
                                    const std::vector<float>& h_b,
                                    std::vector<float>& h_c,
                                    int rows, int cols) {
    h_c.resize(static_cast<size_t>(rows) * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            float val_a = h_a[idx];
            float val_b = h_b[idx];
            float s_a = sigmoid_cpu_test(val_a);
            float silu_a = val_a * s_a;
            h_c[idx] = silu_a * val_b;
        }
    }
}

// CPU implementation for SwiGLU backward
static void swiglu_backward_cpu_test(const std::vector<float>& h_a,
                                     const std::vector<float>& h_b,
                                     const std::vector<float>& h_dc,
                                     std::vector<float>& h_da,
                                     std::vector<float>& h_db,
                                     int rows, int cols) {
    size_t size = static_cast<size_t>(rows) * cols;
    h_da.resize(size);
    h_db.resize(size);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            float val_a = h_a[idx];
            float val_b = h_b[idx];
            float val_dc = h_dc[idx];
            
            float s_a = sigmoid_cpu_test(val_a);
            
            h_da[idx] = val_dc * val_b * s_a * (1.0f + val_a * (1.0f - s_a));
            h_db[idx] = val_dc * val_a * s_a;
        }
    }
}
// --- End CPU Reference Implementations ---

class SwiGLUTest : public ::testing::Test {
protected:
    int rows = 2;
    int cols = 4;
    size_t matrix_size;
    size_t bytes;

    std::vector<float> h_a, h_b, h_dc;
    std::vector<float> h_c_gpu, h_da_gpu, h_db_gpu;
    std::vector<float> h_c_cpu, h_da_cpu, h_db_cpu;

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    float *d_dc = nullptr, *d_da = nullptr, *d_db = nullptr;

    void SetUp() override {
        matrix_size = static_cast<size_t>(rows) * cols;
        bytes = matrix_size * sizeof(float);

        h_a.resize(matrix_size);
        h_b.resize(matrix_size);
        h_dc.resize(matrix_size);

        h_c_gpu.resize(matrix_size);
        h_da_gpu.resize(matrix_size);
        h_db_gpu.resize(matrix_size);
        
        h_c_cpu.resize(matrix_size);
        h_da_cpu.resize(matrix_size);
        h_db_cpu.resize(matrix_size);

        // Initialize input data
        for(size_t i = 0; i < matrix_size; ++i) {
            h_a[i] = static_cast<float>(i % 5) * 0.2f - 0.4f; // Values like -0.4, -0.2, 0.0, 0.2, 0.4
            h_b[i] = static_cast<float>((i+2) % 5) * 0.2f + 0.2f; // Values like 0.2, 0.4, 0.6, 0.8, 1.0
            h_dc[i] = 1.0f; // Gradient of 1
        }

        CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b, bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_c, bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_dc, bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_da, bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_db, bytes));
    }

    void TearDown() override {
        if (d_a) CHECK_CUDA_ERROR(cudaFree(d_a));
        if (d_b) CHECK_CUDA_ERROR(cudaFree(d_b));
        if (d_c) CHECK_CUDA_ERROR(cudaFree(d_c));
        if (d_dc) CHECK_CUDA_ERROR(cudaFree(d_dc));
        if (d_da) CHECK_CUDA_ERROR(cudaFree(d_da));
        if (d_db) CHECK_CUDA_ERROR(cudaFree(d_db));
    }
};

TEST_F(SwiGLUTest, ForwardPassCorrectness) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    launch_swiglu_forward(d_a, d_b, d_c, rows, cols);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    swiglu_forward_cpu_test(h_a, h_b, h_c_cpu, rows, cols);

    for (size_t i = 0; i < matrix_size; ++i) {
        ASSERT_NEAR(h_c_cpu[i], h_c_gpu[i], 1e-4f) << "Mismatch at index " << i;
    }
}

TEST_F(SwiGLUTest, BackwardPassCorrectness) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dc, h_dc.data(), bytes, cudaMemcpyHostToDevice));

    launch_swiglu_backward(d_a, d_b, d_dc, d_da, d_db, rows, cols);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_da_gpu.data(), d_da, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_db_gpu.data(), d_db, bytes, cudaMemcpyDeviceToHost));

    swiglu_backward_cpu_test(h_a, h_b, h_dc, h_da_cpu, h_db_cpu, rows, cols);

    for (size_t i = 0; i < matrix_size; ++i) {
        ASSERT_NEAR(h_da_cpu[i], h_da_gpu[i], 1e-4f) << "dA mismatch at index " << i;
        ASSERT_NEAR(h_db_cpu[i], h_db_gpu[i], 1e-4f) << "dB mismatch at index " << i;
    }
}

// Test with zero rows/cols
TEST(SwiGLUTestEmpty, HandlesZeroSize) {
    int zero_rows = 0, zero_cols = 5;
    size_t zero_matrix_size = 0;
    size_t zero_bytes = 0;

    float *d_a_empty = nullptr, *d_b_empty = nullptr, *d_c_empty = nullptr;
    // No allocation needed for zero size, but kernels should handle it gracefully.
    // The launch wrappers already check for zero rows/cols.

    // Test forward
    ASSERT_NO_THROW(launch_swiglu_forward(d_a_empty, d_b_empty, d_c_empty, zero_rows, zero_cols));
    ASSERT_NO_THROW(launch_swiglu_forward(d_a_empty, d_b_empty, d_c_empty, 5, 0));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure no launch errors occurred

    // Test backward
    float *d_dc_empty = nullptr, *d_da_empty = nullptr, *d_db_empty = nullptr;
    ASSERT_NO_THROW(launch_swiglu_backward(d_a_empty, d_b_empty, d_dc_empty, d_da_empty, d_db_empty, zero_rows, zero_cols));
    ASSERT_NO_THROW(launch_swiglu_backward(d_a_empty, d_b_empty, d_dc_empty, d_da_empty, d_db_empty, 5, 0));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Entry point for running tests
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
// The main for gtest is usually linked by GTest::gtest_main
