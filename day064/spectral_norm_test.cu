#include "spectral_norm.cuh"
#include "gtest/gtest.h"
#include <vector>
#include <cmath> // For fabsf
#include <iostream> // For debug prints if needed
#include <iomanip>  // For std::fixed, std::setprecision

// Helper to compare floats with a tolerance
::testing::AssertionResult AssertNearPred(const char* expr1,
                                          const char* expr2,
                                          const char* abs_error_expr,
                                          float val1,
                                          float val2,
                                          float abs_error) {
    if (fabsf(val1 - val2) <= abs_error) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
           << val1 << " and " << val2 << " are not within " << abs_error
           << " of each other. Difference is " << fabsf(val1 - val2) << ".";
}

// Helper function to copy host data to device and return device pointer
float* create_device_matrix(const std::vector<float>& h_matrix, int rows, int cols) {
    float* d_matrix;
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, rows * cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    return d_matrix;
}

// Test fixture for Spectral Normalization tests
class SpectralNormTest : public ::testing::Test {
protected:
    cublasHandle_t handle;
    float *d_u_test, *d_v_test; // General purpose workspace vectors

    SpectralNormTest() : d_u_test(nullptr), d_v_test(nullptr) {} // Initialize pointers

    void SetUp() override {
        CHECK_CUBLAS_ERROR(cublasCreate(&handle));
    }

    void TearDown() override {
        CHECK_CUBLAS_ERROR(cublasDestroy(handle));
        if (d_u_test) CHECK_CUDA_ERROR(cudaFree(d_u_test));
        if (d_v_test) CHECK_CUDA_ERROR(cudaFree(d_v_test));
        d_u_test = nullptr;
        d_v_test = nullptr;
    }

    // Helper to allocate workspace vectors based on matrix dimensions
    void allocate_workspace(int m, int n) {
        if (d_u_test) CHECK_CUDA_ERROR(cudaFree(d_u_test));
        if (d_v_test) CHECK_CUDA_ERROR(cudaFree(d_v_test));
        CHECK_CUDA_ERROR(cudaMalloc(&d_u_test, m * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_v_test, n * sizeof(float)));
    }
};

TEST_F(SpectralNormTest, EstimateSpectralNorm_Identity2x2) {
    int m = 2, n = 2;
    allocate_workspace(m, n);
    std::vector<float> h_W = {1.0f, 0.0f, 0.0f, 1.0f}; // Identity matrix (col-major)
    float* d_W = create_device_matrix(h_W, m, n);

    float estimated_norm = estimate_spectral_norm(handle, d_W, m, n, d_u_test, d_v_test, 20);
    EXPECT_PRED_FORMAT3(AssertNearPred, estimated_norm, 1.0f, 1e-4f);

    CHECK_CUDA_ERROR(cudaFree(d_W));
}

TEST_F(SpectralNormTest, EstimateSpectralNorm_Simple2x2) {
    int m = 2, n = 2;
    allocate_workspace(m, n);
    // W = [1 0; 0 2] (col-major: {1.0, 0.0, 0.0, 2.0})
    // Singular values are 2, 1. Spectral norm = 2.
    std::vector<float> h_W = {1.0f, 0.0f, 0.0f, 2.0f};
    float* d_W = create_device_matrix(h_W, m, n);

    float estimated_norm = estimate_spectral_norm(handle, d_W, m, n, d_u_test, d_v_test, 20);
    EXPECT_PRED_FORMAT3(AssertNearPred, estimated_norm, 2.0f, 1e-4f);
    
    CHECK_CUDA_ERROR(cudaFree(d_W));
}

TEST_F(SpectralNormTest, EstimateSpectralNorm_3x2Matrix) {
    int m = 3, n = 2;
    allocate_workspace(m, n);
    // W = [1 4; 2 5; 3 6] (col-major: {1,2,3, 4,5,6})
    // Using numpy: np.linalg.svd(np.array([[1,4],[2,5],[3,6]]))[1][0] approx 9.508
    std::vector<float> h_W = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float* d_W = create_device_matrix(h_W, m, n);

    float estimated_norm = estimate_spectral_norm(handle, d_W, m, n, d_u_test, d_v_test, 30); // More iterations
    EXPECT_PRED_FORMAT3(AssertNearPred, estimated_norm, 9.5080f, 1e-3f); // Increased tolerance slightly

    CHECK_CUDA_ERROR(cudaFree(d_W));
}

TEST_F(SpectralNormTest, SpectralNormalizeMatrix_Simple2x2) {
    int m = 2, n = 2;
    allocate_workspace(m, n);
    std::vector<float> h_W_orig = {1.0f, 0.0f, 0.0f, 2.0f}; // Spectral norm = 2
    float* d_W = create_device_matrix(h_W_orig, m, n);

    spectral_normalize_matrix(handle, d_W, m, n, d_u_test, d_v_test, 20);

    // Verify the spectral norm of the normalized matrix is close to 1
    float norm_after_normalization = estimate_spectral_norm(handle, d_W, m, n, d_u_test, d_v_test, 20);
    EXPECT_PRED_FORMAT3(AssertNearPred, norm_after_normalization, 1.0f, 1e-4f);

    // Verify the content of the normalized matrix
    // Expected: W_norm = W / 2 = [0.5 0; 0 1] (col-major: {0.5, 0.0, 0.0, 1.0})
    std::vector<float> h_W_normalized(m * n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_W_normalized.data(), d_W, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_PRED_FORMAT3(AssertNearPred, h_W_normalized[0], 0.5f, 1e-5f);
    EXPECT_PRED_FORMAT3(AssertNearPred, h_W_normalized[1], 0.0f, 1e-5f);
    EXPECT_PRED_FORMAT3(AssertNearPred, h_W_normalized[2], 0.0f, 1e-5f);
    EXPECT_PRED_FORMAT3(AssertNearPred, h_W_normalized[3], 1.0f, 1e-5f);

    CHECK_CUDA_ERROR(cudaFree(d_W));
}

TEST_F(SpectralNormTest, SpectralNormalizeMatrix_ZeroMatrix) {
    int m = 2, n = 2;
    allocate_workspace(m, n);
    std::vector<float> h_W_zero = {0.0f, 0.0f, 0.0f, 0.0f}; // Zero matrix
    float* d_W = create_device_matrix(h_W_zero, m, n);

    // Spectral norm should be 0
    float estimated_norm_before = estimate_spectral_norm(handle, d_W, m, n, d_u_test, d_v_test, 10);
    EXPECT_PRED_FORMAT3(AssertNearPred, estimated_norm_before, 0.0f, 1e-5f);
    
    spectral_normalize_matrix(handle, d_W, m, n, d_u_test, d_v_test, 10);

    // Spectral norm should still be 0 (or very close) after "normalization"
    float norm_after_normalization = estimate_spectral_norm(handle, d_W, m, n, d_u_test, d_v_test, 10);
    EXPECT_PRED_FORMAT3(AssertNearPred, norm_after_normalization, 0.0f, 1e-5f);

    // Matrix should remain zero
    std::vector<float> h_W_normalized(m * n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_W_normalized.data(), d_W, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < m * n; ++i) {
        EXPECT_PRED_FORMAT3(AssertNearPred, h_W_normalized[i], 0.0f, 1e-5f);
    }
    CHECK_CUDA_ERROR(cudaFree(d_W));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
