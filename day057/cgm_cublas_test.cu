#include <gtest/gtest.h>
#include "cgm_cublas.cuh" // Includes cuda_runtime.h, cublas_v2.h, and error checks
#include <vector>
#include <cmath> // For fabs

// Test fixture for CGM tests
class CgmCuBLASTest : public ::testing::Test {
protected:
    cublasHandle_t handle;
    const int n = 2; // Dimension for test case
    const int max_iters = 50;
    const double tolerance = 1e-7;

    // Host data
    std::vector<double> h_A; // Column-major
    std::vector<double> h_b;
    std::vector<double> h_x_initial;
    std::vector<double> h_x_expected;
    std::vector<double> h_x_result;

    // Device data pointers
    double *d_A = nullptr, *d_b = nullptr, *d_x = nullptr;

    void SetUp() override {
        // Initialize cuBLAS
        CHECK_CUBLAS_ERROR(cublasCreate(&handle));

        // Define the test system Ax=b where A = [[2, -1], [-1, 2]], b = [1, 1]
        // Expected solution x = [1, 1]
        h_A = {2.0, -1.0, -1.0, 2.0}; // Column-major
        h_b = {1.0, 1.0};
        h_x_initial = {0.0, 0.0}; // Start with zero vector
        h_x_expected = {1.0, 1.0};
        h_x_result.resize(n);

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_A, n * n * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b, n * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_x, n * sizeof(double)));

        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x_initial.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    }

    void TearDown() override {
        // Free device memory
        if (d_A) CHECK_CUDA_ERROR(cudaFree(d_A));
        if (d_b) CHECK_CUDA_ERROR(cudaFree(d_b));
        if (d_x) CHECK_CUDA_ERROR(cudaFree(d_x));

        // Destroy cuBLAS handle
        if (handle) CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    }
};

// Test case for the conjugateGradientMethodCuBLAS function
TEST_F(CgmCuBLASTest, Solves2x2System) {
    // Run the CGM solver
    int iterations = conjugateGradientMethodCuBLAS(handle, n, d_A, d_b, d_x, max_iters, tolerance);

    // Check if converged
    ASSERT_GE(iterations, 0) << "CGM did not converge within max iterations.";
    ASSERT_LT(iterations, max_iters) << "CGM reached max iterations, likely did not converge.";

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_x_result.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Verify the solution
    ASSERT_EQ(h_x_result.size(), h_x_expected.size());
    for (int i = 0; i < n; ++i) {
        // Use ASSERT_NEAR for floating-point comparisons
        ASSERT_NEAR(h_x_result[i], h_x_expected[i], tolerance * 10) // Allow slightly larger tolerance for result check
            << "Element x[" << i << "] differs.";
    }
}

// Add more tests if needed, e.g., for different matrices, sizes, tolerances, non-convergence cases.
