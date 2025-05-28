#include "gtest/gtest.h"
#include "upper_tri_gemm.cuh"
#include <vector>
#include <numeric> // For std::iota

// Helper function to perform CPU matrix multiplication for verification
template <typename T>
std::vector<T> cpu_upper_tri_gemm(const std::vector<T>& A, const std::vector<T>& B, int n) {
    std::vector<T> C(n * n, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) { // Result is also upper triangular
            T sum = 0;
            for (int k = 0; k < n; ++k) {
                // Only consider non-zero elements for upper triangular matrices
                if (i <= k && k <= j) { // A[i][k] and B[k][j] are non-zero
                    sum += A[i * n + k] * B[k * n + j];
                }
            }
            C[i * n + j] = sum;
        }
    }
    return C;
}

// Test fixture for GPU matrix multiplication
template <typename T>
class UpperTriGemmTest : public ::testing::Test {
protected:
    int n = 4; // Small matrix size for testing
    std::vector<T> h_A, h_B, h_C_gpu, h_C_cpu;

    void SetUp() override {
        h_A.assign(n * n, 0);
        h_B.assign(n * n, 0);
        h_C_gpu.assign(n * n, 0);

        // Initialize A and B as upper triangular matrices
        // Example:
        // A = {{1, 2, 3, 4},
        //      {0, 5, 6, 7},
        //      {0, 0, 8, 9},
        //      {0, 0, 0, 10}}
        // B = {{10, 9, 8, 7},
        //      {0, 6, 5, 4},
        //      {0, 0, 3, 2},
        //      {0, 0, 0, 1}}
        
        int val = 1;
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                h_A[i * n + j] = val++;
            }
        }

        val = 10;
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                h_B[i * n + j] = val--;
            }
        }

        // Compute CPU reference
        h_C_cpu = cpu_upper_tri_gemm(h_A, h_B, n);

        // Perform GPU computation
        T *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, n * n * sizeof(T));
        cudaMalloc(&d_B, n * n * sizeof(T));
        cudaMalloc(&d_C, n * n * sizeof(T));

        cudaMemcpy(d_A, h_A.data(), n * n * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), n * n * sizeof(T), cudaMemcpyHostToDevice);

        dim3 block(2, 2); // Small block size for small matrix
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

        gemm_upper_tri<T, 2><<<grid, block>>>(d_A, d_B, d_C, n); // Use BLOCK_SIZE 2 for small test

        cudaMemcpy(h_C_gpu.data(), d_C, n * n * sizeof(T), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
};

// Define test cases for float and double
using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UpperTriGemmTest, MyTypes);

TYPED_TEST(UpperTriGemmTest, Correctness) {
    for (int i = 0; i < this->n; ++i) {
        for (int j = 0; j < this->n; ++j) {
            if (i > j) { // Elements below diagonal should be zero
                ASSERT_NEAR(this->h_C_gpu[i * this->n + j], 0.0, 1e-6);
            } else {
                ASSERT_NEAR(this->h_C_gpu[i * this->n + j], this->h_C_cpu[i * this->n + j], 1e-6);
            }
        }
    }
}
