#include "muon_optimization.cuh"
#include "gtest/gtest.h"
#include <vector>
#include <cmath>   // For fabsf
#include <numeric> // For std::iota for sequence
#include <algorithm> // For std::equal

// Helper to compare two host matrices
::testing::AssertionResult AreMatricesEqual(const float* A, const float* B, int rows, int cols, float tol = 1e-4f) {
    for (int i = 0; i < rows * cols; ++i) {
        if (fabsf(A[i] - B[i]) > tol) {
            return ::testing::AssertionFailure() << "Matrices differ at index " << i
                                                 << ": A[" << i << "] = " << A[i]
                                                 << ", B[" << i << "] = " << B[i];
        }
    }
    return ::testing::AssertionSuccess();
}

// Host function for matrix multiplication (for verification in tests)
void matrix_multiply_host_test(const float* A, const float* B, float* C, int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            C[i * B_cols + j] = 0.0f;
            for (int k = 0; k < A_cols; ++k) {
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}

// Host function for matrix transpose (for verification in tests)
void matrix_transpose_host_test(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}


TEST(MatrixOpsTest, Transpose) {
    const int rows = 2, cols = 3;
    std::vector<float> h_input = {1, 2, 3, 4, 5, 6};
    std::vector<float> h_expected_output = {1, 4, 2, 5, 3, 6};
    std::vector<float> h_output(rows * cols);

    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, rows * cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, cols * rows * sizeof(float))); // Transposed dimensions

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threads(cols, rows); // Exact threads for small matrix
    dim3 blocks(1, 1);
    matrix_transpose_kernel<<<blocks, threads>>>(d_input, d_output, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, cols * rows * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_TRUE(AreMatricesEqual(h_output.data(), h_expected_output.data(), cols, rows));

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

TEST(MatrixOpsTest, Multiply) {
    const int A_rows = 2, A_cols = 3, B_cols = 2;
    std::vector<float> h_A = {1, 2, 3, 4, 5, 6}; // 2x3
    std::vector<float> h_B = {7, 8, 9, 10, 11, 12}; // 3x2
    std::vector<float> h_C_expected(A_rows * B_cols); // 2x2
    matrix_multiply_host_test(h_A.data(), h_B.data(), h_C_expected.data(), A_rows, A_cols, B_cols);

    std::vector<float> h_C_actual(A_rows * B_cols);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, A_rows * A_cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, A_cols * B_cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, A_rows * B_cols * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), A_rows * A_cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), A_cols * B_cols * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threads(B_cols, A_rows); // Exact threads
    dim3 blocks(1,1);
    matrix_multiply_kernel<<<blocks, threads>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_C_actual.data(), d_C, A_rows * B_cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    EXPECT_TRUE(AreMatricesEqual(h_C_actual.data(), h_C_expected.data(), A_rows, B_cols));

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}


TEST(NewtonSchulzTest, OrthogonalizeSmallMatrix) {
    // Test with a small matrix, e.g., 2x2 or 3x2
    // For a 2x2 matrix, if it's already orthogonal (e.g. identity or rotation), NS should ideally not change it much after normalization.
    // Or, start with a non-orthogonal matrix and check if G_out^T * G_out (or G_out * G_out^T) is closer to identity.
    
    int rows = 2, cols = 2; // Square matrix
    // int rows = 3, cols = 2; // Tall matrix
    // int rows = 2, cols = 3; // Wide matrix

    std::vector<float> h_G_in = {1.0f, 2.0f, 3.0f, 4.0f}; // A simple 2x2 matrix
    if (rows == 3 && cols == 2) h_G_in = {1,2,3,4,5,6};
    if (rows == 2 && cols == 3) h_G_in = {1,2,3,4,5,6};

    std::vector<float> h_G_out(rows * cols);
    int num_ns_iterations = 5;

    float *d_G_in, *d_G_out, *d_O, *d_O_T, *d_prod1, *d_prod2, *d_block_sums;
    CHECK_CUDA_ERROR(cudaMalloc(&d_G_in, rows * cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_G_out, rows * cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_O, rows * cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_O_T, cols * rows * sizeof(float)));
    
    bool tall_or_square = (rows >= cols);
    int prod1_dim1 = tall_or_square ? cols : rows;
    CHECK_CUDA_ERROR(cudaMalloc(&d_prod1, prod1_dim1 * prod1_dim1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_prod2, rows * cols * sizeof(float)));
    
    int N_elements = rows * cols;
    int norm_threads_per_block = 256;
    int num_norm_blocks = (N_elements + norm_threads_per_block - 1) / norm_threads_per_block;
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_sums, num_norm_blocks * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_G_in, h_G_in.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    newton_schulz_iteration_device(d_G_in, d_G_out, rows, cols, num_ns_iterations,
                                   d_O, d_O_T, d_prod1, d_prod2, d_block_sums);
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_G_out.data(), d_G_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    // Verification
    std::vector<float> h_G_out_T_test(cols * rows);
    matrix_transpose_host_test(h_G_out.data(), h_G_out_T_test.data(), rows, cols);

    if (tall_or_square) { // G_out^T * G_out should be Identity_cols
        std::vector<float> h_verify(cols * cols);
        matrix_multiply_host_test(h_G_out_T_test.data(), h_G_out.data(), h_verify.data(), cols, rows, cols);
        // print_matrix_host(h_verify.data(), cols, cols, "Test: G_out^T * G_out");
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < cols; ++j) {
                float target = (i == j) ? 1.0f : 0.0f;
                EXPECT_NEAR(h_verify[i * cols + j], target, 1e-2f) // Looser tolerance due to approximation
                    << "G_out^T * G_out at (" << i << "," << j << ") failed for tall/square matrix.";
            }
        }
    } else { // G_out * G_out^T should be Identity_rows
        std::vector<float> h_verify(rows * rows);
        matrix_multiply_host_test(h_G_out.data(), h_G_out_T_test.data(), h_verify.data(), rows, cols, rows);
        // print_matrix_host(h_verify.data(), rows, rows, "Test: G_out * G_out^T");
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < rows; ++j) {
                float target = (i == j) ? 1.0f : 0.0f;
                EXPECT_NEAR(h_verify[i * rows + j], target, 1e-2f)
                    << "G_out * G_out^T at (" << i << "," << j << ") failed for wide matrix.";
            }
        }
    }

    CHECK_CUDA_ERROR(cudaFree(d_G_in));
    CHECK_CUDA_ERROR(cudaFree(d_G_out));
    CHECK_CUDA_ERROR(cudaFree(d_O));
    CHECK_CUDA_ERROR(cudaFree(d_O_T));
    CHECK_CUDA_ERROR(cudaFree(d_prod1));
    CHECK_CUDA_ERROR(cudaFree(d_prod2));
    CHECK_CUDA_ERROR(cudaFree(d_block_sums));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
