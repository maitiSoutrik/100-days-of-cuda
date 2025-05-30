#include "gtest/gtest.h"
#include "swish_matrix_ops.cuh" // Includes cuda_runtime.h, cmath, CHECK_CUDA_ERROR
#include <vector>
#include <cmath> // For std::abs, expf


// Helper to compare floating point numbers with a tolerance
void EXPECT_FLOAT_EQ_TOL(float actual, float expected, float tol = 1e-5f) {
    EXPECT_NEAR(actual, expected, tol);
}

// CPU implementation of Swish for verification
float swish_cpu(float x, float beta = 1.0f) {
    return x * (1.0f / (1.0f + expf(-beta * x)));
}

// CPU implementation of matrix multiplication C = A * B
void matrix_mul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < K; ++l) {
                sum += A[i * K + l] * B[l * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CPU implementation of matrix_mul_swish_scale for verification
void matrix_mul_swish_scale_cpu(
    const float* A, const float* B, float* C_out,
    int M, int N, int K,
    float scale, float beta) {
    std::vector<float> C_temp(M * N);
    matrix_mul_cpu(A, B, C_temp.data(), M, N, K);
    for (int i = 0; i < M * N; ++i) {
        float scaled_val = C_temp[i] * scale;
        C_out[i] = swish_cpu(scaled_val, beta);
    }
}


TEST(SwishActivationKernelTest, BasicValues) {
    const int size = 5;
    float h_input[size] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
    float h_output[size];
    float h_expected[size];
    float beta = 1.0f;

    for(int i=0; i<size; ++i) {
        h_expected[i] = swish_cpu(h_input[i], beta);
    }

    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

    swish_activation_kernel<<<1, size>>>(d_input, d_output, size, beta);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ_TOL(h_output[i], h_expected[i]);
    }

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

TEST(MatrixMulSwishScaleKernelTest, SmallMatrixIdentity) {
    const int M = 2, N = 2, K = 2;
    const float scale = 1.0f;
    const float beta = 1.0f;

    std::vector<float> h_A = {1.0f, 0.0f, 0.0f, 1.0f}; // Identity
    std::vector<float> h_B = {1.0f, 2.0f, 3.0f, 4.0f}; // Some matrix
    std::vector<float> h_C_gpu(M * N);
    std::vector<float> h_C_expected(M * N);

    // Expected result: C = swish(scale * (I * B)) = swish(scale * B)
    matrix_mul_swish_scale_cpu(h_A.data(), h_B.data(), h_C_expected.data(), M, N, K, scale, beta);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16); // Default, will be oversized but okay for small test
    CHECK_CUDA_ERROR(matrix_mul_swish_scale(d_A, d_B, d_C, M, N, K, scale, beta, threadsPerBlock));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ_TOL(h_C_gpu[i], h_C_expected[i], 1e-4f); // Slightly higher tol for chain of ops
    }

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}

TEST(MatrixMulSwishScaleKernelTest, SmallMatrixSpecificValues) {
    const int M = 2, N = 1, K = 2; // A(2x2) * B(2x1) = C(2x1)
    const float scale = 0.5f;
    const float beta = 1.5f;

    std::vector<float> h_A = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_B = {0.5f, 0.8f};
    std::vector<float> h_C_gpu(M * N);
    std::vector<float> h_C_expected(M * N);

    // CPU calculation:
    // C_temp[0] = 1.0*0.5 + 2.0*0.8 = 0.5 + 1.6 = 2.1
    // C_temp[1] = 3.0*0.5 + 4.0*0.8 = 1.5 + 3.2 = 4.7
    // scaled_C0 = 2.1 * 0.5 = 1.05
    // scaled_C1 = 4.7 * 0.5 = 2.35
    // h_C_expected[0] = swish_cpu(1.05, 1.5)
    // h_C_expected[1] = swish_cpu(2.35, 1.5)
    matrix_mul_swish_scale_cpu(h_A.data(), h_B.data(), h_C_expected.data(), M, N, K, scale, beta);


    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    CHECK_CUDA_ERROR(matrix_mul_swish_scale(d_A, d_B, d_C, M, N, K, scale, beta, threadsPerBlock));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ_TOL(h_C_gpu[i], h_C_expected[i], 1e-4f);
    }

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
