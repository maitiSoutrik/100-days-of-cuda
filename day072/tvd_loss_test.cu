#include "gtest/gtest.h"
#include "tvd_loss.cuh"
#include <vector>
#include <numeric> // For std::iota, std::accumulate
#include <cmath>   // For std::abs
#include <cuda_runtime.h>

// Helper to normalize PMF for tests
void normalize_pmf_test(std::vector<float>& v) {
    if (v.empty()) return;
    double sum = 0.0;
    for (float val : v) {
        sum += std::abs(val); // Ensure positive contributions before normalization
    }
    if (sum == 0.0) {
        if (!v.empty()) {
            float val = 1.0f / v.size();
            for (size_t i = 0; i < v.size(); ++i) v[i] = val;
        }
        return;
    }
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = std::abs(v[i]) / static_cast<float>(sum);
    }
}


TEST(TVD_Loss_CPU, EmptyVectors) {
    std::vector<float> p_empty, q_empty;
    EXPECT_FLOAT_EQ(calculate_tvd_cpu(p_empty, q_empty), 0.0f);
}

TEST(TVD_Loss_CPU, IdenticalVectors) {
    std::vector<float> p = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> q = {0.1f, 0.2f, 0.3f, 0.4f};
    EXPECT_FLOAT_EQ(calculate_tvd_cpu(p, q), 0.0f);
}

TEST(TVD_Loss_CPU, SimpleDisjointVectors) {
    // P = [1, 0], Q = [0, 1] -> TVD = 0.5 * (|1-0| + |0-1|) = 0.5 * (1 + 1) = 1
    std::vector<float> p = {1.0f, 0.0f};
    std::vector<float> q = {0.0f, 1.0f};
    EXPECT_FLOAT_EQ(calculate_tvd_cpu(p, q), 1.0f);
}

TEST(TVD_Loss_CPU, SimpleMixedVectors) {
    // P = [0.5, 0.5], Q = [0.25, 0.75]
    // TVD = 0.5 * (|0.5-0.25| + |0.5-0.75|) = 0.5 * (0.25 + 0.25) = 0.25
    std::vector<float> p = {0.5f, 0.5f};
    std::vector<float> q = {0.25f, 0.75f};
    EXPECT_FLOAT_EQ(calculate_tvd_cpu(p, q), 0.25f);
}

TEST(TVD_Loss_CPU, DifferentSizes) {
    std::vector<float> p = {0.1f, 0.9f};
    std::vector<float> q = {0.1f, 0.2f, 0.7f};
    // Expecting -1.0f or some error indication as per tvd_loss.cu
    // For GTest, it's better if the function throws or returns a clear error code.
    // Given current implementation returns -1.0f:
    EXPECT_FLOAT_EQ(calculate_tvd_cpu(p, q), -1.0f);
}


TEST(TVD_Loss_GPU, IdenticalVectorsGPU) {
    std::vector<float> h_p = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> h_q = {0.1f, 0.2f, 0.3f, 0.4f};
    int n = h_p.size();

    float* d_p;
    float* d_q;
    float* d_tvd;
    CHECK_CUDA_ERROR(cudaMalloc(&d_p, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_q, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_tvd, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_q, h_q.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    calculate_tvd_gpu(d_p, d_q, n, d_tvd);

    float h_tvd_result;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_tvd_result, d_tvd, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_tvd_result, 0.0f);

    CHECK_CUDA_ERROR(cudaFree(d_p));
    CHECK_CUDA_ERROR(cudaFree(d_q));
    CHECK_CUDA_ERROR(cudaFree(d_tvd));
}

TEST(TVD_Loss_GPU, SimpleDisjointVectorsGPU) {
    std::vector<float> h_p = {1.0f, 0.0f};
    std::vector<float> h_q = {0.0f, 1.0f};
    int n = h_p.size();

    float* d_p;
    float* d_q;
    float* d_tvd;
    CHECK_CUDA_ERROR(cudaMalloc(&d_p, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_q, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_tvd, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_q, h_q.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    calculate_tvd_gpu(d_p, d_q, n, d_tvd);

    float h_tvd_result;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_tvd_result, d_tvd, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(h_tvd_result, 1.0f);

    CHECK_CUDA_ERROR(cudaFree(d_p));
    CHECK_CUDA_ERROR(cudaFree(d_q));
    CHECK_CUDA_ERROR(cudaFree(d_tvd));
}

TEST(TVD_Loss_GPU, CompareWithCPU) {
    const int n_elements = 1024;
    std::vector<float> h_p(n_elements);
    std::vector<float> h_q(n_elements);

    std::mt19937 rng(67890); // Different seed from main
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n_elements; ++i) {
        h_p[i] = dist(rng);
        h_q[i] = dist(rng);
    }
    normalize_pmf_test(h_p);
    normalize_pmf_test(h_q);

    float cpu_tvd = calculate_tvd_cpu(h_p, h_q);

    float* d_p;
    float* d_q;
    float* d_tvd;
    CHECK_CUDA_ERROR(cudaMalloc(&d_p, n_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_q, n_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_tvd, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), n_elements * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_q, h_q.data(), n_elements * sizeof(float), cudaMemcpyHostToDevice));

    calculate_tvd_gpu(d_p, d_q, n_elements, d_tvd);

    float gpu_tvd_result;
    CHECK_CUDA_ERROR(cudaMemcpy(&gpu_tvd_result, d_tvd, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(gpu_tvd_result, cpu_tvd, 1e-5); // Using EXPECT_NEAR for float comparison

    CHECK_CUDA_ERROR(cudaFree(d_p));
    CHECK_CUDA_ERROR(cudaFree(d_q));
    CHECK_CUDA_ERROR(cudaFree(d_tvd));
}

// It might be good to test with n that is not a multiple of block size
TEST(TVD_Loss_GPU, CompareWithCPU_OddSize) {
    const int n_elements = 1000; // Not a multiple of 256
    std::vector<float> h_p(n_elements);
    std::vector<float> h_q(n_elements);

    std::mt19937 rng(13579); 
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n_elements; ++i) {
        h_p[i] = dist(rng);
        h_q[i] = dist(rng);
    }
    normalize_pmf_test(h_p);
    normalize_pmf_test(h_q);

    float cpu_tvd = calculate_tvd_cpu(h_p, h_q);

    float* d_p;
    float* d_q;
    float* d_tvd;
    CHECK_CUDA_ERROR(cudaMalloc(&d_p, n_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_q, n_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_tvd, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), n_elements * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_q, h_q.data(), n_elements * sizeof(float), cudaMemcpyHostToDevice));

    calculate_tvd_gpu(d_p, d_q, n_elements, d_tvd);

    float gpu_tvd_result;
    CHECK_CUDA_ERROR(cudaMemcpy(&gpu_tvd_result, d_tvd, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(gpu_tvd_result, cpu_tvd, 1e-5);

    CHECK_CUDA_ERROR(cudaFree(d_p));
    CHECK_CUDA_ERROR(cudaFree(d_q));
    CHECK_CUDA_ERROR(cudaFree(d_tvd));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
