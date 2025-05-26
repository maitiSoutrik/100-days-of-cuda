#include "dyt.cuh"
#include "gtest/gtest.h"
#include <vector>
#include <cmath>
#include <iomanip> // For printing if needed

// Helper function to compare floating point values
void compare_float_vectors(const std::vector<float>& vec1, const std::vector<float>& vec2, float tolerance = 1e-5f) {
    ASSERT_EQ(vec1.size(), vec2.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        EXPECT_NEAR(vec1[i], vec2[i], tolerance) << "Mismatch at index " << i;
    }
}

void compare_floats(float val1, float val2, float tolerance = 1e-5f) {
    EXPECT_NEAR(val1, val2, tolerance);
}

class DyTTest : public ::testing::Test {
protected:
    int n_test_ = 4;
    float alpha_test_ = 2.0f;
    float beta_test_ = 1.5f;

    std::vector<float> h_x_ = {-1.0f, 0.0f, 0.5f, 1.0f};
    std::vector<float> h_upstream_grad_;

    float *d_x_ = nullptr, *d_y_ = nullptr, *d_upstream_grad_ = nullptr, *d_x_grad_ = nullptr;
    float *d_alpha_grad_atomic_ = nullptr, *d_beta_grad_atomic_ = nullptr;

    void SetUp() override {
        h_upstream_grad_.assign(n_test_, 1.0f); // Simple upstream gradient

        CHECK_CUDA_ERROR(cudaMalloc(&d_x_, n_test_ * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_y_, n_test_ * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_upstream_grad_, n_test_ * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_x_grad_, n_test_ * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_alpha_grad_atomic_, sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_beta_grad_atomic_, sizeof(float)));

        CHECK_CUDA_ERROR(cudaMemcpy(d_x_, h_x_.data(), n_test_ * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_upstream_grad_, h_upstream_grad_.data(), n_test_ * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemset(d_alpha_grad_atomic_, 0, sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_beta_grad_atomic_, 0, sizeof(float)));
    }

    void TearDown() override {
        cudaFree(d_x_);
        cudaFree(d_y_);
        cudaFree(d_upstream_grad_);
        cudaFree(d_x_grad_);
        cudaFree(d_alpha_grad_atomic_);
        cudaFree(d_beta_grad_atomic_);
    }

    // Manual calculation for forward pass
    std::vector<float> manual_forward() {
        std::vector<float> expected_y(n_test_);
        for (int i = 0; i < n_test_; ++i) {
            expected_y[i] = alpha_test_ * std::tanh(beta_test_ * h_x_[i]);
        }
        return expected_y;
    }

    // Manual calculation for backward pass
    void manual_backward(std::vector<float>& expected_x_grad, float& expected_alpha_grad, float& expected_beta_grad) {
        expected_x_grad.resize(n_test_);
        expected_alpha_grad = 0.0f;
        expected_beta_grad = 0.0f;

        for (int i = 0; i < n_test_; ++i) {
            float beta_x_val = beta_test_ * h_x_[i];
            float tanh_beta_x_val = std::tanh(beta_x_val);
            float one_minus_tanh_sq_val = 1.0f - tanh_beta_x_val * tanh_beta_x_val;

            expected_x_grad[i] = h_upstream_grad_[i] * alpha_test_ * beta_test_ * one_minus_tanh_sq_val;
            expected_alpha_grad += h_upstream_grad_[i] * tanh_beta_x_val;
            expected_beta_grad += h_upstream_grad_[i] * alpha_test_ * h_x_[i] * one_minus_tanh_sq_val;
        }
    }
};

TEST_F(DyTTest, ForwardPass) {
    int blockSize = 4; // Small block size for small test data
    int numBlocks = (n_test_ + blockSize - 1) / blockSize;

    dyt_forward_kernel<<<numBlocks, blockSize>>>(d_x_, d_y_, alpha_test_, beta_test_, n_test_);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<float> h_y_result(n_test_);
    CHECK_CUDA_ERROR(cudaMemcpy(h_y_result.data(), d_y_, n_test_ * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected_y = manual_forward();
    compare_float_vectors(h_y_result, expected_y);
}

TEST_F(DyTTest, BackwardPass) {
    int blockSize = 4;
    int numBlocks = (n_test_ + blockSize - 1) / blockSize;

    dyt_backward_kernel<<<numBlocks, blockSize>>>(d_upstream_grad_, d_x_, d_x_grad_,
                                                 d_alpha_grad_atomic_, d_beta_grad_atomic_,
                                                 alpha_test_, beta_test_, n_test_);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<float> h_x_grad_result(n_test_);
    float h_alpha_grad_result, h_beta_grad_result;

    CHECK_CUDA_ERROR(cudaMemcpy(h_x_grad_result.data(), d_x_grad_, n_test_ * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_alpha_grad_result, d_alpha_grad_atomic_, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_beta_grad_result, d_beta_grad_atomic_, sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> expected_x_grad;
    float expected_alpha_grad, expected_beta_grad;
    manual_backward(expected_x_grad, expected_alpha_grad, expected_beta_grad);

    compare_float_vectors(h_x_grad_result, expected_x_grad);
    compare_floats(h_alpha_grad_result, expected_alpha_grad);
    compare_floats(h_beta_grad_result, expected_beta_grad);
}

// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
