#include "gtest/gtest.h"
#include "tensor_matrix_mult.cuh" // Includes CHECK_CUDA_ERROR and tensor_matrix_multiply declaration
#include <vector>
#include <numeric>   // For std::iota
#include <algorithm> // For std::abs for comparison
#include <iomanip>   // For printing

// Helper to initialize data for tests
template<typename T>
void initialize_test_data(std::vector<T>& data, T start_val = 1.0f, T increment = 0.1f) {
    T current_val = start_val;
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = current_val;
        current_val += increment;
    }
    // For very small increments or large arrays, consider fixed-point or scaled integers if precision becomes an issue.
    // For this test, simple float increment should be fine.
}

// CPU implementation for verification (can be reused from main or defined here for test isolation)
void tensor_matrix_multiply_cpu_gtest(
    const std::vector<float>& A_host,
    const std::vector<float>& B_host,
    std::vector<float>& C_host,
    size_t B_dim, size_t I_dim, size_t J_dim, size_t L_dim, size_t K_dim) {

    C_host.assign(B_dim * I_dim * J_dim * K_dim, 0.0f); // Ensure C_host is correctly sized and zeroed

    if (L_dim == 0) { // If contraction dimension is 0, result is all zeros.
        return;
    }
    if (B_dim * I_dim * J_dim * K_dim == 0) { // If any output dimension is 0, result is empty.
        return;
    }


    for (size_t b = 0; b < B_dim; ++b) {
        for (size_t i = 0; i < I_dim; ++i) {
            for (size_t j = 0; j < J_dim; ++j) {
                for (size_t k = 0; k < K_dim; ++k) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < L_dim; ++l) {
                        // A_idx = b*I*J*L + i*J*L + j*L + l
                        size_t a_idx = ((b * I_dim + i) * J_dim + j) * L_dim + l;
                        // B_idx = l*K + k
                        size_t b_idx = l * K_dim + k;
                        if (a_idx < A_host.size() && b_idx < B_host.size()){ // Boundary check
                           sum += A_host[a_idx] * B_host[b_idx];
                        }
                    }
                    // C_idx = b*I*J*K + i*J*K + j*K + k
                    size_t c_idx = (((b * I_dim + i) * J_dim + j) * K_dim + k);
                     if (c_idx < C_host.size()){ // Boundary check
                        C_host[c_idx] = sum;
                     }
                }
            }
        }
    }
}

// Test fixture for tensor matrix multiplication tests
class TensorMatrixMultTest : public ::testing::Test {
protected:
    float *d_A_ = nullptr, *d_B_ = nullptr, *d_C_ = nullptr;
    std::vector<float> h_A_, h_B_, h_C_gpu_, h_C_cpu_;

    size_t A_size_ = 0, B_size_ = 0, C_size_ = 0;
    size_t current_B_dim_ = 0, current_I_dim_ = 0, current_J_dim_ = 0, current_L_dim_ = 0, current_K_dim_ = 0;


    void SetupTest(size_t B_dim, size_t I_dim, size_t J_dim, size_t L_dim, size_t K_dim) {
        current_B_dim_ = B_dim;
        current_I_dim_ = I_dim;
        current_J_dim_ = J_dim;
        current_L_dim_ = L_dim;
        current_K_dim_ = K_dim;

        A_size_ = B_dim * I_dim * J_dim * L_dim;
        B_size_ = L_dim * K_dim;
        C_size_ = B_dim * I_dim * J_dim * K_dim;

        h_A_.resize(A_size_);
        h_B_.resize(B_size_);
        h_C_gpu_.resize(C_size_);
        h_C_cpu_.resize(C_size_); // Will be filled by CPU version

        if (A_size_ > 0) initialize_test_data(h_A_, 1.0f, 0.1f);
        if (B_size_ > 0) initialize_test_data(h_B_, 0.5f, 0.05f);
        // h_C_gpu_ and h_C_cpu_ are outputs, so no need to initialize their values here beyond resizing.

        if (A_size_ > 0) CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A_, A_size_ * sizeof(float)));
        if (B_size_ > 0) CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B_, B_size_ * sizeof(float)));
        if (C_size_ > 0) CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C_, C_size_ * sizeof(float)));

        if (A_size_ > 0) CHECK_CUDA_ERROR(cudaMemcpy(d_A_, h_A_.data(), A_size_ * sizeof(float), cudaMemcpyHostToDevice));
        if (B_size_ > 0) CHECK_CUDA_ERROR(cudaMemcpy(d_B_, h_B_.data(), B_size_ * sizeof(float), cudaMemcpyHostToDevice));
        // d_C_ is output, no need to copy to it initially.
    }

    void TearDown() override {
        if (d_A_) CHECK_CUDA_ERROR(cudaFree(d_A_));
        if (d_B_) CHECK_CUDA_ERROR(cudaFree(d_B_));
        if (d_C_) CHECK_CUDA_ERROR(cudaFree(d_C_));
        d_A_ = nullptr; d_B_ = nullptr; d_C_ = nullptr;
    }

    void RunAndVerify() {
        if (C_size_ == 0) { // Handle cases where output is empty (e.g., a dimension is 0)
            tensor_matrix_multiply_cpu_gtest(h_A_, h_B_, h_C_cpu_, current_B_dim_, current_I_dim_, current_J_dim_, current_L_dim_, current_K_dim_);
            ASSERT_EQ(h_C_cpu_.size(), 0); // CPU should also produce empty output
            // No GPU call needed if output is empty, or ensure kernel handles it gracefully.
            // The CUDA kernel implementation has `if (idx < total_elements_C)`
            // and the wrapper has `if (total_output_elements == 0) return;`
            // So, calling it should be safe.
            tensor_matrix_multiply(d_A_, d_B_, d_C_, current_B_dim_, current_I_dim_, current_J_dim_, current_L_dim_, current_K_dim_);
            CHECK_KERNEL_LAUNCH(); // Should still be safe.
            SUCCEED();
            return;
        }
        
        tensor_matrix_multiply(d_A_, d_B_, d_C_, current_B_dim_, current_I_dim_, current_J_dim_, current_L_dim_, current_K_dim_);
        CHECK_KERNEL_LAUNCH(); // Check for kernel errors and synchronize
        
        CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu_.data(), d_C_, C_size_ * sizeof(float), cudaMemcpyDeviceToHost));

        tensor_matrix_multiply_cpu_gtest(h_A_, h_B_, h_C_cpu_, current_B_dim_, current_I_dim_, current_J_dim_, current_L_dim_, current_K_dim_);

        float tolerance = 1e-4f; // Adjust tolerance as needed
        ASSERT_EQ(h_C_gpu_.size(), h_C_cpu_.size());
        for (size_t i = 0; i < C_size_; ++i) {
            ASSERT_NEAR(h_C_gpu_[i], h_C_cpu_[i], tolerance)
                << "Mismatch at index " << i << " for dimensions: "
                << "B=" << current_B_dim_ << ", I=" << current_I_dim_ << ", J=" << current_J_dim_
                << ", L=" << current_L_dim_ << ", K=" << current_K_dim_
                << ". GPU val: " << h_C_gpu_[i] << ", CPU val: " << h_C_cpu_[i];
        }
    }
};


TEST_F(TensorMatrixMultTest, BasicCase) {
    // A: (2, 3, 4, 5), B: (5, 6) -> C: (2, 3, 4, 6)
    SetupTest(2, 3, 4, 5, 6);
    RunAndVerify();
}

TEST_F(TensorMatrixMultTest, SmallDimensions) {
    // A: (1, 1, 1, 1), B: (1, 1) -> C: (1, 1, 1, 1)
    SetupTest(1, 1, 1, 1, 1);
    RunAndVerify();
}

TEST_F(TensorMatrixMultTest, LargerLdim) {
    // A: (1, 2, 2, 10), B: (10, 3) -> C: (1, 2, 2, 3)
    SetupTest(1, 2, 2, 10, 3);
    RunAndVerify();
}

TEST_F(TensorMatrixMultTest, LargerKdim) {
    // A: (1, 2, 2, 3), B: (3, 10) -> C: (1, 2, 2, 10)
    SetupTest(1, 2, 2, 3, 10);
    RunAndVerify();
}

TEST_F(TensorMatrixMultTest, OneIDim) {
    // A: (2, 1, 2, 3), B: (3, 4) -> C: (2, 1, 2, 4)
    SetupTest(2, 1, 2, 3, 4);
    RunAndVerify();
}

TEST_F(TensorMatrixMultTest, OneJDim) {
    // A: (2, 2, 1, 3), B: (3, 4) -> C: (2, 2, 1, 4)
    SetupTest(2, 2, 1, 3, 4);
    RunAndVerify();
}

TEST_F(TensorMatrixMultTest, OneBDim) {
    // A: (1, 2, 2, 3), B: (3, 4) -> C: (1, 2, 2, 4)
    SetupTest(1, 2, 2, 3, 4);
    RunAndVerify();
}

TEST_F(TensorMatrixMultTest, ZeroKDimResultsInZeroCSize) {
    // A: (2,3,4,5), B: (5,0) -> C: (2,3,4,0)
    // C_size will be 0.
    SetupTest(2, 3, 4, 5, 0);
    RunAndVerify(); // Should handle C_size_ == 0 correctly
}

TEST_F(TensorMatrixMultTest, ZeroLDimResultsInZerosInC) {
    // A: (2,3,4,0), B: (0,5) -> C: (2,3,4,5)
    // A_size and B_size will be 0. C_size will be non-zero.
    // The kernel loop `for (size_t l = 0; l < L_dim; ++l)` will not run if L_dim is 0.
    // C elements should all be 0.0f.
    SetupTest(2, 3, 4, 0, 5);
    RunAndVerify(); // CPU version should also produce all zeros for C.
}

TEST_F(TensorMatrixMultTest, ZeroBDimResultsInZeroCSize) {
    SetupTest(0, 3, 4, 5, 6); // B_dim = 0
    RunAndVerify();
}

TEST_F(TensorMatrixMultTest, ZeroIDimResultsInZeroCSize) {
    SetupTest(2, 0, 4, 5, 6); // I_dim = 0
    RunAndVerify();
}

TEST_F(TensorMatrixMultTest, ZeroJDimResultsInZeroCSize) {
    SetupTest(2, 3, 0, 5, 6); // J_dim = 0
    RunAndVerify();
}

// Entry point for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // It's good practice to select a GPU device if multiple are available,
    // or ensure the default device is appropriate.
    // For Jetson Nano, there's typically only one.
    // int deviceCount;
    // cudaGetDeviceCount(&deviceCount);
    // if (deviceCount == 0) {
    //     std::cerr << "No CUDA devices found!" << std::endl;
    //     return 1;
    // }
    // CHECK_CUDA_ERROR(cudaSetDevice(0)); // Select device 0
    return RUN_ALL_TESTS();
}

