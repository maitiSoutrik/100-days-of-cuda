#include "gtest/gtest.h"
#include "negative_cosine_similarity.cuh" // Also includes CHECK_CUDA_ERROR
#include <vector>
#include <cmath>     // For fabs, sqrtf, fmaxf
#include <numeric>   // For std::iota (not used here)
#include <algorithm> // For std::generate (not used here)
#include <iomanip>   // For printing in case of errors

// CPU implementation for verification within tests
void cosine_similarity_cpu_for_test(const std::vector<float>& predictions, const std::vector<float>& targets, std::vector<float>& output, size_t n, size_t d) {
    const float eps = 1e-8f;
    for (size_t i = 0; i < n; ++i) {
        float dot = 0.0f;
        float norm_pred = 0.0f;
        float norm_target = 0.0f;
        size_t offset = i * d;

        for (size_t j = 0; j < d; ++j) {
            float p = predictions[offset + j];
            float t = targets[offset + j];
            dot += p * t;
            norm_pred += p * p;
            norm_target += t * t;
        }
        norm_pred = sqrtf(norm_pred);
        norm_target = sqrtf(norm_target);
        
        float effective_norm_pred = fmaxf(eps, norm_pred);
        float effective_norm_target = fmaxf(eps, norm_target);
        float denom = effective_norm_pred * effective_norm_target;
        
        float cosine_sim = 0.0f;
        if (denom > eps * eps / 2.0f) { 
            cosine_sim = dot / denom;
        } else if (norm_pred == 0.0f && norm_target == 0.0f) {
            cosine_sim = 0.0f; 
        }
        // Clamp cosine_sim to [-1, 1]
        cosine_sim = fmaxf(-1.0f, fminf(1.0f, cosine_sim));
        output[i] = 1.0f - cosine_sim;
    }
}

class NegativeCosineSimilarityTest : public ::testing::Test {
protected:
    float *d_predictions_ = nullptr, *d_targets_ = nullptr, *d_output_ = nullptr;
    cudaError_t err_;

    void AllocateMemory(size_t n, size_t d) {
        err_ = cudaMalloc((void**)&d_predictions_, n * d * sizeof(float)); CHECK_CUDA_ERROR(err_);
        err_ = cudaMalloc((void**)&d_targets_, n * d * sizeof(float));     CHECK_CUDA_ERROR(err_);
        err_ = cudaMalloc((void**)&d_output_, n * sizeof(float));          CHECK_CUDA_ERROR(err_);
    }

    void FreeMemory() {
        if (d_predictions_) cudaFree(d_predictions_);
        if (d_targets_) cudaFree(d_targets_);
        if (d_output_) cudaFree(d_output_);
        d_predictions_ = d_targets_ = d_output_ = nullptr;
    }

    void TearDown() override {
        FreeMemory();
    }
};

TEST_F(NegativeCosineSimilarityTest, HandlesIdenticalVectors) {
    size_t n = 1, d = 3;
    AllocateMemory(n, d);
    std::vector<float> h_predictions = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_targets     = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_output_gpu(n);
    std::vector<float> h_output_cpu(n);

    err_ = cudaMemcpy(d_predictions_, h_predictions.data(), n * d * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(d_targets_, h_targets.data(), n * d * sizeof(float), cudaMemcpyHostToDevice);         CHECK_CUDA_ERROR(err_);

    launch_cosine_similarity_kernel(d_predictions_, d_targets_, d_output_, n, d);
    err_ = cudaDeviceSynchronize(); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(h_output_gpu.data(), d_output_, n * sizeof(float), cudaMemcpyDeviceToHost);           CHECK_CUDA_ERROR(err_);

    cosine_similarity_cpu_for_test(h_predictions, h_targets, h_output_cpu, n, d);

    ASSERT_NEAR(h_output_gpu[0], 0.0f, 1e-6f); 
    ASSERT_NEAR(h_output_gpu[0], h_output_cpu[0], 1e-6f);
}

TEST_F(NegativeCosineSimilarityTest, HandlesOrthogonalVectors) {
    size_t n = 1, d = 2;
    AllocateMemory(n,d);
    std::vector<float> h_predictions = {1.0f, 0.0f};
    std::vector<float> h_targets     = {0.0f, 1.0f};
    std::vector<float> h_output_gpu(n);
    std::vector<float> h_output_cpu(n);
    
    err_ = cudaMemcpy(d_predictions_, h_predictions.data(), n * d * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(d_targets_, h_targets.data(), n * d * sizeof(float), cudaMemcpyHostToDevice);         CHECK_CUDA_ERROR(err_);

    launch_cosine_similarity_kernel(d_predictions_, d_targets_, d_output_, n, d);
    err_ = cudaDeviceSynchronize(); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(h_output_gpu.data(), d_output_, n * sizeof(float), cudaMemcpyDeviceToHost);           CHECK_CUDA_ERROR(err_);
    
    cosine_similarity_cpu_for_test(h_predictions, h_targets, h_output_cpu, n, d);

    ASSERT_NEAR(h_output_gpu[0], 1.0f, 1e-6f); 
    ASSERT_NEAR(h_output_gpu[0], h_output_cpu[0], 1e-6f);
}

TEST_F(NegativeCosineSimilarityTest, HandlesOppositeVectors) {
    size_t n = 1, d = 3;
    AllocateMemory(n,d);
    std::vector<float> h_predictions = {1.0f, 1.0f, 1.0f};
    std::vector<float> h_targets     = {-1.0f, -1.0f, -1.0f};
    std::vector<float> h_output_gpu(n);
    std::vector<float> h_output_cpu(n);

    err_ = cudaMemcpy(d_predictions_, h_predictions.data(), n * d * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(d_targets_, h_targets.data(), n * d * sizeof(float), cudaMemcpyHostToDevice);         CHECK_CUDA_ERROR(err_);

    launch_cosine_similarity_kernel(d_predictions_, d_targets_, d_output_, n, d);
    err_ = cudaDeviceSynchronize(); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(h_output_gpu.data(), d_output_, n * sizeof(float), cudaMemcpyDeviceToHost);           CHECK_CUDA_ERROR(err_);

    cosine_similarity_cpu_for_test(h_predictions, h_targets, h_output_cpu, n, d);
    
    ASSERT_NEAR(h_output_gpu[0], 2.0f, 1e-6f); 
    ASSERT_NEAR(h_output_gpu[0], h_output_cpu[0], 1e-6f);
}

TEST_F(NegativeCosineSimilarityTest, HandlesZeroVectorCases) {
    size_t n = 1, d = 3;
    AllocateMemory(n,d);
    std::vector<float> h_output_gpu(n);
    std::vector<float> h_output_cpu(n);

    // Case 1: Prediction is zero vector
    std::vector<float> h_pred_zero = {0.0f, 0.0f, 0.0f};
    std::vector<float> h_target_non_zero = {1.0f, 2.0f, 3.0f};

    err_ = cudaMemcpy(d_predictions_, h_pred_zero.data(), n * d * sizeof(float), cudaMemcpyHostToDevice);       CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(d_targets_, h_target_non_zero.data(), n * d * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(err_);
    
    launch_cosine_similarity_kernel(d_predictions_, d_targets_, d_output_, n, d);
    err_ = cudaDeviceSynchronize(); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(h_output_gpu.data(), d_output_, n * sizeof(float), cudaMemcpyDeviceToHost);             CHECK_CUDA_ERROR(err_);
    cosine_similarity_cpu_for_test(h_pred_zero, h_target_non_zero, h_output_cpu, n, d);
    
    ASSERT_NEAR(h_output_gpu[0], 1.0f, 1e-6f); // Expect 1.0 (1 - 0)
    ASSERT_NEAR(h_output_gpu[0], h_output_cpu[0], 1e-6f);

    // Case 2: Target is zero vector
    std::vector<float> h_pred_non_zero = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_target_zero = {0.0f, 0.0f, 0.0f};

    err_ = cudaMemcpy(d_predictions_, h_pred_non_zero.data(), n * d * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(d_targets_, h_target_zero.data(), n * d * sizeof(float), cudaMemcpyHostToDevice);     CHECK_CUDA_ERROR(err_);

    launch_cosine_similarity_kernel(d_predictions_, d_targets_, d_output_, n, d);
    err_ = cudaDeviceSynchronize(); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(h_output_gpu.data(), d_output_, n * sizeof(float), cudaMemcpyDeviceToHost);           CHECK_CUDA_ERROR(err_);
    cosine_similarity_cpu_for_test(h_pred_non_zero, h_target_zero, h_output_cpu, n, d);

    ASSERT_NEAR(h_output_gpu[0], 1.0f, 1e-6f); // Expect 1.0 (1 - 0)
    ASSERT_NEAR(h_output_gpu[0], h_output_cpu[0], 1e-6f);

    // Case 3: Both vectors are zero
    err_ = cudaMemcpy(d_predictions_, h_pred_zero.data(), n * d * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(d_targets_, h_target_zero.data(), n * d * sizeof(float), cudaMemcpyHostToDevice);   CHECK_CUDA_ERROR(err_);
    
    launch_cosine_similarity_kernel(d_predictions_, d_targets_, d_output_, n, d);
    err_ = cudaDeviceSynchronize(); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(h_output_gpu.data(), d_output_, n * sizeof(float), cudaMemcpyDeviceToHost);         CHECK_CUDA_ERROR(err_);
    cosine_similarity_cpu_for_test(h_pred_zero, h_target_zero, h_output_cpu, n, d);

    ASSERT_NEAR(h_output_gpu[0], 1.0f, 1e-6f); // Expect 1.0 (1 - 0)
    ASSERT_NEAR(h_output_gpu[0], h_output_cpu[0], 1e-6f);
}

TEST_F(NegativeCosineSimilarityTest, MultipleVectorsBatch) {
    size_t n = 4, d = 2; // Increased n to include a general case
    AllocateMemory(n,d);
    std::vector<float> h_predictions = {
        1.0f, 2.0f,  // Identical
        1.0f, 0.0f,  // Orthogonal
       -1.0f,-1.0f,  // Opposite (using -1,-1 for pred to make it clearer)
        0.5f, 0.8f   // General
    };
    std::vector<float> h_targets = {
        1.0f, 2.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,  // Target for opposite case
       -0.2f, 0.3f   // General
    };
    // For general case:
    // P = [0.5, 0.8], T = [-0.2, 0.3]
    // dot = (0.5*-0.2) + (0.8*0.3) = -0.1 + 0.24 = 0.14
    // normP = sqrt(0.25 + 0.64) = sqrt(0.89) = 0.9434
    // normT = sqrt(0.04 + 0.09) = sqrt(0.13) = 0.3606
    // cos_sim = 0.14 / (0.9434 * 0.3606) = 0.14 / 0.3401 = 0.4116
    // output = 1 - 0.4116 = 0.5884

    std::vector<float> h_output_gpu(n);
    std::vector<float> h_output_cpu(n);

    err_ = cudaMemcpy(d_predictions_, h_predictions.data(), n * d * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(d_targets_, h_targets.data(), n * d * sizeof(float), cudaMemcpyHostToDevice);         CHECK_CUDA_ERROR(err_);

    launch_cosine_similarity_kernel(d_predictions_, d_targets_, d_output_, n, d);
    err_ = cudaDeviceSynchronize(); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(h_output_gpu.data(), d_output_, n * sizeof(float), cudaMemcpyDeviceToHost);           CHECK_CUDA_ERROR(err_);

    cosine_similarity_cpu_for_test(h_predictions, h_targets, h_output_cpu, n, d);

    ASSERT_NEAR(h_output_gpu[0], h_output_cpu[0], 1e-6f); // Identical: 0.0
    ASSERT_NEAR(h_output_gpu[1], h_output_cpu[1], 1e-6f); // Orthogonal: 1.0
    ASSERT_NEAR(h_output_gpu[2], h_output_cpu[2], 1e-6f); // Opposite: 2.0
    ASSERT_NEAR(h_output_gpu[3], h_output_cpu[3], 1e-5f); // General: 0.5883... (tolerance adjusted slightly)
}

// Test with larger N and D to check for race conditions or block/grid issues (simple check)
TEST_F(NegativeCosineSimilarityTest, LargerNAndD) {
    size_t n = 512, d = 64;
    AllocateMemory(n,d);
    std::vector<float> h_predictions(n * d);
    std::vector<float> h_targets(n * d);
    
    // Simple initialization: predictions are multiples of 0.1, targets are multiples of 0.2
    for(size_t i = 0; i < n * d; ++i) {
        h_predictions[i] = static_cast<float>((i % d) + 1) * 0.1f;
        h_targets[i] = static_cast<float>((i % d) + 1) * ( (i / d) % 2 == 0 ? 0.2f : -0.2f); // Alternate positive/negative targets
    }

    std::vector<float> h_output_gpu(n);
    std::vector<float> h_output_cpu(n);

    err_ = cudaMemcpy(d_predictions_, h_predictions.data(), n * d * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR(err_);
    err_ = cudaMemcpy(d_targets_, h_targets.data(), n * d * sizeof(float), cudaMemcpyHostToDevice);         CHECK_CUDA_ERROR(err_);

    launch_cosine_similarity_kernel(d_predictions_, d_targets_, d_output_, n, d);
    err_ = cudaDeviceSynchronize(); CHECK_CUDA_ERROR(err_); // Crucial for catching kernel errors
    cudaError_t kernel_launch_err = cudaGetLastError(); // Check errors specifically after kernel
    ASSERT_EQ(kernel_launch_err, cudaSuccess) << "Kernel launch or execution failed: " << cudaGetErrorString(kernel_launch_err);

    err_ = cudaMemcpy(h_output_gpu.data(), d_output_, n * sizeof(float), cudaMemcpyDeviceToHost);           CHECK_CUDA_ERROR(err_);

    cosine_similarity_cpu_for_test(h_predictions, h_targets, h_output_cpu, n, d);

    for(size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(h_output_gpu[i], h_output_cpu[i], 1e-5f) << "Mismatch at index " << i;
    }
}
