#include "fused_linear_softmax_loss.cuh"
#include "gtest/gtest.h"
#include <vector>
#include <cmath>      // For std::exp, std::log
#include <cfloat>     // For FLT_MAX
#include <numeric>    // For std::accumulate
#include <algorithm>  // For std::max_element, std::transform

// CPU reference calculation (can be simplified or made more robust for tests)
// This is similar to the one in main.cu for consistency.
void calculate_expected_loss_cpu_for_test(
    const float* h_input_features,
    const float* h_weights,
    const float* h_bias,
    const int* h_true_labels,
    float* h_expected_loss_per_sample,
    int M,
    int K,
    int N
) {
    std::vector<float> logits(N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            logits[n] = 0.0f;
            for (int k_val = 0; k_val < K; ++k_val) {
                logits[n] += h_input_features[m * K + k_val] * h_weights[n * K + k_val];
            }
            logits[n] += h_bias[n];
        }

        float max_logit = -FLT_MAX;
        for (int n = 0; n < N; ++n) max_logit = std::max(max_logit, logits[n]);

        float sum_exp_shifted = 0.0f;
        for (int n = 0; n < N; ++n) {
            sum_exp_shifted += std::exp(logits[n] - max_logit);
        }
        
        int true_class_idx = h_true_labels[m];
        float logit_tc = (true_class_idx >=0 && true_class_idx < N) ? logits[true_class_idx] : 0.0f;

        if (sum_exp_shifted > 0 && true_class_idx >=0 && true_class_idx < N) {
            h_expected_loss_per_sample[m] = std::log(sum_exp_shifted) - (logit_tc - max_logit);
        } else {
            h_expected_loss_per_sample[m] = FLT_MAX; // Error or invalid case
        }
    }
}


TEST(FusedLinearSoftmaxLossTest, BasicCorrectness) {
    const int M = 2;
    const int K = 3;
    const int N = 2;

    std::vector<float> h_input_features = {
        1.0f, 2.0f, 3.0f,  // Sample 0
        0.5f, 1.5f, 2.5f   // Sample 1
    };
    std::vector<float> h_weights = {
        0.1f, 0.2f, 0.3f,  // Weights for class 0
        0.4f, 0.5f, 0.6f   // Weights for class 1
    };
    std::vector<float> h_bias = {0.1f, -0.1f}; // Bias for class 0, class 1
    std::vector<int> h_true_labels = {0, 1}; // True label for sample 0 is class 0, for sample 1 is class 1

    std::vector<float> h_output_loss_gpu(M);
    std::vector<float> h_expected_loss_cpu(M);

    // Calculate expected loss using CPU
    calculate_expected_loss_cpu_for_test(
        h_input_features.data(), h_weights.data(), h_bias.data(), 
        h_true_labels.data(), h_expected_loss_cpu.data(), M, K, N);

    // Calculate actual loss using GPU
    compute_fused_linear_softmax_loss_gpu(
        h_input_features.data(), h_weights.data(), h_bias.data(),
        h_true_labels.data(), h_output_loss_gpu.data(), M, K, N);

    for (int i = 0; i < M; ++i) {
        ASSERT_NEAR(h_output_loss_gpu[i], h_expected_loss_cpu[i], 1e-4)
            << "Mismatch in per-sample loss for sample " << i;
    }
}

TEST(FusedLinearSoftmaxLossTest, SingleSampleSingleClass) {
    const int M = 1;
    const int K = 2;
    const int N = 1; // Softmax with N=1 means probability is 1, log(1)=0. Loss should be related to logit.
                     // log(exp(logit - logit)) - (logit - logit) = log(1) - 0 = 0.

    std::vector<float> h_input_features = {1.0f, 2.0f};
    std::vector<float> h_weights = {0.5f, 0.5f};
    std::vector<float> h_bias = {0.1f};
    std::vector<int> h_true_labels = {0};

    std::vector<float> h_output_loss_gpu(M);
    std::vector<float> h_expected_loss_cpu(M);

    calculate_expected_loss_cpu_for_test(
        h_input_features.data(), h_weights.data(), h_bias.data(), 
        h_true_labels.data(), h_expected_loss_cpu.data(), M, K, N);
    
    // Expected for N=1: logit_0 = 1*0.5 + 2*0.5 + 0.1 = 0.5 + 1.0 + 0.1 = 1.6
    // max_logit = 1.6
    // sum_exp_shifted = exp(1.6 - 1.6) = exp(0) = 1.0
    // logit_tc = 1.6
    // loss = log(1.0) - (1.6 - 1.6) = 0 - 0 = 0.0
    // So, h_expected_loss_cpu[0] should be 0.0f

    compute_fused_linear_softmax_loss_gpu(
        h_input_features.data(), h_weights.data(), h_bias.data(),
        h_true_labels.data(), h_output_loss_gpu.data(), M, K, N);

    ASSERT_NEAR(h_expected_loss_cpu[0], 0.0f, 1e-5) << "CPU calculation for N=1 seems off.";
    ASSERT_NEAR(h_output_loss_gpu[0], 0.0f, 1e-5) << "GPU loss for N=1 should be 0.";
}


// Test with a slightly larger, more random case
TEST(FusedLinearSoftmaxLossTest, LargerRandomCase) {
    const int M = 4;
    const int K = 8;
    const int N = 5;

    std::vector<float> h_input_features(M*K);
    std::vector<float> h_weights(N*K);
    std::vector<float> h_bias(N);
    std::vector<int> h_true_labels(M);
    
    std::mt19937 rng(54321); // Different seed from main
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::uniform_int_distribution<int> label_dist(0, N - 1);

    for(int i=0; i<M*K; ++i) h_input_features[i] = dist(rng);
    for(int i=0; i<N*K; ++i) h_weights[i] = dist(rng);
    for(int i=0; i<N; ++i) h_bias[i] = dist(rng);
    for(int i=0; i<M; ++i) h_true_labels[i] = label_dist(rng);

    std::vector<float> h_output_loss_gpu(M);
    std::vector<float> h_expected_loss_cpu(M);

    calculate_expected_loss_cpu_for_test(
        h_input_features.data(), h_weights.data(), h_bias.data(), 
        h_true_labels.data(), h_expected_loss_cpu.data(), M, K, N);

    compute_fused_linear_softmax_loss_gpu(
        h_input_features.data(), h_weights.data(), h_bias.data(),
        h_true_labels.data(), h_output_loss_gpu.data(), M, K, N);

    for (int i = 0; i < M; ++i) {
        ASSERT_NEAR(h_output_loss_gpu[i], h_expected_loss_cpu[i], 1e-4)
            << "Mismatch in per-sample loss for sample " << i << " in larger random case.";
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
