#include "fused_linear_softmax_loss.cuh"
#include <vector>
#include <iostream>
#include <random>
#include <iomanip> // For std::fixed and std::setprecision
#include <numeric> // For std::iota
#include <algorithm> // For std::transform, std::exp
#include <cmath>     // For std::log
#include <cfloat>    // For FLT_MAX

// Helper function to print a matrix (row-major)
template<typename T>
void print_matrix(const T* matrix, int rows, int cols, const std::string& name) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << matrix[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Helper function to print a vector
template<typename T>
void print_vector(const T* vec, int size, const std::string& name) {
    std::cout << name << " (" << size << "):\n";
    for (int i = 0; i < size; ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i] << "\t";
    }
    std::cout << "\n" << std::endl;
}

// CPU implementation for verification
float compute_fused_linear_softmax_loss_cpu(
    const float* h_input_features,
    const float* h_weights,
    const float* h_bias,
    const int* h_true_labels,
    float* h_output_loss_per_sample, // Output parameter
    int M,
    int K,
    int N
) {
    std::vector<float> logits(N);
    double total_loss = 0.0;

    for (int m = 0; m < M; ++m) {
        // Linear transformation: logits = input * weights^T + bias
        for (int n = 0; n < N; ++n) {
            logits[n] = 0.0f;
            for (int k = 0; k < K; ++k) {
                logits[n] += h_input_features[m * K + k] * h_weights[n * K + k];
            }
            logits[n] += h_bias[n];
        }

        // Softmax
        float max_logit = -FLT_MAX;
        for (int n = 0; n < N; ++n) {
            if (logits[n] > max_logit) {
                max_logit = logits[n];
            }
        }

        std::vector<float> exp_logits(N);
        float sum_exp_logits = 0.0f;
        for (int n = 0; n < N; ++n) {
            exp_logits[n] = std::exp(logits[n] - max_logit);
            sum_exp_logits += exp_logits[n];
        }

        // Probabilities (not strictly needed if using log-sum-exp trick for loss)
        // std::vector<float> probabilities(N);
        // for (int n = 0; n < N; ++n) {
        //     probabilities[n] = exp_logits[n] / sum_exp_logits;
        // }

        // Cross-entropy loss: -log(probability_of_true_class)
        // = - ( (logit_true_class - max_logit) - log(sum_exp_logits) )
        // = log(sum_exp_logits) - (logit_true_class - max_logit)
        int true_class_idx = h_true_labels[m];
        if (true_class_idx < 0 || true_class_idx >= N) {
             h_output_loss_per_sample[m] = FLT_MAX; // Invalid label
             total_loss += FLT_MAX;
             continue;
        }
        float logit_tc = logits[true_class_idx];
        
        float sample_loss = 0.0f;
        if (sum_exp_logits > 0) {
            sample_loss = std::log(sum_exp_logits) - (logit_tc - max_logit);
        } else {
            sample_loss = FLT_MAX; // Should not happen
        }
        
        h_output_loss_per_sample[m] = sample_loss;
        total_loss += sample_loss;
    }
    if (M == 0) return 0.0f;
    return static_cast<float>(total_loss / M);
}


int main() {
    // Problem dimensions
    const int M = 4; // Batch size
    const int K = 8; // Input features
    const int N = 5; // Number of classes

    std::cout << "Problem Dimensions:\n";
    std::cout << "Batch Size (M): " << M << "\n";
    std::cout << "Input Features (K): " << K << "\n";
    std::cout << "Number of Classes (N): " << N << "\n\n";

    // Allocate host memory
    std::vector<float> h_input_features(M * K);
    std::vector<float> h_weights(N * K);
    std::vector<float> h_bias(N);
    std::vector<int> h_true_labels(M);
    std::vector<float> h_output_loss_per_sample_gpu(M);
    std::vector<float> h_output_loss_per_sample_cpu(M);

    // Initialize data with random values
    std::mt19937 rng(12345); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> label_dist(0, N - 1);

    for (int i = 0; i < M * K; ++i) h_input_features[i] = dist(rng);
    for (int i = 0; i < N * K; ++i) h_weights[i] = dist(rng);
    for (int i = 0; i < N; ++i) h_bias[i] = dist(rng);
    for (int i = 0; i < M; ++i) h_true_labels[i] = label_dist(rng);
    
    // Print some input data (optional, can be verbose for large M, K, N)
    // print_matrix(h_input_features.data(), M, K, "Input Features");
    // print_matrix(h_weights.data(), N, K, "Weights");
    // print_vector(h_bias.data(), N, "Bias");
    // print_vector(h_true_labels.data(), M, "True Labels");

    // Compute using GPU
    float avg_loss_gpu = compute_fused_linear_softmax_loss_gpu(
        h_input_features.data(),
        h_weights.data(),
        h_bias.data(),
        h_true_labels.data(),
        h_output_loss_per_sample_gpu.data(),
        M, K, N
    );
    std::cout << "GPU Average Loss: " << std::fixed << std::setprecision(6) << avg_loss_gpu << std::endl;
    // print_vector(h_output_loss_per_sample_gpu.data(), M, "GPU Per-Sample Loss");

    // Compute using CPU for verification
    float avg_loss_cpu = compute_fused_linear_softmax_loss_cpu(
        h_input_features.data(),
        h_weights.data(),
        h_bias.data(),
        h_true_labels.data(),
        h_output_loss_per_sample_cpu.data(),
        M, K, N
    );
    std::cout << "CPU Average Loss: " << std::fixed << std::setprecision(6) << avg_loss_cpu << std::endl;
    // print_vector(h_output_loss_per_sample_cpu.data(), M, "CPU Per-Sample Loss");

    // Compare results
    double diff_sum_sq = 0.0;
    for(int i=0; i<M; ++i) {
        double diff = h_output_loss_per_sample_gpu[i] - h_output_loss_per_sample_cpu[i];
        diff_sum_sq += diff * diff;
    }
    double mse = diff_sum_sq / M;
    std::cout << "Mean Squared Error between GPU and CPU per-sample losses: " << mse << std::endl;

    if (std::abs(avg_loss_gpu - avg_loss_cpu) < 1e-4 && mse < 1e-8) { // Adjusted tolerance
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
        std::cout << "Per-sample losses (GPU vs CPU):\n";
        for(int i=0; i<M; ++i) {
            std::cout << "Sample " << i << ": GPU=" << h_output_loss_per_sample_gpu[i] 
                      << ", CPU=" << h_output_loss_per_sample_cpu[i] << std::endl;
        }
    }

    return 0;
}
