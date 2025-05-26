#include "huber_loss.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip> // For std::fixed and std::setprecision

// Helper function to initialize data with random values
void initialize_data(std::vector<float>& predictions, std::vector<float>& targets, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(-5.0f, 5.0f); // Errors can be positive or negative

    predictions.resize(n);
    targets.resize(n);

    for (int i = 0; i < n; ++i) {
        predictions[i] = distrib(gen);
        targets[i] = distrib(gen);
    }
    // Introduce some outliers
    if (n > 10) {
        for(int i = 0; i < n / 20; ++i) { // 5% outliers
            int idx = std::uniform_int_distribution<int>(0, n-1)(gen);
            predictions[idx] *= 10.0f; // Make error larger
        }
    }
}

// Helper function to print a few values for verification
void print_sample_data(const char* title, const std::vector<float>& data, int count = 5) {
    std::cout << title << " (first " << count << " values): ";
    for (int i = 0; i < std::min((int)data.size(), count); ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
    }
    std::cout << std::endl;
}

// Helper function to calculate total loss/gradient sum
float sum_vector(const std::vector<float>& vec) {
    float total = 0.0f;
    for (float val : vec) {
        total += val;
    }
    return total;
}


int main() {
    const int N = 1024 * 1024 * 4; // 4 Million elements
    const float DELTA = 1.0f;      // Huber loss delta parameter

    std::vector<float> h_predictions, h_targets;
    initialize_data(h_predictions, h_targets, N);

    std::vector<float> h_loss_cpu(N);
    std::vector<float> h_gradients_cpu(N);
    std::vector<float> h_loss_gpu(N);
    std::vector<float> h_gradients_gpu(N);

    std::cout << "Running Huber Loss Calculation for N = " << N << " elements, Delta = " << DELTA << std::endl;

    // --- CPU Calculation ---
    std::cout << "\n--- CPU Calculation ---" << std::endl;
    auto start_cpu_loss = std::chrono::high_resolution_clock::now();
    huber_loss_cpu(h_predictions.data(), h_targets.data(), h_loss_cpu.data(), N, DELTA);
    auto end_cpu_loss = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_loss_duration = end_cpu_loss - start_cpu_loss;
    std::cout << "CPU Huber Loss time: " << cpu_loss_duration.count() << " ms" << std::endl;

    auto start_cpu_grad = std::chrono::high_resolution_clock::now();
    huber_loss_derivative_cpu(h_predictions.data(), h_targets.data(), h_gradients_cpu.data(), N, DELTA);
    auto end_cpu_grad = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_grad_duration = end_cpu_grad - start_cpu_grad;
    std::cout << "CPU Huber Loss Derivative time: " << cpu_grad_duration.count() << " ms" << std::endl;

    // --- GPU Calculation ---
    std::cout << "\n--- GPU Calculation ---" << std::endl;
    auto start_gpu_loss = std::chrono::high_resolution_clock::now();
    compute_huber_loss_gpu(h_predictions.data(), h_targets.data(), h_loss_gpu.data(), N, DELTA);
    auto end_gpu_loss = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_loss_duration = end_gpu_loss - start_gpu_loss;
    std::cout << "GPU Huber Loss time: " << gpu_loss_duration.count() << " ms" << std::endl;

    auto start_gpu_grad = std::chrono::high_resolution_clock::now();
    compute_huber_loss_derivative_gpu(h_predictions.data(), h_targets.data(), h_gradients_gpu.data(), N, DELTA);
    auto end_gpu_grad = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_grad_duration = end_gpu_grad - start_gpu_grad;
    std::cout << "GPU Huber Loss Derivative time: " << gpu_grad_duration.count() << " ms" << std::endl;

    // --- Verification (compare a few values and sum) ---
    std::cout << "\n--- Verification ---" << std::endl;
    print_sample_data("Predictions", h_predictions);
    print_sample_data("Targets    ", h_targets);
    
    std::cout << "\nLoss:" << std::endl;
    print_sample_data("CPU Loss   ", h_loss_cpu);
    print_sample_data("GPU Loss   ", h_loss_gpu);

    std::cout << "\nGradients:" << std::endl;
    print_sample_data("CPU Grads  ", h_gradients_cpu);
    print_sample_data("GPU Grads  ", h_gradients_gpu);

    float sum_loss_cpu = sum_vector(h_loss_cpu);
    float sum_loss_gpu = sum_vector(h_loss_gpu);
    float sum_grads_cpu = sum_vector(h_gradients_cpu);
    float sum_grads_gpu = sum_vector(h_gradients_gpu);

    std::cout << "\nTotal Loss (CPU): " << std::fixed << std::setprecision(4) << sum_loss_cpu << std::endl;
    std::cout << "Total Loss (GPU): " << std::fixed << std::setprecision(4) << sum_loss_gpu << std::endl;
    std::cout << "Total Gradients (CPU): " << std::fixed << std::setprecision(4) << sum_grads_cpu << std::endl;
    std::cout << "Total Gradients (GPU): " << std::fixed << std::setprecision(4) << sum_grads_gpu << std::endl;

    // A simple check for correctness
    float loss_diff_threshold = 1e-3 * N; // Allow small aggregate difference
    float grad_diff_threshold = 1e-3 * N;

    if (std::fabs(sum_loss_cpu - sum_loss_gpu) < loss_diff_threshold) {
        std::cout << "\nLoss results VERIFIED (sum comparison)." << std::endl;
    } else {
        std::cout << "\nLoss results MISMATCH (sum comparison)." << std::endl;
        std::cout << "Difference: " << std::fabs(sum_loss_cpu - sum_loss_gpu) << std::endl;
    }

    if (std::fabs(sum_grads_cpu - sum_grads_gpu) < grad_diff_threshold) {
        std::cout << "Gradient results VERIFIED (sum comparison)." << std::endl;
    } else {
        std::cout << "Gradient results MISMATCH (sum comparison)." << std::endl;
        std::cout << "Difference: " << std::fabs(sum_grads_cpu - sum_grads_gpu) << std::endl;
    }
    
    std::cout << "\nSpeedup (Loss): " << cpu_loss_duration.count() / gpu_loss_duration.count() << "x" << std::endl;
    std::cout << "Speedup (Gradient): " << cpu_grad_duration.count() / gpu_grad_duration.count() << "x" << std::endl;

    return 0;
}
