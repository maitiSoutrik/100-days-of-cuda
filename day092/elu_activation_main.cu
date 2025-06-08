#include "elu_activation.cuh"
#include <vector>
#include <random>
#include <algorithm> // For std::generate, std::abs
#include <chrono>
#include <iomanip>   // For std::fixed, std::setprecision
#include <iostream>  // For std::cout, std::endl, std::cerr

// Helper function to print a small portion of a vector for quick inspection
void print_sample_data(const std::string& name, const std::vector<float>& data, int count = 5) {
    std::cout << name << " (first " << count << " elements): [";
    for (int i = 0; i < std::min((int)data.size(), count); ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i]
                  << (i == std::min((int)data.size(), count) - 1 ? "" : ", ");
    }
    if (data.size() > count) {
        std::cout << "...";
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char** argv) {
    const int N_DEFAULT = 1024 * 1024 * 16; // 16 Million elements
    const float ALPHA_DEFAULT = 1.0f;

    int n = N_DEFAULT;
    float alpha = ALPHA_DEFAULT;

    if (argc > 1) {
        n = std::atoi(argv[1]);
        if (n <= 0) {
            std::cerr << "Number of elements must be positive. Using default: " << N_DEFAULT << std::endl;
            n = N_DEFAULT;
        }
    }
    if (argc > 2) {
        alpha = std::atof(argv[2]);
        if (alpha <= 0) {
            std::cerr << "Alpha must be positive. Using default: " << ALPHA_DEFAULT << std::endl;
            alpha = ALPHA_DEFAULT;
        }
    }

    std::cout << "Day 92: ELU Activation Function Benchmark" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Number of elements (N): " << n << std::endl;
    std::cout << "Alpha (α): " << alpha << std::endl;

    // Host vectors
    std::vector<float> h_input(n);
    std::vector<float> h_output_cpu(n);
    std::vector<float> h_output_gpu(n);

    // Initialize input data with random values between -5.0 and 5.0
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::generate(h_input.begin(), h_input.end(), [&]() { return dist(rng); });

    print_sample_data("Sample Input Data", h_input);

    // CPU Execution
    auto start_cpu = std::chrono::high_resolution_clock::now();
    elu_activation_cpu(h_input.data(), h_output_cpu.data(), n, alpha);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "\nCPU ELU execution time: " << std::fixed << std::setprecision(3) << cpu_duration.count() << " ms" << std::endl;
    print_sample_data("Sample CPU Output", h_output_cpu);

    // GPU Execution (with warm-up)
    // Warm-up call to ensure fair timing (e.g., handle JIT compilation, context creation)
    elu_activation_gpu(h_input.data(), h_output_gpu.data(), n, alpha);
    
    auto start_gpu = std::chrono::high_resolution_clock::now();
    elu_activation_gpu(h_input.data(), h_output_gpu.data(), n, alpha);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU ELU execution time: " << std::fixed << std::setprecision(3) << gpu_duration.count() << " ms" << std::endl;
    print_sample_data("Sample GPU Output", h_output_gpu);

    // Verification
    float max_error = 0.0f;
    int errors = 0;
    const float tolerance = 1e-5f; // Tolerance for floating point comparisons

    for (int i = 0; i < n; ++i) {
        float error = std::abs(h_output_cpu[i] - h_output_gpu[i]);
        if (error > tolerance) {
            errors++;
            if (error > max_error) {
                max_error = error;
            }
            if (errors <= 5) { // Print first few errors
                 std::cerr << "Mismatch at index " << i << ": Input=" << h_input[i]
                           << ", CPU=" << h_output_cpu[i] 
                           << ", GPU=" << h_output_gpu[i] 
                           << ", Diff=" << error << std::endl;
            }
        }
    }

    std::cout << "\nVerification Results:" << std::endl;
    if (errors == 0) {
        std::cout << "  PASSED: CPU and GPU results match within tolerance (" << tolerance << ")." << std::endl;
    } else {
        std::cout << "  FAILED: CPU and GPU results differ." << std::endl;
        std::cout << "  Number of mismatches: " << errors << " out of " << n << std::endl;
        std::cout << "  Maximum difference: " << max_error << std::endl;
    }

    // Performance Comparison
    if (cpu_duration.count() > 0 && gpu_duration.count() > 0) {
        std::cout << "\nGPU Speedup over CPU: " << std::fixed << std::setprecision(2)
                  << cpu_duration.count() / gpu_duration.count() << "x" << std::endl;
    }
    std::cout << "-----------------------------------------" << std::endl;

    return (errors == 0) ? 0 : 1;
}
