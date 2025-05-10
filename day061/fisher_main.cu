#include "fisher_matrix.cuh"
#include <iostream>
#include <vector>
#include <iomanip> // For std::setw
#include <cstdlib> // For rand, RAND_MAX, malloc, free
#include <cmath>   // For fabs
#include <cassert> // For assert
#include <ctime>   // For clock

void print_matrix(const float* matrix, int rows, int cols, const std::string& label) {
    std::cout << "\n" << label << ":" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << std::setw(10) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Test function integrated into main for simplicity here
void test_and_benchmark_fisher(int n_samples, int n_params) {
    std::cout << "--- Testing and Benchmarking for N_SAMPLES = " << n_samples 
              << ", N_PARAMS = " << n_params << " ---" << std::endl;

    // Allocate host memory
    std::vector<float> h_log_probs(static_cast<size_t>(n_samples) * n_params);
    std::vector<float> h_fisher_cpu(static_cast<size_t>(n_params) * n_params);
    std::vector<float> h_fisher_gpu(static_cast<size_t>(n_params) * n_params);
    
    // Initialize random data for log_probs (scores)
    for (size_t i = 0; i < static_cast<size_t>(n_samples) * n_params; i++) {
        h_log_probs[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Scores between -1 and 1
    }
    
    // Compute on CPU
    clock_t start_cpu = clock();
    compute_fisher_matrix_cpu(h_log_probs.data(), h_fisher_cpu.data(), n_samples, n_params);
    double cpu_time_ms = (static_cast<double>(clock() - start_cpu) / CLOCKS_PER_SEC) * 1000.0;
    
    // Compute on GPU
    clock_t start_gpu = clock();
    compute_fisher_matrix_gpu(h_log_probs.data(), h_fisher_gpu.data(), n_samples, n_params);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure GPU computation is finished
    double gpu_time_ms = (static_cast<double>(clock() - start_gpu) / CLOCKS_PER_SEC) * 1000.0;
    
    // Verify results
    float max_diff = 0.0f;
    for (size_t i = 0; i < static_cast<size_t>(n_params) * n_params; i++) {
        float diff = std::fabs(h_fisher_cpu[i] - h_fisher_gpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    printf("Verification: Max difference CPU vs GPU = %.6f\n", max_diff);
    assert(max_diff < 1e-5); // Adjust tolerance if necessary

    if (n_params <= 8) { // Print matrices only for small sizes
        print_matrix(h_fisher_cpu.data(), n_params, n_params, "Fisher Matrix (CPU)");
        print_matrix(h_fisher_gpu.data(), n_params, n_params, "Fisher Matrix (GPU)");
    }

    printf("Benchmark: CPU Time = %.2f ms, GPU Time = %.2f ms\n", cpu_time_ms, gpu_time_ms);
    if (gpu_time_ms > 0.001 && cpu_time_ms > 0.001) { // Avoid division by zero or tiny numbers
        printf("Speedup (CPU/GPU) = %.2fx\n", cpu_time_ms / gpu_time_ms);
    } else {
        printf("Speedup: N/A (times too small for meaningful comparison)\n");
    }
    std::cout << "----------------------------------------------------------" << std::endl;
}

int main() {
    std::cout << "Day 61: Fisher Information Matrix - Main Program" << std::endl;
    srand(123); // Seed for reproducibility

    // Test cases (adjust sizes as needed for quick tests vs. thorough benchmarks)
    test_and_benchmark_fisher(1000, 16);   // Small
    test_and_benchmark_fisher(10000, 32);  // Medium
    test_and_benchmark_fisher(50000, 64); // Larger

    // Benchmark cases from user example (can be large)
    // test_and_benchmark_fisher(1000, 64); // Already covered by larger sample test
    // test_and_benchmark_fisher(10000, 128);
    // test_and_benchmark_fisher(100000, 256); // This might be slow for CPU

    std::cout << "\nAll tests and benchmarks completed." << std::endl;
    return 0;
}
