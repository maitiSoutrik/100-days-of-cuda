#include "batched_l2_norm.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip> // For std::fixed, std::setprecision
#include <chrono>  // For timing

// Helper function to print a vector
template<typename T>
void print_vector(const std::vector<T>& vec, const std::string& name, int display_count = 10) {
    std::cout << name << " (first " << std::min((int)vec.size(), display_count) << " elements):" << std::endl;
    for (size_t i = 0; i < std::min(vec.size(), (size_t)display_count); ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i] << " ";
    }
    std::cout << std::endl;
}

// Helper function to compare results
bool compare_results(const std::vector<float>& gpu_norms, const std::vector<float>& cpu_norms, float epsilon = 1e-4f) {
    if (gpu_norms.size() != cpu_norms.size()) {
        std::cerr << "Error: GPU and CPU norm vectors have different sizes!" << std::endl;
        return false;
    }
    for (size_t i = 0; i < gpu_norms.size(); ++i) {
        if (fabsf(gpu_norms[i] - cpu_norms[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": GPU_norm = " << gpu_norms[i]
                      << ", CPU_norm = " << cpu_norms[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Configuration
    const int num_batches = 1024;    // Number of vectors
    const int vector_dim = 512;     // Dimension of each vector

    std::cout << "Batched L2 Norm Calculation" << std::endl;
    std::cout << "Number of batches: " << num_batches << std::endl;
    std::cout << "Vector dimension: " << vector_dim << std::endl;
    std::cout << "------------------------------------" << std::endl;

    // Initialize host data
    std::vector<float> h_vectors(num_batches * vector_dim);
    std::vector<float> h_gpu_norms(num_batches);
    std::vector<float> h_cpu_norms(num_batches);

    // Fill input vectors with random data
    std::mt19937 rng(12345); // Mersenne Twister random number generator
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < h_vectors.size(); ++i) {
        h_vectors[i] = dist(rng);
    }

    // --- GPU Computation ---
    std::cout << "\nRunning GPU computation..." << std::endl;
    auto start_gpu = std::chrono::high_resolution_clock::now();
    compute_batched_l2_norm_gpu(h_vectors.data(), h_gpu_norms.data(), num_batches, vector_dim);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU computation finished." << std::endl;
    print_vector(h_gpu_norms, "GPU Norms");
    std::cout << "GPU Time: " << gpu_duration.count() << " ms" << std::endl;

    // --- CPU Computation (for verification) ---
    std::cout << "\nRunning CPU computation for verification..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    compute_batched_l2_norm_cpu(h_vectors.data(), h_cpu_norms.data(), num_batches, vector_dim);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU computation finished." << std::endl;
    print_vector(h_cpu_norms, "CPU Norms");
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;

    // --- Verification ---
    std::cout << "\nVerifying results..." << std::endl;
    bool success = compare_results(h_gpu_norms, h_cpu_norms);
    if (success) {
        std::cout << "Verification PASSED: GPU and CPU results match." << std::endl;
    } else {
        std::cout << "Verification FAILED: GPU and CPU results differ." << std::endl;
    }
    
    std::cout << "------------------------------------" << std::endl;
    if (gpu_duration.count() > 0 && cpu_duration.count() > 0) {
        std::cout << "Speedup (CPU Time / GPU Time): " << cpu_duration.count() / gpu_duration.count() << "x" << std::endl;
    }

    return 0;
}
