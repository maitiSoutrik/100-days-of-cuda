#include "cumulative_product.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip> // For std::fixed, std::setprecision
#include <chrono>  // For timing
#include <numeric> // For std::iota (if needed for specific data patterns)
#include <algorithm> // For std::generate, std::copy

// Helper function to print arrays
template <typename T>
void print_array(const T* arr, int n, const std::string& label) {
    std::cout << label << ": [";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << (i == n - 1 ? "" : ", ");
        if (i > 10 && i < n - 5) { // Print ellipsis for long arrays
            std::cout << "..., ";
            i = n - 6;
        }
    }
    std::cout << "]" << std::endl;
}

// Helper function to compare arrays
bool compare_arrays(const float* arr1, const float* arr2, int n, float epsilon = 1e-5f) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs(arr1[i] - arr2[i]) > epsilon * std::max(1.0f, std::fabs(arr1[i]))) {
             // Consider relative error for larger numbers, absolute for smaller
            if (std::fabs(arr1[i] - arr2[i]) > epsilon) { // Fallback to absolute for very small numbers
                std::cerr << "Mismatch at index " << i << ": arr1 = " << arr1[i] << ", arr2 = " << arr2[i] << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    srand(time(0)); // Seed for random number generation

    int n = 256; // Array size, keep it within single block limits for this example (e.g., <= 512 for 256 threads)
                 // Max for current kernel: 2 * threads_per_block (e.g. 2 * 256 = 512)

    std::cout << "--- Day 084: Cumulative Product (Prefix Product / Scan) ---" << std::endl;
    std::cout << "Array size: " << n << std::endl;

    // --- Prepare Host Data ---
    std::vector<float> h_input_data(n);
    std::vector<float> h_cpu_output(n);
    std::vector<float> h_gpu_output(n);

    // Initialize with small positive random numbers to avoid quick underflow/overflow
    // and to make products interesting. Values between 0.8 and 1.2 for example.
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(0.9f, 1.1f); // Small values around 1
    // Or simpler:
    // for (int i = 0; i < n; ++i) {
    //    h_input_data[i] = 1.0f + (static_cast<float>(rand()) / RAND_MAX) * 0.2f - 0.1f; // Values around 1.0
    // }
    std::generate(h_input_data.begin(), h_input_data.end(), [&]() { return dist(rng); });


    // Copy input data for CPU and GPU versions
    std::copy(h_input_data.begin(), h_input_data.end(), h_cpu_output.begin());
    std::copy(h_input_data.begin(), h_input_data.end(), h_gpu_output.begin());

    if (n <= 16) { // Print input only for small arrays
        print_array(h_input_data.data(), n, "Input Data");
    }

    // --- CPU Execution & Timing ---
    auto start_cpu = std::chrono::high_resolution_clock::now();
    inclusive_scan_cpu(h_cpu_output.data(), n);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "\nCPU Execution Time: " << std::fixed << std::setprecision(3) << cpu_duration.count() << " ms" << std::endl;
    if (n <= 16) {
        print_array(h_cpu_output.data(), n, "CPU Output");
    }

    // --- GPU Execution & Timing ---
    float* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_gpu_output.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    auto start_gpu = std::chrono::high_resolution_clock::now();
    inclusive_scan_gpu(d_data, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure kernel completion for accurate timing
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;

    CHECK_CUDA_ERROR(cudaMemcpy(h_gpu_output.data(), d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_data));

    std::cout << "GPU Execution Time: " << std::fixed << std::setprecision(3) << gpu_duration.count() << " ms" << std::endl;
    if (n <= 16) {
        print_array(h_gpu_output.data(), n, "GPU Output");
    }

    // --- Verification ---
    bool success = compare_arrays(h_cpu_output.data(), h_gpu_output.data(), n);
    if (success) {
        std::cout << "\nVerification: SUCCESS! CPU and GPU results match." << std::endl;
    } else {
        std::cout << "\nVerification: FAILED! CPU and GPU results differ." << std::endl;
        // For debugging, print both arrays if they are small enough
        if (n <= 32) {
             std::cout << "CPU reference:" << std::endl;
             print_array(h_cpu_output.data(), n, "CPU Output (Mismatch)");
             std::cout << "GPU result:" << std::endl;
             print_array(h_gpu_output.data(), n, "GPU Output (Mismatch)");
        }
    }
    
    if (cpu_duration.count() > 0 && gpu_duration.count() > 0) {
        std::cout << "Speedup (CPU Time / GPU Time): " << std::fixed << std::setprecision(2) 
                  << cpu_duration.count() / gpu_duration.count() << "x" << std::endl;
    }


    std::cout << "\nNote: The current GPU kernel is a single-block implementation." << std::endl;
    std::cout << "It's primarily for demonstrating the scan logic within a block." << std::endl;
    std::cout << "For larger arrays, a multi-block scan algorithm would be necessary for correctness and performance." << std::endl;

    return success ? 0 : 1; // Return 0 on success, 1 on failure for CTest
}
