#include "mse.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip> // For std::fixed and std::setprecision
#include <cmath>   // For std::fabs

// Helper function to generate random float data
void generate_data(std::vector<float>& data, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);
    data.resize(N);
    for (int i = 0; i < N; ++i) {
        data[i] = distrib(gen);
    }
}

int main() {
    // Define the size of the vectors
    // Let's choose a size large enough to see GPU benefits.
    // For example, 2^24 elements (around 16 million)
    const int N = 1 << 24; // 16,777,216 elements

    std::cout << "Mean Squared Error (MSE) Calculation" << std::endl;
    std::cout << "Number of elements (N): " << N << std::endl;
    std::cout << "------------------------------------" << std::endl;

    // Host vectors
    std::vector<float> h_predictions;
    std::vector<float> h_targets;

    // Generate synthetic data
    std::cout << "Generating synthetic data..." << std::endl;
    generate_data(h_predictions, N);
    generate_data(h_targets, N);
    std::cout << "Data generation complete." << std::endl;

    float mse_cpu_result = 0.0f;
    float mse_gpu_result = 0.0f;

    // --- CPU MSE Calculation ---
    std::cout << "\nCalculating MSE on CPU..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    mse_cpu_result = mse_cpu(h_predictions.data(), h_targets.data(), N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU MSE Result: " << std::fixed << std::setprecision(8) << mse_cpu_result << std::endl;
    std::cout << "CPU Execution Time: " << cpu_duration.count() << " ms" << std::endl;

    // --- GPU MSE Calculation ---
    std::cout << "\nCalculating MSE on GPU..." << std::endl;
    // Warm-up GPU (optional, but good for more stable timing)
    mse_gpu(h_predictions.data(), h_targets.data(), 1024, &mse_gpu_result); 

    auto start_gpu = std::chrono::high_resolution_clock::now();
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));

    CHECK_CUDA_ERROR(cudaEventRecord(start_event));
    mse_gpu(h_predictions.data(), h_targets.data(), N, &mse_gpu_result);
    CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));

    float gpu_event_time_ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_event_time_ms, start_event, stop_event));
    
    auto end_gpu_chrono = std::chrono::high_resolution_clock::now(); // For overall host-side timing
    std::chrono::duration<double, std::milli> gpu_duration_chrono = end_gpu_chrono - start_gpu;


    std::cout << "GPU MSE Result: " << std::fixed << std::setprecision(8) << mse_gpu_result << std::endl;
    std::cout << "GPU Execution Time (cudaEvent): " << gpu_event_time_ms << " ms" << std::endl;
    std::cout << "GPU Execution Time (chrono, incl. overhead): " << gpu_duration_chrono.count() << " ms" << std::endl;
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));


    // --- Verification ---
    std::cout << "\n--- Verification ---" << std::endl;
    float tolerance = 1e-5f; // Tolerance for floating point comparison
    if (std::fabs(mse_cpu_result - mse_gpu_result) < tolerance) {
        std::cout << "SUCCESS: CPU and GPU results are close." << std::endl;
    } else {
        std::cout << "FAILURE: CPU and GPU results differ significantly." << std::endl;
        std::cout << "CPU: " << mse_cpu_result << ", GPU: " << mse_gpu_result << std::endl;
    }
    std::cout << "Difference: " << std::fabs(mse_cpu_result - mse_gpu_result) << std::endl;

    // --- Performance Comparison ---
    std::cout << "\n--- Performance Comparison ---" << std::endl;
    if (gpu_event_time_ms > 0 && cpu_duration.count() > 0) {
         double speedup = cpu_duration.count() / gpu_event_time_ms;
         std::cout << "Speedup (CPU Time / GPU Event Time): " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    } else {
        std::cout << "Could not calculate speedup due to zero or invalid timing." << std::endl;
    }
    
    std::cout << "\nExecution finished." << std::endl;

    return 0;
}
