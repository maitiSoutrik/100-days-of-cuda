#include "softplus_activation.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm> // For std::generate
#include <iomanip>   // For std::fixed and std::setprecision
#include <chrono>    // For timing

// Helper function to initialize data with random values
void initializeData(std::vector<float>& data, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // Range for random numbers, e.g., -10.0 to 10.0 for interesting Softplus behavior
    std::uniform_real_distribution<float> distrib(-10.0f, 10.0f);
    std::generate(data.begin(), data.end(), [&]() { return distrib(gen); });
}

// Helper function to compare results
bool verifyResults(const std::vector<float>& cpu_result, const std::vector<float>& gpu_result, float epsilon = 1e-5f) {
    if (cpu_result.size() != gpu_result.size()) {
        std::cerr << "Size mismatch between CPU and GPU results!" << std::endl;
        return false;
    }
    for (size_t i = 0; i < cpu_result.size(); ++i) {
        if (std::abs(cpu_result[i] - gpu_result[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": CPU=" << cpu_result[i]
                      << ", GPU=" << gpu_result[i] << ", Diff=" << std::abs(cpu_result[i] - gpu_result[i])
                      << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 1 << 20; // 2^20 elements, approximately 1 million
    const size_t bytes = N * sizeof(float);

    std::cout << "Softplus Activation Function Demo" << std::endl;
    std::cout << "Number of elements: " << N << std::endl;
    std::cout << "Data size: " << bytes / (1024.0 * 1024.0) << " MB" << std::endl;

    // Host data vectors
    std::vector<float> h_input(N);
    std::vector<float> h_output_cpu(N);
    std::vector<float> h_output_gpu(N);

    // Initialize input data
    initializeData(h_input, N);

    // Device pointers
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, bytes));

    // Transfer input data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // --- GPU Execution and Timing ---
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu));

    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu));
    softplusActivation(d_input, d_output, N); // GPU execution
    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu));

    float milliseconds_gpu = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds_gpu, start_gpu, stop_gpu));
    std::cout << "GPU Execution Time: " << milliseconds_gpu << " ms" << std::endl;

    // Transfer output data from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    // --- CPU Execution and Timing ---
    auto start_cpu_time = std::chrono::high_resolution_clock::now();
    softplusActivationCPU(h_input.data(), h_output_cpu.data(), N); // CPU execution
    auto stop_cpu_time = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu_time - start_cpu_time);
    std::cout << "CPU Execution Time: " << duration_cpu.count() / 1000.0 << " ms" << std::endl;

    // Verify results
    bool success = verifyResults(h_output_cpu, h_output_gpu);
    if (success) {
        std::cout << "Verification successful: CPU and GPU results match." << std::endl;
    } else {
        std::cout << "Verification FAILED: CPU and GPU results differ." << std::endl;
    }

    // Print a few example values
    std::cout << "\nExample values (Input -> CPU Output | GPU Output):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < std::min(N, 10); ++i) {
        std::cout << "Input: " << std::setw(10) << h_input[i]
                  << " -> CPU: " << std::setw(10) << h_output_cpu[i]
                  << " | GPU: " << std::setw(10) << h_output_gpu[i] << std::endl;
    }

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu));

    return success ? 0 : 1;
}