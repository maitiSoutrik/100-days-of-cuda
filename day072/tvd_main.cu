#include "tvd_loss.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <numeric>   // For std::accumulate
#include <iomanip>   // For std::fixed, std::setprecision
#include <cuda_runtime.h>

// Helper function to normalize a vector to sum to 1 (making it a PMF)
void normalize_pmf(std::vector<float>& v) {
    if (v.empty()) return;
    double sum = 0.0;
    for (float val : v) {
        sum += val;
    }
    if (sum == 0.0) { // Avoid division by zero; if all are zero, make it uniform (or handle error)
        if (!v.empty()) {
            float val = 1.0f / v.size();
            for (size_t i = 0; i < v.size(); ++i) v[i] = val;
        }
        return;
    }
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] /= static_cast<float>(sum);
    }
}

// Helper function to print a vector
void print_vector(const std::string& name, const std::vector<float>& v, int limit = 10) {
    std::cout << name << ": [";
    for (int i = 0; i < std::min((int)v.size(), limit); ++i) {
        std::cout << v[i] << (i == std::min((int)v.size(), limit) - 1 ? "" : ", ");
    }
    if (v.size() > limit) {
        std::cout << "...";
    }
    std::cout << "]" << std::endl;
}


int main() {
    const int n = 1024 * 1024; // Size of the probability distributions
    std::vector<float> h_p(n);
    std::vector<float> h_q(n);

    // Initialize with random positive values
    std::mt19937 rng(12345); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        h_p[i] = dist(rng);
        h_q[i] = dist(rng);
    }

    // Normalize to make them valid PMFs
    normalize_pmf(h_p);
    normalize_pmf(h_q);

    // print_vector("P (host, normalized)", h_p);
    // print_vector("Q (host, normalized)", h_q);

    // --- GPU Calculation ---
    float* d_p;
    float* d_q;
    float* d_tvd_gpu_result; // To store the single float result from GPU

    CHECK_CUDA_ERROR(cudaMalloc(&d_p, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_q, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_tvd_gpu_result, sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_q, h_q.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // GPU timing
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu));

    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu));
    calculate_tvd_gpu(d_p, d_q, n, d_tvd_gpu_result);
    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu));

    float milliseconds_gpu = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds_gpu, start_gpu, stop_gpu));

    float h_tvd_gpu_result;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_tvd_gpu_result, d_tvd_gpu_result, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "TVD (GPU): " << h_tvd_gpu_result << std::endl;
    std::cout << "GPU Calculation Time: " << milliseconds_gpu << " ms" << std::endl;

    // --- CPU Calculation ---
    // CPU timing (simple, less accurate than CUDA events for GPU)
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float h_tvd_cpu_result = calculate_tvd_cpu(h_p, h_q);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);
    
    std::cout << "TVD (CPU): " << h_tvd_cpu_result << std::endl;
    std::cout << "CPU Calculation Time: " << duration_cpu.count() / 1000.0 << " ms" << std::endl;

    // --- Verification ---
    float diff = std::abs(h_tvd_gpu_result - h_tvd_cpu_result);
    std::cout << "Difference (GPU - CPU): " << diff << std::endl;
    if (diff < 1e-5) { // Tolerance for floating point differences
        std::cout << "Verification: PASS" << std::endl;
    } else {
        std::cout << "Verification: FAIL" << std::endl;
    }

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_p));
    CHECK_CUDA_ERROR(cudaFree(d_q));
    CHECK_CUDA_ERROR(cudaFree(d_tvd_gpu_result));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu));

    return 0;
}
