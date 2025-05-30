#include "jsd_loss.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip> // For std::fixed and std::setprecision
#include <chrono>  // For CPU timing

// Helper function to initialize probability distributions (sum to 1 per row)
void initialize_distributions(std::vector<float>& P, std::vector<float>& Q, int num_distributions, int num_elements) {
    std::mt19937 rng(12345); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    P.resize(num_distributions * num_elements);
    Q.resize(num_distributions * num_elements);

    for (int i = 0; i < num_distributions; ++i) {
        float p_sum = 0.0f;
        float q_sum = 0.0f;
        for (int j = 0; j < num_elements; ++j) {
            P[i * num_elements + j] = dist(rng);
            Q[i * num_elements + j] = dist(rng);
            p_sum += P[i * num_elements + j];
            q_sum += Q[i * num_elements + j];
        }
        // Normalize each row to sum to 1
        for (int j = 0; j < num_elements; ++j) {
            if (p_sum > 0) P[i * num_elements + j] /= p_sum;
            else P[i * num_elements + j] = 1.0f / num_elements; // Uniform if sum is zero
            
            if (q_sum > 0) Q[i * num_elements + j] /= q_sum;
            else Q[i * num_elements + j] = 1.0f / num_elements; // Uniform if sum is zero
        }
    }
}

void print_matrix(const std::string& name, const std::vector<float>& matrix, int rows, int cols, int print_limit = 5) {
    std::cout << name << " (first " << std::min(rows, print_limit) << "x" << std::min(cols, print_limit) << " elements):\n";
    for (int i = 0; i < std::min(rows, print_limit); ++i) {
        for (int j = 0; j < std::min(cols, print_limit); ++j) {
            std::cout << std::fixed << std::setprecision(4) << matrix[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}


int main() {
    int num_distributions = 1024; // e.g., batch size
    int num_elements = 512;    // e.g., vocabulary size or feature dimension
    float epsilon = 1e-8f;

    std::vector<float> h_P, h_Q;
    initialize_distributions(h_P, h_Q, num_distributions, num_elements);

    // print_matrix("P_host", h_P, num_distributions, num_elements);
    // print_matrix("Q_host", h_Q, num_distributions, num_elements);

    float *d_P, *d_Q, *d_loss_gpu, *d_grad_P, *d_grad_Q;
    size_t matrix_size_bytes = num_distributions * num_elements * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_P, matrix_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_Q, matrix_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_loss_gpu, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_P, matrix_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grad_Q, matrix_size_bytes));

    CHECK_CUDA_ERROR(cudaMemcpy(d_P, h_P.data(), matrix_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Q, h_Q.data(), matrix_size_bytes, cudaMemcpyHostToDevice));

    std::vector<float> betas = {0.0f, 0.5f, 1.0f}; // Forward KL, Symmetric JSD, Reverse KL (interpretation based on problem desc)
                                                // Beta = 0.0 for D_KL(Q || M)
                                                // Beta = 1.0 for D_KL(P || M)
                                                // Beta = 0.5 for symmetric JSD

    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu));
    float gpu_time_ms = 0;

    std::cout << "Running JSD Loss Computations:\n";
    std::cout << "Num Distributions: " << num_distributions << ", Num Elements per Distribution: " << num_elements << std::endl;
    std::cout << "-----------------------------------------------------------\n";
    std::cout << std::setw(10) << "Beta" << std::setw(20) << "GPU Loss" << std::setw(20) << "CPU Loss (Fwd)"
              << std::setw(15) << "GPU Time (ms)" << std::setw(15) << "CPU Time (ms)" << std::endl;
    std::cout << "-----------------------------------------------------------\n";


    for (float beta : betas) {
        // GPU Computation
        CHECK_CUDA_ERROR(cudaMemset(d_grad_P, 0, matrix_size_bytes)); // Zero out gradients
        CHECK_CUDA_ERROR(cudaMemset(d_grad_Q, 0, matrix_size_bytes));
        CHECK_CUDA_ERROR(cudaEventRecord(start_gpu));
        jsd_loss_gpu(d_P, d_Q, d_loss_gpu, d_grad_P, d_grad_Q, num_distributions, num_elements, beta, epsilon);
        CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu));

        float h_loss_gpu;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_loss_gpu, d_loss_gpu, sizeof(float), cudaMemcpyDeviceToHost));
        
        // Optionally, copy gradients back and print/verify a few values
        // std::vector<float> h_grad_P(num_distributions * num_elements);
        // CHECK_CUDA_ERROR(cudaMemcpy(h_grad_P.data(), d_grad_P, matrix_size_bytes, cudaMemcpyDeviceToHost));
        // print_matrix("Grad_P_GPU (beta=" + std::to_string(beta) + ")", h_grad_P, num_distributions, num_elements, 3);


        // CPU Computation (Forward Pass Only for timing comparison)
        auto start_cpu = std::chrono::high_resolution_clock::now();
        float h_loss_cpu = jsd_loss_forward_cpu(h_P, h_Q, num_distributions, num_elements, beta, epsilon);
        auto stop_cpu = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);
        float cpu_time_ms = cpu_duration.count() / 1000.0f;

        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(10) << beta
                  << std::setw(20) << h_loss_gpu
                  << std::setw(20) << h_loss_cpu
                  << std::setw(15) << gpu_time_ms
                  << std::setw(15) << cpu_time_ms
                  << std::endl;
    }
    std::cout << "-----------------------------------------------------------\n";

    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu));
    CHECK_CUDA_ERROR(cudaFree(d_P));
    CHECK_CUDA_ERROR(cudaFree(d_Q));
    CHECK_CUDA_ERROR(cudaFree(d_loss_gpu));
    CHECK_CUDA_ERROR(cudaFree(d_grad_P));
    CHECK_CUDA_ERROR(cudaFree(d_grad_Q));

    return 0;
}
