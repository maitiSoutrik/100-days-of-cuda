#include <iostream>
#include <vector>
#include <chrono>
#include <random> // For initializing matrices
#include <iomanip> // For std::fixed and std::setprecision

#include "../include/swish_matrix_ops.cuh" // Includes cuda_runtime.h and CHECK_CUDA_ERROR

// Helper function to print a matrix (or a portion of it)
void print_matrix_sample(const float* matrix, int rows, int cols, int sample_rows, int sample_cols, const std::string& label) {
    std::cout << "\n" << label << " (first " << sample_rows << "x" << sample_cols << " sample):" << std::endl;
    for (int i = 0; i < std::min(rows, sample_rows); ++i) {
        for (int j = 0; j < std::min(cols, sample_cols); ++j) {
            std::cout << std::fixed << std::setprecision(4) << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

// Helper function to initialize a matrix with random values
void initialize_matrix(float* matrix, int num_elements) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-1.0, 1.0);
    for (int i = 0; i < num_elements; ++i) {
        matrix[i] = static_cast<float>(distrib(gen));
    }
}

int main() {
    // Matrix dimensions
    const int M = 256; // Rows of A and C
    const int N = 256; // Columns of B and C
    const int K = 256; // Columns of A and Rows of B

    // Swish and scaling parameters
    const float scale_factor = 1.0f;
    const float beta_swish = 1.0f; // Beta for Swish: x * sigmoid(beta * x)

    std::cout << "Day 80: Matrix Multiplication with Swish Activation and Scaling" << std::endl;
    std::cout << "Matrix dimensions: A(" << M << "x" << K << "), B(" << K << "x" << N << "), C(" << M << "x" << N << ")" << std::endl;
    std::cout << "Scale factor: " << scale_factor << ", Swish beta: " << beta_swish << std::endl;

    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    // Initialize host matrices
    initialize_matrix(h_A.data(), M * K);
    initialize_matrix(h_B.data(), K * N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Define CUDA execution configuration
    dim3 threadsPerBlock(16, 16); // As used in the kernel's shared memory tiling

    // Warm-up run (optional, but good for more stable timing)
    CHECK_CUDA_ERROR(matrix_mul_swish_scale(d_A, d_B, d_C, M, N, K, scale_factor, beta_swish, threadsPerBlock));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    // Benchmark the kernel execution
    auto start_time = std::chrono::high_resolution_clock::now();

    CHECK_CUDA_ERROR(matrix_mul_swish_scale(d_A, d_B, d_C, M, N, K, scale_factor, beta_swish, threadsPerBlock));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for the kernel to finish

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

    std::cout << "\nCUDA Kernel Execution Time: " << duration_ms << " ms" << std::endl;

    // Copy result from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print a sample of the resulting matrix C
    print_matrix_sample(h_C.data(), M, N, 5, 5, "Result Matrix C");

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    std::cout << "\nBenchmark completed successfully." << std::endl;

    return 0;
}
