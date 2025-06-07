#include "frobenius_norm.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip> // For std::fixed and std::setprecision
#include <chrono>  // For timing

// Helper function to initialize a matrix with random values
void initializeMatrix(float* matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-10.0, 10.0);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(distrib(gen));
    }
}

// Helper function to print a matrix (optional, for debugging small matrices)
void printMatrix(const float* matrix, int rows, int cols, const std::string& name) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    if (rows * cols == 0) {
        std::cout << " (empty matrix)\n";
        return;
    }
    if (rows * cols > 100) { // Don't print large matrices
        std::cout << " (matrix too large to print)\n";
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(2) << matrix[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    int rows = 1024;
    int cols = 1024;
    int total_elements = rows * cols;
    size_t matrix_size_bytes = total_elements * sizeof(float);

    // Allocate host memory
    float* h_matrix = (float*)malloc(matrix_size_bytes);
    if (h_matrix == NULL) {
        std::cerr << "Failed to allocate host memory for matrix." << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize matrix
    initializeMatrix(h_matrix, rows, cols);
    // printMatrix(h_matrix, rows, cols, "Input Matrix"); // Optional: print for small matrices

    // Allocate device memory
    float* d_matrix;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix, matrix_size_bytes));

    // Copy matrix from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix, matrix_size_bytes, cudaMemcpyHostToDevice));

    // --- GPU Calculation ---
    float norm_gpu;
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu));

    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu));
    norm_gpu = frobeniusNormGPU(d_matrix, rows, cols);
    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu));

    float milliseconds_gpu = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds_gpu, start_gpu, stop_gpu));

    std::cout << "Frobenius Norm (GPU): " << std::fixed << std::setprecision(6) << norm_gpu << std::endl;
    std::cout << "GPU Calculation Time: " << milliseconds_gpu << " ms" << std::endl;

    // --- CPU Calculation ---
    float norm_cpu;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    norm_cpu = frobeniusNormCPU(h_matrix, rows, cols);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu - start_cpu);

    std::cout << "Frobenius Norm (CPU): " << std::fixed << std::setprecision(6) << norm_cpu << std::endl;
    std::cout << "CPU Calculation Time: " << duration_cpu.count() / 1000.0 << " ms" << std::endl;

    // --- Verification ---
    float tolerance = 1e-3; // Tolerance for floating point comparison
    if (fabs(norm_gpu - norm_cpu) < tolerance) {
        std::cout << "Verification: PASSED" << std::endl;
    } else {
        std::cout << "Verification: FAILED" << std::endl;
        std::cout << "Difference: " << fabs(norm_gpu - norm_cpu) << std::endl;
    }

    // Clean up
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    free(h_matrix);
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu));

    return EXIT_SUCCESS;
}
