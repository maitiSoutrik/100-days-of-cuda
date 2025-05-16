#include "swiglu.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath> // For expf, fabsf
#include <numeric> // For std::iota
#include <algorithm> // For std::generate

// Helper function to print a matrix
void print_matrix(const std::vector<float>& matrix, int rows, int cols, const std::string& name) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// CPU implementation for sigmoid
float sigmoid_cpu(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// CPU implementation for SwiGLU forward
void swiglu_forward_cpu(const std::vector<float>& h_a,
                        const std::vector<float>& h_b,
                        std::vector<float>& h_c,
                        int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            float val_a = h_a[idx];
            float val_b = h_b[idx];
            float s_a = sigmoid_cpu(val_a);
            float silu_a = val_a * s_a;
            h_c[idx] = silu_a * val_b;
        }
    }
}

// CPU implementation for SwiGLU backward
void swiglu_backward_cpu(const std::vector<float>& h_a,
                         const std::vector<float>& h_b,
                         const std::vector<float>& h_dc,
                         std::vector<float>& h_da,
                         std::vector<float>& h_db,
                         int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            float val_a = h_a[idx];
            float val_b = h_b[idx];
            float val_dc = h_dc[idx];
            
            float s_a = sigmoid_cpu(val_a);
            
            h_da[idx] = val_dc * val_b * s_a * (1.0f + val_a * (1.0f - s_a));
            h_db[idx] = val_dc * val_a * s_a;
        }
    }
}

// Function to compare matrices
bool compare_matrices(const std::vector<float>& m1, const std::vector<float>& m2, float epsilon = 1e-4f) {
    if (m1.size() != m2.size()) return false;
    for (size_t i = 0; i < m1.size(); ++i) {
        if (fabsf(m1[i] - m2[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": " << m1[i] << " vs " << m2[i] << std::endl;
            return false;
        }
    }
    return true;
}


int main() {
    int rows = 4;
    int cols = 8; // Ensure cols is reasonable for blockDim.x
    size_t matrix_size = static_cast<size_t>(rows) * cols;
    size_t bytes = matrix_size * sizeof(float);

    // Host vectors
    std::vector<float> h_a(matrix_size);
    std::vector<float> h_b(matrix_size);
    std::vector<float> h_c_gpu(matrix_size);
    std::vector<float> h_dc(matrix_size);
    std::vector<float> h_da_gpu(matrix_size);
    std::vector<float> h_db_gpu(matrix_size);

    std::vector<float> h_c_cpu(matrix_size);
    std::vector<float> h_da_cpu(matrix_size);
    std::vector<float> h_db_cpu(matrix_size);

    // Initialize input data
    // std::iota(h_a.begin(), h_a.end(), 0.1f); // Simple sequence
    // std::iota(h_b.begin(), h_b.end(), 0.5f);
    // std::iota(h_dc.begin(), h_dc.end(), 1.0f);
    for(size_t i = 0; i < matrix_size; ++i) {
        h_a[i] = static_cast<float>(i % 10) * 0.1f - 0.5f; // Values between -0.5 and 0.4
        h_b[i] = static_cast<float>((i+5) % 10) * 0.1f + 0.1f; // Values between 0.1 and 1.0
        h_dc[i] = 1.0f; // Gradient of 1 for simplicity
    }


    print_matrix(h_a, rows, cols, "Input A (Host)");
    print_matrix(h_b, rows, cols, "Input B (Host)");
    print_matrix(h_dc, rows, cols, "Input dC (Host)");

    // Device pointers
    float *d_a, *d_b, *d_c, *d_dc, *d_da, *d_db;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dc, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_da, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_db, bytes));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dc, h_dc.data(), bytes, cudaMemcpyHostToDevice));

    // Launch forward kernel
    std::cout << "Launching SwiGLU forward kernel...\n";
    launch_swiglu_forward(d_a, d_b, d_c, rows, cols);
    
    // Launch backward kernel
    std::cout << "Launching SwiGLU backward kernel...\n";
    launch_swiglu_backward(d_a, d_b, d_dc, d_da, d_db, rows, cols);

    // Synchronize and check for errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy results from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_c_gpu.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_da_gpu.data(), d_da, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_db_gpu.data(), d_db, bytes, cudaMemcpyDeviceToHost));

    // Print GPU results
    print_matrix(h_c_gpu, rows, cols, "Output C (GPU)");
    print_matrix(h_da_gpu, rows, cols, "Output dA (GPU)");
    print_matrix(h_db_gpu, rows, cols, "Output dB (GPU)");

    // CPU computation for verification
    std::cout << "\nPerforming CPU computation for verification...\n";
    swiglu_forward_cpu(h_a, h_b, h_c_cpu, rows, cols);
    swiglu_backward_cpu(h_a, h_b, h_dc, h_da_cpu, h_db_cpu, rows, cols);

    print_matrix(h_c_cpu, rows, cols, "Output C (CPU)");
    print_matrix(h_da_cpu, rows, cols, "Output dA (CPU)");
    print_matrix(h_db_cpu, rows, cols, "Output dB (CPU)");

    // Compare results
    bool c_match = compare_matrices(h_c_gpu, h_c_cpu);
    bool da_match = compare_matrices(h_da_gpu, h_da_cpu);
    bool db_match = compare_matrices(h_db_gpu, h_db_cpu);

    std::cout << "Verification Results:\n";
    std::cout << "Forward pass (C) matches CPU: " << (c_match ? "Yes" : "No") << std::endl;
    std::cout << "Backward pass (dA) matches CPU: " << (da_match ? "Yes" : "No") << std::endl;
    std::cout << "Backward pass (dB) matches CPU: " << (db_match ? "Yes" : "No") << std::endl;

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    CHECK_CUDA_ERROR(cudaFree(d_dc));
    CHECK_CUDA_ERROR(cudaFree(d_da));
    CHECK_CUDA_ERROR(cudaFree(d_db));

    if (c_match && da_match && db_match) {
        std::cout << "\nAll computations verified successfully!\n";
        return 0;
    } else {
        std::cout << "\nComputations mismatch. Please check.\n";
        return 1;
    }
}
