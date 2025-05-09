#include "muon_optimization.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip> // For std::fixed and std::setprecision
#include <cmath>   // For fabsf in verification

// Function to initialize a host matrix with random values
void initialize_host_matrix_random(float* matrix, int rows, int cols, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(min_val, max_val);
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = distrib(gen);
    }
}

// Host function for matrix multiplication (for verification)
void matrix_multiply_host(const float* A, const float* B, float* C, int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            C[i * B_cols + j] = 0.0f;
            for (int k = 0; k < A_cols; ++k) {
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}

// Host function for matrix transpose (for verification)
void matrix_transpose_host(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// Host function to check if a matrix is close to identity
bool is_identity_host(const float* matrix, int N, float tol = 1e-3f) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float target = (i == j) ? 1.0f : 0.0f;
            if (fabsf(matrix[i * N + j] - target) > tol) {
                return false;
            }
        }
    }
    return true;
}


int main() {
    int rows = 4; // Example: a "tall" matrix
    int cols = 3;
    // int rows = 3; // Example: a "wide" matrix
    // int cols = 4;
    // int rows = 3; // Example: a "square" matrix
    // int cols = 3;

    int num_ns_iterations = 10; // Number of Newton-Schulz iterations

    std::cout << "Simulating Muon Optimization (Newton-Schulz Iteration) for a "
              << rows << "x" << cols << " matrix." << std::endl;
    std::cout << "Number of NS iterations: " << num_ns_iterations << std::endl << std::endl;

    // Allocate host memory
    std::vector<float> h_G_in(rows * cols);
    std::vector<float> h_G_out(rows * cols);

    // Initialize input matrix (simulated gradient)
    initialize_host_matrix_random(h_G_in.data(), rows, cols);
    print_matrix_host(h_G_in.data(), rows, cols, "Initial Simulated Gradient (G_in) on Host");

    // Allocate device memory
    float *d_G_in, *d_G_out, *d_O, *d_O_T, *d_prod1, *d_prod2, *d_block_sums;

    CHECK_CUDA_ERROR(cudaMalloc(&d_G_in, rows * cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_G_out, rows * cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_O, rows * cols * sizeof(float)));
    
    // O_T is cols x rows
    CHECK_CUDA_ERROR(cudaMalloc(&d_O_T, cols * rows * sizeof(float)));

    // d_prod1 is (O_k^T * O_k) (cols x cols) or (O_k * O_k^T) (rows x rows)
    bool tall_or_square = (rows >= cols);
    int prod1_dim1 = tall_or_square ? cols : rows;
    int prod1_dim2 = prod1_dim1;
    CHECK_CUDA_ERROR(cudaMalloc(&d_prod1, prod1_dim1 * prod1_dim2 * sizeof(float)));

    // d_prod2 is O_k * (O_k^T * O_k) or (O_k * O_k^T) * O_k (rows x cols)
    CHECK_CUDA_ERROR(cudaMalloc(&d_prod2, rows * cols * sizeof(float)));
    
    // d_block_sums for Frobenius norm reduction
    int N_elements = rows * cols;
    int norm_threads_per_block = 256; // Must match kernel
    int num_norm_blocks = (N_elements + norm_threads_per_block - 1) / norm_threads_per_block;
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_sums, num_norm_blocks * sizeof(float)));

    // Copy input matrix to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_G_in, h_G_in.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    // Call Newton-Schulz iteration
    std::cout << "Running Newton-Schulz iterations on GPU..." << std::endl;
    newton_schulz_iteration_device(d_G_in, d_G_out, rows, cols, num_ns_iterations,
                                   d_O, d_O_T, d_prod1, d_prod2, d_block_sums);
    std::cout << "Newton-Schulz iterations completed." << std::endl << std::endl;
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_G_out.data(), d_G_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    print_matrix_host(h_G_out.data(), rows, cols, "Orthogonalized Matrix (G_out) on Host");

    // Verification: Check G_out * G_out^T (if wide) or G_out^T * G_out (if tall/square)
    // This should be close to an identity matrix.
    std::cout << "Verifying orthogonality property on host..." << std::endl;
    if (tall_or_square) { // G_out is rows x cols (tall or square), G_out^T is cols x rows
                          // G_out^T * G_out should be cols x cols Identity
        std::vector<float> h_G_out_T(cols * rows);
        std::vector<float> h_verification_matrix(cols * cols);
        matrix_transpose_host(h_G_out.data(), h_G_out_T.data(), rows, cols);
        matrix_multiply_host(h_G_out_T.data(), h_G_out.data(), h_verification_matrix.data(), cols, rows, cols);
        print_matrix_host(h_verification_matrix.data(), cols, cols, "G_out^T * G_out (should be Identity)");
        if (is_identity_host(h_verification_matrix.data(), cols)) {
            std::cout << "Verification PASSED: G_out^T * G_out is close to Identity." << std::endl;
        } else {
            std::cout << "Verification FAILED: G_out^T * G_out is NOT close to Identity." << std::endl;
        }
    } else { // G_out is rows x cols (wide), G_out^T is cols x rows
             // G_out * G_out^T should be rows x rows Identity
        std::vector<float> h_G_out_T(cols * rows);
        std::vector<float> h_verification_matrix(rows * rows);
        matrix_transpose_host(h_G_out.data(), h_G_out_T.data(), rows, cols);
        matrix_multiply_host(h_G_out.data(), h_G_out_T.data(), h_verification_matrix.data(), rows, cols, rows);
        print_matrix_host(h_verification_matrix.data(), rows, rows, "G_out * G_out^T (should be Identity)");
        if (is_identity_host(h_verification_matrix.data(), rows)) {
            std::cout << "Verification PASSED: G_out * G_out^T is close to Identity." << std::endl;
        } else {
            std::cout << "Verification FAILED: G_out * G_out^T is NOT close to Identity." << std::endl;
        }
    }
    std::cout << std::endl;


    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_G_in));
    CHECK_CUDA_ERROR(cudaFree(d_G_out));
    CHECK_CUDA_ERROR(cudaFree(d_O));
    CHECK_CUDA_ERROR(cudaFree(d_O_T));
    CHECK_CUDA_ERROR(cudaFree(d_prod1));
    CHECK_CUDA_ERROR(cudaFree(d_prod2));
    CHECK_CUDA_ERROR(cudaFree(d_block_sums));

    std::cout << "Execution finished successfully." << std::endl;
    return 0;
}
