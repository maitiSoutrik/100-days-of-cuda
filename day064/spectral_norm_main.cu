#include "spectral_norm.cuh"
#include <iostream>
#include <vector>
#include <iomanip> // For std::fixed and std::setprecision

// Helper function to print a matrix (column-major)
void print_matrix(const char* title, const float* matrix, int rows, int cols) {
    std::cout << title << " (" << rows << "x" << cols << "):\n";
    std::vector<float> h_matrix(rows * cols);
    CHECK_CUDA_ERROR(cudaMemcpy(h_matrix.data(), matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Access element (i, j) in column-major format: matrix[j * rows + i]
            std::cout << std::fixed << std::setprecision(4) << h_matrix[j * rows + i] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Helper function to print a vector
void print_vector(const char* title, const float* vec, int n) {
    std::cout << title << " (" << n << "x1):\n";
    std::vector<float> h_vec(n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_vec.data(), vec, n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        std::cout << std::fixed << std::setprecision(4) << h_vec[i] << std::endl;
    }
    std::cout << std::endl;
}


int main() {
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    // --- Test Case 1: A simple 2x2 matrix ---
    std::cout << "--- Test Case 1: 2x2 Matrix ---" << std::endl;
    int m1 = 2, n1 = 2;
    std::vector<float> h_W1 = {1.0f, 0.0f, 0.0f, 2.0f}; // Column-major: [1 0; 0 2]
                                                       // Singular values should be 2 and 1. Spectral norm = 2.
    float* d_W1;
    float* d_u1;
    float* d_v1;

    CHECK_CUDA_ERROR(cudaMalloc(&d_W1, m1 * n1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_u1, m1 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_v1, n1 * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_W1, h_W1.data(), m1 * n1 * sizeof(float), cudaMemcpyHostToDevice));

    print_matrix("Original Matrix W1", d_W1, m1, n1);

    float sigma1_before = estimate_spectral_norm(handle, d_W1, m1, n1, d_u1, d_v1);
    std::cout << "Estimated Spectral Norm (before normalization) for W1: " << sigma1_before << std::endl << std::endl;

    spectral_normalize_matrix(handle, d_W1, m1, n1, d_u1, d_v1);
    print_matrix("Normalized Matrix W1_norm", d_W1, m1, n1);

    // Re-estimate spectral norm of the normalized matrix (should be close to 1)
    // Need to re-initialize u and v or use different ones if estimate_spectral_norm modifies them in a way
    // that affects subsequent calls for the *same* matrix. Here, d_W1 is modified, so it's fine.
    float sigma1_after = estimate_spectral_norm(handle, d_W1, m1, n1, d_u1, d_v1);
    std::cout << "Estimated Spectral Norm (after normalization) for W1_norm: " << sigma1_after << std::endl << std::endl;

    CHECK_CUDA_ERROR(cudaFree(d_W1));
    CHECK_CUDA_ERROR(cudaFree(d_u1));
    CHECK_CUDA_ERROR(cudaFree(d_v1));

    // --- Test Case 2: A 3x2 matrix ---
    std::cout << "\n--- Test Case 2: 3x2 Matrix ---" << std::endl;
    int m2 = 3, n2 = 2;
    // Column-major: [1 4; 2 5; 3 6]
    std::vector<float> h_W2 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float* d_W2;
    float* d_u2;
    float* d_v2;

    CHECK_CUDA_ERROR(cudaMalloc(&d_W2, m2 * n2 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_u2, m2 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_v2, n2 * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_W2, h_W2.data(), m2 * n2 * sizeof(float), cudaMemcpyHostToDevice));

    print_matrix("Original Matrix W2", d_W2, m2, n2);

    float sigma2_before = estimate_spectral_norm(handle, d_W2, m2, n2, d_u2, d_v2, 20); // More iterations for stability
    std::cout << "Estimated Spectral Norm (before normalization) for W2: " << sigma2_before << std::endl << std::endl;

    spectral_normalize_matrix(handle, d_W2, m2, n2, d_u2, d_v2, 20);
    print_matrix("Normalized Matrix W2_norm", d_W2, m2, n2);

    float sigma2_after = estimate_spectral_norm(handle, d_W2, m2, n2, d_u2, d_v2, 20);
    std::cout << "Estimated Spectral Norm (after normalization) for W2_norm: " << sigma2_after << std::endl << std::endl;

    CHECK_CUDA_ERROR(cudaFree(d_W2));
    CHECK_CUDA_ERROR(cudaFree(d_u2));
    CHECK_CUDA_ERROR(cudaFree(d_v2));
    
    // --- Test Case 3: A 2x3 matrix ---
    std::cout << "\n--- Test Case 3: 2x3 Matrix ---" << std::endl;
    int m3 = 2, n3 = 3;
    // Column-major: [1 2; 3 4; 5 6]  -> Transpose of previous W2's row-major representation
    // W = [1 3 5; 2 4 6]
    std::vector<float> h_W3 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float* d_W3;
    float* d_u3;
    float* d_v3;

    CHECK_CUDA_ERROR(cudaMalloc(&d_W3, m3 * n3 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_u3, m3 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_v3, n3 * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_W3, h_W3.data(), m3 * n3 * sizeof(float), cudaMemcpyHostToDevice));

    print_matrix("Original Matrix W3", d_W3, m3, n3);

    float sigma3_before = estimate_spectral_norm(handle, d_W3, m3, n3, d_u3, d_v3, 20);
    std::cout << "Estimated Spectral Norm (before normalization) for W3: " << sigma3_before << std::endl << std::endl;

    spectral_normalize_matrix(handle, d_W3, m3, n3, d_u3, d_v3, 20);
    print_matrix("Normalized Matrix W3_norm", d_W3, m3, n3);

    float sigma3_after = estimate_spectral_norm(handle, d_W3, m3, n3, d_u3, d_v3, 20);
    std::cout << "Estimated Spectral Norm (after normalization) for W3_norm: " << sigma3_after << std::endl << std::endl;

    CHECK_CUDA_ERROR(cudaFree(d_W3));
    CHECK_CUDA_ERROR(cudaFree(d_u3));
    CHECK_CUDA_ERROR(cudaFree(d_v3));


    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    return 0;
}
