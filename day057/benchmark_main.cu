#include "cgm_cublas.cuh" // For the solver function and error checks
#include <iostream>
#include <vector>
#include <cmath> // For sqrt in verification
#include <chrono> // For timing
#include <string> // For print function names

// Helper function to print a vector (moved from cgm_cublas.cu)
void printVector(const std::string& name, const std::vector<double>& vec) {
    std::cout << name << " = [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

// Helper function to print a matrix (column-major) (moved from cgm_cublas.cu)
void printMatrixColMajor(const std::string& name, const std::vector<double>& mat, int n) {
    std::cout << name << " (Column-Major) = [" << std::endl;
    for (int i = 0; i < n; ++i) { // Row index
        std::cout << "  ";
        for (int j = 0; j < n; ++j) { // Column index
            std::cout << mat[j * n + i] << (j == n - 1 ? "" : ", ");
        }
        std::cout << (i == n - 1 ? "" : ";") << std::endl;
    }
    std::cout << "]" << std::endl;
}

// Main function for the benchmark executable (moved from cgm_cublas.cu)
int main() {
    const int N = 4; // Small dimension for demonstration
    const int MAX_ITERS = 100;
    const double TOLERANCE = 1e-6;

    // --- Host Data ---
    // Define a symmetric positive-definite matrix A (column-major)
    // A = [ 4  1  0  0 ]
    //     [ 1  4  1  0 ]
    //     [ 0  1  4  1 ]
    //     [ 0  0  1  4 ]
    std::vector<double> h_A = {
        4.0, 1.0, 0.0, 0.0,
        1.0, 4.0, 1.0, 0.0,
        0.0, 1.0, 4.0, 1.0,
        0.0, 0.0, 1.0, 4.0
    };

    // Define vector b
    std::vector<double> h_b = {1.0, 2.0, 3.0, 4.0};

    // Initial guess for x (zeros)
    std::vector<double> h_x(N, 0.0);

    // Solution vector (for verification)
    std::vector<double> h_x_sol(N);

    printMatrixColMajor("Matrix A", h_A, N);
    printVector("Vector b", h_b);
    printVector("Initial x", h_x);

    // --- Device Data ---
    double *d_A, *d_b, *d_x;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, N * N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, N * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, N * sizeof(double)));

    // --- cuBLAS Initialization ---
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublasHandle));

    // --- Copy Data to Device ---
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // --- Solve Ax = b using CGM ---
    std::cout << "\nStarting Conjugate Gradient Method (cuBLAS)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Call the solver function (defined in cgm_cublas.cu, compiled into the library)
    int iterations = conjugateGradientMethodCuBLAS(cublasHandle, N, d_A, d_b, d_x, MAX_ITERS, TOLERANCE);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;

    if (iterations >= 0) {
        std::cout << "Converged in " << iterations << " iterations." << std::endl;
        std::cout << "Execution Time: " << duration_ms.count() << " ms" << std::endl;

        // --- Copy Result Back to Host ---
        CHECK_CUDA_ERROR(cudaMemcpy(h_x_sol.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
        printVector("Solution x", h_x_sol);

        // --- Basic Verification (Calculate Ax_sol and compare with b) ---
        std::vector<double> h_Ax_sol(N, 0.0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                h_Ax_sol[i] += h_A[j * N + i] * h_x_sol[j]; // A is column-major
            }
        }
        printVector("Verification (A*x_sol)", h_Ax_sol);
        double diff_norm = 0.0;
        for(int i=0; i<N; ++i) {
            diff_norm += (h_Ax_sol[i] - h_b[i]) * (h_Ax_sol[i] - h_b[i]);
        }
        std::cout << "Norm of difference ||Ax_sol - b||: " << sqrt(diff_norm) << std::endl;

    } else {
        std::cout << "CGM failed to converge." << std::endl;
    }


    // --- Cleanup ---
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUBLAS_ERROR(cublasDestroy(cublasHandle));

    return 0;
}
