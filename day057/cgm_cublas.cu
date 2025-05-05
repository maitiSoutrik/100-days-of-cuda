#include "cgm_cublas.cuh"
#include <iostream>
#include <vector>
#include <cmath> // For fabs, sqrt
#include <chrono> // For timing

// Implementation of the Conjugate Gradient Method using cuBLAS
int conjugateGradientMethodCuBLAS(cublasHandle_t handle, int n, const double *d_A, const double *d_b, double *d_x,
                                  int max_iters, double tolerance) {
    double *d_r, *d_p, *d_Ap; // Residual, direction, A*p vectors on device
    CHECK_CUDA_ERROR(cudaMalloc(&d_r, n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_p, n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_Ap, n * sizeof(double)));

    double alpha, beta, r_dot_r, r_dot_r_new, p_dot_Ap;
    const double one = 1.0;
    const double zero = 0.0;
    double neg_alpha, neg_beta_alpha; // For cuBLAS calls

    // Initial calculation: r = b - A*x
    // 1. Calculate A*x
    CHECK_CUBLAS_ERROR(cublasDgemv(handle, CUBLAS_OP_N, n, n, &one, d_A, n, d_x, 1, &zero, d_r, 1));
    // 2. Calculate r = b - A*x (using daxpy: r = -1.0*r + b)
    const double minus_one = -1.0;
    CHECK_CUBLAS_ERROR(cublasDaxpy(handle, n, &minus_one, d_r, 1, d_b, 1)); // d_b now holds b
    CHECK_CUDA_ERROR(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice)); // Copy b into r (r = b - Ax)

    // Initial p = r
    CHECK_CUDA_ERROR(cudaMemcpy(d_p, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));

    // Initial residual norm squared: r_dot_r = r' * r
    CHECK_CUBLAS_ERROR(cublasDdot(handle, n, d_r, 1, d_r, 1, &r_dot_r));

    double initial_residual_norm = sqrt(r_dot_r);
    double tolerance_sq = tolerance * tolerance * initial_residual_norm * initial_residual_norm; // Relative tolerance squared

    if (initial_residual_norm < 1e-10) { // Check if initial guess is already the solution
         tolerance_sq = tolerance * tolerance;
    }


    int k = 0;
    for (k = 0; k < max_iters; ++k) {
        if (r_dot_r < tolerance_sq) {
            break; // Converged
        }

        // Calculate Ap = A * p
        CHECK_CUBLAS_ERROR(cublasDgemv(handle, CUBLAS_OP_N, n, n, &one, d_A, n, d_p, 1, &zero, d_Ap, 1));

        // Calculate alpha = (r' * r) / (p' * Ap)
        CHECK_CUBLAS_ERROR(cublasDdot(handle, n, d_p, 1, d_Ap, 1, &p_dot_Ap));
        if (fabs(p_dot_Ap) < 1e-12) { // Avoid division by zero or near-zero
             fprintf(stderr, "Warning: p' * Ap is close to zero (%.6e) at iteration %d. Potential breakdown.\n", p_dot_Ap, k);
             // Handle breakdown: maybe return k or specific error code
             break;
        }
        alpha = r_dot_r / p_dot_Ap;

        // Update x: x = x + alpha * p
        CHECK_CUBLAS_ERROR(cublasDaxpy(handle, n, &alpha, d_p, 1, d_x, 1));

        // Update r: r = r - alpha * Ap
        neg_alpha = -alpha;
        CHECK_CUBLAS_ERROR(cublasDaxpy(handle, n, &neg_alpha, d_Ap, 1, d_r, 1));

        // Calculate new residual norm squared: r_dot_r_new = r' * r
        CHECK_CUBLAS_ERROR(cublasDdot(handle, n, d_r, 1, d_r, 1, &r_dot_r_new));

        // Calculate beta = (r_new' * r_new) / (r_old' * r_old)
        if (fabs(r_dot_r) < 1e-12) { // Avoid division by zero if previous residual was tiny
            fprintf(stderr, "Warning: r_dot_r is close to zero (%.6e) at iteration %d before beta calculation.\n", r_dot_r, k);
            // Could indicate convergence or stagnation
            beta = 0.0; // Reset direction
        } else {
            beta = r_dot_r_new / r_dot_r;
        }


        // Update p: p = r + beta * p
        CHECK_CUBLAS_ERROR(cublasDscal(handle, n, &beta, d_p, 1)); // p = beta * p
        CHECK_CUBLAS_ERROR(cublasDaxpy(handle, n, &one, d_r, 1, d_p, 1)); // p = r + p (which is now beta*p)

        // Update residual norm squared for next iteration
        r_dot_r = r_dot_r_new;
    }

    // Cleanup device memory
    CHECK_CUDA_ERROR(cudaFree(d_r));
    CHECK_CUDA_ERROR(cudaFree(d_p));
    CHECK_CUDA_ERROR(cudaFree(d_Ap));

    if (k == max_iters) {
        fprintf(stderr, "Warning: Conjugate Gradient Method did not converge within %d iterations.\n", max_iters);
        fprintf(stderr, "Final residual norm squared: %.6e (Tolerance squared: %.6e)\n", r_dot_r, tolerance_sq);
        return -1; // Indicate non-convergence
    }

    return k; // Return number of iterations
}


// Helper function to print a vector
void printVector(const std::string& name, const std::vector<double>& vec) {
    std::cout << name << " = [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

// Helper function to print a matrix (column-major)
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
