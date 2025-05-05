#include "cgm_cublas.cuh"
#include <iostream>
#include <vector>
#include <cmath> // For fabs, sqrt
#include <chrono> // For timing

// Implementation of the Conjugate Gradient Method using cuBLAS
int conjugateGradientMethodCuBLAS(cublasHandle_t handle, int n, const double *d_A, const double *d_b, double *d_x,
                                  int max_iters, double tolerance) {
    double *d_r, *d_p, *d_Ap, *d_Ax; // Residual, direction, A*p, A*x vectors on device
    CHECK_CUDA_ERROR(cudaMalloc(&d_r, n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_p, n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_Ap, n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_Ax, n * sizeof(double))); // Allocate for initial A*x

    double alpha, beta, r_dot_r, r_dot_r_new, p_dot_Ap;
    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0; // Constant for alpha = -1.0
    double neg_alpha; // For cuBLAS calls

    // Initial calculation: r = b - A*x
    // 1. Calculate d_Ax = A * d_x (initial guess)
    CHECK_CUBLAS_ERROR(cublasDgemv(handle, CUBLAS_OP_N, n, n, &one, d_A, n, d_x, 1, &zero, d_Ax, 1));
    // 2. Copy b to r: d_r = d_b
    CHECK_CUDA_ERROR(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
    // 3. Calculate r = r - Ax (r = 1*r + (-1)*Ax)
    CHECK_CUBLAS_ERROR(cublasDaxpy(handle, n, &minus_one, d_Ax, 1, d_r, 1));

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
    CHECK_CUDA_ERROR(cudaFree(d_Ax)); // Free the temporary d_Ax

    if (k == max_iters) {
        fprintf(stderr, "Warning: Conjugate Gradient Method did not converge within %d iterations.\n", max_iters);
        fprintf(stderr, "Final residual norm squared: %.6e (Tolerance squared: %.6e)\n", r_dot_r, tolerance_sq);
        return -1; // Indicate non-convergence
    }

    return k; // Return number of iterations
}
