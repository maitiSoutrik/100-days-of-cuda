#include "dyt.cuh"
#include <iostream>
#include <vector>
#include <cmath>   // For host-side math if any, though tanhf is mainly for device here
#include <iomanip> // For std::fixed and std::setprecision
#include <cstdio>  // For fprintf in CHECK_CUDA_ERROR (if not already via dyt.cuh)
#include <cstdlib> // For exit in CHECK_CUDA_ERROR (if not already via dyt.cuh)


int main() {
    // Parameters
    int n = 1024; // Number of elements
    float alpha_val = 1.5f;
    float beta_val = 0.5f;

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Dynamic Tanh (DyT) Operation - Main Demo" << std::endl;
    std::cout << "Parameters: n = " << n << ", alpha = " << alpha_val << ", beta = " << beta_val << std::endl;

    // Host data
    std::vector<float> h_x(n);
    std::vector<float> h_upstream_grad(n);

    // Initialize host data
    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>((i % 20) - 10) / 5.0f; // Sample values e.g. -2.0, -1.8, ..., 1.8
        h_upstream_grad[i] = 1.0f; // Simplest upstream gradient
    }

    // Device data pointers
    float *d_x, *d_y, *d_upstream_grad, *d_x_grad;
    float *d_alpha_grad_atomic, *d_beta_grad_atomic;

    // Allocate memory on device
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_upstream_grad, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x_grad, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_alpha_grad_atomic, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_beta_grad_atomic, sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_upstream_grad, h_upstream_grad.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize atomic gradient accumulators to zero
    CHECK_CUDA_ERROR(cudaMemset(d_alpha_grad_atomic, 0, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_beta_grad_atomic, 0, sizeof(float)));

    // Kernel launch parameters
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // --- Forward Pass ---
    std::cout << "\n--- Running Forward Pass ---" << std::endl;
    dyt_forward_kernel<<<numBlocks, blockSize>>>(d_x, d_y, alpha_val, beta_val, n);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for errors after kernel launch
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    // Copy result of forward pass back to host
    std::vector<float> h_y(n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Print some forward pass results
    std::cout << "Forward Pass Results (first 5 elements):" << std::endl;
    for (int i = 0; i < 5 && i < n; ++i) {
        std::cout << "x[" << i << "] = " << h_x[i] << ", y[" << i << "] = " << h_y[i] << std::endl;
    }

    // --- Backward Pass ---
    std::cout << "\n--- Running Backward Pass ---" << std::endl;
    dyt_backward_kernel<<<numBlocks, blockSize>>>(d_upstream_grad, d_x, d_x_grad,
                                                 d_alpha_grad_atomic, d_beta_grad_atomic,
                                                 alpha_val, beta_val, n);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for errors after kernel launch
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    // Copy results of backward pass back to host
    std::vector<float> h_x_grad(n);
    float h_alpha_grad, h_beta_grad;
    CHECK_CUDA_ERROR(cudaMemcpy(h_x_grad.data(), d_x_grad, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_alpha_grad, d_alpha_grad_atomic, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_beta_grad, d_beta_grad_atomic, sizeof(float), cudaMemcpyDeviceToHost));

    // Print some backward pass results
    std::cout << "\nBackward Pass Results:" << std::endl;
    std::cout << "First 5 x_grad elements:" << std::endl;
    for (int i = 0; i < 5 && i < n; ++i) {
        std::cout << "x_grad[" << i << "] = " << h_x_grad[i] << std::endl;
    }
    std::cout << "Total alpha_grad = " << h_alpha_grad << std::endl;
    std::cout << "Total beta_grad  = " << h_beta_grad << std::endl;

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_y));
    CHECK_CUDA_ERROR(cudaFree(d_upstream_grad));
    CHECK_CUDA_ERROR(cudaFree(d_x_grad));
    CHECK_CUDA_ERROR(cudaFree(d_alpha_grad_atomic));
    CHECK_CUDA_ERROR(cudaFree(d_beta_grad_atomic));

    std::cout << "\nDyT operation demo completed." << std::endl;

    return 0;
}
