#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- CUDA Error Checking Macro ---
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// --- Kernels ---

// Kernel to project latent variables using tanh
__global__ void projection_kernel(float* d_x_projected, const float* d_x_latent, float beta, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_x_projected[idx] = tanhf(beta * d_x_latent[idx]);
    }
}

// Kernel to calculate the gradient of the Sphere function f(x) = sum(x_i^2)
// Gradient: d(f)/dx_i = 2 * x_i
__global__ void gradient_kernel(float* d_gradient, const float* d_x_projected, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_gradient[idx] = 2.0f * d_x_projected[idx];
    }
}

// Kernel to update the latent variables (dual space update)
__global__ void update_latent_kernel(float* d_x_latent, const float* d_gradient, float learning_rate, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_x_latent[idx] = d_x_latent[idx] - learning_rate * d_gradient[idx];
    }
}

// --- Host Logic ---
int main() {
    // Parameters
    const int N = 1024; // Size of the vector x
    const int num_iterations = 5000;
    const float learning_rate = 0.01f;
    float beta = 1.0f; // Initial annealing parameter
    const float rho = 1.001f; // Annealing factor (beta = beta * rho)
    const int print_interval = 500;

    printf("Starting Mirror Descent (STE) Optimization for Sphere Function\n");
    printf("N = %d, Iterations = %d, LR = %.4f, Initial Beta = %.2f, Rho = %.4f\n",
           N, num_iterations, learning_rate, beta, rho);

    // Allocate host memory
    float* h_x_latent = (float*)malloc(N * sizeof(float));
    float* h_x_projected = (float*)malloc(N * sizeof(float)); // For final result verification
    if (!h_x_latent || !h_x_projected) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize latent variables randomly on host (-1 to 1)
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        h_x_latent[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Allocate device memory
    float *d_x_latent, *d_x_projected, *d_gradient;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x_latent, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x_projected, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gradient, N * sizeof(float)));

    // Copy initial latent variables from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_x_latent, h_x_latent, N * sizeof(float), cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Optimization loop
    for (int iter = 0; iter < num_iterations; ++iter) {
        // 1. Project latent variables to get current x
        projection_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x_projected, d_x_latent, beta, N);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // 2. Calculate gradient based on projected x
        gradient_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_gradient, d_x_projected, N);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // 3. Update latent variables (dual space update)
        update_latent_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x_latent, d_gradient, learning_rate, N);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // 4. Anneal beta
        beta *= rho;

        // Optional: Print progress
        if ((iter + 1) % print_interval == 0 || iter == 0) {
            CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure kernels are finished before printing
             printf("Iteration %d, Beta = %.4f\n", iter + 1, beta);
             // Optionally copy back d_x_projected to check intermediate values
             // CHECK_CUDA_ERROR(cudaMemcpy(h_x_projected, d_x_projected, N * sizeof(float), cudaMemcpyDeviceToHost));
             // printf("  Sample projected[0]: %.4f\n", h_x_projected[0]);
        }
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for all iterations to complete

    printf("\nOptimization finished.\n");
    printf("Final Beta = %.4f\n", beta);

    // Copy final results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_x_latent, d_x_latent, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_x_projected, d_x_projected, N * sizeof(float), cudaMemcpyDeviceToHost)); // Get final projected values

    // Print some final values
    printf("Final Latent Variables (first 10):\n");
    for (int i = 0; i < fmin(10, N); ++i) {
        printf("  x_latent[%d] = %.6f\n", i, h_x_latent[i]);
    }
    printf("Final Projected Variables (first 10):\n");
    for (int i = 0; i < fmin(10, N); ++i) {
        printf("  x_projected[%d] = %.6f\n", i, h_x_projected[i]);
    }

    // Calculate final function value using the projected values
    double final_f_val = 0.0;
    for (int i = 0; i < N; ++i) {
        final_f_val += h_x_projected[i] * h_x_projected[i];
    }
    printf("\nFinal function value f(x_projected) = %f\n", final_f_val);

    // Calculate function value using sign(latent) as the "true" quantized result
    double final_f_val_quantized = 0.0;
     for (int i = 0; i < N; ++i) {
        float sign_val = (h_x_latent[i] >= 0.0f) ? 1.0f : -1.0f;
        final_f_val_quantized += sign_val * sign_val; // Will be N if perfectly quantized
    }
    printf("Final function value f(sign(x_latent)) = %f (Should approach N=%d)\n", final_f_val_quantized, N);


    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_x_latent));
    CHECK_CUDA_ERROR(cudaFree(d_x_projected));
    CHECK_CUDA_ERROR(cudaFree(d_gradient));

    // Free host memory
    free(h_x_latent);
    free(h_x_projected);

    printf("\nDay 27 Mirror Descent finished successfully.\n");
    return EXIT_SUCCESS;
}
