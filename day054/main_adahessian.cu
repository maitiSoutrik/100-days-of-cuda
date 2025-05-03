#include "adahessian.h"
#include <stdio.h>
#include <stdlib.h> // For malloc, free
#include <math.h>   // For basic math if needed in main

int main() {
    const int N = 1024 * 1024;      // Number of parameters (increased for a better test)
    const size_t bytes = N * sizeof(float);
    const float lr = 0.01f;         // Learning rate
    const float beta1 = 0.9f;       // Decay rate for first moment
    const float beta2 = 0.999f;     // Decay rate for second moment
    const float epsilon = 1e-7f;    // Small constant for numerical stability
    const float delta = 1e-4f;      // Perturbation for finite differences

    printf("AdaHessian Main Example\n");
    printf("Number of parameters: %d\n", N);
    printf("Learning Rate: %.4f, Beta1: %.3f, Beta2: %.3f, Epsilon: %.1e, Delta: %.1e\n",
           lr, beta1, beta2, epsilon, delta);

    // Host arrays
    float *h_theta, *h_grad, *h_gradPerturbed, *h_m, *h_v;
    h_theta         = (float*)malloc(bytes);
    h_grad          = (float*)malloc(bytes);
    h_gradPerturbed = (float*)malloc(bytes);
    h_m             = (float*)malloc(bytes);
    h_v             = (float*)malloc(bytes);

    if (!h_theta || !h_grad || !h_gradPerturbed || !h_m || !h_v) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize arrays with dummy data
    printf("Initializing host data...\n");
    for (int i = 0; i < N; i++) {
        h_theta[i] = 1.0f;                     // Initial parameter value
        h_grad[i] = 0.1f + (float)i / N * 0.01f; // Slightly varying gradient
        // Simulate perturbed gradient: a small change from h_grad related to Hessian approx
        float approx_hessian_diag = (float)i / N * 0.5f + 0.1f; // Arbitrary "true" Hessian diag
        h_gradPerturbed[i] = h_grad[i] + approx_hessian_diag * delta;
        h_m[i] = 0.0f;                         // Initialize first moment
        h_v[i] = 0.0f;                         // Initialize second moment
    }
    printf("Host data initialization complete.\n");

    // Device arrays
    float *d_theta, *d_grad, *d_gradPerturbed, *d_m, *d_v;
    printf("Allocating device memory...\n");
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_theta, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gradPerturbed, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_m, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_v, bytes));
    printf("Device memory allocation complete.\n");

    // Copy data from host to device
    printf("Copying data from host to device...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_theta, h_theta, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_grad, h_grad, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_gradPerturbed, h_gradPerturbed, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice));
    printf("Data copy complete.\n");

    // Launch the kernel with an appropriate grid size
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    printf("Launching kernel with grid size %d and block size %d\n", gridSize, blockSize);

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    adaHessianUpdateKernel<<<gridSize, blockSize>>>(
        d_theta, d_grad, d_gradPerturbed, d_m, d_v,
        lr, beta1, beta2, epsilon, delta, N
    );
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for launch errors immediately after <<<>>>
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    // Wait for the kernel and timing to complete
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy updated parameters and moment estimates back to host
    printf("Copying results from device to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(h_theta, d_theta, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_m, d_m, bytes, cudaMemcpyDeviceToHost)); // Also copy m and v if needed for inspection
    CHECK_CUDA_ERROR(cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost));
    printf("Result copy complete.\n");

    // Print a few updated theta values for verification
    printf("\nUpdated theta values (first/last 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", h_theta[i]);
    }
    if (N > 20) printf("... ");
    for (int i = (N > 10 ? N - 10 : 10); i < N; i++) {
        printf("%.6f ", h_theta[i]);
    }
    printf("\n");

    // Free device memory
    printf("Freeing device memory...\n");
    CHECK_CUDA_ERROR(cudaFree(d_theta));
    CHECK_CUDA_ERROR(cudaFree(d_grad));
    CHECK_CUDA_ERROR(cudaFree(d_gradPerturbed));
    CHECK_CUDA_ERROR(cudaFree(d_m));
    CHECK_CUDA_ERROR(cudaFree(d_v));
    printf("Device memory freed.\n");

    // Free host memory
    printf("Freeing host memory...\n");
    free(h_theta);
    free(h_grad);
    free(h_gradPerturbed);
    free(h_m);
    free(h_v);
    printf("Host memory freed.\n");

    // Destroy events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    printf("\nExecution finished successfully.\n");
    return 0;
}
