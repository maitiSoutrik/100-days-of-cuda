#include "adahessian.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h> // For bool type

// Function to compare floats with tolerance
bool compareFloats(float a, float b, float tolerance) {
    return fabsf(a - b) < tolerance;
}

// CPU version for verification (single element)
void adaHessianUpdateCPU(
    float* theta,
    const float grad,
    const float gradPerturbed,
    float* m,
    float* v,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float delta
) {
    float h_diag = (delta != 0.0f) ? (gradPerturbed - grad) / delta : 0.0f;
    *m = beta1 * (*m) + (1.0f - beta1) * grad;
    *v = beta2 * (*v) + (1.0f - beta2) * (h_diag * h_diag);
    float denom = sqrtf(*v) + epsilon;
    if (denom != 0.0f) {
        *theta -= lr * (*m) / denom;
    }
}


int main() {
    const int N = 10;             // Small number of parameters for testing
    const size_t bytes = N * sizeof(float);
    const float lr = 0.01f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-7f;
    const float delta = 1e-4f;
    const float verification_tolerance = 1e-6f;

    printf("AdaHessian Test\n");
    printf("Number of parameters: %d\n", N);

    // Host arrays
    float h_theta[N], h_grad[N], h_gradPerturbed[N], h_m[N], h_v[N];
    float h_theta_initial[N], h_m_initial[N], h_v_initial[N]; // For CPU verification

    // Initialize arrays with simple, predictable data
    for (int i = 0; i < N; i++) {
        h_theta_initial[i] = h_theta[i] = 1.0f;
        h_grad[i] = 0.1f * (i + 1); // Simple gradient
        h_gradPerturbed[i] = h_grad[i] + (0.05f * (i+1)) * delta; // Simple perturbation
        h_m_initial[i] = h_m[i] = 0.0f;
        h_v_initial[i] = h_v[i] = 0.0f;
    }

    // --- GPU Execution ---
    printf("\n--- Running GPU Kernel ---\n");
    // Device arrays
    float *d_theta, *d_grad, *d_gradPerturbed, *d_m, *d_v;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_theta, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gradPerturbed, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_m, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_v, bytes));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_theta, h_theta, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_grad, h_grad, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_gradPerturbed, h_gradPerturbed, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice));

    // Launch the kernel
    int blockSize = 32; // Smaller block size for small N
    int gridSize = (N + blockSize - 1) / blockSize;
    adaHessianUpdateKernel<<<gridSize, blockSize>>>(
        d_theta, d_grad, d_gradPerturbed, d_m, d_v,
        lr, beta1, beta2, epsilon, delta, N
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel completion

    // Copy results back to host
    float h_theta_gpu[N], h_m_gpu[N], h_v_gpu[N];
    CHECK_CUDA_ERROR(cudaMemcpy(h_theta_gpu, d_theta, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_m_gpu, d_m, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_v_gpu, d_v, bytes, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(d_theta));
    CHECK_CUDA_ERROR(cudaFree(d_grad));
    CHECK_CUDA_ERROR(cudaFree(d_gradPerturbed));
    CHECK_CUDA_ERROR(cudaFree(d_m));
    CHECK_CUDA_ERROR(cudaFree(d_v));

    // --- CPU Verification (for element 0) ---
    printf("\n--- Running CPU Verification (for element 0) ---\n");
    float theta_cpu = h_theta_initial[0];
    float m_cpu = h_m_initial[0];
    float v_cpu = h_v_initial[0];
    adaHessianUpdateCPU(&theta_cpu, h_grad[0], h_gradPerturbed[0], &m_cpu, &v_cpu,
                        lr, beta1, beta2, epsilon, delta);

    printf("GPU Result (theta[0]): %.8f\n", h_theta_gpu[0]);
    printf("CPU Result (theta[0]): %.8f\n", theta_cpu);
    printf("GPU Result (m[0]):     %.8f\n", h_m_gpu[0]);
    printf("CPU Result (m[0]):     %.8f\n", m_cpu);
    printf("GPU Result (v[0]):     %.8f\n", h_v_gpu[0]);
    printf("CPU Result (v[0]):     %.8f\n", v_cpu);

    // --- Comparison ---
    bool passed = compareFloats(h_theta_gpu[0], theta_cpu, verification_tolerance) &&
                  compareFloats(h_m_gpu[0], m_cpu, verification_tolerance) &&
                  compareFloats(h_v_gpu[0], v_cpu, verification_tolerance);

    printf("\n--- Test Result ---\n");
    if (passed) {
        printf("PASS: GPU results match CPU verification for element 0.\n");
        return 0; // Success
    } else {
        printf("FAIL: GPU results DO NOT match CPU verification for element 0.\n");
        return 1; // Failure
    }
}
