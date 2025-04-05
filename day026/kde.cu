#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define PI 3.14159265358979323846f

// CUDA kernel to compute KDE
__global__ void compute_kde_kernel(const float* d_values, const float* d_query_points, float* d_density_estimates, int N, int M, float h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        float query_x = d_query_points[idx];
        float density_sum = 0.0f;

        for (int j = 0; j < N; ++j) {
            float value_j = d_values[j];
            float diff = query_x - value_j;
            float exponent = -0.5f * (diff / h) * (diff / h);
            density_sum += expf(exponent);
        }

        float final_density = density_sum / (N * h * sqrtf(2.0f * PI));
        d_density_estimates[idx] = final_density;
    }
}

int main() {
    // Host parameters
    int N = 10000; // Number of input values
    int M = 1000;  // Number of query points
    float h = 0.1f; // Bandwidth

    // Generate sample data (example: random values)
    std::vector<float> values(N);
    for (int i = 0; i < N; ++i) {
        values[i] = (float)rand() / RAND_MAX; // Values between 0 and 1
    }

    // Generate query points (example: evenly spaced)
    std::vector<float> query_points(M);
    for (int i = 0; i < M; ++i) {
        query_points[i] = (float)i / (M - 1); // Query points between 0 and 1
    }

    // Allocate device memory
    float* d_values;
    float* d_query_points;
    float* d_density_estimates;
    cudaMalloc(&d_values, N * sizeof(float));
    cudaMalloc(&d_query_points, M * sizeof(float));
    cudaMalloc(&d_density_estimates, M * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_values, values.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_points, query_points.data(), M * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel configuration
    int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;

    // Launch kernel
    compute_kde_kernel<<<grid_size, block_size>>>(d_values, query_points, d_density_estimates, N, M, h);

    // Error check
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(cuda_status) << std::endl;
        return 1;
    }

    // Copy results back to host
    std::vector<float> density_estimates(M);
    cudaMemcpy(density_estimates.data(), d_density_estimates, M * sizeof(float), cudaMemcpyDeviceToHost);

    // (Optional) CPU calculation for verification
    // ... (Implementation of CPU KDE calculation) ...

    // Print/analyze results (example: print first 10)
    std::cout << "KDE Results (first 10):" << std::endl;
    for (int i = 0; i < std::min(10, M); ++i) {
        std::cout << "Query Point " << query_points[i] << ": " << density_estimates[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_values);
    cudaFree(d_query_points);
    cudaFree(d_density_estimates);

    return 0;
}
