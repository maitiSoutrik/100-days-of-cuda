#include <iostream>
#include <vector>
#include "upper_tri_gemm.cuh"

template <typename T>
void gemm_upper_tri_host(const std::vector<T>& A, const std::vector<T>& B, std::vector<T>& C, int n) {
    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(T));
    cudaMalloc(&d_B, n * n * sizeof(T));
    cudaMalloc(&d_C, n * n * sizeof(T));

    cudaMemcpy(d_A, A.data(), n * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), n * n * sizeof(T), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    gemm_upper_tri<T, 16><<<grid, block>>>(d_A, d_B, d_C, n);

    cudaMemcpy(C.data(), d_C, n * n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    const int n = 1024;
    std::vector<float> A(n * n), B(n * n), C(n * n);

    // Initialize A and B as upper triangular matrices
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            A[i * n + j] = i + j;
            B[i * n + j] = i - j;
        }
    }

    gemm_upper_tri_host(A, B, C, n);

    // Print a few elements of the result
    for (int i = 0; i < 5; ++i) {
        for (int j = i; j < 5; ++j) {
            std::cout << C[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
