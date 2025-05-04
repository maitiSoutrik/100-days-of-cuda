#include "quantization_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// --- Benchmarking ---
// Generic benchmark function (can stay here or move to a utility file)
template <typename KernelType, typename InTypeA, typename InTypeB, typename OutType>
float benchmark_kernel(KernelType kernel, const InTypeA *d_A, const InTypeB *d_B, OutType *d_C, int N, dim3 blocks, dim3 threads) {
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure previous work is done
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop)); // Wait for kernel completion

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return milliseconds;
}


// --- Host Code ---
int main(int argc, char **argv) {
    int N = 1024; // Default Matrix size
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            printf("Invalid matrix size provided. Using default N=1024.\n");
            N = 1024;
        }
    }

    if (N % TILE_DIM != 0) {
        printf("Error: Matrix size N (%d) must be divisible by TILE_DIM (%d)\n", N, TILE_DIM);
        // Adjust N to the nearest lower multiple of TILE_DIM for demonstration?
        // N = (N / TILE_DIM) * TILE_DIM;
        // printf("Adjusting N to %d\n", N);
        // Or simply exit:
         return 1;
    }
    printf("Starting Quantization Comparison for %dx%d Matrices\n", N, N);

    size_t bytes_fp32 = (size_t)N * N * sizeof(float);
    size_t bytes_fp16 = (size_t)N * N * sizeof(__half);
    size_t bytes_fp8_sim = (size_t)N * N * sizeof(uint8_t);

    // Host memory allocation
    float *h_A = (float*)malloc(bytes_fp32);
    float *h_B = (float*)malloc(bytes_fp32);
    float *h_C_fp32_gpu = (float*)malloc(bytes_fp32);
    float *h_C_fp16_gpu = (float*)malloc(bytes_fp32); // Store FP16 result converted back to FP32
    float *h_C_fp8_sim_gpu = (float*)malloc(bytes_fp32);
    float *h_C_cpu_ref = (float*)malloc(bytes_fp32); // CPU reference result

    if (!h_A || !h_B || !h_C_fp32_gpu || !h_C_fp16_gpu || !h_C_fp8_sim_gpu || !h_C_cpu_ref) {
        fprintf(stderr, "Failed to allocate host memory\n");
        // Free any already allocated memory before exiting
        free(h_A); free(h_B); free(h_C_fp32_gpu); free(h_C_fp16_gpu); free(h_C_fp8_sim_gpu); free(h_C_cpu_ref);
        return 1;
    }

    // Initialize matrices A and B
    srand(0); // for reproducibility
    for (int i = 0; i < N * N; i++) {
        h_A[i] = ((rand() / (float)RAND_MAX) * 100.0f) - 50.0f; // Range [-50, 50]
        h_B[i] = ((rand() / (float)RAND_MAX) * 100.0f) - 50.0f;
    }
    printf("Host matrices initialized.\n");

    // Device memory allocation
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    __half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    uint8_t *d_A_fp8_sim, *d_B_fp8_sim;
    float *d_C_fp8_sim; // Output for FP8 sim is FP32

    CHECK_CUDA_ERROR(cudaMalloc(&d_A_fp32, bytes_fp32));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B_fp32, bytes_fp32));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C_fp32, bytes_fp32));

    CHECK_CUDA_ERROR(cudaMalloc(&d_A_fp16, bytes_fp16));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B_fp16, bytes_fp16));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C_fp16, bytes_fp16));

    CHECK_CUDA_ERROR(cudaMalloc(&d_A_fp8_sim, bytes_fp8_sim));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B_fp8_sim, bytes_fp8_sim));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C_fp8_sim, bytes_fp32));

    printf("Device memory allocated.\n");

    // Copy input data from host to device (FP32)
    CHECK_CUDA_ERROR(cudaMemcpy(d_A_fp32, h_A, bytes_fp32, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B_fp32, h_B, bytes_fp32, cudaMemcpyHostToDevice));
    printf("Input data copied to device (FP32).\n");

    // --- Setup Grid/Block Dimensions ---
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks(N / TILE_DIM, N / TILE_DIM);
    dim3 conversion_threads(256); // For 1D kernels
    dim3 conversion_blocks((N * N + conversion_threads.x - 1) / conversion_threads.x);

    // --- FP32 Execution & Benchmark ---
    printf("\n==== FP32 Execution ====\n");
    float fp32_time = benchmark_kernel(matmul_fp32_kernel, d_A_fp32, d_B_fp32, d_C_fp32, N, blocks, threads);
    printf("FP32 GPU Time: %.3f ms\n", fp32_time);
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_fp32_gpu, d_C_fp32, bytes_fp32, cudaMemcpyDeviceToHost));

    // --- FP16 Execution & Benchmark ---
    printf("\n==== FP16 Execution ====\n");
    fp32_to_fp16_kernel<<<conversion_blocks, conversion_threads>>>(d_A_fp32, d_A_fp16, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    fp32_to_fp16_kernel<<<conversion_blocks, conversion_threads>>>(d_B_fp32, d_B_fp16, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("Inputs converted to FP16.\n");

    float fp16_time = benchmark_kernel(matmul_fp16_kernel, d_A_fp16, d_B_fp16, d_C_fp16, N, blocks, threads);
    printf("FP16 GPU Time: %.3f ms\n", fp16_time);

    fp16_to_fp32_kernel<<<conversion_blocks, conversion_threads>>>(d_C_fp16, d_C_fp32, N); // Reuse d_C_fp32 buffer
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_fp16_gpu, d_C_fp32, bytes_fp32, cudaMemcpyDeviceToHost));
    printf("FP16 result converted back to FP32.\n");

    // --- Simulated FP8 Execution & Benchmark ---
    printf("\n==== Simulated FP8 Execution ====\n");
    fp32_to_fp8_sim_kernel<<<conversion_blocks, conversion_threads>>>(d_A_fp32, d_A_fp8_sim, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    fp32_to_fp8_sim_kernel<<<conversion_blocks, conversion_threads>>>(d_B_fp32, d_B_fp8_sim, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    printf("Inputs quantized to simulated FP8.\n");

    float fp8_sim_time = benchmark_kernel(matmul_fp8_sim_kernel, d_A_fp8_sim, d_B_fp8_sim, d_C_fp8_sim, N, blocks, threads);
    printf("Simulated FP8 GPU Time: %.3f ms\n", fp8_sim_time);
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_fp8_sim_gpu, d_C_fp8_sim, bytes_fp32, cudaMemcpyDeviceToHost));

    // --- CPU Reference Calculation ---
    printf("\n==== CPU Reference Calculation ====\n");
    printf("Calculating reference result on CPU (N=%d)...\n", N);
    matmul_cpu(h_A, h_B, h_C_cpu_ref, N);
    printf("CPU calculation complete.\n");

    // --- Verification ---
    printf("\n==== Verification vs CPU Reference ====\n");
    verify_results(h_C_cpu_ref, h_C_fp32_gpu, N, "FP32");
    verify_results(h_C_cpu_ref, h_C_fp16_gpu, N, "FP16");
    verify_results(h_C_cpu_ref, h_C_fp8_sim_gpu, N, "SimFP8");

    // --- Memory Usage Summary ---
    printf("\n==== Memory Usage per Element ====\n");
    printf("FP32: %zu bytes\n", sizeof(float));
    printf("FP16: %zu bytes\n", sizeof(__half));
    printf("FP8 (Simulated Storage): %zu byte\n", sizeof(uint8_t));

    // --- Cleanup ---
    printf("\nCleaning up resources...\n");
    cudaFree(d_A_fp32); cudaFree(d_B_fp32); cudaFree(d_C_fp32);
    cudaFree(d_A_fp16); cudaFree(d_B_fp16); cudaFree(d_C_fp16);
    cudaFree(d_A_fp8_sim); cudaFree(d_B_fp8_sim); cudaFree(d_C_fp8_sim);

    free(h_A); free(h_B);
    free(h_C_fp32_gpu); free(h_C_fp16_gpu); free(h_C_fp8_sim_gpu);
    free(h_C_cpu_ref);

    printf("Cleanup complete. Exiting.\n");
    // cudaDeviceReset(); // Optional: Reset device at the very end
    return 0;
}
