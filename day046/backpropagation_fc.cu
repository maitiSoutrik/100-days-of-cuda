#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> // For fabs comparison
#include <chrono> // For CPU timing

// --- Error Checking Macros ---
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUBLAS_ERROR(status) \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: Status %d at %s:%d\n", status, __FILE__, __LINE__); \
        /* Note: cuBLAS doesn't have a direct equivalent to cudaGetErrorString */ \
        /* You might need a custom function to map status codes to strings */ \
        exit(EXIT_FAILURE); \
    }

// --- Bias Gradient Kernel ---
// Calculates dL_dBias by summing dL_dOutput across the batch dimension.
// dL_dOutput has dimensions (output_features, batch_size) - Column-Major for cuBLAS GEMM!
// dL_dBias has dimensions (output_features, 1)
__global__ void calculate_bias_gradients(const float* dL_dOutput, float* dL_dBias, int output_features, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Index corresponds to output feature

    if (idx < output_features) {
        float sum = 0.0f;
        // Sum across the batch dimension for this output feature
        // dL_dOutput[feature * batch_size + batch_idx]  <-- Incorrect for Column-Major
        // dL_dOutput[batch_idx * output_features + feature] <-- Column-Major access
        for (int i = 0; i < batch_size; ++i) {
            sum += dL_dOutput[i * output_features + idx];
        }
        dL_dBias[idx] = sum;
    }
}

// --- Host Calculation & Verification Functions ---

// Calculate bias gradients on CPU
void calculate_bias_gradients_cpu(const float* h_dL_dOutput, float* h_dL_dBias_cpu, int output_features, int batch_size) {
    for (int j = 0; j < output_features; ++j) { // Iterate over output features
        h_dL_dBias_cpu[j] = 0.0f;
        for (int i = 0; i < batch_size; ++i) { // Sum across batch dimension
             // Accessing Column-Major dL_dOutput
             h_dL_dBias_cpu[j] += h_dL_dOutput[i * output_features + j];
        }
    }
}

// Compare CPU and GPU results
bool compare_gradients(const float* arr1, const float* arr2, size_t size, const char* name, float threshold = 1e-4) {
    float max_error = 0.0f;
    int errors = 0;
    for (size_t i = 0; i < size; ++i) {
        float error = fabs(arr1[i] - arr2[i]);
        if (error > max_error) max_error = error;
        if (error > threshold) {
           // printf("%s Error at index %zu: Arr1=%.6f, Arr2=%.6f, Error=%.6f\n", name, i, arr1[i], arr2[i], error);
           errors++;
        }
    }

    if (errors == 0) {
        printf("%s Verification PASSED. Max Error: %.6f\n", name, max_error);
        return true;
    } else {
        printf("%s Verification FAILED. Found %d discrepancies. Max Error: %.6f\n", name, errors, max_error);
        return false;
    }
}


// Calculate weight gradients on CPU
void calculate_weight_gradients_cpu(const float* h_dL_dOutput, const float* h_Input, float* h_dL_dW_cpu, int input_features, int output_features, int batch_size) {
    // dL/dW = dL/dOutput * Input^T (thinking in Column-Major)
    // A = dL_dOutput (output_features x batch_size)
    // B = Input (input_features x batch_size)
    // C = dL_dW (output_features x input_features) = A * B^T

    // Perform C = A * B^T using Column-Major indexing
    // C[col, row] = sum_k (A[k, row] * B[k, col])
     for (int row = 0; row < output_features; ++row) { // Output feature index (row of C)
        for (int col = 0; col < input_features; ++col) { // Input feature index (col of C)
            float sum = 0.0f;
            for (int k = 0; k < batch_size; ++k) { // Batch index (inner dimension)
                // A[k * lda + row] = dL_dOutput[k * output_features + row]
                // B[k * ldb + col] = Input[k * input_features + col]
                sum += h_dL_dOutput[k * output_features + row] * h_Input[k * input_features + col];
            }
            // C[col * ldc + row] = dL_dW[col * output_features + row]
            h_dL_dW_cpu[col * output_features + row] = sum;
        }
    }
}


// --- Main Function ---
int main() {
    // --- Configuration ---
    const int BATCH_SIZE = 64;
    const int INPUT_FEATURES = 128;
    const int OUTPUT_FEATURES = 256;
    printf("Configuration:\n");
    printf("  Batch Size: %d\n", BATCH_SIZE);
    printf("  Input Features: %d\n", INPUT_FEATURES);
    printf("  Output Features: %d\n", OUTPUT_FEATURES);

    // --- Matrix Dimensions (Column-Major for cuBLAS) ---
    // Input: (input_features, batch_size) -> Transposed for GEMM becomes (batch_size, input_features)
    // dL/dOutput: (output_features, batch_size)
    // Weights (Not directly used here, but defines FC layer shape): (output_features, input_features)
    // dL/dW: (output_features, input_features) - Result of GEMM
    // dL/dBias: (output_features, 1) - Result of reduction kernel

    // --- Host Memory Allocation ---
    float *h_Input, *h_dL_dOutput, *h_dL_dW_gpu, *h_dL_dBias_gpu, *h_dL_dW_cpu, *h_dL_dBias_cpu;
    size_t input_size = (size_t)INPUT_FEATURES * BATCH_SIZE * sizeof(float);
    size_t dl_doutput_size = (size_t)OUTPUT_FEATURES * BATCH_SIZE * sizeof(float);
    size_t dl_dw_size = (size_t)OUTPUT_FEATURES * INPUT_FEATURES * sizeof(float);
    size_t dl_dbias_size = (size_t)OUTPUT_FEATURES * sizeof(float); // Bias is per output feature

    h_Input = (float*)malloc(input_size);
    h_dL_dOutput = (float*)malloc(dl_doutput_size);
    h_dL_dW_gpu = (float*)malloc(dl_dw_size);      // To store GPU result
    h_dL_dBias_gpu = (float*)malloc(dl_dbias_size); // To store GPU result
    h_dL_dW_cpu = (float*)malloc(dl_dw_size);      // For CPU calculation
    h_dL_dBias_cpu = (float*)malloc(dl_dbias_size); // For CPU calculation


    if (!h_Input || !h_dL_dOutput || !h_dL_dW_gpu || !h_dL_dBias_gpu || !h_dL_dW_cpu || !h_dL_dBias_cpu) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // --- Initialize Host Data ---
    printf("Initializing host data...\n");
    // Simple initialization (e.g., sequential or random)
    for (int i = 0; i < INPUT_FEATURES * BATCH_SIZE; ++i) {
        h_Input[i] = (float)(rand() % 100) / 100.0f; // Small random values
    }
    for (int i = 0; i < OUTPUT_FEATURES * BATCH_SIZE; ++i) {
        h_dL_dOutput[i] = (float)(rand() % 100) / 50.0f - 1.0f; // Small random values around 0
    }

    // --- Device Memory Allocation ---
    float *d_Input, *d_dL_dOutput, *d_dL_dW, *d_dL_dBias;
    CHECK_CUDA_ERROR(cudaMalloc(&d_Input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dL_dOutput, dl_doutput_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dL_dW, dl_dw_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dL_dBias, dl_dbias_size));

    // --- Copy Data Host to Device ---
    printf("Copying data from host to device...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_Input, h_Input, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dL_dOutput, h_dL_dOutput, dl_doutput_size, cudaMemcpyHostToDevice));

    // --- cuBLAS Initialization ---
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle));

    // --- CUDA Events for GPU Timing ---
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu));
    float gpu_weight_time_ms = 0.0f;
    float gpu_bias_time_ms = 0.0f;

    // --- GPU Weight Gradient Calculation (dL/dW = dL/dOutput * Input^T) ---
    printf("Calculating weight gradients (dL/dW) using cuBLAS Sgemm (GPU)...\n");
    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu));
    // cuBLAS assumes column-major ordering.
    // A = dL_dOutput (Matrix dimensions: OUTPUT_FEATURES x BATCH_SIZE) -> m=OUTPUT_FEATURES, k=BATCH_SIZE
    // B = Input      (Matrix dimensions: INPUT_FEATURES x BATCH_SIZE) -> Transpose -> (BATCH_SIZE x INPUT_FEATURES) -> k=BATCH_SIZE, n=INPUT_FEATURES
    // C = dL_dW      (Matrix dimensions: OUTPUT_FEATURES x INPUT_FEATURES) -> m=OUTPUT_FEATURES, n=INPUT_FEATURES

    // Parameters for sgemm (C = alpha*op(A)*op(B) + beta*C)
    // op(A): No transpose (N)
    // op(B): Transpose (T)
    // m: rows of op(A) and C = OUTPUT_FEATURES
    // n: columns of op(B) and C = INPUT_FEATURES
    // k: columns of op(A) and rows of op(B) = BATCH_SIZE
    // lda: leading dimension of A = OUTPUT_FEATURES (since column-major)
    // ldb: leading dimension of B = INPUT_FEATURES (since column-major)
    // ldc: leading dimension of C = OUTPUT_FEATURES (since column-major)
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS_ERROR(cublasSgemm(cublas_handle,
                                   CUBLAS_OP_N,        // op(A) = dL_dOutput, no transpose
                                   CUBLAS_OP_T,        // op(B) = Input, transpose
                                   OUTPUT_FEATURES,    // m
                                   INPUT_FEATURES,     // n
                                   BATCH_SIZE,         // k
                                   &alpha,             // alpha
                                   d_dL_dOutput,       // A (dL_dOutput)
                                   OUTPUT_FEATURES,    // lda
                                   d_Input,            // B (Input)
                                   INPUT_FEATURES,     // ldb
                                   &beta,              // beta
                                   d_dL_dW,            // C (dL_dW)
                                   OUTPUT_FEATURES));  // ldc
    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_weight_time_ms, start_gpu, stop_gpu));


    // --- GPU Bias Gradient Calculation (dL/dBias = sum(dL/dOutput over batch)) ---
    printf("Calculating bias gradients (dL/dBias) using custom kernel (GPU)...\n");
    int threads_per_block = 256;
    int blocks_per_grid = (OUTPUT_FEATURES + threads_per_block - 1) / threads_per_block;

    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu));
    calculate_bias_gradients<<<blocks_per_grid, threads_per_block>>>(d_dL_dOutput, d_dL_dBias, OUTPUT_FEATURES, BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu)); // Wait for kernel completion and ensure timing is accurate
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_bias_time_ms, start_gpu, stop_gpu));


    // --- Copy Results Device to Host ---
    printf("Copying results from device to host...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(h_dL_dW_gpu, d_dL_dW, dl_dw_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_dL_dBias_gpu, d_dL_dBias, dl_dbias_size, cudaMemcpyDeviceToHost));

    // --- CPU Calculation for Verification & Benchmarking ---
    printf("\nCalculating gradients on CPU for verification...\n");
    using Clock = std::chrono::high_resolution_clock;
    auto cpu_start_time = Clock::now();

    calculate_weight_gradients_cpu(h_dL_dOutput, h_Input, h_dL_dW_cpu, INPUT_FEATURES, OUTPUT_FEATURES, BATCH_SIZE);
    calculate_bias_gradients_cpu(h_dL_dOutput, h_dL_dBias_cpu, OUTPUT_FEATURES, BATCH_SIZE);

    auto cpu_end_time = Clock::now();
    double cpu_total_time_ms = std::chrono::duration<double, std::milli>(cpu_end_time - cpu_start_time).count();


    // --- Verification ---
    printf("\n--- Verification ---\n");
    bool weight_check = compare_gradients(h_dL_dW_cpu, h_dL_dW_gpu, dl_dw_size / sizeof(float), "Weight Gradients (dL/dW)");
    bool bias_check = compare_gradients(h_dL_dBias_cpu, h_dL_dBias_gpu, dl_dbias_size / sizeof(float), "Bias Gradients (dL/dBias)");
    printf("--------------------\n");

    // --- Benchmarking Results ---
    printf("\n--- Benchmarking ---\n");
    printf("CPU Total Time: %.3f ms\n", cpu_total_time_ms);
    printf("GPU Weight Gradient Time (cuBLAS): %.3f ms\n", gpu_weight_time_ms);
    printf("GPU Bias Gradient Time (Kernel):   %.3f ms\n", gpu_bias_time_ms);
    printf("GPU Total Computation Time:        %.3f ms\n", gpu_weight_time_ms + gpu_bias_time_ms);
    if (weight_check && bias_check) {
         printf("Speedup Factor (CPU Time / GPU Time): %.2fx\n", cpu_total_time_ms / (gpu_weight_time_ms + gpu_bias_time_ms));
    } else {
        printf("Speedup factor not calculated due to verification failure.\n");
    }
    printf("---------------------\n");


    /*
    // --- Optional: Print small section of results ---
    printf("\n--- Sample Results (GPU) ---\n");
    printf("dL/dW (first 5x5 elements):\n");
    for (int i = 0; i < 5 && i < OUTPUT_FEATURES; ++i) {
        for (int j = 0; j < 5 && j < INPUT_FEATURES; ++j) {
            // Accessing column-major dL_dW[col * rows + row]
            printf("%8.4f ", h_dL_dW_gpu[j * OUTPUT_FEATURES + i]);
        }
        printf("\n");
    }
    printf("\ndL/dBias (first 10 elements):\n");
    for (int i = 0; i < 10 && i < OUTPUT_FEATURES; ++i) {
        printf("%8.4f ", h_dL_dBias_gpu[i]);
    }
    printf("\n--------------------------\n");
    */

    // --- Cleanup ---
    printf("Cleaning up resources...\n");
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu));
    CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle));
    CHECK_CUDA_ERROR(cudaFree(d_Input));
    CHECK_CUDA_ERROR(cudaFree(d_dL_dOutput));
    CHECK_CUDA_ERROR(cudaFree(d_dL_dW));
    CHECK_CUDA_ERROR(cudaFree(d_dL_dBias));
    free(h_Input);
    free(h_dL_dOutput);
    free(h_dL_dW_gpu);
    free(h_dL_dBias_gpu);
    free(h_dL_dW_cpu);    // Free CPU result arrays
    free(h_dL_dBias_cpu); // Free CPU result arrays

    printf("Day 46 Finished Successfully.\n");
    return EXIT_SUCCESS;
}
