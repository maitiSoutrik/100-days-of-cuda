#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include <math.h>

// Error checking macros
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS_ERROR(err) \
    do { \
        cublasStatus_t err_ = (err); \
        if (err_ != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            /* Provide more detail for common cuBLAS errors */ \
            if (err_ == CUBLAS_STATUS_NOT_INITIALIZED) fprintf(stderr, " Error detail: CUBLAS_STATUS_NOT_INITIALIZED\n"); \
            else if (err_ == CUBLAS_STATUS_ALLOC_FAILED) fprintf(stderr, " Error detail: CUBLAS_STATUS_ALLOC_FAILED\n"); \
            else if (err_ == CUBLAS_STATUS_INVALID_VALUE) fprintf(stderr, " Error detail: CUBLAS_STATUS_INVALID_VALUE\n"); \
            else if (err_ == CUBLAS_STATUS_ARCH_MISMATCH) fprintf(stderr, " Error detail: CUBLAS_STATUS_ARCH_MISMATCH\n"); \
            else if (err_ == CUBLAS_STATUS_MAPPING_ERROR) fprintf(stderr, " Error detail: CUBLAS_STATUS_MAPPING_ERROR\n"); \
            else if (err_ == CUBLAS_STATUS_EXECUTION_FAILED) fprintf(stderr, " Error detail: CUBLAS_STATUS_EXECUTION_FAILED\n"); \
            else if (err_ == CUBLAS_STATUS_INTERNAL_ERROR) fprintf(stderr, " Error detail: CUBLAS_STATUS_INTERNAL_ERROR\n"); \
            else if (err_ == CUBLAS_STATUS_NOT_SUPPORTED) fprintf(stderr, " Error detail: CUBLAS_STATUS_NOT_SUPPORTED\n"); \
            else fprintf(stderr, " Error detail: Unknown cuBLAS error code %d\n", err_); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// Kernel to add bias and apply activation function (ReLU example) using 2D grid/block
__global__ void add_bias_activate_2d(float* output, const float* bias, int M_batch_size, int N_output_features) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // corresponds to N (output_features)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // corresponds to M (batch_size)

    // Check bounds
    if (row < M_batch_size && col < N_output_features) {
         int idx = row * N_output_features + col; // Linear index for the output matrix (MxN)
         float bias_val = bias[col]; // Bias applies per output feature/neuron (index `col`)
         output[idx] = fmaxf(0.0f, output[idx] + bias_val); // Add bias and apply ReLU
    }
}

// Helper function to initialize matrix with random values
void initialize_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Random values between -1 and 1
    }
}

// Helper function to print matrix (optional, for debugging)
void print_matrix(const float *mat, int rows, int cols, const char *label) {
    printf("%s (%d x %d):\n", label, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.4f ", mat[i * cols + j]); // Assuming row-major
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    srand(time(NULL));

    // --- Configuration ---
    // Larger dimensions to highlight GPU vs CPU difference
    int batch_size = 64;      // M: Number of input samples in the batch
    int input_features = 1024;// K: Dimension of input features
    int output_features = 512;// N: Dimension of output features (number of neurons)

    printf("Configuration:\n");
    printf(" Batch Size (M): %d\n", batch_size);
    printf(" Input Features (K): %d\n", input_features);
    printf(" Output Features (N): %d\n", output_features);
    printf("-----------------------------\n");

    // --- Host Data Initialization ---
    float *h_input = (float *)malloc(batch_size * input_features * sizeof(float));
    float *h_weights = (float *)malloc(input_features * output_features * sizeof(float));
    float *h_bias = (float *)malloc(output_features * sizeof(float));
    float *h_output_gpu = (float *)malloc(batch_size * output_features * sizeof(float)); // To store GPU result
    float *h_output_cpu = (float *)malloc(batch_size * output_features * sizeof(float)); // For CPU verification

    if (!h_input || !h_weights || !h_bias || !h_output_gpu || !h_output_cpu) {
        fprintf(stderr, "Failed to allocate host memory\n");
        // Consider adding cleanup for partially allocated buffers if needed
        return EXIT_FAILURE;
    }

    initialize_matrix(h_input, batch_size, input_features);
    initialize_matrix(h_weights, input_features, output_features);
    initialize_matrix(h_bias, 1, output_features); // Bias vector

    // print_matrix(h_input, batch_size, input_features, "Host Input");
    // print_matrix(h_weights, input_features, output_features, "Host Weights");
    // print_matrix(h_bias, 1, output_features, "Host Bias");

    // --- Device Data Allocation ---
    float *d_input, *d_weights, *d_bias, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, batch_size * input_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights, input_features * output_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bias, output_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, batch_size * output_features * sizeof(float)));

    // --- Data Transfer: Host -> Device ---
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, batch_size * input_features * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, h_weights, input_features * output_features * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias, h_bias, output_features * sizeof(float), cudaMemcpyHostToDevice));

    // --- cuBLAS Initialization ---
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    // --- GEMM Calculation (W * input -> output = C = alpha*A*B + beta*C) ---
    // W (weights) is K x N (input_features x output_features)
    // input is M x K (batch_size x input_features)
    // output = input * W -> (M x K) * (K x N) = M x N (batch_size x output_features)
    // cuBLAS expects column-major by default. Let's treat our row-major matrices as column-major
    // from cuBLAS's perspective by swapping dimensions and operands.
    // C = B^T * A^T, where B = W (N x K) and A = input (K x M) --> Result C is N x M
    // OR, we can tell cuBLAS our matrices are row-major if supported, or handle transposition.
    // Let's stick to the common approach: A=Weights(KxN), B=Input(MxK).
    // We want C(MxN) = B(MxK) * A(KxN).
    // In cuBLAS (column-major): C'(NxM) = A'(NxK) * B'(KxM).
    // Here A' is Weights (col-major view), B' is Input (col-major view).
    // Let's use the standard row-major interpretation and pass appropriate parameters to GEMM.
    // C(m,n) = alpha * A(m,k) * B(k,n) + beta * C(m,n)
    // A = h_input (M x K) -> rows=M, cols=K
    // B = h_weights (K x N) -> rows=K, cols=N
    // C = h_output_gpu (M x N) -> rows=M, cols=N
    // In cuBLAS: we need op(A), op(B). Let's use non-transposed for row-major:
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_weights, N, d_input, K, &beta, d_output, N)
    // This interprets them as column-major.
    // Let's try the convention C = alpha * A * B + beta * C where A=input, B=weights
    // A: d_input (M rows, K cols)
    // B: d_weights (K rows, N cols)
    // C: d_output (M rows, N cols)
    // cublasSgemm expects matrices in COLUMN MAJOR.
    // To multiply row-major A (MxK) * B (KxN) = C (MxN), use:
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_weights, N, d_input, K, &beta, d_output, N)
    // This seems counter-intuitive but is a common pattern. It computes C_colmajor = B_colmajor * A_colmajor
    // where B_colmajor is our d_weights (KxN interpreted as NxK col-major), A_colmajor is our d_input (MxK interpreted as KxM col-major)
    // The result C_colmajor (NxM) when read row-by-row gives the correct C (MxN) result for A*B.

    float alpha = 1.0f;
    float beta = 0.0f;
    // Parameters: handle, op(B), op(A), N, M, K, alpha, B_ptr, ldb, A_ptr, lda, beta, C_ptr, ldc
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   output_features, batch_size, input_features, // N, M, K
                                   &alpha,
                                   d_weights, output_features,                  // B matrix (Weights), leading dim N
                                   d_input, input_features,                     // A matrix (Input), leading dim K
                                   &beta,
                                   d_output, output_features));                 // C matrix (Output), leading dim N

    // --- Bias Addition and Activation Kernel Launch ---
    int num_output_elements = batch_size * output_features;
    int threads_per_block = 256;
    int blocks_per_grid = (num_output_elements + threads_per_block - 1) / threads_per_block;

    // We need to reshape bias for element-wise add if thinking per batch item
    // Or simply add bias[neuron_index] to output[batch_item * num_neurons + neuron_index]
    // The kernel needs modification to handle batch dimension correctly.

    // --- V2 Kernel: Add bias and activate ---
    // Let's make the kernel handle the 2D structure explicitly for clarity
    dim3 gridDim( (output_features + 15) / 16, (batch_size + 15) / 16 );
    dim3 blockDim(16, 16); // Example 2D block

    // Kernel signature needs update to match 2D launch
    // __global__ void add_bias_activate_2d(float* output, const float* bias, int M, int N)
    // int col = blockIdx.x * blockDim.x + threadIdx.x; // corresponds to N (output_features)
    // int row = blockIdx.y * blockDim.y + threadIdx.y; // corresponds to M (batch_size)
    // if (row < M && col < N) {
    //      int idx = row * N + col;
    //      float bias_val = bias[col]; // Bias applies per output feature
    //      output[idx] = fmaxf(0.0f, output[idx] + bias_val); // ReLU
    // }

    // Replace 1D kernel launch with 2D launch using the modified kernel
    add_bias_activate_2d<<<gridDim, blockDim>>>(d_output, d_bias, batch_size, output_features);

    // Remove the old 1D kernel launch call:
    // add_bias_and_activate_relu<<<blocks_per_grid, threads_per_block>>>(d_output, d_bias, num_output_elements);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel completion

    // --- Data Transfer: Device -> Host ---
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, batch_size * output_features * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nGPU computation complete. Output received.\n");

    // --- CPU Verification ---
    printf("Performing CPU computation for verification...\n");
    for (int m = 0; m < batch_size; ++m) { // Loop over batch items
        for (int n = 0; n < output_features; ++n) { // Loop over output neurons
            float sum = 0.0f;
            for (int k = 0; k < input_features; ++k) { // Dot product
                sum += h_input[m * input_features + k] * h_weights[k * output_features + n];
            }
            // Add bias and apply ReLU
            h_output_cpu[m * output_features + n] = fmaxf(0.0f, sum + h_bias[n]);
        }
    }
    printf("CPU computation complete.\n");

    // Compare CPU and GPU results
    float max_diff = 0.0f;
    float mse = 0.0f;
    for (int i = 0; i < batch_size * output_features; ++i) {
        float diff = fabsf(h_output_gpu[i] - h_output_cpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        mse += diff * diff;
    }
    mse /= (batch_size * output_features);

    printf("\nVerification Results:\n");
    printf(" Max Absolute Difference: %e\n", max_diff);
    printf(" Mean Squared Error (MSE): %e\n", mse);

    // Check if results are close enough
    float tolerance = 1e-5f;
    if (max_diff < tolerance) {
        printf(" Verification PASSED (Max Diff < %e)\n", tolerance);
    } else {
        printf(" Verification FAILED (Max Diff >= %e)\n", tolerance);
        // Optionally print matrices if verification fails
        // print_matrix(h_output_gpu, batch_size, output_features, "GPU Output");
        // print_matrix(h_output_cpu, batch_size, output_features, "CPU Output");
    }

    // --- Cleanup ---
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_weights));
    CHECK_CUDA_ERROR(cudaFree(d_bias));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    free(h_input);
    free(h_weights);
    free(h_bias);
    free(h_output_gpu);
    free(h_output_cpu); // Free the CPU output buffer

    printf("\nResources freed. Exiting.\n");
    return EXIT_SUCCESS;
}
