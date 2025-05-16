#include "lora.cuh"
#include <random> // For std::mt19937 and std::normal_distribution
#include <vector>
#include <cmath> // For std::sqrt

// Helper function for CHECK_CUBLAS_ERROR macro
const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "Unknown cuBLAS error";
    }
}

// Helper function to initialize a matrix with random values (He initialization style for A, zeros for B)
void initialize_matrix_random(float* matrix, int rows, int cols, bool is_A) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if (is_A) {
        // He initialization for matrix A (often used for ReLU-like activations)
        // std::normal_distribution<> distrib(0.0, std::sqrt(2.0 / cols)); // cols is fan_in
        // For LoRA, a common initialization for A is Kaiming uniform or normal, and B is zeros.
        // Let's use a simpler normal distribution for A for now.
        std::normal_distribution<> distrib(0.0, 0.02); // Small random values
        for (int i = 0; i < rows * cols; ++i) {
            matrix[i] = static_cast<float>(distrib(gen));
        }
    } else {
        // Initialize matrix B to zeros
        for (int i = 0; i < rows * cols; ++i) {
            matrix[i] = 0.0f;
        }
    }
}

void initializeLoRAParameters(LoRAParameters& params, int d_model, int rank, float alpha) {
    params.d_model = d_model;
    params.rank = rank;
    params.alpha = alpha;

    // Allocate host memory
    // A: rank x d_model
    // B: d_model x rank
    params.h_A = new float[rank * d_model];
    params.h_B = new float[d_model * rank];

    // Initialize host matrices
    // A: rank x d_model (down-projection)
    initialize_matrix_random(params.h_A, rank, d_model, true); // Initialize A with small random numbers
    // B: d_model x rank (up-projection)
    initialize_matrix_random(params.h_B, d_model, rank, false); // Initialize B with zeros

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&params.d_A, rank * d_model * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&params.d_B, d_model * rank * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(params.d_A, params.h_A, rank * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(params.d_B, params.h_B, d_model * rank * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create cuBLAS handle
    CHECK_CUBLAS_ERROR(cublasCreate(&params.cublas_handle));

    // Note: h_A/d_A are (rank x d_model), h_B/d_B are (d_model x rank).
}

void freeLoRAParameters(LoRAParameters& params) {
    // Free host memory
    delete[] params.h_A;
    delete[] params.h_B;
    params.h_A = nullptr;
    params.h_B = nullptr;

    // Free device memory
    if (params.d_A) CHECK_CUDA_ERROR(cudaFree(params.d_A));
    if (params.d_B) CHECK_CUDA_ERROR(cudaFree(params.d_B));
    params.d_A = nullptr;
    params.d_B = nullptr;

    // Destroy cuBLAS handle
    if (params.cublas_handle) CHECK_CUBLAS_ERROR(cublasDestroy(params.cublas_handle));
}

// --- Kernels (Kept for reference, but loraForwardGPU will use cuBLAS) ---

// Generic Matrix Multiplication Kernel: C = A * B
// A (m x k), B (k x n), C (m x n)
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Kernel for temp_vec = A * input_vec
// A: rank x d_model (down-projection matrix)
// x: d_model x 1 (input vector)
// temp_output: rank x 1 (intermediate vector)
__global__ void loraMatVecMulKernel_A_x(const float* A, const float* x, float* temp_output, int d_model, int rank) {
    int r = blockIdx.x * blockDim.x + threadIdx.x; // Index for the output (row of A, element of temp_output)

    if (r < rank) {
        float sum = 0.0f;
        for (int i = 0; i < d_model; ++i) {
            sum += A[r * d_model + i] * x[i]; // A is row-major: A[r][i]
        }
        temp_output[r] = sum;
    }
}

// Kernel for final_output = B * temp_input
// B: d_model x rank (up-projection matrix)
// temp_input: rank x 1 (intermediate vector)
// final_output: d_model x 1 (LoRA adjustment vector)
__global__ void loraMatVecMulKernel_B_temp(const float* B, const float* temp_input, float* final_output, int d_model, int rank) {
    int d = blockIdx.x * blockDim.x + threadIdx.x; // Index for the output (row of B, element of final_output)

    if (d < d_model) {
        float sum = 0.0f;
        for (int i = 0; i < rank; ++i) {
            sum += B[d * rank + i] * temp_input[i];
        }
        final_output[d] = sum;
    }
}

// Kernel to scale a vector
__global__ void scaleVectorKernel(float* vector, int size, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vector[idx] *= scale_factor;
    }
}


void loraForwardGPU(const float* d_input, float* d_lora_output, const LoRAParameters& params) {
    // d_input is (d_model x 1)
    // params.d_A is (rank x d_model)
    // params.d_B is (d_model x rank)

    // Temporary device memory for A*x result (rank x 1)
    float* d_temp_vec;
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_vec, params.rank * sizeof(float)));

    const float const_alpha_one = 1.0f; // Alpha for sgemv
    const float const_beta_zero = 0.0f; // Beta for sgemv

    // 1. Compute temp_vec = A * input
    // A is (rank x d_model) row-major. x is (d_model x 1). temp_vec is (rank x 1).
    // cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    // For A (M rows, N cols) row-major, to compute Y = A*X:
    // trans = CUBLAS_OP_N (no transpose)
    // m = M (number of rows in A, which is params.rank)
    // n = N (number of columns in A, which is params.d_model)
    // A = params.d_A
    // lda = N (leading dimension of A, which is params.d_model for row-major)
    // x = d_input
    // incx = 1
    // y = d_temp_vec
    // incy = 1
    CHECK_CUBLAS_ERROR(cublasSgemv(params.cublas_handle, 
                                   CUBLAS_OP_N, 
                                   params.rank, params.d_model, 
                                   &const_alpha_one, 
                                   params.d_A, params.d_model, // A is rank x d_model, lda is d_model
                                   d_input, 1, 
                                   &const_beta_zero, 
                                   d_temp_vec, 1));

    // 2. Compute lora_output = B * temp_vec
    // B is (d_model x rank) row-major. temp_vec is (rank x 1). lora_output is (d_model x 1).
    // m = params.d_model (number of rows in B)
    // n = params.rank (number of columns in B)
    // B = params.d_B
    // lda = params.rank (leading dimension of B, which is params.rank for row-major)
    // x = d_temp_vec
    // y = d_lora_output
    CHECK_CUBLAS_ERROR(cublasSgemv(params.cublas_handle,
                                   CUBLAS_OP_N,
                                   params.d_model, params.rank,
                                   &const_alpha_one,
                                   params.d_B, params.rank, // B is d_model x rank, lda is rank
                                   d_temp_vec, 1,
                                   &const_beta_zero,
                                   d_lora_output, 1));

    // 3. Scale lora_output by (alpha / rank)
    float scale_factor = params.alpha / static_cast<float>(params.rank);
    // cublasSscal(handle, n, alpha, x, incx)
    CHECK_CUBLAS_ERROR(cublasSscal(params.cublas_handle,
                                   params.d_model,
                                   &scale_factor,
                                   d_lora_output, 1));
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure cuBLAS operations are complete before freeing d_temp_vec

    // Free temporary device memory
    CHECK_CUDA_ERROR(cudaFree(d_temp_vec));
}


void loraForwardCPU(const float* h_input, float* h_lora_output, const LoRAParameters& params) {
    // h_A is (rank x d_model)
    // h_B is (d_model x rank)
    // h_input is (d_model x 1)

    std::vector<float> temp_vec(params.rank);

    // 1. temp_vec = A * input
    for (int r = 0; r < params.rank; ++r) {
        float sum = 0.0f;
        for (int i = 0; i < params.d_model; ++i) {
            sum += params.h_A[r * params.d_model + i] * h_input[i];
        }
        temp_vec[r] = sum;
    }

    // 2. lora_output = B * temp_vec
    for (int d = 0; d < params.d_model; ++d) {
        float sum = 0.0f;
        for (int i = 0; i < params.rank; ++i) {
            sum += params.h_B[d * params.rank + i] * temp_vec[i];
        }
        h_lora_output[d] = sum;
    }

    // 3. Scale lora_output by (alpha / rank)
    float scale_factor = params.alpha / static_cast<float>(params.rank);
    for (int d = 0; d < params.d_model; ++d) {
        h_lora_output[d] *= scale_factor;
    }
}
