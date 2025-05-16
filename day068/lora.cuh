#ifndef LORA_CUH
#define LORA_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h> // For cuBLAS
#include <vector>
#include <stdexcept> // For std::runtime_error
#include <iostream> // For std::cerr

// Error checking macro (as per .clinerules)
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                      << ": " << cudaGetErrorString(err_) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err_)); \
        } \
    } while (0)

// cuBLAS error checking macro
const char* cublasGetErrorString(cublasStatus_t status); // Forward declaration for helper
#define CHECK_CUBLAS_ERROR(status) \
    do { \
        cublasStatus_t status_ = (status); \
        if (status_ != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << __FILE__ << " at line " << __LINE__ \
                      << ": " << cublasGetErrorString(status_) << std::endl; \
            throw std::runtime_error(cublasGetErrorString(status_)); \
        } \
    } while (0)


// Structure to hold LoRA parameters
struct LoRAParameters {
    // h_A and d_A represent the down-projection matrix A (rank x d_model)
    // h_B and d_B represent the up-projection matrix B (d_model x rank)
    float* h_A;     // Host pointer for matrix A (rank x d_model)
    float* h_B;     // Host pointer for matrix B (d_model x rank)
    float* d_A;     // Device pointer for matrix A (rank x d_model)
    float* d_B;     // Device pointer for matrix B (d_model x rank)

    int d_model;    // Original model dimension (input/output features)
    int rank;       // Rank of the LoRA decomposition
    float alpha;    // Scaling factor

    cublasHandle_t cublas_handle; // cuBLAS handle
};

// Function declarations

/**
 * @brief Initializes LoRA parameters on both host and device.
 * @param params Reference to LoRAParameters struct.
 * @param d_model Dimension of the original model.
 * @param rank Rank of the LoRA decomposition.
 * @param alpha Scaling factor.
 */
void initializeLoRAParameters(LoRAParameters& params, int d_model, int rank, float alpha);

/**
 * @brief Frees LoRA parameters from host and device memory.
 * @param params Reference to LoRAParameters struct.
 */
void freeLoRAParameters(LoRAParameters& params);

/**
 * @brief Applies LoRA transformation to an input vector on the GPU.
 *        output = input + (B * A * input) * (alpha / rank)
 *        More accurately, for a weight matrix W_0 and its update W_0 + BA:
 *        h = W_0 x + (alpha/rank) * B A x
 *        Here, we compute the LoRA part: (alpha/rank) * B A x
 *        The input 'x' is a vector of size d_model.
 *        The output 'lora_output' is a vector of size d_model.
 *
 * @param d_input Pointer to the device input vector (size d_model).
 * @param d_lora_output Pointer to the device output vector where LoRA result is stored (size d_model).
 * @param params LoRAParameters struct containing d_A (rank x d_model), d_B (d_model x rank), d_model, rank, alpha.
 */
void loraForwardGPU(const float* d_input, float* d_lora_output, const LoRAParameters& params);

/**
 * @brief Applies LoRA transformation to an input vector on the CPU.
 *        Computes (alpha/rank) * B A x
 *
 * @param h_input Pointer to the host input vector (size d_model).
 * @param h_lora_output Pointer to the host output vector where LoRA result is stored (size d_model).
 * @param params LoRAParameters struct containing h_A (rank x d_model), h_B (d_model x rank), d_model, rank, alpha.
 */
void loraForwardCPU(const float* h_input, float* h_lora_output, const LoRAParameters& params);


// Kernel for matrix multiplication: C = A * B
// A: m x k, B: k x n, C: m x n
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int m, int n, int k);

// Kernel for LoRA: computes B * (A * input_vec)
// Standard LoRA: A is (rank x d_model), B is (d_model x rank)
// input_vec: d_model x 1
// temp_vec (A * input_vec): rank x 1
// output_vec (B * temp_vec): d_model x 1

// Kernel to compute temp_output = A * x
// A: rank x d_model
// x: d_model x 1
// temp_output: rank x 1
__global__ void loraMatVecMulKernel_A_x(const float* A, const float* x, float* temp_output, int d_model, int rank);

// Kernel to compute final_output = B * temp_input
// B: d_model x rank
// temp_input: rank x 1
// final_output: d_model x 1
__global__ void loraMatVecMulKernel_B_temp(const float* B, const float* temp_input, float* final_output, int d_model, int rank);

// Kernel to scale and add: final_output = final_output * (alpha/rank)
// This can be part of loraMatVecMulKernel_B_temp or a separate small kernel.
__global__ void scaleVectorKernel(float* vector, int size, float scale_factor);


#endif // LORA_CUH
