#ifndef RMS_NORM_CUH
#define RMS_NORM_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Error checking macros
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUBLAS_ERROR(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, status); \
            exit(1); \
        } \
    } while(0)

// Constants
#define EPSILON 1e-5f
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// RMS Normalization functions
void rms_norm_cpu(const float* input, float* output, const float* gamma, 
                  int batch_size, int seq_len, int hidden_dim);

void rms_norm_gpu(const float* input, float* output, const float* gamma,
                  int batch_size, int seq_len, int hidden_dim);

void layer_norm_cpu(const float* input, float* output, const float* gamma, const float* beta,
                    int batch_size, int seq_len, int hidden_dim);

void layer_norm_gpu(const float* input, float* output, const float* gamma, const float* beta,
                    int batch_size, int seq_len, int hidden_dim);

// Utility functions
void initialize_data(float* data, int size, float mean = 0.0f, float std = 1.0f);
void print_tensor(const float* data, int batch_size, int seq_len, int hidden_dim, 
                  const char* name, int max_elements = 10);
bool verify_results(const float* cpu_result, const float* gpu_result, int size, 
                    float tolerance = 1e-4f);

// Performance benchmarking
struct BenchmarkResult {
    float cpu_time_ms;
    float gpu_time_ms;
    float speedup;
    bool correctness_passed;
};

BenchmarkResult benchmark_normalization(int batch_size, int seq_len, int hidden_dim, 
                                       int num_iterations = 100);

#endif // RMS_NORM_CUH
