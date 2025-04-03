#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Sigmoid activation function for CPU
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Sigmoid activation function for GPU
__device__ float sigmoid_device(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// GLU kernel implementation
__global__ void glu_kernel(float* output, const float* input, 
                          const float* W, const float* b,
                          const float* V, const float* c,
                          int batch_size, int input_dim, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * output_dim) {
        int batch_idx = idx / output_dim;
        int feature_idx = idx % output_dim;
        
        // Calculate linear transformation A = Wx + b
        float A = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            A += W[feature_idx * input_dim + i] * input[batch_idx * input_dim + i];
        }
        A += b[feature_idx];
        
        // Calculate linear transformation B = Vx + c
        float B = 0.0f;
        for (int i = 0; i < input_dim; i++) {
            B += V[feature_idx * input_dim + i] * input[batch_idx * input_dim + i];
        }
        B += c[feature_idx];
        
        // Apply sigmoid to B to get the gate
        float gate = sigmoid_device(B);
        
        // Apply gating: output = A ⊙ sigmoid(B)
        output[idx] = A * gate;
    }
}

// CPU implementation of GLU for verification
void glu_cpu(float* output, const float* input, 
             const float* W, const float* b,
             const float* V, const float* c,
             int batch_size, int input_dim, int output_dim) {
    
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (int feature_idx = 0; feature_idx < output_dim; feature_idx++) {
            // Calculate linear transformation A = Wx + b
            float A = 0.0f;
            for (int i = 0; i < input_dim; i++) {
                A += W[feature_idx * input_dim + i] * input[batch_idx * input_dim + i];
            }
            A += b[feature_idx];
            
            // Calculate linear transformation B = Vx + c
            float B = 0.0f;
            for (int i = 0; i < input_dim; i++) {
                B += V[feature_idx * input_dim + i] * input[batch_idx * input_dim + i];
            }
            B += c[feature_idx];
            
            // Apply sigmoid to B to get the gate
            float gate = sigmoid(B);
            
            // Apply gating: output = A ⊙ sigmoid(B)
            output[batch_idx * output_dim + feature_idx] = A * gate;
        }
    }
}

// Initialize weights and biases with random values
void initialize_parameters(float* W, float* b, float* V, float* c, 
                          int input_dim, int output_dim) {
    // Simple Xavier/Glorot initialization for weights
    float scale_w = sqrtf(2.0f / (input_dim + output_dim));
    float scale_b = 0.1f;
    
    for (int i = 0; i < output_dim * input_dim; i++) {
        W[i] = scale_w * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        V[i] = scale_w * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
    
    for (int i = 0; i < output_dim; i++) {
        b[i] = scale_b * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        c[i] = scale_b * ((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
}

// Generate random input data
void generate_input_data(float* input, int batch_size, int input_dim) {
    for (int i = 0; i < batch_size * input_dim; i++) {
        input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

// Calculate mean squared error between two arrays
float calculate_mse(const float* arr1, const float* arr2, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = arr1[i] - arr2[i];
        sum += diff * diff;
    }
    return sum / size;
}

int main() {
    // Set random seed for reproducibility
    srand(42);
    
    // Define dimensions
    const int batch_size = 32;
    const int input_dim = 128;
    const int output_dim = 64;
    
    // Allocate host memory
    float *h_input, *h_W, *h_b, *h_V, *h_c;
    float *h_output_gpu, *h_output_cpu;
    
    h_input = (float*)malloc(batch_size * input_dim * sizeof(float));
    h_W = (float*)malloc(output_dim * input_dim * sizeof(float));
    h_b = (float*)malloc(output_dim * sizeof(float));
    h_V = (float*)malloc(output_dim * input_dim * sizeof(float));
    h_c = (float*)malloc(output_dim * sizeof(float));
    h_output_gpu = (float*)malloc(batch_size * output_dim * sizeof(float));
    h_output_cpu = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Initialize parameters and input data
    initialize_parameters(h_W, h_b, h_V, h_c, input_dim, output_dim);
    generate_input_data(h_input, batch_size, input_dim);
    
    // Allocate device memory
    float *d_input, *d_W, *d_b, *d_V, *d_c, *d_output;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_W, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, output_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_V, output_dim * input_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, output_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, batch_size * output_dim * sizeof(float)));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_W, h_W, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, output_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, h_V, output_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_c, h_c, output_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Define grid and block dimensions
    int block_size = 256;
    int grid_size = (batch_size * output_dim + block_size - 1) / block_size;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Measure GPU execution time
    cudaEventRecord(start);
    
    // Launch GLU kernel
    glu_kernel<<<grid_size, block_size>>>(d_output, d_input, d_W, d_b, d_V, d_c, 
                                         batch_size, input_dim, output_dim);
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Measure CPU execution time
    clock_t cpu_start = clock();
    
    // Run CPU implementation for verification
    glu_cpu(h_output_cpu, h_input, h_W, h_b, h_V, h_c, batch_size, input_dim, output_dim);
    
    clock_t cpu_end = clock();
    float cpu_time = 1000.0f * (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    
    // Calculate MSE between CPU and GPU results
    float mse = calculate_mse(h_output_cpu, h_output_gpu, batch_size * output_dim);
    
    // Print results
    printf("GLU Implementation Results:\n");
    printf("Batch Size: %d, Input Dimension: %d, Output Dimension: %d\n", 
           batch_size, input_dim, output_dim);
    printf("GPU Execution Time: %.4f ms\n", gpu_time);
    printf("CPU Execution Time: %.4f ms\n", cpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Mean Squared Error between CPU and GPU results: %.10f\n", mse);
    
    // Print a few sample outputs for verification
    printf("\nSample Outputs (first 5 elements of first batch):\n");
    printf("CPU: ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", h_output_cpu[i]);
    }
    printf("\nGPU: ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", h_output_gpu[i]);
    }
    printf("\n");
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_V);
    cudaFree(d_c);
    cudaFree(d_output);
    
    // Free host memory
    free(h_input);
    free(h_W);
    free(h_b);
    free(h_V);
    free(h_c);
    free(h_output_gpu);
    free(h_output_cpu);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
