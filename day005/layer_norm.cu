#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU implementation of Layer Normalization
void layerNormCPU(const float* input, float* output, int rows, int cols, float epsilon = 1e-5) {
    for (int row = 0; row < rows; row++) {
        // Compute mean
        float mean = 0.0f;
        for (int col = 0; col < cols; col++) {
            mean += input[row * cols + col];
        }
        mean /= cols;
        
        // Compute variance
        float variance = 0.0f;
        for (int col = 0; col < cols; col++) {
            float diff = input[row * cols + col] - mean;
            variance += diff * diff;
        }
        variance /= cols;
        
        // Normalize
        float stddev = sqrtf(variance + epsilon);
        for (int col = 0; col < cols; col++) {
            output[row * cols + col] = (input[row * cols + col] - mean) / stddev;
        }
    }
}

// Basic Layer Normalization kernel (one thread per row)
__global__ void layerNormBasicKernel(const float* input, float* output, int rows, int cols, float epsilon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Compute mean
        float mean = 0.0f;
        for (int col = 0; col < cols; col++) {
            mean += input[row * cols + col];
        }
        mean /= cols;
        
        // Compute variance
        float variance = 0.0f;
        for (int col = 0; col < cols; col++) {
            float diff = input[row * cols + col] - mean;
            variance += diff * diff;
        }
        variance /= cols;
        
        // Normalize
        float stddev = sqrtf(variance + epsilon);
        for (int col = 0; col < cols; col++) {
            output[row * cols + col] = (input[row * cols + col] - mean) / stddev;
        }
    }
}

// Optimized Layer Normalization kernel using shared memory
__global__ void layerNormSharedKernel(const float* input, float* output, int rows, int cols, float epsilon) {
    // Calculate row index - one thread block handles one row
    int row = blockIdx.x;
    int tid = threadIdx.x;  // Thread ID within the block
    
    if (row < rows) {
        // Use shared memory for row-wise computation
        extern __shared__ float shared[];
        
        // Copy row data to shared memory with coalesced access
        for (int col = tid; col < cols; col += blockDim.x) {
            shared[col] = input[row * cols + col];
        }
        __syncthreads();
        
        // First thread in block computes mean and variance
        if (tid == 0) {
            // Compute mean
            float mean = 0.0f;
            for (int col = 0; col < cols; col++) {
                mean += shared[col];
            }
            mean /= cols;
            
            // Store mean in shared memory for other threads to use
            shared[cols] = mean;
            
            // Compute variance
            float variance = 0.0f;
            for (int col = 0; col < cols; col++) {
                float diff = shared[col] - mean;
                variance += diff * diff;
            }
            variance /= cols;
            
            // Store stddev in shared memory for other threads to use
            shared[cols + 1] = sqrtf(variance + epsilon);
        }
        __syncthreads();
        
        // All threads read the mean and stddev computed by thread 0
        float mean = shared[cols];
        float stddev = shared[cols + 1];
        
        // All threads normalize their portion of the data
        for (int col = tid; col < cols; col += blockDim.x) {
            output[row * cols + col] = (shared[col] - mean) / stddev;
        }
    }
}



// Utility function to initialize matrix with random values
void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // Range [-1, 1]
    }
}

// Utility function to verify results
bool verifyResults(const float* cpuOutput, const float* gpuOutput, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; i++) {
        if (std::abs(cpuOutput[i] - gpuOutput[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": CPU = " << cpuOutput[i] 
                      << ", GPU = " << gpuOutput[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Utility function to print matrix (for small matrices)
void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    int rows = (argc > 1) ? atoi(argv[1]) : 1024;
    int cols = (argc > 2) ? atoi(argv[2]) : 256;
    float epsilon = 1e-5f;
    
    std::cout << "Layer Normalization on matrix of size " << rows << "x" << cols << std::endl;
    
    // Allocate host memory
    size_t matrix_size = rows * cols * sizeof(float);
    float* h_input = (float*)malloc(matrix_size);
    float* h_output_cpu = (float*)malloc(matrix_size);
    float* h_output_basic = (float*)malloc(matrix_size);
    float* h_output_shared = (float*)malloc(matrix_size);

    
    // Initialize input matrix
    srand(42); // For reproducibility
    initializeMatrix(h_input, rows * cols);
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_output, matrix_size));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, matrix_size, cudaMemcpyHostToDevice));
    
    // CPU implementation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    layerNormCPU(h_input, h_output_cpu, rows, cols, epsilon);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = cpu_end - cpu_start;
    
    // Basic GPU implementation
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up run
    layerNormBasicKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols, epsilon);
    
    // Timed run - Basic kernel
    cudaEventRecord(start);
    layerNormBasicKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols, epsilon);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float basic_time_ms = 0;
    cudaEventElapsedTime(&basic_time_ms, start, stop);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output_basic, d_output, matrix_size, cudaMemcpyDeviceToHost));
    
    // Shared memory implementation
    int threadsPerBlock = 256; // Use 256 threads per block
    int blocksPerGrid = rows;  // One block per row
    // Need space for row data + mean + stddev
    size_t shared_mem_size = (cols + 2) * sizeof(float);
    
    // Timed run - Shared memory kernel
    cudaEventRecord(start);
    layerNormSharedKernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_input, d_output, rows, cols, epsilon);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float shared_time_ms = 0;
    cudaEventElapsedTime(&shared_time_ms, start, stop);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output_shared, d_output, matrix_size, cudaMemcpyDeviceToHost));
    

    
    // Verify results
    bool basic_correct = verifyResults(h_output_cpu, h_output_basic, rows * cols);
    bool shared_correct = verifyResults(h_output_cpu, h_output_shared, rows * cols);

    
    // Print results
    std::cout << "CPU Time: " << cpu_time.count() * 1000 << " ms" << std::endl;
    std::cout << "GPU Basic Kernel Time: " << basic_time_ms << " ms, Verification: " 
              << (basic_correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "GPU Shared Memory Kernel Time: " << shared_time_ms << " ms, Verification: " 
              << (shared_correct ? "PASSED" : "FAILED") << std::endl;

    
    // Calculate speedups
    float cpu_time_ms = cpu_time.count() * 1000;
    std::cout << "\nSpeedups:" << std::endl;
    std::cout << "Basic Kernel vs CPU: " << cpu_time_ms / basic_time_ms << "x" << std::endl;
    std::cout << "Shared Memory Kernel vs CPU: " << cpu_time_ms / shared_time_ms << "x" << std::endl;

    
    // Print a small sample of the matrices if they are large
    if (rows <= 5 && cols <= 10) {
        printMatrix(h_input, rows, cols, "Input");
        printMatrix(h_output_cpu, rows, cols, "CPU Output");
        printMatrix(h_output_shared, rows, cols, "GPU Shared Memory Output");
    }
    
    // Clean up
    free(h_input);
    free(h_output_cpu);
    free(h_output_basic);
    free(h_output_shared);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
