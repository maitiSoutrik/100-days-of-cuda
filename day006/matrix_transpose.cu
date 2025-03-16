#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <iomanip>

// Define the size of the matrix
#define WIDTH 1024
#define HEIGHT 1024

// CUDA kernel for matrix transposition
__global__ void transposeMatrix(const float* input, float* output, int width, int height) {
    // Calculate the row and column index of the element
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Perform the transposition if within bounds
    if (x < width && y < height) {
        int inputIndex = y * width + x;
        int outputIndex = x * height + y;
        output[outputIndex] = input[inputIndex];
    }
}

// Macro to check for CUDA errors
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(error) << " (" << error << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU implementation of matrix transpose
void transposeMatrixCPU(const float* input, float* output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int inputIndex = y * width + x;
            int outputIndex = x * height + y;
            output[outputIndex] = input[inputIndex];
        }
    }
}

// Function to verify results
bool verifyResults(const float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

// Function to print matrix (for small matrices only)
void printMatrix(const float* matrix, int width, int height, const char* name) {
    if (width > 16 || height > 16) {
        std::cout << name << " is too large to print" << std::endl;
        return;
    }
    
    std::cout << name << ":" << std::endl;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << std::setw(6) << matrix[y * width + x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    int width = WIDTH;
    int height = HEIGHT;
    size_t size = width * height * sizeof(float);
    
    std::cout << "Matrix size: " << width << "x" << height << std::endl;

    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output_gpu = (float*)malloc(size);
    float* h_output_cpu = (float*)malloc(size);

    // Initialize the input matrix with some values
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(i % 100); // Use modulo to keep values small
    }

    // Print a small portion of the input matrix for verification (if small enough)
    int small_width = std::min(width, 8);
    int small_height = std::min(height, 8);
    if (width <= 16 && height <= 16) {
        printMatrix(h_input, width, height, "Input Matrix");
    } else {
        std::cout << "Input Matrix (top-left corner):" << std::endl;
        for (int y = 0; y < small_height; y++) {
            for (int x = 0; x < small_width; x++) {
                std::cout << std::setw(6) << h_input[y * width + x] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // ==================== CPU Implementation ====================
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    transposeMatrixCPU(h_input, h_output_cpu, width, height);
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    
    std::cout << "CPU Transpose Time: " << cpu_time.count() << " ms" << std::endl;

    // ==================== GPU Implementation ====================
    // Allocate device memory
    float* d_input;
    float* d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, size));

    // Start GPU timing (including memory transfers)
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    transposeMatrix<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost));
    
    // Stop GPU timing
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float gpu_time = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time, start, stop));
    
    std::cout << "GPU Transpose Time (including memory transfers): " << gpu_time << " ms" << std::endl;
    
    // Calculate speedup
    float speedup = cpu_time.count() / gpu_time;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // Verify the result
    bool success = verifyResults(h_output_cpu, h_output_gpu, width * height);
    std::cout << (success ? "Matrix transposition succeeded!" : "Matrix transposition failed!") << std::endl;

    // Print a small portion of the output matrix for verification (if small enough)
    if (width <= 16 && height <= 16) {
        printMatrix(h_output_gpu, height, width, "Output Matrix (GPU)");
    } else {
        std::cout << "Output Matrix (GPU, top-left corner):" << std::endl;
        for (int y = 0; y < small_height; y++) {
            for (int x = 0; x < small_width; x++) {
                std::cout << std::setw(6) << h_output_gpu[y * height + x] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Clean up
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);

    return 0;
}
