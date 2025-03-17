#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>

// Define the size of the 1D signal and 2D image
#define SIGNAL_SIZE 1024
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define KERNEL_SIZE 5

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

// CUDA kernel for 1D convolution
__global__ void convolution1D(const float* input, float* output, const float* kernel, 
                            int inputSize, int kernelSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes one output element
    if (idx < inputSize) {
        float result = 0.0f;
        int radius = kernelSize / 2;
        
        // Apply the kernel
        for (int k = 0; k < kernelSize; k++) {
            int pos = idx + (k - radius);
            
            // Handle boundary conditions (zero padding)
            if (pos >= 0 && pos < inputSize) {
                result += input[pos] * kernel[k];
            }
        }
        
        output[idx] = result;
    }
}

// CUDA kernel for 2D convolution
__global__ void convolution2D(const float* input, float* output, const float* kernel,
                            int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Each thread computes one output element
    if (x < width && y < height) {
        float result = 0.0f;
        int radius = kernelSize / 2;
        
        // Apply the 2D kernel
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                int imgX = x + (kx - radius);
                int imgY = y + (ky - radius);
                
                // Handle boundary conditions (zero padding)
                if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
                    result += input[imgY * width + imgX] * 
                              kernel[ky * kernelSize + kx];
                }
            }
        }
        
        output[y * width + x] = result;
    }
}

// CPU implementation of 1D convolution
void convolution1DCPU(const float* input, float* output, const float* kernel, 
                    int inputSize, int kernelSize) {
    int radius = kernelSize / 2;
    
    for (int i = 0; i < inputSize; i++) {
        float result = 0.0f;
        
        for (int k = 0; k < kernelSize; k++) {
            int pos = i + (k - radius);
            
            // Handle boundary conditions (zero padding)
            if (pos >= 0 && pos < inputSize) {
                result += input[pos] * kernel[k];
            }
        }
        
        output[i] = result;
    }
}

// CPU implementation of 2D convolution
void convolution2DCPU(const float* input, float* output, const float* kernel,
                    int width, int height, int kernelSize) {
    int radius = kernelSize / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float result = 0.0f;
            
            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    int imgX = x + (kx - radius);
                    int imgY = y + (ky - radius);
                    
                    // Handle boundary conditions (zero padding)
                    if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
                        result += input[imgY * width + imgX] * 
                                  kernel[ky * kernelSize + kx];
                    }
                }
            }
            
            output[y * width + x] = result;
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

// Function to print 1D array (for small arrays only)
void print1DArray(const float* array, int size, const char* name) {
    int printSize = std::min(size, 16);
    
    std::cout << name << " (first " << printSize << " elements):" << std::endl;
    for (int i = 0; i < printSize; i++) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << array[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

// Function to print 2D array (for small arrays only)
void print2DArray(const float* array, int width, int height, const char* name) {
    int printWidth = std::min(width, 8);
    int printHeight = std::min(height, 8);
    
    std::cout << name << " (top-left " << printWidth << "x" << printHeight << " corner):" << std::endl;
    for (int y = 0; y < printHeight; y++) {
        for (int x = 0; x < printWidth; x++) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << array[y * width + x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // ==================== 1D Convolution ====================
    std::cout << "========== 1D Convolution ==========" << std::endl;
    
    int signalSize = SIGNAL_SIZE;
    int kernelSize1D = KERNEL_SIZE;
    size_t signalBytes = signalSize * sizeof(float);
    size_t kernelBytes1D = kernelSize1D * sizeof(float);
    
    // Allocate host memory for 1D convolution
    float* h_signal = (float*)malloc(signalBytes);
    float* h_kernel1D = (float*)malloc(kernelBytes1D);
    float* h_output1D_cpu = (float*)malloc(signalBytes);
    float* h_output1D_gpu = (float*)malloc(signalBytes);
    
    // Initialize the signal with a sine wave
    for (int i = 0; i < signalSize; i++) {
        h_signal[i] = sinf(0.1f * i);
    }
    
    // Initialize the kernel with a Gaussian filter
    float sigma = 1.0f;
    float sum = 0.0f;
    int radius = kernelSize1D / 2;
    
    for (int i = 0; i < kernelSize1D; i++) {
        int x = i - radius;
        h_kernel1D[i] = expf(-(x * x) / (2 * sigma * sigma));
        sum += h_kernel1D[i];
    }
    
    // Normalize the kernel
    for (int i = 0; i < kernelSize1D; i++) {
        h_kernel1D[i] /= sum;
    }
    
    // Print the signal and kernel
    print1DArray(h_signal, signalSize, "Input Signal");
    print1DArray(h_kernel1D, kernelSize1D, "1D Kernel");
    
    // ==================== 1D CPU Implementation ====================
    auto cpu_start_1d = std::chrono::high_resolution_clock::now();
    
    convolution1DCPU(h_signal, h_output1D_cpu, h_kernel1D, signalSize, kernelSize1D);
    
    auto cpu_end_1d = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time_1d = cpu_end_1d - cpu_start_1d;
    
    std::cout << "1D CPU Convolution Time: " << cpu_time_1d.count() << " ms" << std::endl;
    
    // ==================== 1D GPU Implementation ====================
    // Allocate device memory for 1D convolution
    float* d_signal;
    float* d_kernel1D;
    float* d_output1D;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_signal, signalBytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_kernel1D, kernelBytes1D));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output1D, signalBytes));
    
    // Start GPU timing (including memory transfers)
    cudaEvent_t start_1d, stop_1d;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_1d));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_1d));
    CHECK_CUDA_ERROR(cudaEventRecord(start_1d));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_signal, h_signal, signalBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel1D, h_kernel1D, kernelBytes1D, cudaMemcpyHostToDevice));
    
    // Define block and grid sizes for 1D convolution
    int blockSize1D = 256;
    int gridSize1D = (signalSize + blockSize1D - 1) / blockSize1D;
    
    // Launch the 1D convolution kernel
    convolution1D<<<gridSize1D, blockSize1D>>>(d_signal, d_output1D, d_kernel1D, signalSize, kernelSize1D);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output1D_gpu, d_output1D, signalBytes, cudaMemcpyDeviceToHost));
    
    // Stop GPU timing
    CHECK_CUDA_ERROR(cudaEventRecord(stop_1d));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_1d));
    float gpu_time_1d = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_1d, start_1d, stop_1d));
    
    std::cout << "1D GPU Convolution Time (including memory transfers): " << gpu_time_1d << " ms" << std::endl;
    
    // Calculate speedup for 1D convolution
    float speedup_1d = cpu_time_1d.count() / gpu_time_1d;
    std::cout << "1D Speedup: " << speedup_1d << "x" << std::endl;
    
    // Verify the 1D convolution result
    bool success_1d = verifyResults(h_output1D_cpu, h_output1D_gpu, signalSize);
    std::cout << (success_1d ? "1D Convolution succeeded!" : "1D Convolution failed!") << std::endl;
    
    // Print the 1D output
    print1DArray(h_output1D_gpu, signalSize, "1D Output Signal (GPU)");
    
    // ==================== 2D Convolution ====================
    std::cout << "\n========== 2D Convolution ==========" << std::endl;
    
    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;
    int kernelSize2D = KERNEL_SIZE;
    size_t imageBytes = width * height * sizeof(float);
    size_t kernelBytes2D = kernelSize2D * kernelSize2D * sizeof(float);
    
    // Allocate host memory for 2D convolution
    float* h_image = (float*)malloc(imageBytes);
    float* h_kernel2D = (float*)malloc(kernelBytes2D);
    float* h_output2D_cpu = (float*)malloc(imageBytes);
    float* h_output2D_gpu = (float*)malloc(imageBytes);
    
    // Initialize the image with a pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Create a simple pattern (checkerboard, gradient, etc.)
            h_image[y * width + x] = (sinf(0.1f * x) + cosf(0.1f * y)) * 0.5f;
        }
    }
    
    // Initialize the 2D kernel with a Gaussian filter
    sum = 0.0f;
    radius = kernelSize2D / 2;
    
    for (int y = 0; y < kernelSize2D; y++) {
        for (int x = 0; x < kernelSize2D; x++) {
            int dx = x - radius;
            int dy = y - radius;
            h_kernel2D[y * kernelSize2D + x] = expf(-(dx * dx + dy * dy) / (2 * sigma * sigma));
            sum += h_kernel2D[y * kernelSize2D + x];
        }
    }
    
    // Normalize the 2D kernel
    for (int i = 0; i < kernelSize2D * kernelSize2D; i++) {
        h_kernel2D[i] /= sum;
    }
    
    // Print the image and kernel
    print2DArray(h_image, width, height, "Input Image");
    print2DArray(h_kernel2D, kernelSize2D, kernelSize2D, "2D Kernel");
    
    // ==================== 2D CPU Implementation ====================
    auto cpu_start_2d = std::chrono::high_resolution_clock::now();
    
    convolution2DCPU(h_image, h_output2D_cpu, h_kernel2D, width, height, kernelSize2D);
    
    auto cpu_end_2d = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time_2d = cpu_end_2d - cpu_start_2d;
    
    std::cout << "2D CPU Convolution Time: " << cpu_time_2d.count() << " ms" << std::endl;
    
    // ==================== 2D GPU Implementation ====================
    // Allocate device memory for 2D convolution
    float* d_image;
    float* d_kernel2D;
    float* d_output2D;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_image, imageBytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_kernel2D, kernelBytes2D));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output2D, imageBytes));
    
    // Start GPU timing (including memory transfers)
    cudaEvent_t start_2d, stop_2d;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_2d));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_2d));
    CHECK_CUDA_ERROR(cudaEventRecord(start_2d));
    
    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_image, h_image, imageBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_kernel2D, h_kernel2D, kernelBytes2D, cudaMemcpyHostToDevice));
    
    // Define block and grid sizes for 2D convolution
    dim3 blockSize2D(16, 16);
    dim3 gridSize2D((width + blockSize2D.x - 1) / blockSize2D.x, 
                   (height + blockSize2D.y - 1) / blockSize2D.y);
    
    // Launch the 2D convolution kernel
    convolution2D<<<gridSize2D, blockSize2D>>>(d_image, d_output2D, d_kernel2D, width, height, kernelSize2D);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output2D_gpu, d_output2D, imageBytes, cudaMemcpyDeviceToHost));
    
    // Stop GPU timing
    CHECK_CUDA_ERROR(cudaEventRecord(stop_2d));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_2d));
    float gpu_time_2d = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_2d, start_2d, stop_2d));
    
    std::cout << "2D GPU Convolution Time (including memory transfers): " << gpu_time_2d << " ms" << std::endl;
    
    // Calculate speedup for 2D convolution
    float speedup_2d = cpu_time_2d.count() / gpu_time_2d;
    std::cout << "2D Speedup: " << speedup_2d << "x" << std::endl;
    
    // Verify the 2D convolution result
    bool success_2d = verifyResults(h_output2D_cpu, h_output2D_gpu, width * height);
    std::cout << (success_2d ? "2D Convolution succeeded!" : "2D Convolution failed!") << std::endl;
    
    // Print the 2D output
    print2DArray(h_output2D_gpu, width, height, "2D Output Image (GPU)");
    
    // Clean up 1D resources
    CHECK_CUDA_ERROR(cudaEventDestroy(start_1d));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_1d));
    CHECK_CUDA_ERROR(cudaFree(d_signal));
    CHECK_CUDA_ERROR(cudaFree(d_kernel1D));
    CHECK_CUDA_ERROR(cudaFree(d_output1D));
    free(h_signal);
    free(h_kernel1D);
    free(h_output1D_cpu);
    free(h_output1D_gpu);
    
    // Clean up 2D resources
    CHECK_CUDA_ERROR(cudaEventDestroy(start_2d));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_2d));
    CHECK_CUDA_ERROR(cudaFree(d_image));
    CHECK_CUDA_ERROR(cudaFree(d_kernel2D));
    CHECK_CUDA_ERROR(cudaFree(d_output2D));
    free(h_image);
    free(h_kernel2D);
    free(h_output2D_cpu);
    free(h_output2D_gpu);
    
    return 0;
}
