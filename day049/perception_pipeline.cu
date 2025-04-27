#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp> // Requires OpenCV installed

// --- Error Checking ---
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_NULL(ptr) \
    if (ptr == nullptr) { \
        fprintf(stderr, "CUDA Error: Memory allocation failed at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// --- Kernel: Grayscale Conversion (uchar3 BGR -> uchar1 Gray) ---
__global__ void grayscale_kernel(const uchar3* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = input[idx];
        // Standard luminance calculation (BGR order from OpenCV)
        output[idx] = (unsigned char)(0.114f * pixel.z + 0.587f * pixel.y + 0.299f * pixel.x);
    }
}

// --- Kernel: Gaussian Blur (5x5) ---
// Simple version without shared memory optimization
__global__ void gaussian_blur_kernel(const unsigned char* input, unsigned char* output, int width, int height, const float* filter, int filterWidth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        int halfFilterWidth = filterWidth / 2;

        for (int fy = -halfFilterWidth; fy <= halfFilterWidth; ++fy) {
            for (int fx = -halfFilterWidth; fx <= halfFilterWidth; ++fx) {
                int currentX = x + fx;
                int currentY = y + fy;

                // Clamp coordinates to image boundaries (border replication)
                currentX = max(0, min(width - 1, currentX));
                currentY = max(0, min(height - 1, currentY));

                int filterIdx = (fy + halfFilterWidth) * filterWidth + (fx + halfFilterWidth);
                int imageIdx = currentY * width + currentX;

                sum += (float)input[imageIdx] * filter[filterIdx];
            }
        }
        output[y * width + x] = (unsigned char)max(0.0f, min(255.0f, sum));
    }
}

// --- Kernel: Sobel Edge Detection (calculates magnitude) ---
// Simple version without shared memory optimization
__global__ void sobel_kernel(const unsigned char* input, float* output_mag, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Sobel operators (3x3)
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float sumX = 0.0f;
        float sumY = 0.0f;

        for (int fy = -1; fy <= 1; ++fy) {
            for (int fx = -1; fx <= 1; ++fx) {
                int currentX = x + fx;
                int currentY = y + fy;
                int imageIdx = currentY * width + currentX;
                float pixelValue = (float)input[imageIdx];

                sumX += pixelValue * Gx[fy + 1][fx + 1];
                sumY += pixelValue * Gy[fy + 1][fx + 1];
            }
        }

        // Calculate magnitude: sqrt(Gx^2 + Gy^2)
        output_mag[y * width + x] = sqrtf(sumX * sumX + sumY * sumY);
    } else {
        // Handle borders (set magnitude to 0)
        output_mag[y * width + x] = 0.0f;
    }
}

// --- Kernel: Reduction (Count pixels above threshold) ---
// Basic reduction kernel - assumes blockDim.x is power of 2 and <= 1024
__global__ void reduction_count_kernel(const float* input_mag, unsigned int* output_count, int n, float threshold) {
    extern __shared__ unsigned int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    unsigned int count = 0;
    while (i < n) {
        if (input_mag[i] > threshold) {
            count++;
        }
        i += gridSize;
    }
    sdata[tid] = count;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        atomicAdd(output_count, sdata[0]); // Use atomicAdd for inter-block summation
    }
}


// --- Host Code ---
int main(int argc, char **argv) {
    printf("Day 49: Accelerated Perception Pipeline (Internal Camera Capture)\n");

    // 1. Initialize Camera using OpenCV
    cv::VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened()) {
        fprintf(stderr, "Error: Could not open camera.\n");
        return EXIT_FAILURE;
    }

    // Capture a single frame
    cv::Mat h_input_img_bgr;
    printf("Attempting to capture frame from camera...\n");
    cap >> h_input_img_bgr; // or cap.read(h_input_img_bgr);

    if (h_input_img_bgr.empty()) {
        fprintf(stderr, "Error: Captured empty frame.\n");
        cap.release(); // Release camera resource
        return EXIT_FAILURE;
    }
    printf("Frame captured successfully.\n");
    cap.release(); // Release camera immediately after capture

    int width = h_input_img_bgr.cols;
    int height = h_input_img_bgr.rows;
    size_t num_pixels = (size_t)width * height;
    size_t image_size_bgr = num_pixels * sizeof(uchar3);
    size_t image_size_gray = num_pixels * sizeof(unsigned char);
    size_t image_size_float = num_pixels * sizeof(float);

    printf("Captured Frame Info: (Width: %d, Height: %d, Channels: %d)\n",
           width, height, h_input_img_bgr.channels());

     // Ensure the captured frame is suitable (e.g., 3 channels)
    if (h_input_img_bgr.channels() != 3) {
         fprintf(stderr, "Error: Captured frame is not a 3-channel color image.\n");
         // cap is already released
         return EXIT_FAILURE;
    }

    // 2. Allocate Device Memory
    uchar3* d_input_img_bgr = nullptr;
    unsigned char* d_gray_img = nullptr;
    unsigned char* d_blurred_img = nullptr;
    float* d_sobel_mag = nullptr;
    unsigned int* d_edge_count = nullptr; // Single value for final count
    unsigned int* d_block_sums = nullptr; // Intermediate sums for reduction

    CHECK_CUDA_ERROR(cudaMalloc(&d_input_img_bgr, image_size_bgr));
    CHECK_CUDA_ERROR(cudaMalloc(&d_gray_img, image_size_gray));
    CHECK_CUDA_ERROR(cudaMalloc(&d_blurred_img, image_size_gray));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sobel_mag, image_size_float));
    CHECK_CUDA_ERROR(cudaMalloc(&d_edge_count, sizeof(unsigned int)));

    CHECK_NULL(d_input_img_bgr);
    CHECK_NULL(d_gray_img);
    CHECK_NULL(d_blurred_img);
    CHECK_NULL(d_sobel_mag);
    CHECK_NULL(d_edge_count);

    // Initialize edge count to 0 on device
    CHECK_CUDA_ERROR(cudaMemset(d_edge_count, 0, sizeof(unsigned int)));

    // 3. Copy Input Image to Device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_img_bgr, h_input_img_bgr.ptr(), image_size_bgr, cudaMemcpyHostToDevice));

    // 4. Define Kernels Launch Configuration
    dim3 threadsPerBlock(16, 16); // 256 threads per block
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // For Reduction Kernel
    unsigned int reduction_threads_per_block = 256; // Must be power of 2 <= 1024
    unsigned int reduction_num_blocks = (num_pixels + reduction_threads_per_block - 1) / reduction_threads_per_block;
     // Allocate intermediate buffer for reduction if needed (simpler atomic version used here)
    size_t reduction_shared_mem_size = reduction_threads_per_block * sizeof(unsigned int);


    // --- Define Gaussian Filter (5x5) ---
    const int filterWidth = 5;
    // Precompute a simple 5x5 Gaussian kernel (unnormalized for simplicity, normalization happens implicitly via uchar cast)
    // Values roughly approximate a Gaussian distribution sigma=1.0
     float h_gaussian_filter[filterWidth * filterWidth] = {
        1,  4,  7,  4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1,  4,  7,  4, 1
    };
     // Normalize the filter
    float filterSum = 0;
    for(int i=0; i < filterWidth * filterWidth; ++i) filterSum += h_gaussian_filter[i];
    for(int i=0; i < filterWidth * filterWidth; ++i) h_gaussian_filter[i] /= filterSum;

    float* d_gaussian_filter = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_gaussian_filter, filterWidth * filterWidth * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_gaussian_filter, h_gaussian_filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_NULL(d_gaussian_filter);


    // 5. Launch Kernels (Pipeline Stages)
    // --- Stage 1: Grayscale ---
    grayscale_kernel<<<numBlocks, threadsPerBlock>>>(d_input_img_bgr, d_gray_img, width, height);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

    // --- Stage 2: Gaussian Blur ---
    gaussian_blur_kernel<<<numBlocks, threadsPerBlock>>>(d_gray_img, d_blurred_img, width, height, d_gaussian_filter, filterWidth);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // --- Stage 3: Sobel Edge Detection ---
    sobel_kernel<<<numBlocks, threadsPerBlock>>>(d_blurred_img, d_sobel_mag, width, height);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // --- Stage 4: Reduction Count ---
    float edge_threshold = 100.0f; // Threshold for counting an edge pixel
    reduction_count_kernel<<<reduction_num_blocks, reduction_threads_per_block, reduction_shared_mem_size>>>(
        d_sobel_mag, d_edge_count, num_pixels, edge_threshold
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Synchronize to ensure all kernels are finished before copying result
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // 6. Copy Result Back to Host
    unsigned int h_edge_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_edge_count, d_edge_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // 7. Print Result
    printf("----------------------------------------\n");
    printf("Perception Pipeline Completed.\n");
    printf("Edge Threshold: %.1f\n", edge_threshold);
    printf("Detected Edge Pixels: %u\n", h_edge_count);
    printf("----------------------------------------\n");


    // 8. Free Memory
    CHECK_CUDA_ERROR(cudaFree(d_input_img_bgr));
    CHECK_CUDA_ERROR(cudaFree(d_gray_img));
    CHECK_CUDA_ERROR(cudaFree(d_blurred_img));
    CHECK_CUDA_ERROR(cudaFree(d_sobel_mag));
    CHECK_CUDA_ERROR(cudaFree(d_edge_count));
    CHECK_CUDA_ERROR(cudaFree(d_gaussian_filter));


    printf("Finished successfully.\n");
    return 0;
}
