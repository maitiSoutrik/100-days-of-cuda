#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp> // Include OpenCV headers
#include <opencv2/videoio.hpp>

// Error checking macro (as used in previous days)
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t err, const char *const func, const char *const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<int>(err), cudaGetErrorName(err), func);
        exit(EXIT_FAILURE);
    }
}

// Kernel to compute image gradients Ix, Iy (Sobel) and It (temporal difference)
// Assumes input images are grayscale (single channel)
__global__ void computeGradientsKernel(const unsigned char* frame1, const unsigned char* frame2,
                                       float* Ix, float* Iy, float* It,
                                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check boundary conditions
    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;

    // Compute temporal gradient It = Frame2(x,y) - Frame1(x,y)
    It[idx] = static_cast<float>(frame2[idx]) - static_cast<float>(frame1[idx]);

    // Compute spatial gradients Ix, Iy using Sobel operators (simple version, ignoring borders for now)
    // For a robust implementation, handle borders carefully (clamp, mirror, etc.)
    // and potentially use shared memory for the stencil operation.
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel X kernel: [-1 0 1; -2 0 2; -1 0 1]
        Ix[idx] = -1.0f * frame1[(y - 1) * width + (x - 1)] + 1.0f * frame1[(y - 1) * width + (x + 1)]
                  -2.0f * frame1[ y      * width + (x - 1)] + 2.0f * frame1[ y      * width + (x + 1)]
                  -1.0f * frame1[(y + 1) * width + (x - 1)] + 1.0f * frame1[(y + 1) * width + (x + 1)];

        // Sobel Y kernel: [-1 -2 -1; 0 0 0; 1 2 1]
        Iy[idx] = -1.0f * frame1[(y - 1) * width + (x - 1)] - 2.0f * frame1[(y - 1) * width +  x     ] - 1.0f * frame1[(y - 1) * width + (x + 1)]
                  +1.0f * frame1[(y + 1) * width + (x - 1)] + 2.0f * frame1[(y + 1) * width +  x     ] + 1.0f * frame1[(y + 1) * width + (x + 1)];
    } else {
        // Handle borders (e.g., set gradients to 0)
        Ix[idx] = 0.0f;
        Iy[idx] = 0.0f;
    }
}

// Host function to manage CUDA operations
void computeGradients(const unsigned char* h_frame1, const unsigned char* h_frame2,
                      float* h_Ix, float* h_Iy, float* h_It,
                      int width, int height) {
    size_t imageBytes = width * height * sizeof(unsigned char);
    size_t gradientBytes = width * height * sizeof(float);

    unsigned char *d_frame1 = nullptr, *d_frame2 = nullptr;
    float *d_Ix = nullptr, *d_Iy = nullptr, *d_It = nullptr;

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_frame1, imageBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_frame2, imageBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_Ix, gradientBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_Iy, gradientBytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_It, gradientBytes));

    // Copy input frames from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_frame1, h_frame1, imageBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_frame2, h_frame2, imageBytes, cudaMemcpyHostToDevice));

    // Define kernel launch parameters
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    computeGradientsKernel<<<gridSize, blockSize>>>(d_frame1, d_frame2, d_Ix, d_Iy, d_It, width, height);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_Ix, d_Ix, gradientBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_Iy, d_Iy, gradientBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_It, d_It, gradientBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_frame1));
    CHECK_CUDA_ERROR(cudaFree(d_frame2));
    CHECK_CUDA_ERROR(cudaFree(d_Ix));
    CHECK_CUDA_ERROR(cudaFree(d_Iy));
    CHECK_CUDA_ERROR(cudaFree(d_It));
}


int main() {
    printf("Day 45: Optical Flow Gradient Calculation (CUDA with OpenCV Camera Input)\n");

    // --- OpenCV Camera Setup ---
    int cameraIndex = 0; // Default camera
    cv::VideoCapture cap(cameraIndex);

    if (!cap.isOpened()) {
        fprintf(stderr, "Error: Could not open camera %d\n", cameraIndex);
        return EXIT_FAILURE;
    }

    // --- Frame Acquisition and Processing ---
    cv::Mat frame1_bgr, frame2_bgr;
    cv::Mat frame1_gray, frame2_gray;

    // Capture frame 1
    cap >> frame1_bgr;
    if (frame1_bgr.empty()) {
        fprintf(stderr, "Error: Could not capture frame 1 from camera\n");
        cap.release();
        return EXIT_FAILURE;
    }

    // Capture frame 2
    cap >> frame2_bgr;
    if (frame2_bgr.empty()) {
        fprintf(stderr, "Error: Could not capture frame 2 from camera\n");
        cap.release();
        return EXIT_FAILURE;
    }
    cap.release(); // Release camera after capturing frames
    printf("Captured two consecutive frames from camera %d.\n", cameraIndex);

    // --- Define processing dimensions ---
    // Resize frames for consistent processing size (adjust as needed)
    int processWidth = 640;
    int processHeight = 480;
    cv::Size processSize(processWidth, processHeight);

    // Resize and convert to grayscale
    cv::resize(frame1_bgr, frame1_bgr, processSize);
    cv::resize(frame2_bgr, frame2_bgr, processSize);
    cv::cvtColor(frame1_bgr, frame1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2_bgr, frame2_gray, cv::COLOR_BGR2GRAY);

    // --- Memory Allocation ---
    int width = frame1_gray.cols;
    int height = frame1_gray.rows;
    size_t numPixels = width * height;
    printf("Processing Dimensions: %d x %d\n", width, height);

    // Allocate host memory for gradient results
    float *h_Ix = (float*)malloc(numPixels * sizeof(float));
    float *h_Iy = (float*)malloc(numPixels * sizeof(float));
    float *h_It = (float*)malloc(numPixels * sizeof(float));

    if (!h_Ix || !h_Iy || !h_It) {
        fprintf(stderr, "Failed to allocate host memory for gradients!\n");
        return EXIT_FAILURE;
    }

    // Input frame data pointers (no extra allocation needed for input, use cv::Mat data)
    // Ensure Mats are continuous for direct pointer use if possible, although
    // computeGradients handles non-continuous via cudaMemcpy implicitly.
    unsigned char *h_frame1_data = frame1_gray.data;
    unsigned char *h_frame2_data = frame2_gray.data;


    // --- Compute Gradients on GPU ---
    printf("Computing gradients on GPU...\n");
    computeGradients(h_frame1_data, h_frame2_data, h_Ix, h_Iy, h_It, width, height);
    printf("GPU computation complete.\n");

    // --- Output Sample Results ---
    // Print some sample results (e.g., center pixel)
    int centerX = width / 2;
    int centerY = height / 2;
    int centerIdx = centerY * width + centerX;

    printf("\nSample Gradient Values (Center Pixel (%d, %d)):\n", centerX, centerY);
    printf("    Ix: %.2f, Iy: %.2f, It: %.2f\n", h_Ix[centerIdx], h_Iy[centerIdx], h_It[centerIdx]);

    // --- Verification (More complex for real camera data) ---
    // Simple check: Count non-zero gradients (expect many)
    int nonZeroIx = 0, nonZeroIy = 0, nonZeroIt = 0;
    float maxAbsIt = 0.0f;
    for(size_t i=0; i < numPixels; ++i) {
        if (fabs(h_Ix[i]) > 1e-6) nonZeroIx++;
        if (fabs(h_Iy[i]) > 1e-6) nonZeroIy++;
        if (fabs(h_It[i]) > 1e-6) {
            nonZeroIt++;
            if(fabs(h_It[i]) > maxAbsIt) maxAbsIt = fabs(h_It[i]);
        }
    }
    printf("\nVerification (counts of non-zero gradients):\n");
    printf("  Ix > 0: %d / %zu pixels\n", nonZeroIx, numPixels);
    printf("  Iy > 0: %d / %zu pixels\n", nonZeroIy, numPixels);
    printf("  It > 0: %d / %zu pixels\n", nonZeroIt, numPixels);
    printf("  Max absolute It: %.2f\n", maxAbsIt);


    // --- Cleanup ---
    // Free host memory for gradients
    free(h_Ix);
    free(h_Iy);
    free(h_It);
    // cv::Mat objects (frameX_bgr, frameX_gray) are automatically managed

    printf("\nDay 45 finished successfully.\n");
    return EXIT_SUCCESS;
}
