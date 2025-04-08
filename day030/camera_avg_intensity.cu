#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <unistd.h> // For usleep

// Error checking macro (adapted from Day 4)
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel to convert RGB image to grayscale
__global__ void rgb_to_gray_kernel(uchar3* rgb, float* gray, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = width * height;

    if (idx >= num_pixels) return;

    uchar3 pixel = rgb[idx];
    // Standard luminance calculation, result in [0, 255] float range
    gray[idx] = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
}

// Kernel for parallel sum reduction (adapted from Day 4)
__global__ void reduce_sum_kernel(float *input, float *output, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid; // Stride by blockDim*2 for first load
    unsigned int gridSize = blockDim.x * 2 * gridDim.x; // Total elements covered by grid in one go

    float mySum = 0;

    // We reduce multiple elements per thread. Reduce elements assigned to this thread.
    while (i < N) {
        mySum += input[i];
        // Handle the second element if the block covers more than N
        if (i + blockDim.x < N) {
            mySum += input[i + blockDim.x];
        }
        i += gridSize;
    }
    sdata[tid] = mySum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}


int main(int argc, char *argv[]) {
    const int BLOCK_SIZE = 256; // Block size for reduction and grayscale
    const int MAX_FRAMES = 100; // Number of frames to process
    int camera_index = 0;       // Default camera index

    if (argc > 1) {
        camera_index = atoi(argv[1]);
        printf("Using camera index: %d\n", camera_index);
    }
     if (argc > 2) {
        fprintf(stderr, "Usage: %s [camera_index]\n", argv[0]);
        return 1;
    }


    // --- Camera Initialization ---
    cv::VideoCapture cap(camera_index);
    if (!cap.isOpened()) {
        fprintf(stderr, "Error: Could not open camera with index %d\n", camera_index);
        return -1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (width <= 0 || height <= 0) {
         fprintf(stderr, "Error: Could not get valid frame dimensions (%dx%d)\n", width, height);
         cap.release();
         return -1;
    }
    int num_pixels = width * height;
    printf("Camera opened successfully. Frame dimensions: %d x %d (%d pixels)\n", width, height, num_pixels);


    // --- Memory Allocation ---
    cv::Mat frame; // Host frame buffer (OpenCV managed)
    float h_avg_intensity = 0.0f;
    float h_final_sum = 0.0f; // Sum from GPU reduction

    uchar3* d_rgb_input = nullptr;
    float* d_gray_output = nullptr;
    float* d_sum_output = nullptr; // Partial sums from first reduction pass
    float* d_final_sum = nullptr; // Single final sum value

    CHECK_CUDA_ERROR(cudaMalloc(&d_rgb_input, num_pixels * sizeof(uchar3)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_gray_output, num_pixels * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_final_sum, 1 * sizeof(float))); // Allocate space for the single final sum

    // Calculate grid/block sizes for kernels
    dim3 block(BLOCK_SIZE);
    dim3 grid_gray((num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Reduction requires careful grid/block calculation
    // First pass: Reduce num_pixels down to num_blocks_for_reduction partial sums
    int num_blocks_for_reduction = (num_pixels + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);
     if (num_blocks_for_reduction == 0) num_blocks_for_reduction = 1; // Ensure at least one block
    CHECK_CUDA_ERROR(cudaMalloc(&d_sum_output, num_blocks_for_reduction * sizeof(float)));
    dim3 grid_reduce1(num_blocks_for_reduction);
    dim3 block_reduce(BLOCK_SIZE); // Use BLOCK_SIZE threads per block


    printf("Starting frame capture and processing for %d frames...\n", MAX_FRAMES);
    // --- Main Loop ---
    for (int i = 0; i < MAX_FRAMES; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 1. Capture frame
        cap >> frame;
        if (frame.empty()) {
            fprintf(stderr, "Warning: Captured empty frame, skipping. (Frame %d)\n", i);
            continue; // Skip this iteration
        }
         // Ensure frame is 3-channel uchar (CV_8UC3)
        if (frame.type() != CV_8UC3) {
             fprintf(stderr, "Warning: Frame type is not CV_8UC3 (Type: %d), attempting conversion. (Frame %d)\n", frame.type(), i);
             cv::Mat temp_frame;
             if (frame.channels() == 1) {
                 cv::cvtColor(frame, temp_frame, cv::COLOR_GRAY2BGR);
             } else {
                 // Attempt a general conversion, might fail if format is unusual
                frame.convertTo(temp_frame, CV_8UC3);
             }
             if(temp_frame.empty() || temp_frame.type() != CV_8UC3){
                 fprintf(stderr, "Error: Could not convert frame to CV_8UC3. Skipping Frame %d.\n", i);
                 continue;
             }
              frame = temp_frame; // Replace original frame with converted one
        }


        // 2. Transfer H->D
        CHECK_CUDA_ERROR(cudaMemcpy(d_rgb_input, frame.ptr<uchar>(), num_pixels * sizeof(uchar3), cudaMemcpyHostToDevice));

        // 3. Launch Kernel 1: RGB to Grayscale
        rgb_to_gray_kernel<<<grid_gray, block>>>(d_rgb_input, d_gray_output, width, height);
        CHECK_CUDA_ERROR(cudaGetLastError()); // Check for launch errors

        // 4. Launch Kernel 2: Reduction (Pass 1)
        // Input: d_gray_output (num_pixels), Output: d_sum_output (num_blocks_for_reduction)
        reduce_sum_kernel<<<grid_reduce1, block_reduce, block_reduce.x * sizeof(float)>>>(d_gray_output, d_sum_output, num_pixels);
        CHECK_CUDA_ERROR(cudaGetLastError()); // Check for launch errors

        // 5. Launch Kernel 2: Reduction (Pass 2) - Reduce the partial sums
        // Input: d_sum_output (num_blocks_for_reduction), Output: d_final_sum (1 element)
        // We only need 1 block for this final reduction step.
        if (num_blocks_for_reduction > 1) {
            reduce_sum_kernel<<<1, block_reduce, block_reduce.x * sizeof(float)>>>(d_sum_output, d_final_sum, num_blocks_for_reduction);
            CHECK_CUDA_ERROR(cudaGetLastError());
        } else {
             // If only one block in the first pass, the result is already in d_sum_output[0]
             CHECK_CUDA_ERROR(cudaMemcpy(d_final_sum, d_sum_output, sizeof(float), cudaMemcpyDeviceToDevice));
        }


        // 6. Transfer D->H (Final Sum)
        CHECK_CUDA_ERROR(cudaMemcpy(&h_final_sum, d_final_sum, sizeof(float), cudaMemcpyDeviceToHost));

        // 7. Calculate Average Intensity
        h_avg_intensity = h_final_sum / static_cast<float>(num_pixels);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();


        // 8. Log Output (Headless)
        printf("Frame: %d, Timestamp_ms: %lld, Avg Intensity: %.4f, Frame Time: %.2f ms\n",
               i, timestamp, h_avg_intensity, elapsed_ms);

        // 9. Optional Delay
        // usleep(10000); // 10 ms delay
    }

    printf("Processing complete.\n");

    // --- Cleanup ---
    printf("Releasing camera and freeing memory...\n");
    cap.release();
    CHECK_CUDA_ERROR(cudaFree(d_rgb_input));
    CHECK_CUDA_ERROR(cudaFree(d_gray_output));
    CHECK_CUDA_ERROR(cudaFree(d_sum_output));
    CHECK_CUDA_ERROR(cudaFree(d_final_sum));
    printf("Cleanup complete.\n");

    return 0;
}
