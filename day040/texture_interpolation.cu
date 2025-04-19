#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp> // For imread/imwrite
#include <opencv2/imgproc.hpp> // For cvtColor, resize
#include <chrono>
#include <string>      // <<< Added missing include
#include <sys/stat.h> // For checking/creating directory
#include <opencv2/videoio.hpp> // For VideoCapture, CAP_V4L2
#include <unistd.h> // For usleep

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel for bilinear interpolation using texture memory
// Output pixel (x, y) maps to input texture coordinate (u, v)
// We use normalized coordinates (0.0 to 1.0) for tex2D.
__global__ void bilinear_interpolation_kernel(cudaTextureObject_t texObj, float* output, int outWidth, int outHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outWidth || y >= outHeight) {
        return;
    }

    // Calculate normalized coordinates [0, 1] corresponding to the center of the output pixel
    float u = (float)(x + 0.5f) / (float)outWidth;
    float v = (float)(y + 0.5f) / (float)outHeight;

    // tex2D with normalized coordinates returns the interpolated value
    output[y * outWidth + x] = tex2D<float>(texObj, u, v);
}

// Simple kernel to convert uchar3 RGB to float grayscale
__global__ void rgb_to_gray_float_kernel(uchar3* rgb, float* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x >= width || y >= height) return;

    uchar3 pixel = rgb[idx];
    // Standard luminance calculation
    gray[idx] = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
}


int main(int argc, char *argv[]) {
    // --- Parameters ---
    std::string input_filename = "../day014/lena_gray.png"; // Default input file
    std::string output_dir = "./output_interpolated";
    std::string output_filename_gpu = ""; // Determined later
    std::string output_filename_cpu = ""; // Determined later
    float upscale_factor = 2.0f; // Upscale by 2x
    int camera_index = 0; // Default camera index
    bool use_camera = false;
    const int MAX_FRAMES = 100; // Number of frames to process from camera

    // --- Argument Parsing ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--camera") {
            use_camera = true;
            if (i + 1 < argc && argv[i + 1][0] != '-' && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
                camera_index = std::stoi(argv[++i]);
            }
            printf("Using camera input (index: %d).\n", camera_index);
        } else if (arg == "--input" && i + 1 < argc) {
            input_filename = argv[++i];
        } else if (arg == "--factor" && i + 1 < argc) {
            upscale_factor = std::stof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            // Allow specifying a base name for output, we'll add suffixes
            output_filename_gpu = argv[++i]; // Store base path temporarily
        } else {
            fprintf(stderr, "Usage: %s [--camera [index]] [--input <path>] [--factor <f>] [--output <base_output_path>]\n", argv[0]);
            return 1;
        }
    }

     // --- Determine Output Filenames ---
    std::string base_output_name;
    if (!output_filename_gpu.empty()) { // User specified a base path
        // If user provided output, assume it's a full path or base name without suffixes
         size_t last_dot_idx = output_filename_gpu.find_last_of('.');
         if (last_dot_idx != std::string::npos && output_filename_gpu.substr(last_dot_idx) == ".png") {
             // Assume it includes '.png', treat part before as base
            base_output_name = output_filename_gpu.substr(0, last_dot_idx);
             // Remove potential existing _gpu or _cpu suffixes from user input if present
            size_t suffix_pos = base_output_name.rfind("_gpu");
            if (suffix_pos != std::string::npos) base_output_name.resize(suffix_pos);
            suffix_pos = base_output_name.rfind("_cpu");
            if (suffix_pos != std::string::npos) base_output_name.resize(suffix_pos);
         } else {
             // Assume it's just a base name (e.g., "output/my_image")
             base_output_name = output_filename_gpu;
         }
    } else if (use_camera) {
        base_output_name = output_dir + "/camera_frame_interpolated";
    } else { // Use input filename base
        size_t last_slash_idx = input_filename.find_last_of('/');
        std::string base_name = (last_slash_idx == std::string::npos) ? input_filename : input_filename.substr(last_slash_idx + 1);
        size_t dot_idx = base_name.find_last_of('.');
        std::string name_without_ext = (dot_idx == std::string::npos) ? base_name : base_name.substr(0, dot_idx);
        base_output_name = output_dir + "/" + name_without_ext + "_interpolated";
    }
    // Add suffixes
    // Format factor nicely, e.g., x2.0, x1.5
    char factor_buf[10];
    snprintf(factor_buf, sizeof(factor_buf), "%.1f", upscale_factor);
    std::string factor_str = "_x" + std::string(factor_buf);

    output_filename_gpu = base_output_name + "_gpu" + factor_str + ".png";
    output_filename_cpu = base_output_name + "_cpu" + factor_str + ".png";


    if (use_camera) {
         printf("Mode: Camera Input\n");
         printf("Camera Index: %d\n", camera_index);
    } else {
         printf("Mode: File Input\n");
         printf("Input image: %s\n", input_filename.c_str());
    }
    printf("Upscale factor: %.2f\n", upscale_factor);
    printf("Output image GPU (last frame for camera): %s\n", output_filename_gpu.c_str());
    printf("Output image CPU (last frame for camera): %s\n", output_filename_cpu.c_str());


    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat(output_dir.c_str(), &st) == -1) {
        if (mkdir(output_dir.c_str(), 0755) == -1) { // Use 0755 for permissions
             fprintf(stderr, "Error: Could not create output directory: %s\n", output_dir.c_str());
             // Attempt to continue if it already exists (race condition?)
             if (stat(output_dir.c_str(), &st) == -1) return -1; // Exit if still not there
        } else {
             printf("Created output directory: %s\n", output_dir.c_str());
        }
    }


    // --- Initialization ---
    cv::VideoCapture cap;
    cv::Mat h_frame_bgr; // Host BGR frame from camera or file
    cv::Mat h_frame_gray; // Host Grayscale uchar frame
    cv::Mat h_input_float; // Host Grayscale float frame
    uchar3* d_bgr_input = nullptr; // Device BGR input (for camera mode)
    float* d_gray_input = nullptr; // Device Grayscale float input (used for texture binding)

    int inWidth = 0, inHeight = 0;
    int outWidth = 0, outHeight = 0;
    size_t inputGrayBytes = 0;


    if (use_camera) {
        // --- Camera Initialization ---
        cap.open(camera_index, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            fprintf(stderr, "Error: Could not open camera with index %d using V4L2. Trying default...\n", camera_index);
            cap.open(camera_index);
            if (!cap.isOpened()) {
                fprintf(stderr, "Error: Could not open camera with index %d using default backend.\n", camera_index);
                return -1;
            }
            fprintf(stderr, "Warning: Opened camera with default backend after V4L2 failed.\n");
        } else {
             printf("Camera opened successfully using cv::CAP_V4L2 backend.\n");
        }

        inWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        inHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (inWidth <= 0 || inHeight <= 0) {
             fprintf(stderr, "Error: Could not get valid frame dimensions (%dx%d)\n", inWidth, inHeight);
             cap.release();
             return -1;
        }
        printf("Camera frame dimensions: %d x %d\n", inWidth, inHeight);
        // Allocate device memory for BGR input
        CHECK_CUDA_ERROR(cudaMalloc(&d_bgr_input, (size_t)inWidth * inHeight * sizeof(uchar3)));
    } else {
        // --- Load Input Image (Host) ---
        h_frame_bgr = cv::imread(input_filename, cv::IMREAD_COLOR); // Load as color first
        if (h_frame_bgr.empty()) {
            fprintf(stderr, "Error: Could not load input image: %s\n", input_filename.c_str());
            return -1;
        }
         if (h_frame_bgr.channels() == 1) {
             // If already grayscale, create a 3-channel version for consistency
             cv::cvtColor(h_frame_bgr, h_frame_bgr, cv::COLOR_GRAY2BGR);
             printf("Input image was grayscale, converted to BGR for processing pipeline.\n");
         } else if (h_frame_bgr.channels() != 3) {
             fprintf(stderr, "Error: Input image has %d channels, expected 3 (BGR) or 1 (Grayscale).\n", h_frame_bgr.channels());
             return -1;
         }
        inWidth = h_frame_bgr.cols;
        inHeight = h_frame_bgr.rows;
        printf("Loaded input image: %d x %d channels: %d\n", inWidth, inHeight, h_frame_bgr.channels());
        // Allocate device memory for BGR input and copy
        CHECK_CUDA_ERROR(cudaMalloc(&d_bgr_input, (size_t)inWidth * inHeight * sizeof(uchar3)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_bgr_input, h_frame_bgr.ptr<uchar>(), (size_t)inWidth * inHeight * sizeof(uchar3), cudaMemcpyHostToDevice));
    }

    // --- Common Initialization ---
    outWidth = static_cast<int>(inWidth * upscale_factor);
    outHeight = static_cast<int>(inHeight * upscale_factor);
    inputGrayBytes = (size_t)inWidth * inHeight * sizeof(float);
    printf("Input dimensions: %d x %d\n", inWidth, inHeight);
    printf("Output dimensions: %d x %d\n", outWidth, outHeight);

    // Allocate device memory for grayscale float input (will be texture source)
    CHECK_CUDA_ERROR(cudaMalloc(&d_gray_input, inputGrayBytes));

    // --- Texture Object Setup ---
    // Resource description for binding texture to linear memory (d_gray_input)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_gray_input;
    resDesc.res.pitch2D.desc = channelDesc;
    resDesc.res.pitch2D.width = inWidth;
    resDesc.res.pitch2D.height = inHeight;
    resDesc.res.pitch2D.pitchInBytes = (size_t)inWidth * sizeof(float); // Pitch for linear memory

    // Texture description (filtering, addressing)
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0; // Will be created/destroyed inside loop/processing


    // --- Allocate Output Memory (Device) ---
    float* d_output = nullptr;
    size_t outputBytes = (size_t)outWidth * outHeight * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, outputBytes));

    // Kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridGray((inWidth + blockSize.x - 1) / blockSize.x, (inHeight + blockSize.y - 1) / blockSize.y);
    dim3 gridInterpolate((outWidth + blockSize.x - 1) / blockSize.x, (outHeight + blockSize.y - 1) / blockSize.y);

    // Host buffer for final output
    cv::Mat h_output_float(outHeight, outWidth, CV_32FC1);
    cv::Mat h_output_img; // uchar version for saving

    // --- Processing Loop (Camera or File) ---
    int frame_count = use_camera ? MAX_FRAMES : 1;
    printf("Starting processing for %d frame(s)...\n", frame_count);

    for (int i = 0; i < frame_count; ++i) {
        auto frame_start_time = std::chrono::high_resolution_clock::now();

        // 1. Get BGR Frame (Capture or use loaded one)
        if (use_camera) {
            if (!cap.read(h_frame_bgr)) { // More robust check
                 fprintf(stderr, "Warning: Failed to capture frame from camera, skipping. (Frame %d)\n", i);
                 usleep(50000); // Wait briefly before trying again
                 continue;
            }
            if (h_frame_bgr.empty()) {
                fprintf(stderr, "Warning: Captured empty frame, skipping. (Frame %d)\n", i);
                continue;
            }
            // Ensure frame is 3-channel uchar (CV_8UC3) - Check might be redundant with cap.read if format is set
            if (h_frame_bgr.type() != CV_8UC3) {
                fprintf(stderr, "Warning: Camera frame type is not CV_8UC3 (Type: %d), attempting conversion. (Frame %d)\n", h_frame_bgr.type(), i);
                cv::Mat temp_frame;
                if (h_frame_bgr.channels() == 1) {
                    cv::cvtColor(h_frame_bgr, temp_frame, cv::COLOR_GRAY2BGR);
                } else {
                    h_frame_bgr.convertTo(temp_frame, CV_8UC3);
                }
                 if(temp_frame.empty() || temp_frame.type() != CV_8UC3){
                     fprintf(stderr, "Error: Could not convert camera frame to CV_8UC3. Skipping Frame %d.\n", i);
                     continue;
                 }
                 h_frame_bgr = temp_frame;
            }
             // Transfer H->D for camera frame
            CHECK_CUDA_ERROR(cudaMemcpy(d_bgr_input, h_frame_bgr.ptr<uchar>(), (size_t)inWidth * inHeight * sizeof(uchar3), cudaMemcpyHostToDevice));
        } else if (i > 0) {
            // File mode only processes once
            break;
        }
        // If file mode (i=0), d_bgr_input was already populated during initialization.

        // 2. Convert BGR to Grayscale Float (GPU)
        rgb_to_gray_float_kernel<<<gridGray, blockSize>>>(d_bgr_input, d_gray_input, inWidth, inHeight);
        CHECK_CUDA_ERROR(cudaGetLastError()); // Check kernel launch error

        // 3. Create/Update Texture Object (Bound to d_gray_input)
        // Destroy previous texture object if it exists (necessary in loop)
        if (texObj != 0) {
            CHECK_CUDA_ERROR(cudaDestroyTextureObject(texObj));
            texObj = 0; // Reset handle
        }
        // Update resource description pointer (in case d_gray_input changed, though it doesn't here)
        resDesc.res.pitch2D.devPtr = d_gray_input;
        // Create Texture Object for the current frame's data in d_gray_input
        CHECK_CUDA_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        // 4. Launch Interpolation Kernel
        bilinear_interpolation_kernel<<<gridInterpolate, blockSize>>>(texObj, d_output, outWidth, outHeight);
        CHECK_CUDA_ERROR(cudaGetLastError()); // Check kernel launch error
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for GPU to finish

        auto frame_end_time = std::chrono::high_resolution_clock::now();
        auto frame_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(frame_end_time - frame_start_time).count();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        // 5. Log Output (Headless)
        // Use %ld for timestamp (long int) instead of %lld (long long int)
        printf("Frame: %d, Timestamp_ms: %ld, GPU Interpolation Time: %.3f ms\n",
               i, timestamp, frame_duration_us / 1000.0);


        // 6. If it's the last frame, copy result back and save
        if (i == frame_count - 1) {
            printf("Processing last frame, copying result and saving...\n");
            // --- Copy Result Back to Host ---
            CHECK_CUDA_ERROR(cudaMemcpy(h_output_float.ptr<float>(), d_output, outputBytes, cudaMemcpyDeviceToHost));

            // --- Convert GPU result to 8-bit and Save Output ---
            // Clamp values before converting to prevent wrap-around or saturation issues
            cv::threshold(h_output_float, h_output_float, 255.0, 255.0, cv::THRESH_TRUNC); // Values > 255 become 255
            cv::threshold(h_output_float, h_output_float, 0.0, 0.0, cv::THRESH_TOZERO);   // Values < 0 become 0 (though grayscale shouldn't be < 0)
            h_output_float.convertTo(h_output_img, CV_8UC1);

            if (!cv::imwrite(output_filename_gpu, h_output_img)) {
                fprintf(stderr, "Error: Could not save GPU output image: %s\n", output_filename_gpu.c_str());
            } else {
                printf("Saved GPU interpolated image to: %s\n", output_filename_gpu.c_str());
            }

            // --- Optional: CPU Comparison for the last frame ---
            // Need the original grayscale uchar frame for cv::resize.
            // Get it from the host BGR frame used in this iteration.
            if (!h_frame_bgr.empty()) {
                 cv::cvtColor(h_frame_bgr, h_frame_gray, cv::COLOR_BGR2GRAY);

                 if (!h_frame_gray.empty()) {
                     cv::Mat h_output_cpu;
                     auto cpu_start = std::chrono::high_resolution_clock::now();
                     cv::resize(h_frame_gray, h_output_cpu, cv::Size(outWidth, outHeight), 0, 0, cv::INTER_LINEAR);
                     auto cpu_stop = std::chrono::high_resolution_clock::now();
                     auto cpu_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count();

                     if (!cv::imwrite(output_filename_cpu, h_output_cpu)) {
                          fprintf(stderr, "Warning: Could not save CPU interpolated image: %s\n", output_filename_cpu.c_str());
                     } else {
                         printf("Saved CPU interpolated image to: %s\n", output_filename_cpu.c_str());
                         printf("CPU Interpolation Time (last frame): %.3f ms\n", cpu_duration_us / 1000.0);
                     }
                 } else {
                     fprintf(stderr, "Warning: Could not convert host BGR frame to grayscale for CPU comparison.\n");
                 }
             } else {
                 fprintf(stderr, "Warning: Host BGR frame was empty, cannot perform CPU comparison.\n");
             }
        } // End if last frame

        // Optional delay for camera mode to avoid overwhelming system
        if (use_camera) {
            usleep(10000); // 10 ms delay
        }

    } // End frame loop

    printf("Processing complete.\n");


    // --- Cleanup ---
    printf("Cleaning up resources...\n");
    if (cap.isOpened()) {
        cap.release();
        printf("Camera released.\n");
    }
    if (texObj != 0) { // Destroy the last used texture object
        CHECK_CUDA_ERROR(cudaDestroyTextureObject(texObj));
    }
    // Free device memory
    if (d_bgr_input) CHECK_CUDA_ERROR(cudaFree(d_bgr_input));
    if (d_gray_input) CHECK_CUDA_ERROR(cudaFree(d_gray_input));
    if (d_output) CHECK_CUDA_ERROR(cudaFree(d_output));
    printf("Cleanup complete.\n");

    return 0;
}
