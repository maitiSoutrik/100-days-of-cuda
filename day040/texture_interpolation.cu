#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp> // For imread/imwrite
#include <opencv2/imgproc.hpp> // For cvtColor, resize
#include <chrono>
#include <string>
#include <sys/stat.h> // For checking/creating directory

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


int main(int argc, char *argv[]) {
    // --- Parameters ---
    std::string input_filename = "../day014/lena_gray.png"; // Default input
    std.string output_dir = "./output_interpolated";
    std::string output_filename = output_dir + "/lena_interpolated_gpu.png";
    float upscale_factor = 2.0f; // Upscale by 2x

    if (argc > 1) {
        input_filename = argv[1];
        printf("Using input image: %s\n", input_filename.c_str());
         // Basic output filename generation if input changes
        size_t last_slash_idx = input_filename.find_last_of('/');
        std::string base_name = (last_slash_idx == std::string::npos) ? input_filename : input_filename.substr(last_slash_idx + 1);
        size_t dot_idx = base_name.find_last_of('.');
        std::string name_without_ext = (dot_idx == std::string::npos) ? base_name : base_name.substr(0, dot_idx);
        output_filename = output_dir + "/" + name_without_ext + "_interpolated_gpu.png";
    }
     if (argc > 2) {
        upscale_factor = std::stof(argv[2]);
         printf("Using upscale factor: %.2f\n", upscale_factor);
          // Update output filename suffix if upscale factor changes
         size_t last_dot_idx = output_filename.find_last_of('.');
         std::string base_output_name = output_filename.substr(0, last_dot_idx);
         std::string extension = output_filename.substr(last_dot_idx);
         output_filename = base_output_name + "_x" + std::to_string(static_cast<int>(upscale_factor)) + extension;

    }
    if (argc > 3) {
        output_filename = argv[3];
         printf("Using output filename: %s\n", output_filename.c_str());
    }

    printf("Input image: %s\n", input_filename.c_str());
    printf("Upscale factor: %.2f\n", upscale_factor);
    printf("Output image: %s\n", output_filename.c_str());


    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat(output_dir.c_str(), &st) == -1) {
        mkdir(output_dir.c_str(), 0700); // Create directory with read/write/execute permissions for owner
        printf("Created output directory: %s\n", output_dir.c_str());
    }


    // --- Load Input Image (Host) ---
    cv::Mat h_input_img = cv::imread(input_filename, cv::IMREAD_GRAYSCALE);
    if (h_input_img.empty()) {
        fprintf(stderr, "Error: Could not load input image: %s\n", input_filename.c_str());
        return -1;
    }
    printf("Loaded input image: %d x %d channels: %d\n", h_input_img.cols, h_input_img.rows, h_input_img.channels());

    // Convert to 32-bit float for interpolation precision
    cv::Mat h_input_float;
    h_input_img.convertTo(h_input_float, CV_32FC1);

    int inWidth = h_input_float.cols;
    int inHeight = h_input_float.rows;
    int outWidth = static_cast<int>(inWidth * upscale_factor);
    int outHeight = static_cast<int>(inHeight * upscale_factor);
    printf("Input dimensions (float): %d x %d\n", inWidth, inHeight);
    printf("Output dimensions: %d x %d\n", outWidth, outHeight);


    // --- Allocate CUDA Array for Texture ---
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); // float texture
    cudaArray* d_inputArray = nullptr;
    CHECK_CUDA_ERROR(cudaMallocArray(&d_inputArray, &channelDesc, inWidth, inHeight));

    // --- Copy Host Image to CUDA Array ---
    size_t pitch = inWidth * sizeof(float);
    CHECK_CUDA_ERROR(cudaMemcpy2DToArray(d_inputArray, 0, 0, h_input_float.ptr<float>(), pitch, inWidth * sizeof(float), inHeight, cudaMemcpyHostToDevice));

    // --- Create Texture Object ---
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_inputArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp; // Clamp coordinates to edge
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;     // Bilinear filtering
    texDesc.readMode = cudaReadModeElementType;    // Read elements as float
    texDesc.normalizedCoords = 1;                  // Use normalized coordinates (0.0 to 1.0)

    cudaTextureObject_t texObj = 0;
    CHECK_CUDA_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));


    // --- Allocate Output Memory (Device) ---
    float* d_output = nullptr;
    size_t outputBytes = outWidth * outHeight * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, outputBytes));

    // --- Launch Kernel ---
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((outWidth + blockSize.x - 1) / blockSize.x, (outHeight + blockSize.y - 1) / blockSize.y);

    printf("Launching kernel with Grid: (%d, %d), Block: (%d, %d)\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    auto start = std::chrono::high_resolution_clock::now();

    bilinear_interpolation_kernel<<<gridSize, blockSize>>>(texObj, d_output, outWidth, outHeight);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    printf("GPU Interpolation Time: %.3f ms\n", duration.count() / 1000.0);


    // --- Copy Result Back to Host ---
    cv::Mat h_output_float(outHeight, outWidth, CV_32FC1);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_float.ptr<float>(), d_output, outputBytes, cudaMemcpyDeviceToHost));

    // --- Convert to 8-bit and Save Output ---
    cv::Mat h_output_img;
    // Clamp values to [0, 255] before converting to uchar
    cv::threshold(h_output_float, h_output_float, 255.0, 255.0, cv::THRESH_TRUNC); // Clamp > 255 to 255
    cv::threshold(h_output_float, h_output_float, 0.0, 0.0, cv::THRESH_TOZERO);   // Clamp < 0 to 0
    h_output_float.convertTo(h_output_img, CV_8UC1);

    if (!cv::imwrite(output_filename, h_output_img)) {
        fprintf(stderr, "Error: Could not save output image: %s\n", output_filename.c_str());
    } else {
        printf("Saved interpolated image to: %s\n", output_filename.c_str());
    }

    // --- Optional: CPU Comparison ---
    cv::Mat h_output_cpu;
    start = std::chrono::high_resolution_clock::now();
    cv::resize(h_input_img, h_output_cpu, cv::Size(outWidth, outHeight), 0, 0, cv::INTER_LINEAR);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::string cpu_output_filename = output_dir + "/lena_interpolated_cpu.png"; // Adjust naming as needed
     if (argc > 1) {
         size_t last_slash_idx = input_filename.find_last_of('/');
        std::string base_name = (last_slash_idx == std::string::npos) ? input_filename : input_filename.substr(last_slash_idx + 1);
        size_t dot_idx = base_name.find_last_of('.');
        std::string name_without_ext = (dot_idx == std::string::npos) ? base_name : base_name.substr(0, dot_idx);
         std::string suffix = (argc > 2) ? ("_x" + std::to_string(static_cast<int>(upscale_factor))) : "";
         cpu_output_filename = output_dir + "/" + name_without_ext + "_interpolated_cpu" + suffix + ".png";
     }

    if (!cv::imwrite(cpu_output_filename, h_output_cpu)) {
         fprintf(stderr, "Warning: Could not save CPU interpolated image: %s\n", cpu_output_filename.c_str());
    } else {
        printf("Saved CPU interpolated image to: %s\n", cpu_output_filename.c_str());
        printf("CPU Interpolation Time: %.3f ms\n", duration.count() / 1000.0);
    }


    // --- Cleanup ---
    printf("Cleaning up resources...\n");
    CHECK_CUDA_ERROR(cudaDestroyTextureObject(texObj));
    CHECK_CUDA_ERROR(cudaFreeArray(d_inputArray));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    printf("Cleanup complete.\n");

    return 0;
}
