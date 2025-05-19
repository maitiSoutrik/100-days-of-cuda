#include "group_norm_forward.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip> // For std::fixed and std::setprecision
#include <chrono> // For timing

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

// Helper function to initialize gamma and beta
void initializeGammaBeta(float* data, int size, float val_mean = 1.0f, float val_stddev = 0.1f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(val_mean, val_stddev);
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

// Helper function to print a small part of the tensor for verification
void printTensor(const float* tensor, int N, int C, int H, int W, const std::string& name, int print_n=1, int print_c=1, int print_h=4, int print_w=4) {
    std::cout << name << " (first " << print_n << " N, " << print_c << " C, " << print_h << " H, " << print_w << " W):\n";
    for (int n = 0; n < std::min(N, print_n); ++n) {
        for (int c_idx = 0; c_idx < std::min(C, print_c); ++c_idx) {
            std::cout << "  N=" << n << ", C=" << c_idx << ":\n";
            for (int h = 0; h < std::min(H, print_h); ++h) {
                std::cout << "    ";
                for (int w = 0; w < std::min(W, print_w); ++w) {
                    int idx = n * C * H * W + c_idx * H * W + h * W + w;
                    std::cout << std::fixed << std::setprecision(4) << tensor[idx] << " ";
                }
                std::cout << "\n";
            }
        }
    }
    std::cout << std::endl;
}

// Helper function to compare CPU and GPU results
bool compareResults(const float* cpu_result, const float* gpu_result, int size, float tolerance = 1e-3f) { // Increased tolerance for float images
    for (int i = 0; i < size; ++i) {
        if (fabsf(cpu_result[i] - gpu_result[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": CPU=" << cpu_result[i]
                      << ", GPU=" << gpu_result[i] << ", Diff=" << fabsf(cpu_result[i] - gpu_result[i]) << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Parameters
    int N_param = 1; // Batch size is 1 for single camera frame
    int C_param = 1; // Channels: 1 for grayscale, 3 for color. Let's start with grayscale.
    int H_param = 240; // Desired Height, can be adjusted or taken from frame
    int W_param = 320; // Desired Width
    int G_param = 1;   // Groups. If C_param=1, G_param must be 1. If C_param=3, G_param can be 1 or 3.
    float epsilon = 1e-5f;

    cv::VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }
    std::cout << "Camera opened successfully." << std::endl;

    cv::Mat frame, processed_frame;
    cap.read(frame); // Capture a frame
    if (frame.empty()) {
        std::cerr << "Error: Captured empty frame." << std::endl;
        return -1;
    }
    std::cout << "Frame captured: " << frame.cols << "x" << frame.rows << " Channels: " << frame.channels() << std::endl;

    // Preprocess frame: resize and convert to grayscale float
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(W_param, H_param));

    if (C_param == 1) {
        cv::cvtColor(resized_frame, processed_frame, cv::COLOR_BGR2GRAY);
        std::cout << "Converted to grayscale." << std::endl;
    } else if (C_param == 3) {
        processed_frame = resized_frame; // Use as BGR
         if (G_param != 1 && G_param != 3) {
            std::cerr << "For C=3, G must be 1 or 3. Setting G=1." << std::endl;
            G_param = 1;
        }
        std::cout << "Using BGR color image." << std::endl;
    } else {
        std::cerr << "Unsupported C_param = " << C_param <&lt ". Must be 1 or 3 for camera input." << std::endl;
        return -1;
    }
    
    processed_frame.convertTo(processed_frame, CV_32F, 1.0/255.0); // Normalize to [0,1]

    N_param = 1; // Batch size
    C_param = processed_frame.channels(); // Update C based on processed image
    H_param = processed_frame.rows;
    W_param = processed_frame.cols;

    if (C_param % G_param != 0) {
        std::cerr << "Error: Number of channels C (" << C_param << ") from image must be divisible by G (" << G_param << ")." << std::endl;
        std::cerr << "Try setting G=1 or ensure C is compatible with G." << std::endl;
        if (C_param == 1) G_param = 1;
        else if (C_param == 3 && (G_param != 1 && G_param !=3)) G_param = 1;
        else if (C_param % G_param != 0) {
             std::cerr << "Cannot automatically adjust G. Exiting." << std::endl;
             return 1;
        }
        std::cout << "Adjusted G to " << G_param << " for C=" << C_param << std::endl;
    }


    int input_size = N_param * C_param * H_param * W_param;
    int params_size = C_param;

    std::vector<float> h_input(input_size);
    // Convert cv::Mat to NCHW float vector
    if (C_param == 1) {
        memcpy(h_input.data(), processed_frame.data, input_size * sizeof(float));
    } else { // C_param == 3 (BGR)
        for (int h = 0; h < H_param; ++h) {
            for (int w = 0; w < W_param; ++w) {
                cv::Vec3f pixel = processed_frame.at<cv::Vec3f>(h, w);
                h_input[0 * H_param * W_param + h * W_param + w] = pixel[0]; // B
                h_input[1 * H_param * W_param + h * W_param + w] = pixel[1]; // G
                h_input[2 * H_param * W_param + h * W_param + w] = pixel[2]; // R
            }
        }
    }
    std::cout << "Input tensor prepared: N=" << N_param << " C=" << C_param << " H=" << H_param << " W=" << W_param << std::endl;

    std::vector<float> h_gamma(params_size);
    std::vector<float> h_beta(params_size);
    std::vector<float> h_output_gpu(input_size);
    std::vector<float> h_output_cpu(input_size);

    initializeGammaBeta(h_gamma.data(), params_size, 1.0f, 0.01f);
    initializeGammaBeta(h_beta.data(), params_size, 0.0f, 0.01f);

    float *d_input, *d_gamma, *d_beta, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, input_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gamma, params_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_beta, params_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, input_size * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_gamma, h_gamma.data(), params_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_beta, h_beta.data(), params_size * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "Running GPU Group Normalization..." << std::endl;
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu));

    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu));
    groupNormForward(d_output, d_input, N_param, C_param, H_param, W_param, G_param, d_gamma, d_beta, epsilon);
    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu));

    float milliseconds_gpu = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds_gpu, start_gpu, stop_gpu));
    std::cout << "GPU Execution Time: " << milliseconds_gpu << " ms" << std::endl;

    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\nRunning CPU Group Normalization for verification..." << std::endl;
    auto start_cpu_chrono = std::chrono::high_resolution_clock::now();
    groupNormForwardCPU(h_output_cpu.data(), h_input.data(), N_param, C_param, H_param, W_param, G_param, h_gamma.data(), h_beta.data(), epsilon);
    auto stop_cpu_chrono = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu_chrono - start_cpu_chrono);
    std::cout << "CPU Execution Time: " << duration_cpu.count() / 1000.0 << " ms" << std::endl;
    
    // printTensor(h_input.data(), N_param, C_param, H_param, W_param, "Input Host");
    // printTensor(h_output_cpu.data(), N_param, C_param, H_param, W_param, "Output CPU");
    // printTensor(h_output_gpu.data(), N_param, C_param, H_param, W_param, "Output GPU");

    bool success = compareResults(h_output_cpu.data(), h_output_gpu.data(), input_size);
    if (success) {
        std::cout << "\nVerification Successful: GPU and CPU results match." << std::endl;
    } else {
        std::cout << "\nVerification Failed: GPU and CPU results differ." << std::endl;
    }
    
    // Display output (optional, first channel if multiple)
    cv::Mat output_display_mat(H_param, W_param, CV_32FC1);
    // Copy the first channel of h_output_gpu to output_display_mat
    for (int h = 0; h < H_param; ++h) {
        for (int w = 0; w < W_param; ++w) {
            output_display_mat.at<float>(h,w) = h_output_gpu[h * W_param + w]; // Assuming N=1, first channel
        }
    }
    // Normalize for display if needed (e.g. if values are not in [0,1])
    double minVal, maxVal;
    cv::minMaxLoc(output_display_mat, &minVal, &maxVal);
    if (maxVal > minVal) {
         output_display_mat.convertTo(output_display_mat, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
    } else {
        output_display_mat.convertTo(output_display_mat, CV_8U);
    }

    cv::imshow("Original Grayscale", processed_frame); // Show processed input
    cv::imshow("Group Norm Output (GPU)", output_display_mat);
    std::cout << "Displaying images. Press any key to close." << std::endl;
    cv::waitKey(0);


    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_gamma));
    CHECK_CUDA_ERROR(cudaFree(d_beta));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu));
    cap.release();
    cv::destroyAllWindows();

    return success ? 0 : 1;
}
