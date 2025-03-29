#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <chrono>
#include <complex>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(error), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// cuFFT error checking macro
#define CHECK_CUFFT_ERROR(call) \
{ \
    cufftResult error = call; \
    if (error != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error: %d at line %d\n", error, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Define complex data type
typedef float2 Complex;

// Utility function to create a complex number
__host__ __device__ Complex make_complex(float re, float im) {
    Complex c;
    c.x = re;
    c.y = im;
    return c;
}

// Complex multiplication
__host__ __device__ Complex complex_mul(Complex a, Complex b) {
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex addition
__host__ __device__ Complex complex_add(Complex a, Complex b) {
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex subtraction
__host__ __device__ Complex complex_sub(Complex a, Complex b) {
    Complex c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

// Generate twiddle factors for FFT
__global__ void generate_twiddle_factors(Complex* twiddle_factors, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float angle = -2.0f * M_PI * idx / n;
        twiddle_factors[idx] = make_complex(cosf(angle), sinf(angle));
    }
}

// Cooley-Tukey FFT kernel for power-of-two sizes
__global__ void fft_kernel(Complex* input, Complex* output, Complex* twiddle_factors, int n, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n / 2) {
        // Calculate butterfly indices
        int even_idx = idx;
        int odd_idx = idx + n / 2;
        
        // Get input values
        Complex even = input[even_idx];
        Complex odd = input[odd_idx];
        
        // Get twiddle factor
        Complex twiddle = twiddle_factors[(idx * step) % n];
        
        // Perform butterfly operation
        Complex temp = complex_mul(odd, twiddle);
        output[idx] = complex_add(even, temp);
        output[idx + n / 2] = complex_sub(even, temp);
    }
}

// Iterative Cooley-Tukey FFT implementation
void custom_fft(Complex* h_input, Complex* h_output, int n) {
    // Allocate device memory
    Complex *d_input, *d_output, *d_twiddle_factors, *d_temp;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(Complex)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(Complex)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_twiddle_factors, n * sizeof(Complex)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp, n * sizeof(Complex)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, n * sizeof(Complex), cudaMemcpyHostToDevice));
    
    // Generate twiddle factors
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    generate_twiddle_factors<<<grid_size, block_size>>>(d_twiddle_factors, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Bit-reversal permutation (simplified for power-of-two sizes)
    // For simplicity, we'll do this on the CPU
    std::vector<Complex> bit_reversed(n);
    for (int i = 0; i < n; i++) {
        int reversed = 0;
        int bits = log2(n);
        for (int j = 0; j < bits; j++) {
            reversed = (reversed << 1) | ((i >> j) & 1);
        }
        bit_reversed[reversed] = h_input[i];
    }
    
    // Copy bit-reversed data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, bit_reversed.data(), n * sizeof(Complex), cudaMemcpyHostToDevice));
    
    // Perform FFT iterations
    Complex *d_src = d_input;
    Complex *d_dst = d_output;
    
    for (int stage = 1; stage <= log2(n); stage++) {
        int step = 1 << (stage - 1);
        int butterfly_size = 1 << stage;
        
        grid_size = (butterfly_size / 2 + block_size - 1) / block_size;
        
        for (int offset = 0; offset < n; offset += butterfly_size) {
            fft_kernel<<<grid_size, block_size>>>(
                d_src + offset, 
                d_dst + offset, 
                d_twiddle_factors, 
                butterfly_size, 
                n / butterfly_size
            );
        }
        
        // Swap source and destination for next iteration
        Complex *temp = d_src;
        d_src = d_dst;
        d_dst = temp;
        
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    
    // Copy result back to host
    // If we did an odd number of iterations, the result is in d_src
    if ((int)log2(n) % 2 == 1) {
        CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_src, n * sizeof(Complex), cudaMemcpyDeviceToHost));
    } else {
        CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_dst, n * sizeof(Complex), cudaMemcpyDeviceToHost));
    }
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_twiddle_factors));
    CHECK_CUDA_ERROR(cudaFree(d_temp));
}

// Perform FFT using cuFFT library
void cufft_fft(Complex* h_input, Complex* h_output, int n) {
    // Allocate device memory
    cufftComplex *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(cufftComplex)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(cufftComplex)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, n * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    
    // Create cuFFT plan
    cufftHandle plan;
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan, n, CUFFT_C2C, 1));
    
    // Execute FFT
    CHECK_CUFFT_ERROR(cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD));
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, n * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    
    // Clean up
    CHECK_CUFFT_ERROR(cufftDestroy(plan));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

// Generate a complex signal with multiple frequencies
void generate_complex_signal(Complex* signal, int n, float sample_rate) {
    for (int i = 0; i < n; i++) {
        float t = (float)i / sample_rate;
        // Sum of three sine waves with different frequencies
        signal[i].x = sinf(2.0f * M_PI * 50.0f * t) +  // 50 Hz
                      0.5f * sinf(2.0f * M_PI * 120.0f * t) +  // 120 Hz
                      0.25f * sinf(2.0f * M_PI * 200.0f * t);  // 200 Hz
        signal[i].y = 0.0f;
    }
}

// Calculate the magnitude of a complex number
float magnitude(Complex c) {
    return sqrtf(c.x * c.x + c.y * c.y);
}

// Calculate the phase of a complex number in radians
float phase(Complex c) {
    return atan2f(c.y, c.x);
}

// Apply FFT to an image (2D FFT using row-column decomposition)
void apply_fft_to_image(const cv::Mat& input_image, cv::Mat& magnitude_spectrum) {
    // Convert input image to grayscale
    cv::Mat gray_image;
    if (input_image.channels() == 3) {
        cv::cvtColor(input_image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = input_image.clone();
    }
    
    // Expand input image to optimal size (power of 2)
    int rows = cv::getOptimalDFTSize(gray_image.rows);
    int cols = cv::getOptimalDFTSize(gray_image.cols);
    
    cv::Mat padded;
    cv::copyMakeBorder(gray_image, padded, 0, rows - gray_image.rows, 0, cols - gray_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    // Create complex image (real and imaginary parts)
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complex_image;
    cv::merge(planes, 2, complex_image);
    
    // Perform FFT
    cv::dft(complex_image, complex_image);
    
    // Split the complex result into real and imaginary parts
    cv::split(complex_image, planes);
    
    // Compute magnitude
    cv::magnitude(planes[0], planes[1], planes[0]);
    magnitude_spectrum = planes[0];
    
    // Switch to logarithmic scale for better visualization
    magnitude_spectrum += cv::Scalar::all(1);
    cv::log(magnitude_spectrum, magnitude_spectrum);
    
    // Normalize for display
    cv::normalize(magnitude_spectrum, magnitude_spectrum, 0, 1, cv::NORM_MINMAX);
    
    // Rearrange quadrants to have the origin at the center
    int cx = magnitude_spectrum.cols / 2;
    int cy = magnitude_spectrum.rows / 2;
    
    cv::Mat q0(magnitude_spectrum, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magnitude_spectrum, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magnitude_spectrum, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magnitude_spectrum, cv::Rect(cx, cy, cx, cy));
    
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

// Apply a low-pass filter to an image in the frequency domain
void apply_low_pass_filter(const cv::Mat& input_image, cv::Mat& output_image, float cutoff_radius) {
    // Convert input image to grayscale
    cv::Mat gray_image;
    if (input_image.channels() == 3) {
        cv::cvtColor(input_image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = input_image.clone();
    }
    
    // Expand input image to optimal size (power of 2)
    int rows = cv::getOptimalDFTSize(gray_image.rows);
    int cols = cv::getOptimalDFTSize(gray_image.cols);
    
    cv::Mat padded;
    cv::copyMakeBorder(gray_image, padded, 0, rows - gray_image.rows, 0, cols - gray_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    // Create complex image (real and imaginary parts)
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complex_image;
    cv::merge(planes, 2, complex_image);
    
    // Perform FFT
    cv::dft(complex_image, complex_image);
    
    // Create the low-pass filter mask
    cv::Mat mask = cv::Mat::zeros(complex_image.size(), CV_32F);
    cv::Point center(mask.cols / 2, mask.rows / 2);
    
    // Create a circle with the specified cutoff radius
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            float distance = sqrtf(powf((i - center.y), 2) + powf((j - center.x), 2));
            if (distance <= cutoff_radius) {
                mask.at<float>(i, j) = 1.0f;
            }
        }
    }
    
    // Split the complex image
    cv::split(complex_image, planes);
    
    // Apply the mask to both real and imaginary parts
    planes[0] = planes[0].mul(mask);
    planes[1] = planes[1].mul(mask);
    
    // Merge the filtered components
    cv::merge(planes, 2, complex_image);
    
    // Perform inverse FFT
    cv::dft(complex_image, complex_image, cv::DFT_INVERSE | cv::DFT_SCALE);
    
    // Split the result into real and imaginary parts
    cv::split(complex_image, planes);
    
    // The real part contains the filtered image
    output_image = planes[0];
    
    // Crop the result to the original size
    output_image = output_image(cv::Rect(0, 0, gray_image.cols, gray_image.rows));
    
    // Normalize for display
    cv::normalize(output_image, output_image, 0, 255, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8U);
}

// Apply a high-pass filter to an image in the frequency domain
void apply_high_pass_filter(const cv::Mat& input_image, cv::Mat& output_image, float cutoff_radius) {
    // Convert input image to grayscale
    cv::Mat gray_image;
    if (input_image.channels() == 3) {
        cv::cvtColor(input_image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = input_image.clone();
    }
    
    // Expand input image to optimal size (power of 2)
    int rows = cv::getOptimalDFTSize(gray_image.rows);
    int cols = cv::getOptimalDFTSize(gray_image.cols);
    
    cv::Mat padded;
    cv::copyMakeBorder(gray_image, padded, 0, rows - gray_image.rows, 0, cols - gray_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    // Create complex image (real and imaginary parts)
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complex_image;
    cv::merge(planes, 2, complex_image);
    
    // Perform FFT
    cv::dft(complex_image, complex_image);
    
    // Create the high-pass filter mask
    cv::Mat mask = cv::Mat::ones(complex_image.size(), CV_32F);
    cv::Point center(mask.cols / 2, mask.rows / 2);
    
    // Create a circle with the specified cutoff radius
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            float distance = sqrtf(powf((i - center.y), 2) + powf((j - center.x), 2));
            if (distance <= cutoff_radius) {
                mask.at<float>(i, j) = 0.0f;
            }
        }
    }
    
    // Split the complex image
    cv::split(complex_image, planes);
    
    // Apply the mask to both real and imaginary parts
    planes[0] = planes[0].mul(mask);
    planes[1] = planes[1].mul(mask);
    
    // Merge the filtered components
    cv::merge(planes, 2, complex_image);
    
    // Perform inverse FFT
    cv::dft(complex_image, complex_image, cv::DFT_INVERSE | cv::DFT_SCALE);
    
    // Split the result into real and imaginary parts
    cv::split(complex_image, planes);
    
    // The real part contains the filtered image
    output_image = planes[0];
    
    // Crop the result to the original size
    output_image = output_image(cv::Rect(0, 0, gray_image.cols, gray_image.rows));
    
    // Normalize for display
    cv::normalize(output_image, output_image, 0, 255, cv::NORM_MINMAX);
    output_image.convertTo(output_image, CV_8U);
}

// Visualize 1D FFT results
void visualize_1d_fft(Complex* signal, Complex* fft_result, int n, float sample_rate) {
    // Create time domain plot
    cv::Mat time_domain_plot(400, 800, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Draw axes
    cv::line(time_domain_plot, cv::Point(50, 350), cv::Point(750, 350), cv::Scalar(0, 0, 0), 2);
    cv::line(time_domain_plot, cv::Point(50, 50), cv::Point(50, 350), cv::Scalar(0, 0, 0), 2);
    
    // Draw axis labels
    cv::putText(time_domain_plot, "Time (s)", cv::Point(400, 390), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(time_domain_plot, "Amplitude", cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    
    // Draw time domain signal
    for (int i = 0; i < n - 1; i++) {
        float x1 = 50 + i * 700.0f / n;
        float y1 = 350 - signal[i].x * 100.0f;
        float x2 = 50 + (i + 1) * 700.0f / n;
        float y2 = 350 - signal[i + 1].x * 100.0f;
        
        cv::line(time_domain_plot, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
    }
    
    // Create frequency domain plot
    cv::Mat freq_domain_plot(400, 800, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Draw axes
    cv::line(freq_domain_plot, cv::Point(50, 350), cv::Point(750, 350), cv::Scalar(0, 0, 0), 2);
    cv::line(freq_domain_plot, cv::Point(50, 50), cv::Point(50, 350), cv::Scalar(0, 0, 0), 2);
    
    // Draw axis labels
    cv::putText(freq_domain_plot, "Frequency (Hz)", cv::Point(400, 390), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(freq_domain_plot, "Magnitude", cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    
    // Draw frequency domain signal (only the first half, as the second half is symmetric for real signals)
    float max_magnitude = 0.0f;
    for (int i = 0; i < n / 2; i++) {
        float mag = magnitude(fft_result[i]);
        if (mag > max_magnitude) {
            max_magnitude = mag;
        }
    }
    
    for (int i = 0; i < n / 2; i++) {
        float freq = i * sample_rate / n;
        float mag = magnitude(fft_result[i]) / max_magnitude;
        
        float x = 50 + i * 700.0f / (n / 2);
        float y = 350 - mag * 300.0f;
        
        cv::line(freq_domain_plot, cv::Point(x, 350), cv::Point(x, y), cv::Scalar(0, 0, 255), 2);
    }
    
    // Display plots
    cv::imshow("Time Domain Signal", time_domain_plot);
    cv::imshow("Frequency Domain (Magnitude Spectrum)", freq_domain_plot);
    
    // Save plots
    cv::imwrite("time_domain.png", time_domain_plot);
    cv::imwrite("frequency_domain.png", freq_domain_plot);
}

int main() {
    printf("CUDA FFT Visualization\n");
    printf("======================\n\n");
    
    // 1. Visualize 1D FFT
    printf("1. Visualizing 1D FFT of a complex signal\n");
    
    // Set FFT size (must be a power of 2)
    const int n = 1024;
    const float sample_rate = 1000.0f;  // 1000 Hz
    
    // Allocate host memory
    Complex* h_signal = (Complex*)malloc(n * sizeof(Complex));
    Complex* h_fft_result = (Complex*)malloc(n * sizeof(Complex));
    
    // Generate test signal
    generate_complex_signal(h_signal, n, sample_rate);
    
    // Perform FFT using cuFFT (more accurate)
    cufft_fft(h_signal, h_fft_result, n);
    
    // Visualize 1D FFT results
    visualize_1d_fft(h_signal, h_fft_result, n, sample_rate);
    
    // 2. Image FFT and filtering
    printf("2. Applying FFT to an image and demonstrating filtering\n");
    
    // Load an image
    cv::Mat input_image = cv::imread("input_image.jpg", cv::IMREAD_GRAYSCALE);
    
    // If no image is loaded, create a synthetic test image
    if (input_image.empty()) {
        printf("No input image found, creating a synthetic test image\n");
        input_image = cv::Mat(512, 512, CV_8UC1, cv::Scalar(0));
        
        // Draw some shapes
        cv::circle(input_image, cv::Point(256, 256), 100, cv::Scalar(255), -1);
        cv::rectangle(input_image, cv::Rect(100, 100, 100, 100), cv::Scalar(255), -1);
        cv::line(input_image, cv::Point(400, 100), cv::Point(500, 200), cv::Scalar(255), 5);
        
        // Add some noise
        cv::Mat noise(512, 512, CV_8UC1);
        cv::randn(noise, 0, 25);
        input_image += noise;
        
        // Save the synthetic image
        cv::imwrite("synthetic_image.png", input_image);
    }
    
    // Compute and visualize the FFT magnitude spectrum
    cv::Mat magnitude_spectrum;
    apply_fft_to_image(input_image, magnitude_spectrum);
    
    // Apply low-pass filter
    cv::Mat low_pass_result;
    apply_low_pass_filter(input_image, low_pass_result, 30.0f);
    
    // Apply high-pass filter
    cv::Mat high_pass_result;
    apply_high_pass_filter(input_image, high_pass_result, 30.0f);
    
    // Display results
    cv::imshow("Original Image", input_image);
    cv::imshow("Magnitude Spectrum", magnitude_spectrum);
    cv::imshow("Low-Pass Filtered", low_pass_result);
    cv::imshow("High-Pass Filtered", high_pass_result);
    
    // Save results
    cv::imwrite("original_image.png", input_image);
    cv::imwrite("magnitude_spectrum.png", magnitude_spectrum * 255);
    cv::imwrite("low_pass_filtered.png", low_pass_result);
    cv::imwrite("high_pass_filtered.png", high_pass_result);
    
    printf("\nPress any key to exit...\n");
    cv::waitKey(0);
    
    // Free host memory
    free(h_signal);
    free(h_fft_result);
    
    return 0;
}