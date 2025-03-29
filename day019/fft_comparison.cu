#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <chrono>
#include <complex>
#include <vector>
#include <iostream>
#include <iomanip>

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
// This is a simple implementation that works for small sizes
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

// Generate a simple signal (sine wave)
void generate_sine_wave(Complex* signal, int n, float frequency, float sample_rate) {
    for (int i = 0; i < n; i++) {
        float t = (float)i / sample_rate;
        signal[i].x = sinf(2.0f * M_PI * frequency * t);
        signal[i].y = 0.0f;
    }
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

// Print the first few elements of a complex array
void print_complex_array(const char* name, Complex* array, int n, int max_print = 10) {
    printf("%s:\n", name);
    for (int i = 0; i < std::min(n, max_print); i++) {
        printf("[%d] (%.4f, %.4f) - Magnitude: %.4f, Phase: %.4f rad\n", 
               i, array[i].x, array[i].y, magnitude(array[i]), phase(array[i]));
    }
    printf("\n");
}

// Calculate the mean squared error between two complex arrays
float calculate_mse(Complex* a, Complex* b, int n) {
    float mse = 0.0f;
    for (int i = 0; i < n; i++) {
        float re_diff = a[i].x - b[i].x;
        float im_diff = a[i].y - b[i].y;
        mse += re_diff * re_diff + im_diff * im_diff;
    }
    return mse / n;
}

int main() {
    // Set FFT size (must be a power of 2)
    const int fft_sizes[] = {256, 1024, 4096, 16384};
    const int num_sizes = sizeof(fft_sizes) / sizeof(fft_sizes[0]);
    
    // Sample rate for our synthetic signal
    const float sample_rate = 1000.0f;  // 1000 Hz
    
    printf("CUDA FFT Comparison: Custom Implementation vs. cuFFT\n");
    printf("===================================================\n\n");
    
    for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
        int n = fft_sizes[size_idx];
        printf("FFT Size: %d\n", n);
        printf("-----------\n");
        
        // Allocate host memory
        Complex* h_signal = (Complex*)malloc(n * sizeof(Complex));
        Complex* h_custom_fft_result = (Complex*)malloc(n * sizeof(Complex));
        Complex* h_cufft_result = (Complex*)malloc(n * sizeof(Complex));
        
        // Generate test signal
        generate_complex_signal(h_signal, n, sample_rate);
        
        // Print the first few elements of the input signal
        if (size_idx == 0) {  // Only print for the smallest size
            print_complex_array("Input Signal", h_signal, n);
        }
        
        // Perform custom FFT and measure time
        auto custom_start = std::chrono::high_resolution_clock::now();
        custom_fft(h_signal, h_custom_fft_result, n);
        auto custom_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> custom_duration = custom_end - custom_start;
        
        // Perform cuFFT and measure time
        auto cufft_start = std::chrono::high_resolution_clock::now();
        cufft_fft(h_signal, h_cufft_result, n);
        auto cufft_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cufft_duration = cufft_end - cufft_start;
        
        // Print timing results
        printf("Custom FFT Execution Time: %.3f ms\n", custom_duration.count());
        printf("cuFFT Execution Time: %.3f ms\n", cufft_duration.count());
        printf("Speedup (cuFFT vs Custom): %.2fx\n", custom_duration.count() / cufft_duration.count());
        
        // Calculate and print MSE between the two results
        float mse = calculate_mse(h_custom_fft_result, h_cufft_result, n);
        printf("Mean Squared Error: %.6e\n\n", mse);
        
        // Print the first few elements of the FFT results for the smallest size
        if (size_idx == 0) {
            print_complex_array("Custom FFT Result", h_custom_fft_result, n);
            print_complex_array("cuFFT Result", h_cufft_result, n);
            
            // Print the magnitude spectrum for the first few frequencies
            printf("Frequency Spectrum (Magnitude):\n");
            printf("Frequency (Hz) | Custom FFT | cuFFT\n");
            printf("----------------------------------------\n");
            
            for (int i = 0; i < std::min(n/2, 10); i++) {
                float freq = i * sample_rate / n;
                printf("%8.2f Hz   | %9.4f | %9.4f\n", 
                       freq, 
                       magnitude(h_custom_fft_result[i]), 
                       magnitude(h_cufft_result[i]));
            }
            printf("\n");
        }
        
        // Free host memory
        free(h_signal);
        free(h_custom_fft_result);
        free(h_cufft_result);
    }
    
    return 0;
}