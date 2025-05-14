#include "geglu.cuh"
#include <iostream>
#include <vector>
#include <cstdlib> // For rand
#include <iomanip> // For std::fixed, std::setprecision
#include <cmath>   // For M_PI, fabsf

// Define M_PI_F for CPU side if not already (should match .cu file)
#ifndef M_PI_F
#define M_PI_F ((float)M_PI)
#endif

// CPU GELU approximation (for verification)
float cpu_gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI_F) * (x + 0.044715f * x * x * x)));
}

// CPU GEGLU implementation (for verification)
void cpu_geglu(const std::vector<float>& input_a, const std::vector<float>& input_b, std::vector<float>& output) {
    if (input_a.size() != input_b.size() || input_a.size() != output.size()) {
        std::cerr << "CPU GEGLU: Input/Output size mismatch!" << std::endl;
        return;
    }
    for (size_t i = 0; i < input_a.size(); ++i) {
        output[i] = cpu_gelu_approx(input_a[i]) * input_b[i];
    }
}

void print_vector_sample(const std::string& name, const std::vector<float>& vec, int count = 5) {
    std::cout << name << ": [";
    for (int i = 0; i < std::min((int)vec.size(), count); ++i) {
        std::cout << std::fixed << std::setprecision(6) << vec[i] << (i == std::min((int)vec.size(), count) - 1 ? "" : ", ");
    }
    if (vec.size() > count) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
}

int main() {
    const int n = 1024; // Example size
    const float epsilon = 1e-5f; // Tolerance for comparison

    std::vector<float> h_input_a(n);
    std::vector<float> h_input_b(n);
    std::vector<float> h_output_gpu(n);
    std::vector<float> h_output_cpu(n);

    // Initialize input data (e.g., with random floats between -1 and 1)
    for (int i = 0; i < n; ++i) {
        h_input_a[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f)) - 1.0f;
        h_input_b[i] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f)) - 1.0f;
    }

    // Allocate device memory
    float *d_input_a, *d_input_b, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_a, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_b, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_a, h_input_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_b, h_input_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Launch GEGLU kernel
    int threads_per_block = 256;
    launch_geglu_kernel(d_input_a, d_input_b, d_output, n, threads_per_block);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for errors after kernel launch
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    // Copy results from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute CPU reference
    cpu_geglu(h_input_a, h_input_b, h_output_cpu);

    // Print samples
    std::cout << "--- GEGLU Kernel Verification ---" << std::endl;
    std::cout << "Problem size (n): " << n << std::endl;
    print_vector_sample("Input A (Host)", h_input_a);
    print_vector_sample("Input B (Host)", h_input_b);
    print_vector_sample("Output (GPU)", h_output_gpu);
    print_vector_sample("Output (CPU Ref)", h_output_cpu);

    // Verification
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_output_gpu[i] - h_output_cpu[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << h_output_gpu[i] 
                      << ", CPU=" << h_output_cpu[i] 
                      << ", Diff=" << fabsf(h_output_gpu[i] - h_output_cpu[i]) << std::endl;
            success = false;
            break; 
        }
    }

    if (success) {
        std::cout << "\nVerification Successful: GPU and CPU results match within tolerance." << std::endl;
    } else {
        std::cout << "\nVerification Failed: GPU and CPU results differ." << std::endl;
    }

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_input_a));
    CHECK_CUDA_ERROR(cudaFree(d_input_b));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    std::cout << "GEGLU main finished." << std::endl;
    return 0;
}
