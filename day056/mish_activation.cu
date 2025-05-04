// day056/mish_activation.cu
#include "mish_activation.cuh" // Include the header

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip> // For std::setprecision

// Implementation of CUDA error checking function
void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") "
                  << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// CPU implementation of Mish activation (definition)
void mish_cpu(const std::vector<float>& input, std::vector<float>& output) {
    if (input.size() != output.size()) {
         std::cerr << "Error: Input and output vector sizes must match for mish_cpu." << std::endl;
         return; // Or throw an exception
    }
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = mish(input[i]); // Uses the inline function from the header
    }
}

// GPU kernel for Mish activation (definition)
// (Declaration is in the header)
__global__ void mish_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = mish(input[idx]); // Uses the inline function from the header
    }
}

// GPU wrapper function (definition)
void mish_gpu_wrapper(const float* d_input, float* d_output, int n, cudaEvent_t start, cudaEvent_t stop) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    if (start) CHECK_CUDA_ERROR(cudaEventRecord(start));

    mish_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

    if (stop) CHECK_CUDA_ERROR(cudaEventRecord(stop));
}


// Function to verify results (moved here from main, could also be in header/separate util file)
bool verify_results(const std::vector<float>& cpu_output, const std::vector<float>& gpu_output, float tolerance = 1e-5) {
    if (cpu_output.size() != gpu_output.size()) {
        std::cerr << "Verification failed: Output sizes differ! CPU=" << cpu_output.size() << ", GPU=" << gpu_output.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < cpu_output.size(); ++i) {
        if (fabsf(cpu_output[i] - gpu_output[i]) > tolerance) {
            std::cerr << "Verification failed at index " << i << ": CPU=" << cpu_output[i]
                      << ", GPU=" << gpu_output[i] << ", Diff=" << fabsf(cpu_output[i] - gpu_output[i]) << std::endl;
            return false;
        }
    }
    return true;
}

// Main function remains largely the same, but calls the wrapper
int main() {
    // --- Configuration ---
    int n = 1 << 24; // Number of elements (e.g., 16 million)
    size_t bytes = n * sizeof(float);
    std::cout << "Processing " << n << " elements (" << bytes / (1024.0 * 1024.0) << " MB)" << std::endl;

    // --- Host Memory Allocation and Initialization ---
    std::vector<float> h_input(n);
    std::vector<float> h_output_cpu(n);
    std::vector<float> h_output_gpu(n);

    // Initialize input data with random values (e.g., between -5 and 5)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
    for (int i = 0; i < n; ++i) {
        h_input[i] = dis(gen);
    }
    std::cout << "Host input data initialized." << std::endl;

    // --- Device Memory Allocation ---
    float *d_input = nullptr, *d_output = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, bytes));
    std::cout << "Device memory allocated." << std::endl;

    // --- Copy Input Data to Device ---
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    std::cout << "Input data copied to device." << std::endl;

    // --- CPU Benchmark ---
    std::cout << "\n--- CPU Execution ---" << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    mish_cpu(h_input, h_output_cpu);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = stop_cpu - start_cpu;
    std::cout << "CPU execution time: " << std::fixed << std::setprecision(3) << cpu_duration.count() << " ms" << std::endl;

    // --- GPU Benchmark ---
    std::cout << "\n--- GPU Execution ---" << std::endl;
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu));

    // Warm-up run (optional but good practice)
    mish_gpu_wrapper(d_input, d_output, n); // Use wrapper, no events needed for warm-up timing
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Synchronize after warm-up

    // Actual benchmark run using the wrapper with events
    mish_gpu_wrapper(d_input, d_output, n, start_gpu, stop_gpu);
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu)); // Wait for GPU to finish

    float gpu_duration_ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_duration_ms, start_gpu, stop_gpu));
    std::cout << "GPU execution time: " << std::fixed << std::setprecision(3) << gpu_duration_ms << " ms" << std::endl;

    // --- Copy Results Back to Host ---
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    std::cout << "GPU results copied back to host." << std::endl;

    // --- Verification ---
    std::cout << "\n--- Verification ---" << std::endl;
    bool success = verify_results(h_output_cpu, h_output_gpu);
    if (success) {
        std::cout << "Verification successful: CPU and GPU results match." << std::endl;
    } else {
        std::cout << "Verification failed: CPU and GPU results differ." << std::endl;
    }

    // --- Performance Comparison ---
    std::cout << "\n--- Performance ---" << std::endl;
    if (gpu_duration_ms > 0) { // Avoid division by zero
        double speedup = cpu_duration.count() / gpu_duration_ms;
        std::cout << "GPU Speedup over CPU: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    } else {
        std::cout << "GPU execution time was zero or negative, cannot calculate speedup." << std::endl;
    }

    // --- Cleanup ---
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    std::cout << "\nCleanup complete." << std::endl;

    return success ? 0 : 1; // Return 0 on success, 1 on failure
}
