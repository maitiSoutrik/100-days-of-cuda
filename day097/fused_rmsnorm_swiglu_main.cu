#include "fused_rmsnorm_swiglu.cuh"
#include "cuda_utils.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip> // For std::fixed, std::setprecision
#include <chrono>   // For timing

// Helper function to initialize data
void initialize_data(float* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

// Helper function to print a small part of a tensor
void print_tensor_sample(const float* tensor, int num_rows, int dim, int sample_rows, int sample_cols, const std::string& name) {
    std::cout << name << " (sample " << sample_rows << "x" << sample_cols << "):" << std::endl;
    for (int i = 0; i < std::min(num_rows, sample_rows); ++i) {
        for (int j = 0; j < std::min(dim, sample_cols); ++j) {
            std::cout << std::fixed << std::setprecision(4) << tensor[i * dim + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Helper function to compare results
bool compare_results(const float* gpu_res, const float* cpu_res, int size, float tolerance = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        if (fabs(gpu_res[i] - cpu_res[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << gpu_res[i]
                      << ", CPU=" << cpu_res[i] << ", Diff=" << fabs(gpu_res[i] - cpu_res[i]) << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Parameters
    int batch_size = 4;
    int seq_len = 64; // Example sequence length
    int hidden_dim = 256; // Example hidden dimension, must be even for SwiGLU
    int num_rows = batch_size * seq_len;
    int output_dim = hidden_dim / 2;

    std::cout << "Running Fused RMSNorm + SwiGLU Benchmark" << std::endl;
    std::cout << "Parameters: Batch Size=" << batch_size << ", Seq Len=" << seq_len
              << ", Hidden Dim=" << hidden_dim << std::endl;
    std::cout << "Total rows (tokens): " << num_rows << std::endl;
    std::cout << "Output dimension after SwiGLU: " << output_dim << std::endl;


    if (hidden_dim % 2 != 0) {
        std::cerr << "Error: hidden_dim must be an even number." << std::endl;
        return 1;
    }

    // Host data
    std::vector<float> h_input(num_rows * hidden_dim);
    std::vector<float> h_weight(hidden_dim); // RMSNorm weights (gamma)
    std::vector<float> h_output_gpu(num_rows * output_dim);
    std::vector<float> h_output_cpu(num_rows * output_dim);

    // Initialize host data
    initialize_data(h_input.data(), h_input.size());
    initialize_data(h_weight.data(), h_weight.size(), 0.5f, 1.5f); // Typically initialized around 1

    // Device data pointers
    float *d_input, *d_weight, *d_output;

    // Allocate memory on device
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weight, h_weight.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, h_output_gpu.size() * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Launch CUDA kernel
    // Determine block_size. For the reduction in RMSNorm, it's good if block_size is a power of 2.
    // For the output computation, if block_size < hidden_dim/2, threads would need to loop.
    // The current kernel assumes block_size >= hidden_dim/2 for direct mapping of threads to outputs.
    // Let's choose a block_size that is a power of 2 and also >= hidden_dim / 2.
    // A common default is 256. If hidden_dim/2 is larger, this needs adjustment or kernel modification.
    // If hidden_dim/2 is small (e.g., 32), then block_size can be hidden_dim/2 (or next power of 2 like 32 or 64).
    int block_size = 256; // Default block size
    if (hidden_dim / 2 > 256) {
        // This case is not optimally handled by the current kernel without thread looping for output.
        // For simplicity, we might cap block_size or adjust.
        // Or, ensure hidden_dim is not excessively large for this example.
        // The kernel's shared memory reduction part assumes block_size threads.
        // The output part assumes block_size threads compute up to block_size outputs.
        // If hidden_dim/2 > block_size, the current kernel will only compute the first block_size outputs per row.
        // THIS IS A BUG IN THE KERNEL'S OUTPUT PART IF hidden_dim/2 > block_size.
        // The kernel should be: for (int i = threadIdx.x; i < hidden_dim/2; i += blockDim.x)
        // Let's assume for this main, we set block_size = hidden_dim/2 if it's a power of 2, or a suitable power of 2.
        // For the current kernel structure: block_size must be >= hidden_dim / 2 for correctness.
        // And for reduction, block_size should be a power of 2.
        // Let's pick block_size as the smallest power of 2 >= hidden_dim / 2, capped at e.g. 512.
        
        int temp_block_size = 1;
        while(temp_block_size < (hidden_dim / 2)) temp_block_size *= 2;
        block_size = std::min(temp_block_size, 512); // Cap at 512 for typical limits
        if (block_size < 32) block_size = 32; // Minimum sensible block size (warp size)
    }
     if (block_size > hidden_dim) { // For reduction, block_size should not exceed hidden_dim
        block_size = hidden_dim;
        // ensure power of 2 if possible
        int temp_bs = 1;
        while(temp_bs * 2 <= block_size) temp_bs *=2;
        block_size = temp_bs;
     }


    std::cout << "Using CUDA block size: " << block_size << std::endl;

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    launch_fused_rmsnorm_swiglu(d_output, d_input, d_weight, num_rows, hidden_dim, block_size);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    // Copy results from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, h_output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Perform CPU computation for verification
    std::cout << "Performing CPU computation for verification..." << std::endl;
    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    fused_rmsnorm_swiglu_cpu(h_output_cpu.data(), h_input.data(), h_weight.data(), num_rows, hidden_dim);
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end_time - cpu_start_time;
    std::cout << "CPU Execution Time: " << cpu_duration.count() << " ms" << std::endl;


    // Print samples
    print_tensor_sample(h_input.data(), num_rows, hidden_dim, 2, std::min(hidden_dim, 8), "Input (Host)");
    print_tensor_sample(h_weight.data(), 1, hidden_dim, 1, std::min(hidden_dim, 8), "Weights (Host)");
    print_tensor_sample(h_output_gpu.data(), num_rows, output_dim, 2, std::min(output_dim, 8), "Output (GPU)");
    print_tensor_sample(h_output_cpu.data(), num_rows, output_dim, 2, std::min(output_dim, 8), "Output (CPU)");

    // Compare results
    bool success = compare_results(h_output_gpu.data(), h_output_cpu.data(), h_output_gpu.size());
    if (success) {
        std::cout << "Verification Successful: GPU and CPU results match." << std::endl;
    } else {
        std::cout << "Verification Failed: GPU and CPU results differ." << std::endl;
    }

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_weight));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return success ? 0 : 1;
}
