#include "lora.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip> // For std::fixed and std::setprecision

// Helper function to generate random input data
void generate_random_vector(float* vec, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(-1.0, 1.0);
    for (int i = 0; i < size; ++i) {
        vec[i] = static_cast<float>(distrib(gen));
    }
}

// Helper function to print a vector (for debugging)
void print_vector(const float* vec, int size, const std::string& label) {
    std::cout << label << ": [";
    for (int i = 0; i < std::min(size, 10); ++i) { // Print at most 10 elements
        std::cout << vec[i] << (i == std::min(size, 10) - 1 ? "" : ", ");
    }
    if (size > 10) {
        std::cout << "...";
    }
    std::cout << "]" << std::endl;
}

// Helper function to compare two vectors
bool compare_vectors(const float* vec1, const float* vec2, int size, float tolerance = 1e-5f) {
    for (int i = 0; i < size; ++i) {
        if (std::abs(vec1[i] - vec2[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": " << vec1[i] << " vs " << vec2[i] << std::endl;
            return false;
        }
    }
    return true;
}


int main() {
    // --- Configuration ---
    const int d_model = 4096;   // Dimension of the model's features
    const int rank = 64;        // Rank for LoRA decomposition (Increased further)
    const float alpha = 1.0f;   // Scaling factor for LoRA
    const int num_iterations = 1000; // Number of forward passes for benchmarking

    std::cout << "--- LoRA Implementation Benchmark ---" << std::endl;
    std::cout << "Model Dimension (d_model): " << d_model << std::endl;
    std::cout << "LoRA Rank (rank):          " << rank << std::endl;
    std::cout << "LoRA Alpha (alpha):        " << alpha << std::endl;
    std::cout << "Benchmark Iterations:    " << num_iterations << std::endl;
    std::cout << std::endl;

    LoRAParameters params;
    try {
        initializeLoRAParameters(params, d_model, rank, alpha);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error during LoRA parameter initialization: " << e.what() << std::endl;
        return 1;
    }

    // --- Prepare Input Data ---
    std::vector<float> h_input_data(d_model);
    generate_random_vector(h_input_data.data(), d_model);

    std::vector<float> h_lora_output_cpu(d_model);
    std::vector<float> h_lora_output_gpu(d_model);

    // Device input and output
    float* d_input_data;
    float* d_lora_output_gpu;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_data, d_model * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_lora_output_gpu, d_model * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_data, h_input_data.data(), d_model * sizeof(float), cudaMemcpyHostToDevice));

    // --- CPU Benchmark ---
    std::cout << "Running CPU LoRA Forward Pass..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        loraForwardCPU(h_input_data.data(), h_lora_output_cpu.data(), params);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_cpu = end_cpu - start_cpu;
    std::cout << "CPU LoRA Forward Pass (avg per iteration): " 
              << std::fixed << std::setprecision(6) << (duration_cpu.count() / num_iterations) << " ms" << std::endl;
    // print_vector(h_lora_output_cpu.data(), d_model, "CPU Output (first run)");


    // --- GPU Benchmark ---
    std::cout << "\nRunning GPU LoRA Forward Pass..." << std::endl;
    // Warm-up run for GPU
    loraForwardGPU(d_input_data, d_lora_output_gpu, params);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure warm-up is complete

    cudaEvent_t start_gpu_event, stop_gpu_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu_event));

    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu_event));
    for (int i = 0; i < num_iterations; ++i) {
        loraForwardGPU(d_input_data, d_lora_output_gpu, params);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu_event));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu_event));

    float milliseconds_gpu = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds_gpu, start_gpu_event, stop_gpu_event));
    std::cout << "GPU LoRA Forward Pass (avg per iteration): " 
              << std::fixed << std::setprecision(6) << (milliseconds_gpu / num_iterations) << " ms" << std::endl;
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_lora_output_gpu.data(), d_lora_output_gpu, d_model * sizeof(float), cudaMemcpyDeviceToHost));
    // print_vector(h_lora_output_gpu.data(), d_model, "GPU Output (last run)");

    // --- Verification ---
    std::cout << "\nVerifying CPU and GPU results..." << std::endl;
    bool results_match = compare_vectors(h_lora_output_cpu.data(), h_lora_output_gpu.data(), d_model);
    if (results_match) {
        std::cout << "SUCCESS: CPU and GPU results match." << std::endl;
    } else {
        std::cout << "FAILURE: CPU and GPU results DO NOT match." << std::endl;
    }

    // --- Cleanup ---
    std::cout << "\nCleaning up resources..." << std::endl;
    freeLoRAParameters(params);
    CHECK_CUDA_ERROR(cudaFree(d_input_data));
    CHECK_CUDA_ERROR(cudaFree(d_lora_output_gpu));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu_event));

    std::cout << "\nBenchmark complete." << std::endl;

    return 0;
}
