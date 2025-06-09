#include "rms_norm.cuh"
#include <iostream>
#include <iomanip>
#include <vector>

void print_device_info() {
    int device_count;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    
    std::cout << "=== CUDA Device Information ===\n";
    std::cout << "Number of CUDA devices: " << device_count << "\n";
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "\nDevice " << i << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
        std::cout << "  Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB\n";
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Warp Size: " << prop.warpSize << "\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
    }
    std::cout << "\n";
}

void demonstrate_rms_norm_concept() {
    std::cout << "=== RMS Normalization Concept Demonstration ===\n";
    
    // Small example to show the mathematical concept
    const int batch_size = 1;
    const int seq_len = 1;
    const int hidden_dim = 4;
    const int total_elements = batch_size * seq_len * hidden_dim;
    
    // Create simple input data
    float h_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float h_beta[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float h_output_rms[4];
    float h_output_layer[4];
    
    std::cout << "Input vector: [";
    for (int i = 0; i < hidden_dim; i++) {
        std::cout << h_input[i];
        if (i < hidden_dim - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // Compute RMS Norm manually to show the math
    float sum_squares = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        sum_squares += h_input[i] * h_input[i];
    }
    float mean_square = sum_squares / hidden_dim;
    float rms = sqrtf(mean_square);
    float rms_norm_factor = 1.0f / sqrtf(mean_square + EPSILON);
    
    std::cout << "\nRMS Normalization Math:\n";
    std::cout << "  Sum of squares: " << sum_squares << "\n";
    std::cout << "  Mean square: " << mean_square << "\n";
    std::cout << "  RMS: " << rms << "\n";
    std::cout << "  RMS norm factor: " << rms_norm_factor << "\n";
    
    // Apply RMS normalization
    rms_norm_cpu(h_input, h_output_rms, h_gamma, batch_size, seq_len, hidden_dim);
    
    std::cout << "\nRMS Normalized output: [";
    for (int i = 0; i < hidden_dim; i++) {
        std::cout << std::fixed << std::setprecision(6) << h_output_rms[i];
        if (i < hidden_dim - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // Compare with Layer Normalization
    layer_norm_cpu(h_input, h_output_layer, h_gamma, h_beta, batch_size, seq_len, hidden_dim);
    
    std::cout << "\nLayer Normalized output: [";
    for (int i = 0; i < hidden_dim; i++) {
        std::cout << std::fixed << std::setprecision(6) << h_output_layer[i];
        if (i < hidden_dim - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // Show the difference
    std::cout << "\nDifference (RMS - Layer): [";
    for (int i = 0; i < hidden_dim; i++) {
        float diff = h_output_rms[i] - h_output_layer[i];
        std::cout << std::fixed << std::setprecision(6) << diff;
        if (i < hidden_dim - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";
}

void run_comprehensive_benchmark() {
    std::cout << "=== Comprehensive Performance Benchmark ===\n";
    
    // Test different configurations typical in transformer models
    struct TestConfig {
        int batch_size;
        int seq_len;
        int hidden_dim;
        const char* description;
    };
    
    std::vector<TestConfig> configs = {
        {1, 128, 512, "Small: Single sequence, 512 hidden"},
        {8, 128, 512, "Medium: 8 sequences, 512 hidden"},
        {16, 256, 768, "Large: 16 sequences, 768 hidden (BERT-base)"},
        {4, 512, 1024, "XL: 4 sequences, 1024 hidden (BERT-large)"},
        {32, 64, 2048, "Wide: 32 sequences, 2048 hidden"},
        {1, 1024, 4096, "Long: Single long sequence, 4096 hidden"}
    };
    
    std::cout << std::setw(40) << "Configuration" 
              << std::setw(12) << "CPU (ms)" 
              << std::setw(12) << "GPU (ms)" 
              << std::setw(10) << "Speedup" 
              << std::setw(12) << "Correct" << "\n";
    std::cout << std::string(86, '-') << "\n";
    
    for (const auto& config : configs) {
        BenchmarkResult result = benchmark_normalization(
            config.batch_size, config.seq_len, config.hidden_dim, 50);
        
        std::cout << std::setw(40) << config.description
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.cpu_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.gpu_time_ms
                  << std::setw(10) << std::fixed << std::setprecision(2) << result.speedup << "x"
                  << std::setw(12) << (result.correctness_passed ? "PASS" : "FAIL") << "\n";
    }
    std::cout << "\n";
}

void compare_rms_vs_layer_norm() {
    std::cout << "=== RMS Norm vs Layer Norm Comparison ===\n";
    
    const int batch_size = 8;
    const int seq_len = 128;
    const int hidden_dim = 768;
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int num_iterations = 100;
    
    // Allocate memory
    float* h_input = new float[total_elements];
    float* h_gamma = new float[hidden_dim];
    float* h_beta = new float[hidden_dim];
    float* h_output_rms = new float[total_elements];
    float* h_output_layer = new float[total_elements];
    
    // Initialize data
    initialize_data(h_input, total_elements, 0.0f, 1.0f);
    initialize_data(h_gamma, hidden_dim, 1.0f, 0.1f);
    initialize_data(h_beta, hidden_dim, 0.0f, 0.1f);
    
    // Allocate device memory
    float *d_input, *d_gamma, *d_beta, *d_output_rms, *d_output_layer;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, total_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_gamma, hidden_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_beta, hidden_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_rms, total_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_layer, total_elements * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_gamma, h_gamma, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_beta, h_beta, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Warm up
    rms_norm_gpu(d_input, d_output_rms, d_gamma, batch_size, seq_len, hidden_dim);
    layer_norm_gpu(d_input, d_output_layer, d_gamma, d_beta, batch_size, seq_len, hidden_dim);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Benchmark RMS Norm
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        rms_norm_gpu(d_input, d_output_rms, d_gamma, batch_size, seq_len, hidden_dim);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    float rms_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    // Benchmark Layer Norm
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        layer_norm_gpu(d_input, d_output_layer, d_gamma, d_beta, batch_size, seq_len, hidden_dim);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    float layer_time = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    // Copy results back
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_rms, d_output_rms, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_layer, d_output_layer, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "Configuration: " << batch_size << "x" << seq_len << "x" << hidden_dim << "\n";
    std::cout << "RMS Norm GPU time: " << std::fixed << std::setprecision(3) << rms_time << " ms\n";
    std::cout << "Layer Norm GPU time: " << std::fixed << std::setprecision(3) << layer_time << " ms\n";
    std::cout << "RMS Norm speedup: " << std::fixed << std::setprecision(2) << (layer_time / rms_time) << "x faster\n";
    std::cout << "Efficiency gain: " << std::fixed << std::setprecision(1) << (100.0f * (layer_time - rms_time) / layer_time) << "%\n";
    
    // Show sample outputs
    print_tensor(h_input, 1, 1, min(hidden_dim, 8), "Sample Input", 8);
    print_tensor(h_output_rms, 1, 1, min(hidden_dim, 8), "RMS Norm Output", 8);
    print_tensor(h_output_layer, 1, 1, min(hidden_dim, 8), "Layer Norm Output", 8);
    
    // Cleanup
    delete[] h_input;
    delete[] h_gamma;
    delete[] h_beta;
    delete[] h_output_rms;
    delete[] h_output_layer;
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_gamma));
    CHECK_CUDA_ERROR(cudaFree(d_beta));
    CHECK_CUDA_ERROR(cudaFree(d_output_rms));
    CHECK_CUDA_ERROR(cudaFree(d_output_layer));
    
    std::cout << "\n";
}

int main() {
    std::cout << "Day 93: RMS Normalization Implementation\n";
    std::cout << "========================================\n\n";
    
    // Print device information
    print_device_info();
    
    // Demonstrate the concept with a simple example
    demonstrate_rms_norm_concept();
    
    // Run comprehensive benchmarks
    run_comprehensive_benchmark();
    
    // Compare RMS Norm vs Layer Norm directly
    compare_rms_vs_layer_norm();
    
    std::cout << "=== Summary ===\n";
    std::cout << "RMS Normalization successfully implemented with:\n";
    std::cout << "✓ Efficient CUDA kernels with warp-level reductions\n";
    std::cout << "✓ CPU reference implementation for verification\n";
    std::cout << "✓ Comprehensive performance benchmarking\n";
    std::cout << "✓ Direct comparison with Layer Normalization\n";
    std::cout << "✓ Mathematical correctness validation\n\n";
    
    std::cout << "Key Benefits of RMS Normalization:\n";
    std::cout << "• Reduced computational overhead (no mean calculation)\n";
    std::cout << "• Faster training and inference\n";
    std::cout << "• Maintains re-scaling invariance\n";
    std::cout << "• Simpler implementation\n";
    std::cout << "• Comparable performance to Layer Normalization\n\n";
    
    return 0;
}
