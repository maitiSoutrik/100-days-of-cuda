#include "rms_norm.cuh"
#include <cmath>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

// ============================================================================
// CUDA Kernels
// ============================================================================

// Warp-level reduction for sum of squares
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level reduction for sum of squares
__device__ __forceinline__ float block_reduce_sum(float val) {
    // Reduce within the warp. After this, lane 0 of each warp has the sum for that warp.
    val = warp_reduce_sum(val);

    if (blockDim.x <= WARP_SIZE) {
        // If the entire block is one warp or smaller,
        // the sum computed by warp_reduce_sum is the final block sum.
        // This sum is in 'val' of thread 0 (lane 0 of the first/only warp).
        // The calling kernel uses the 'val' from thread 0.
        return val;
    }

    // For blocks with multiple warps:
    static __shared__ float shared_warp_sums[WARP_SIZE]; // Max WARP_SIZE warps in a block (e.g. 256 threads = 8 warps, 1024 threads = 32 warps)
    
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Each warp leader (lane_id == 0) writes its partial sum to shared memory.
    if (lane_id == 0) {
        shared_warp_sums[warp_id] = val;
    }
    __syncthreads(); // Ensure all warp sums are written before reading.

    // The first warp (warp_id == 0) sums the results from shared memory.
    if (warp_id == 0) {
        // Each thread in the first warp picks up a sum from shared memory if it corresponds to an actual warp.
        int num_active_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        if (lane_id < num_active_warps) {
            val = shared_warp_sums[lane_id];
        } else {
            val = 0.0f; // Threads beyond num_active_warps contribute 0 to this sum.
        }
        val = warp_reduce_sum(val); // Reduce these sums within the first warp. Result in thread 0.
    }
    // After this, 'val' in thread 0 (lane 0 of warp 0) contains the total sum for the block.
    // Other threads' 'val' are not guaranteed to hold the final sum here.
    // The calling kernel (rms_norm_kernel_optimized) correctly uses thread 0's 'val'.
    return val;
}

// RMS Normalization kernel
__global__ void rms_norm_kernel(const float* input, float* output, const float* gamma,
                               int batch_size, int seq_len, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len;
    
    if (idx >= total_elements) return;
    
    // Calculate offset for this sequence element
    int offset = idx * hidden_dim;
    
    // Step 1: Compute sum of squares for this sequence element
    float sum_squares = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        if (i < hidden_dim) {
            float val = input[offset + i];
            sum_squares += val * val;
        }
    }
    
    // Reduce sum of squares across the block
    sum_squares = block_reduce_sum(sum_squares);
    
    // Broadcast the result to all threads in the block
    __shared__ float shared_rms;
    if (threadIdx.x == 0) {
        float mean_square = sum_squares / hidden_dim;
        shared_rms = rsqrtf(mean_square + EPSILON);
    }
    __syncthreads();
    
    // Step 2: Apply normalization
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        if (i < hidden_dim) {
            output[offset + i] = input[offset + i] * shared_rms * gamma[i];
        }
    }
}

// Optimized RMS Normalization kernel (one block per sequence element)
__global__ void rms_norm_kernel_optimized(const float* input, float* output, const float* gamma,
                                         int batch_size, int seq_len, int hidden_dim) {
    int seq_idx = blockIdx.x;
    int total_sequences = batch_size * seq_len;
    
    if (seq_idx >= total_sequences) return;
    
    int offset = seq_idx * hidden_dim;
    
    // Step 1: Compute sum of squares
    float sum_squares = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float val = input[offset + i];
        sum_squares += val * val;
    }
    
    // Reduce across block
    sum_squares = block_reduce_sum(sum_squares);
    
    // Compute RMS normalization factor
    __shared__ float rms_norm;
    if (threadIdx.x == 0) {
        float mean_square = sum_squares / hidden_dim;
        rms_norm = rsqrtf(mean_square + EPSILON);
    }
    __syncthreads();
    
    // Step 2: Apply normalization
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        output[offset + i] = input[offset + i] * rms_norm * gamma[i];
    }
}

// Layer Normalization kernel for comparison
__global__ void layer_norm_kernel(const float* input, float* output, 
                                 const float* gamma, const float* beta,
                                 int batch_size, int seq_len, int hidden_dim) {
    int seq_idx = blockIdx.x;
    int total_sequences = batch_size * seq_len;
    
    if (seq_idx >= total_sequences) return;
    
    int offset = seq_idx * hidden_dim;
    
    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        sum += input[offset + i];
    }
    sum = block_reduce_sum(sum);
    
    __shared__ float mean;
    if (threadIdx.x == 0) {
        mean = sum / hidden_dim;
    }
    __syncthreads();
    
    // Step 2: Compute variance
    float sum_squares = 0.0f;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float diff = input[offset + i] - mean;
        sum_squares += diff * diff;
    }
    sum_squares = block_reduce_sum(sum_squares);
    
    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        float variance = sum_squares / hidden_dim;
        inv_std = rsqrtf(variance + EPSILON);
    }
    __syncthreads();
    
    // Step 3: Apply normalization
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float normalized = (input[offset + i] - mean) * inv_std;
        output[offset + i] = normalized * gamma[i] + beta[i];
    }
}

// ============================================================================
// CPU Reference Implementations
// ============================================================================

void rms_norm_cpu(const float* input, float* output, const float* gamma,
                  int batch_size, int seq_len, int hidden_dim) {
    int total_sequences = batch_size * seq_len;
    
    for (int seq = 0; seq < total_sequences; seq++) {
        int offset = seq * hidden_dim;
        
        // Step 1: Compute sum of squares
        float sum_squares = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            float val = input[offset + i];
            sum_squares += val * val;
        }
        
        // Step 2: Compute RMS normalization factor
        float mean_square = sum_squares / hidden_dim;
        float rms_norm = 1.0f / sqrtf(mean_square + EPSILON);
        
        // Step 3: Apply normalization
        for (int i = 0; i < hidden_dim; i++) {
            output[offset + i] = input[offset + i] * rms_norm * gamma[i];
        }
    }
}

void layer_norm_cpu(const float* input, float* output, const float* gamma, const float* beta,
                    int batch_size, int seq_len, int hidden_dim) {
    int total_sequences = batch_size * seq_len;
    
    for (int seq = 0; seq < total_sequences; seq++) {
        int offset = seq * hidden_dim;
        
        // Step 1: Compute mean
        float sum = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            sum += input[offset + i];
        }
        float mean = sum / hidden_dim;
        
        // Step 2: Compute variance
        float sum_squares = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            float diff = input[offset + i] - mean;
            sum_squares += diff * diff;
        }
        float variance = sum_squares / hidden_dim;
        float inv_std = 1.0f / sqrtf(variance + EPSILON);
        
        // Step 3: Apply normalization
        for (int i = 0; i < hidden_dim; i++) {
            float normalized = (input[offset + i] - mean) * inv_std;
            output[offset + i] = normalized * gamma[i] + beta[i];
        }
    }
}

// ============================================================================
// GPU Wrapper Functions
// ============================================================================

void rms_norm_gpu(const float* input, float* output, const float* gamma,
                  int batch_size, int seq_len, int hidden_dim) {
    int total_sequences = batch_size * seq_len;
    
    // Use optimized kernel with one block per sequence
    dim3 grid(total_sequences);
    dim3 block(min(BLOCK_SIZE, hidden_dim));
    
    rms_norm_kernel_optimized<<<grid, block>>>(input, output, gamma, 
                                              batch_size, seq_len, hidden_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void layer_norm_gpu(const float* input, float* output, const float* gamma, const float* beta,
                    int batch_size, int seq_len, int hidden_dim) {
    int total_sequences = batch_size * seq_len;
    
    dim3 grid(total_sequences);
    dim3 block(min(BLOCK_SIZE, hidden_dim));
    
    layer_norm_kernel<<<grid, block>>>(input, output, gamma, beta,
                                      batch_size, seq_len, hidden_dim);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// ============================================================================
// Utility Functions
// ============================================================================

void initialize_data(float* data, int size, float mean, float std) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    
    for (int i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
}

void print_tensor(const float* data, int batch_size, int seq_len, int hidden_dim,
                  const char* name, int max_elements) {
    std::cout << "\n" << name << " (showing first " << max_elements << " elements):\n";
    std::cout << std::fixed << std::setprecision(6);
    
    int count = 0;
    for (int b = 0; b < batch_size && count < max_elements; b++) {
        for (int s = 0; s < seq_len && count < max_elements; s++) {
            std::cout << "  [" << b << "," << s << "]: ";
            for (int h = 0; h < min(hidden_dim, max_elements - count); h++) {
                int idx = b * seq_len * hidden_dim + s * hidden_dim + h;
                std::cout << data[idx] << " ";
                count++;
            }
            std::cout << "\n";
        }
    }
}

bool verify_results(const float* cpu_result, const float* gpu_result, int size, float tolerance) {
    float max_diff = 0.0f;
    int error_count = 0;
    
    for (int i = 0; i < size; i++) {
        float diff = fabsf(cpu_result[i] - gpu_result[i]);
        max_diff = fmaxf(max_diff, diff);
        
        if (diff > tolerance) {
            error_count++;
            if (error_count <= 5) {  // Print first 5 errors
                std::cout << "  Error at index " << i << ": CPU=" << cpu_result[i] 
                         << ", GPU=" << gpu_result[i] << ", diff=" << diff << "\n";
            }
        }
    }
    
    std::cout << "Verification: max_diff=" << max_diff << ", errors=" << error_count 
              << "/" << size << " (" << (100.0f * error_count / size) << "%)\n";
    
    return error_count == 0;
}

// ============================================================================
// Benchmarking
// ============================================================================

BenchmarkResult benchmark_normalization(int batch_size, int seq_len, int hidden_dim, int num_iterations) {
    BenchmarkResult result = {0.0f, 0.0f, 0.0f, false};
    
    int total_elements = batch_size * seq_len * hidden_dim;
    
    // Allocate host memory
    float* h_input = new float[total_elements];
    float* h_gamma = new float[hidden_dim];
    float* h_beta = new float[hidden_dim];
    float* h_output_cpu_rms = new float[total_elements];
    float* h_output_cpu_layer = new float[total_elements];
    float* h_output_gpu_rms = new float[total_elements];
    float* h_output_gpu_layer = new float[total_elements];
    
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
    
    // Warm up GPU
    rms_norm_gpu(d_input, d_output_rms, d_gamma, batch_size, seq_len, hidden_dim);
    layer_norm_gpu(d_input, d_output_layer, d_gamma, d_beta, batch_size, seq_len, hidden_dim);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Benchmark CPU RMS Norm
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        rms_norm_cpu(h_input, h_output_cpu_rms, h_gamma, batch_size, seq_len, hidden_dim);
    }
    auto end = std::chrono::high_resolution_clock::now();
    result.cpu_time_ms = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    // Benchmark GPU RMS Norm
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        rms_norm_gpu(d_input, d_output_rms, d_gamma, batch_size, seq_len, hidden_dim);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    result.gpu_time_ms = std::chrono::duration<float, std::milli>(end - start).count() / num_iterations;
    
    result.speedup = result.cpu_time_ms / result.gpu_time_ms;
    
    // Copy results back for verification
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu_rms, d_output_rms, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify correctness
    result.correctness_passed = verify_results(h_output_cpu_rms, h_output_gpu_rms, total_elements);
    
    // Cleanup
    delete[] h_input;
    delete[] h_gamma;
    delete[] h_beta;
    delete[] h_output_cpu_rms;
    delete[] h_output_cpu_layer;
    delete[] h_output_gpu_rms;
    delete[] h_output_gpu_layer;
    
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_gamma));
    CHECK_CUDA_ERROR(cudaFree(d_beta));
    CHECK_CUDA_ERROR(cudaFree(d_output_rms));
    CHECK_CUDA_ERROR(cudaFree(d_output_layer));
    
    return result;
}
