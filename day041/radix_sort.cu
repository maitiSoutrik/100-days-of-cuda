#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// Error checking macro
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " code=" << err << " \"" << cudaGetErrorString(err) << "\" for " << func << std::endl;
        exit(EXIT_FAILURE);
    }
}

// --- Kernels ---

// Kernel to calculate histogram for a specific bit pass
__global__ void histogram_kernel(const unsigned int* input, unsigned int* histogram, int n, int bit_shift, int num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ unsigned int local_hist[]; // Size = num_buckets

    // Initialize shared memory histogram
    if (threadIdx.x < num_buckets) {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    // Calculate histogram for elements handled by this block
    if (idx < n) {
        unsigned int value = input[idx];
        unsigned int bucket = (value >> bit_shift) & (num_buckets - 1); // Extract bits for this pass
        atomicAdd(&local_hist[bucket], 1);
    }
    __syncthreads();

    // Write block histogram to global memory
    if (threadIdx.x < num_buckets) {
        atomicAdd(&histogram[threadIdx.x], local_hist[threadIdx.x]);
    }
}

// Kernel to scatter elements based on scanned histogram offsets
__global__ void scatter_kernel(const unsigned int* input, unsigned int* output, const unsigned int* offsets, int n, int bit_shift, int num_buckets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ unsigned int local_hist_offsets[]; // Size = num_buckets
    extern __shared__ unsigned int local_scatter_pos[]; // Size = num_buckets

    // Load histogram offsets into shared memory
    if (threadIdx.x < num_buckets) {
        local_hist_offsets[threadIdx.x] = offsets[threadIdx.x];
        local_scatter_pos[threadIdx.x] = offsets[threadIdx.x]; // Initialize scatter positions
    }
    __syncthreads();

    if (idx < n) {
        unsigned int value = input[idx];
        unsigned int bucket = (value >> bit_shift) & (num_buckets - 1);

        // Atomically determine the position for this element within its bucket
        unsigned int pos = atomicAdd(&local_scatter_pos[bucket], 1);
        output[pos] = value;
    }
}


// --- Host Functions ---

// CPU Radix Sort (for comparison - simplified, single pass logic)
void cpu_radix_sort_pass(const std::vector<unsigned int>& input, std::vector<unsigned int>& output, int bit_shift, int num_buckets) {
    int n = input.size();
    std::vector<unsigned int> histogram(num_buckets, 0);
    std::vector<unsigned int> offsets(num_buckets, 0);

    // 1. Histogram
    for (int i = 0; i < n; ++i) {
        unsigned int bucket = (input[i] >> bit_shift) & (num_buckets - 1);
        histogram[bucket]++;
    }

    // 2. Scan (Prefix Sum)
    offsets[0] = 0;
    for (int i = 1; i < num_buckets; ++i) {
        offsets[i] = offsets[i - 1] + histogram[i - 1];
    }

    // 3. Scatter
    std::vector<unsigned int> current_offsets = offsets; // Copy offsets to track current position
    for (int i = 0; i < n; ++i) {
        unsigned int bucket = (input[i] >> bit_shift) & (num_buckets - 1);
        output[current_offsets[bucket]] = input[i];
        current_offsets[bucket]++;
    }
}

// Function to verify sort results
bool verify_sort(const std::vector<unsigned int>& sorted_data) {
    for (size_t i = 1; i < sorted_data.size(); ++i) {
        if (sorted_data[i] < sorted_data[i - 1]) {
            std::cerr << "Verification failed at index " << i << ": " << sorted_data[i] << " < " << sorted_data[i-1] << std::endl;
            return false;
        }
    }
    return true;
}


int main() {
    // --- Parameters ---
    const int N_POWER = 20; // Increased array size
    const int N = 1 << N_POWER;
    const int BITS_PER_PASS = 4; // Process 4 bits (1 hex digit) per pass
    const int NUM_BUCKETS = 1 << BITS_PER_PASS; // 16 buckets
    const int TOTAL_BITS = 32; // For unsigned int
    const int NUM_PASSES = TOTAL_BITS / BITS_PER_PASS;

    std::cout << "Radix Sort (Single Pass Example)" << std::endl;
    std::cout << "Array size: " << N << " (2^" << N_POWER << ")" << std::endl;
    std::cout << "Bits per pass: " << BITS_PER_PASS << std::endl;
    std::cout << "Number of buckets: " << NUM_BUCKETS << std::endl;

    // --- Data Initialization ---
    std::vector<unsigned int> h_input(N);
    std::vector<unsigned int> h_output_gpu(N);
    std::vector<unsigned int> h_output_cpu(N);
    std::vector<unsigned int> h_temp(N); // For multi-pass CPU sort

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> distrib(0, std::numeric_limits<unsigned int>::max());

    std::cout << "Generating random data..." << std::endl;
    for (int i = 0; i < N; ++i) {
        h_input[i] = distrib(gen);
    }
    h_temp = h_input; // Copy for CPU sort

    // --- GPU Radix Sort (Single Pass Example) ---
    std::cout << "\n--- GPU Radix Sort (Pass 0) ---" << std::endl;

    unsigned int *d_input, *d_output, *d_histogram, *d_offsets;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, N * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, N * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_histogram, NUM_BUCKETS * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_offsets, NUM_BUCKETS * sizeof(unsigned int))); // For scanned histogram

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Timing
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu));

    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu));

    // 1. Histogram Calculation
    const int BLOCK_SIZE_HIST = 256;
    const int GRID_SIZE_HIST = (N + BLOCK_SIZE_HIST - 1) / BLOCK_SIZE_HIST;
    CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, NUM_BUCKETS * sizeof(unsigned int)));
    histogram_kernel<<<GRID_SIZE_HIST, BLOCK_SIZE_HIST, NUM_BUCKETS * sizeof(unsigned int)>>>(d_input, d_histogram, N, 0, NUM_BUCKETS); // Pass 0 (bits 0-3)
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

    // 2. Scan (Prefix Sum) using Thrust
    thrust::device_ptr<unsigned int> d_hist_ptr(d_histogram);
    thrust::device_ptr<unsigned int> d_offsets_ptr(d_offsets);
    thrust::exclusive_scan(thrust::device, d_hist_ptr, d_hist_ptr + NUM_BUCKETS, d_offsets_ptr);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure scan is complete

    // 3. Scatter Elements
    const int BLOCK_SIZE_SCATTER = 256;
    const int GRID_SIZE_SCATTER = (N + BLOCK_SIZE_SCATTER - 1) / BLOCK_SIZE_SCATTER;
    // Shared memory: offsets + current scatter positions
    size_t shmem_scatter = 2 * NUM_BUCKETS * sizeof(unsigned int);
    scatter_kernel<<<GRID_SIZE_SCATTER, BLOCK_SIZE_SCATTER, shmem_scatter>>>(d_input, d_output, d_offsets, N, 0, NUM_BUCKETS); // Pass 0
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu));

    float ms_gpu = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms_gpu, start_gpu, stop_gpu));
    std::cout << "GPU Pass 0 Time: " << ms_gpu << " ms" << std::endl;

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // --- CPU Radix Sort (Single Pass Example) ---
    std::cout << "\n--- CPU Radix Sort (Pass 0) ---" << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_radix_sort_pass(h_input, h_output_cpu, 0, NUM_BUCKETS); // Pass 0
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(stop_cpu - start_cpu);
    std::cout << "CPU Pass 0 Time: " << duration_cpu.count() << " ms" << std::endl;

    // --- Verification (Optional - only meaningful after full sort) ---
    // Note: Verifying after a single pass doesn't make sense unless input was specifically crafted.
    // A full sort implementation would be needed for proper verification.
    std::cout << "\n--- Full CPU Sort (for reference) ---" << std::endl;
    auto start_full_cpu = std::chrono::high_resolution_clock::now();
    std::sort(h_temp.begin(), h_temp.end()); // Use std::sort for reliable baseline
    auto stop_full_cpu = std::chrono::high_resolution_clock::now();
    auto duration_full_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(stop_full_cpu - start_full_cpu);
    std::cout << "std::sort Time: " << duration_full_cpu.count() << " ms" << std::endl;
    bool cpu_sorted_correctly = verify_sort(h_temp);
    std::cout << "std::sort verification: " << (cpu_sorted_correctly ? "Passed" : "Failed") << std::endl;


    // --- Cleanup ---
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_histogram));
    CHECK_CUDA_ERROR(cudaFree(d_offsets));

    std::cout << "\nFinished." << std::endl;

    return 0;
}
