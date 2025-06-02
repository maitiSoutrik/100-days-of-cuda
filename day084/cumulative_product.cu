#include "cumulative_product.cuh"
#include <vector>
#include <cmath> // For fabs in comparison
#include <iostream> // For std::cout, std::endl

// CPU implementation for reference and testing
void inclusive_scan_cpu(float* h_data, int n) {
    if (n == 0) return;
    for (int i = 1; i < n; ++i) {
        h_data[i] = h_data[i] * h_data[i-1];
    }
}

// Kernel for single-block inclusive scan (prefix product) using Blelloch-style scan
// n_actual: actual number of elements in d_data to be scanned.
// Kernel assumes n_actual <= 2 * blockDim.x.
__global__ void inclusive_scan_kernel_blelloch(float* d_data, int n_actual) {
    extern __shared__ float s_temp[]; // Shared memory for intermediate products

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    // scan_elements_count is the effective number of elements processed by the scan in shared memory.
    // This is typically a power of two and matches the shared memory allocation.
    int scan_elements_count = 2 * block_size; 

    // Load data into shared memory. Each thread loads up to two elements.
    // Pad with identity (1.0f for product) if index is beyond n_actual.
    int s_idx1 = tid;
    int s_idx2 = tid + block_size;

    // Load first element for this thread
    if (s_idx1 < scan_elements_count) { // Check against shared memory capacity
        s_temp[s_idx1] = (s_idx1 < n_actual) ? d_data[s_idx1] : 1.0f;
    }
    // Load second element for this thread
    if (s_idx2 < scan_elements_count) { // Check against shared memory capacity
        s_temp[s_idx2] = (s_idx2 < n_actual) ? d_data[s_idx2] : 1.0f;
    }
    __syncthreads();

    // Up-sweep (Reduction phase) - Operates on scan_elements_count
    for (int stride = 1; stride < scan_elements_count; stride *= 2) {
        int current_idx = (tid + 1) * (2 * stride) - 1; // Element to update
        int prev_idx    = current_idx - stride;         // Element to read from
        
        if (current_idx < scan_elements_count && prev_idx >= 0) { // Operate within shared memory bounds
            s_temp[current_idx] = s_temp[prev_idx] * s_temp[current_idx];
        }
        __syncthreads(); 
    }

    // Down-sweep phase - Operates on scan_elements_count
    if (scan_elements_count > 0) {
        if (tid == 0) { 
            s_temp[scan_elements_count - 1] = 1.0f; // Set last element of scan range to identity
        }
    }
    __syncthreads();

    for (int stride = scan_elements_count / 2; stride > 0; stride /= 2) {
        int current_offset = (tid * 2 * stride) + stride - 1; // Index of left element in pair
        int next_offset    = current_offset + stride;         // Index of right element in pair

        if (next_offset < scan_elements_count) { // Operate within shared memory bounds
            float val_left = s_temp[current_offset];
            float val_right = s_temp[next_offset];
            s_temp[current_offset] = val_right;
            s_temp[next_offset] = val_left * val_right;
        }
        __syncthreads(); 
    }
    
    // s_temp now holds the exclusive scan for scan_elements_count elements.
    // Convert to inclusive scan and write to global memory for the first n_actual elements.
    // d_data still holds the original input values.
    if (s_idx1 < n_actual) { // Write first element handled by this thread
        d_data[s_idx1] = s_temp[s_idx1] * d_data[s_idx1];
    }
    if (s_idx2 < n_actual) { // Write second element handled by this thread
        d_data[s_idx2] = s_temp[s_idx2] * d_data[s_idx2];
    }
}


// GPU implementation wrapper
void inclusive_scan_gpu(float* d_data, int n_actual) {
    if (n_actual == 0) return;

    int threads_per_block = 256; // A common choice, can be tuned
    int scan_elements_per_block = 2 * threads_per_block;

    if (n_actual > scan_elements_per_block) {
        std::cerr << "Error: Input size n_actual=" << n_actual 
                  << " is too large for this single-block scan. Max supported: " << scan_elements_per_block << "." << std::endl;
        std::cerr << "A multi-block scan algorithm would be required for larger inputs." << std::endl;
        // For robust error handling, one might throw an exception or return an error code.
        // For this project, we'll proceed if tests use smaller N, but this is a hard limit.
        // If this function is called with n_actual > scan_elements_per_block, behavior is undefined for the kernel.
        // It's better to return or throw to prevent incorrect execution.
        throw std::runtime_error("Input size exceeds single-block scan capacity."); 
        // return; // Or simply return, depending on error strategy.
    }

    // Shared memory size: enough for 2 elements per thread for the full block.
    size_t shared_mem_size = (size_t)scan_elements_per_block * sizeof(float);

    // Launch kernel
    inclusive_scan_kernel_blelloch<<<1, threads_per_block, shared_mem_size>>>(d_data, n_actual);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete
}
