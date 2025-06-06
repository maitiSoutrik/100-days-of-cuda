#include "memory_coalescing.cuh"

// Kernel that performs a scaled copy with coalesced memory access
__global__ void coalesced_access_kernel(const float* input, float* output, int n, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = input[idx] * scalar;
    }
}

// Kernel that performs a scaled copy with uncoalesced memory access
__global__ void uncoalesced_access_kernel(const float* input, float* output, int n, float scalar, int stride_factor) {
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Number of elements each stride_factor "column" would have if data were 2D
    int elements_per_stride_group = n / stride_factor;
    
    // Handle cases where n is small or not perfectly divisible, ensure elements_per_stride_group is at least 1 if n > 0
    if (elements_per_stride_group == 0 && n > 0) elements_per_stride_group = 1; 

    if (global_thread_idx < n && elements_per_stride_group > 0) {
        // Conceptual column index for this thread
        int group_idx = global_thread_idx / elements_per_stride_group;
        // Conceptual row index for this thread
        int idx_in_group = global_thread_idx % elements_per_stride_group;
        
        // Check if the calculated group_idx (column) is valid before forming uncoalesced_idx
        if (group_idx < stride_factor) { 
            int uncoalesced_idx = idx_in_group * stride_factor + group_idx;
            // Final boundary check for the uncoalesced index itself
            if (uncoalesced_idx < n) { 
                 output[uncoalesced_idx] = input[uncoalesced_idx] * scalar;
            }
        }
    }
}

