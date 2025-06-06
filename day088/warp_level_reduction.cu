#include "warp_level_reduction.cuh"

// Device function for warp-level sum reduction using __shfl_down_sync
// Each thread in the warp calls this function with its partial sum
// The final sum for the warp is returned to lane 0 of the warp
__device__ int warpReduceSum(int val) {
    // Iteratively add values from higher lanes to lower lanes
    // Threads with lane_id >= offset will send their 'val' to lane_id - offset
    // Threads with lane_id < offset will receive and add to their 'val'
    // The active mask ensures only participating threads are involved.
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset); // 0xFFFFFFFF is the full warp mask
    }
    return val; // Lane 0 has the final sum for the warp
}

__global__ void warpSumReductionKernel(const int *input_data, int *output_data, int num_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements) {
        int thread_val = input_data[tid];

        // Perform warp-level sum reduction
        int warp_sum = warpReduceSum(thread_val);

        // Lane 0 of each warp writes the result
        if ((threadIdx.x % warpSize) == 0) {
            int warp_id = tid / warpSize;
            output_data[warp_id] = warp_sum;
        }
    }
}
