#include "warp_level_reduction.cuh"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

void printArray(const std::string& name, const int* arr, int size) {
    std::cout << name << ": [ ";
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << (i == size - 1 ? " " : ", ");
    }
    std::cout << "]" << std::endl;
}

int main() {
    const int hostWarpSize = 32; // Define warpSize for host code
    const int num_elements = 256; // Example: 8 warps if blockDim.x is 256
    const int block_size = 256;   // Threads per block

    if (num_elements % hostWarpSize != 0) {
        std::cerr << "Error: num_elements must be a multiple of warpSize (" << hostWarpSize << ") for this example." << std::endl;
        return 1;
    }
    if (block_size % hostWarpSize != 0) {
        std::cerr << "Error: block_size must be a multiple of warpSize (" << hostWarpSize << ") for this example." << std::endl;
        return 1;
    }

    const int num_warps = num_elements / hostWarpSize;

    std::vector<int> h_input(num_elements);
    std::iota(h_input.begin(), h_input.end(), 1); // Fill with 1, 2, ..., num_elements

    std::vector<int> h_output_gpu(num_warps);
    std::vector<int> h_output_cpu(num_warps);

    int *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, num_elements * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, num_warps * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice));

    // Kernel launch: 1 block, block_size threads
    // Ensure block_size is a multiple of warpSize
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks( (num_elements + threadsPerBlock.x -1) / threadsPerBlock.x ); 
    // For this example, if num_elements == block_size, numBlocks will be 1.

    warpSumReductionKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, num_elements);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to complete

    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, num_warps * sizeof(int), cudaMemcpyDeviceToHost));

    // CPU verification
    for (int i = 0; i < num_warps; ++i) {
        h_output_cpu[i] = 0;
        for (int j = 0; j < hostWarpSize; ++j) {
            h_output_cpu[i] += h_input[i * hostWarpSize + j];
        }
    }

    printArray("Input Data (first 32)", h_input.data(), std::min(num_elements, 32));
    printArray("GPU Output (Warp Sums)", h_output_gpu.data(), num_warps);
    printArray("CPU Expected (Warp Sums)", h_output_cpu.data(), num_warps);

    // Verification
    bool success = true;
    for (int i = 0; i < num_warps; ++i) {
        if (h_output_gpu[i] != h_output_cpu[i]) {
            std::cerr << "Mismatch at warp " << i << ": GPU = " << h_output_gpu[i]
                      << ", CPU = " << h_output_cpu[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
    }

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    return 0;
}
