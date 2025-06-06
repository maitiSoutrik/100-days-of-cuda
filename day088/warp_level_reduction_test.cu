#include "gtest/gtest.h"
#include "warp_level_reduction.cuh"
#include <vector>
#include <numeric>
#include <algorithm>

// Helper to initialize data and run the kernel for testing
void run_warp_reduction_test(const std::vector<int>& h_input, std::vector<int>& h_output_gpu) {
    const int num_elements = h_input.size();
    if (num_elements == 0 || num_elements % warpSize != 0) {
        ADD_FAILURE() << "Test input size must be non-zero and a multiple of warpSize.";
        return;
    }
    const int num_warps = num_elements / warpSize;
    h_output_gpu.resize(num_warps);

    int *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, num_elements * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, num_warps * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice));

    // Determine block and grid dimensions
    // For simplicity in testing, let's try to fit into one block if possible,
    // or ensure blockDim.x is a multiple of warpSize.
    int threads_per_block = std::min(num_elements, 256); // Max 256 threads for this test, can be adjusted
    if (threads_per_block % warpSize != 0) { // Ensure threads_per_block is multiple of warpSize
        threads_per_block = ((threads_per_block + warpSize - 1) / warpSize) * warpSize;
        threads_per_block = std::min(threads_per_block, 1024); // Cap at max threads per block
    }
    if (num_elements < threads_per_block && num_elements > 0) {
         threads_per_block = num_elements; // if num_elements is small, e.g. 32 or 64
    }
    
    dim3 threadsPerBlock(threads_per_block);
    dim3 numBlocks((num_elements + threadsPerBlock.x - 1) / threadsPerBlock.x);

    warpSumReductionKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, num_elements);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, num_warps * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
}

TEST(WarpReductionTest, SingleWarpFull) {
    const int num_elements = 32; // Exactly one warp
    std::vector<int> h_input(num_elements);
    std::iota(h_input.begin(), h_input.end(), 1); // 1, 2, ..., 32

    std::vector<int> h_output_gpu;
    run_warp_reduction_test(h_input, h_output_gpu);

    ASSERT_EQ(h_output_gpu.size(), 1);
    int expected_sum = 0;
    for (int x : h_input) expected_sum += x; // Sum of 1 to 32 is 32*33/2 = 528
    EXPECT_EQ(h_output_gpu[0], expected_sum);
}

TEST(WarpReductionTest, MultipleWarps) {
    const int num_elements = 64; // Two warps
    std::vector<int> h_input(num_elements);
    std::iota(h_input.begin(), h_input.end(), 1); // 1, 2, ..., 64

    std::vector<int> h_output_gpu;
    run_warp_reduction_test(h_input, h_output_gpu);

    ASSERT_EQ(h_output_gpu.size(), 2);

    int expected_sum_warp0 = 0;
    for (int i = 0; i < 32; ++i) expected_sum_warp0 += h_input[i]; // 1 to 32 -> 528
    EXPECT_EQ(h_output_gpu[0], expected_sum_warp0);

    int expected_sum_warp1 = 0;
    for (int i = 32; i < 64; ++i) expected_sum_warp1 += h_input[i]; // 33 to 64
    EXPECT_EQ(h_output_gpu[1], expected_sum_warp1);
}

TEST(WarpReductionTest, MultipleWarpsPartialBlock) {
    const int num_elements = 128; // Four warps
    std::vector<int> h_input(num_elements);
    // Fill with alternating 1s and 2s for variety
    for(int i=0; i < num_elements; ++i) h_input[i] = (i % 2) + 1;

    std::vector<int> h_output_gpu;
    run_warp_reduction_test(h_input, h_output_gpu);

    ASSERT_EQ(h_output_gpu.size(), 4);

    for (int warp_idx = 0; warp_idx < 4; ++warp_idx) {
        int expected_sum_warp = 0;
        for (int i = 0; i < 32; ++i) {
            expected_sum_warp += h_input[warp_idx * 32 + i];
        }
        EXPECT_EQ(h_output_gpu[warp_idx], expected_sum_warp)
            << "Mismatch in warp " << warp_idx;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
