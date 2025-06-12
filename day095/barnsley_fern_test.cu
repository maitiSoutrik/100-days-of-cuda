#include "gtest/gtest.h"
#include "barnsley_fern.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h> // For curandState, though not directly used in this basic test
#include <vector>
#include <iostream>

// Basic smoke test to ensure the kernel can be launched and completes without error
// for a minimal configuration.
TEST(BarnsleyFernTest, KernelExecutionSmokeTest) {
    const int image_width = 64;
    const int image_height = 64;
    const int num_iterations_per_thread = 10;
    const int warmup_iterations = 5;
    const int threads_per_block = 32;
    const int num_blocks = 1;
    const int total_threads = num_blocks * threads_per_block;

    const float x_min_fern = -2.1820f;
    const float x_max_fern = 2.6558f;
    const float y_min_fern = 0.0f;
    const float y_max_fern = 9.9983f;

    unsigned int* d_image_buffer = nullptr;
    curandState* d_rand_states = nullptr;

    ASSERT_EQ(cudaMalloc(&d_image_buffer, image_width * image_height * sizeof(unsigned int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_image_buffer, 0, image_width * image_height * sizeof(unsigned int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_rand_states, total_threads * sizeof(curandState)), cudaSuccess);

    unsigned long long seed = 12345ULL;
    setup_kernel<<<num_blocks, threads_per_block>>>(d_rand_states, seed, total_threads);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "CUDA error after setup_kernel launch.";
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA error after setup_kernel synchronize.";

    generate_fern_kernel<<<num_blocks, threads_per_block>>>(
        d_image_buffer, 
        image_width, image_height, 
        num_iterations_per_thread, 
        d_rand_states,
        x_min_fern, x_max_fern, y_min_fern, y_max_fern,
        warmup_iterations);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "CUDA error after generate_fern_kernel launch.";
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA error after generate_fern_kernel synchronize.";

    // Basic check: copy a small part of the buffer and see if it's not all zero (optional)
    // For a true smoke test, just ensuring no CUDA errors is often sufficient.
    // std::vector<unsigned int> h_image_buffer(image_width * image_height);
    // EXPECT_EQ(cudaMemcpy(h_image_buffer.data(), d_image_buffer, image_width * image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost), cudaSuccess);

    ASSERT_EQ(cudaFree(d_image_buffer), cudaSuccess);
    ASSERT_EQ(cudaFree(d_rand_states), cudaSuccess);

    SUCCEED(); // If all asserts and expects pass
}

// It would be good to add more specific tests if parts of the fern logic
// (e.g., individual affine transformations) were refactored into __device__ functions.
// For example:
// __device__ void apply_f1(float& x, float& y) { ... }
// Then a test kernel could call these and check outputs.

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
