#include "gtest/gtest.h"
#include "bidirectional_lstm.h" // Include the CUDA implementation header
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm> // For std::min

// Simple test case for CUDA memory allocation and deallocation
TEST(BidirectionalLSTMTest, CudaMemoryAllocation) {
    float *d_input, *d_h_forward, *d_c_forward, *d_h_backward, *d_c_backward, *d_W, *d_U, *d_b, *d_output;

    // Allocate device memory
    ASSERT_EQ(cudaMalloc(&d_input, SEQ_LEN * BATCH_SIZE * INPUT_SIZE * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_h_forward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c_forward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_h_backward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_c_backward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_W, 4 * HIDDEN_SIZE * INPUT_SIZE * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_U, 4 * HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, 4 * HIDDEN_SIZE * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, 2 * BATCH_SIZE * HIDDEN_SIZE * sizeof(float)), cudaSuccess);

    // Free device memory
    ASSERT_EQ(cudaFree(d_input), cudaSuccess);
    ASSERT_EQ(cudaFree(d_h_forward), cudaSuccess);
    ASSERT_EQ(cudaFree(d_c_forward), cudaSuccess);
    ASSERT_EQ(cudaFree(d_h_backward), cudaSuccess);
    ASSERT_EQ(cudaFree(d_c_backward), cudaSuccess);
    ASSERT_EQ(cudaFree(d_W), cudaSuccess);
    ASSERT_EQ(cudaFree(d_U), cudaSuccess);
    ASSERT_EQ(cudaFree(d_b), cudaSuccess);
    ASSERT_EQ(cudaFree(d_output), cudaSuccess);
}

// Add more tests here for data transfer, kernel launch, etc.
// Note: Testing the actual LSTM logic with random weights is complex.
// Ideally, use fixed small inputs/weights for verifiable kernel tests.

// Basic test for the bidirectional_lstm host function execution flow
TEST(BidirectionalLSTMTest, HostFunctionExecution) {
    // This test primarily checks if the function runs without crashing and if CUDA calls within it succeed.
    // It doesn't verify computational correctness due to random initialization.

    std::vector<float> h_input(SEQ_LEN * BATCH_SIZE * INPUT_SIZE, 1.0f);
    std::vector<float> h_W(4 * HIDDEN_SIZE * INPUT_SIZE);
    std::vector<float> h_U(4 * HIDDEN_SIZE * HIDDEN_SIZE);
    std::vector<float> h_b(4 * HIDDEN_SIZE);

    srand(time(NULL));
    for (auto& w : h_W) w = ((float) rand() / RAND_MAX) * 0.1f;
    for (auto& u : h_U) u = ((float) rand() / RAND_MAX) * 0.1f;
    for (auto& b : h_b) b = 0.0f;

    float *d_input, *d_h_forward, *d_c_forward, *d_h_backward, *d_c_backward, *d_W, *d_U, *d_b, *d_output;

    cudaMalloc(&d_input, SEQ_LEN * BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_h_forward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_c_forward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_h_backward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_c_backward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_W, 4 * HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_U, 4 * HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_b, 4 * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output, 2 * BATCH_SIZE * HIDDEN_SIZE * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), SEQ_LEN * BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W.data(), 4 * HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U.data(), 4 * HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), 4 * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_h_forward, 0, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_c_forward, 0, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_h_backward, 0, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_c_backward, 0, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));

    // Call the function that launches kernels
    bidirectional_lstm(d_input, d_h_forward, d_c_forward, d_h_backward, d_c_backward, d_W, d_U, d_b, d_output);

    // The main function's copy to host is removed, testing device side concatenation
    ASSERT_EQ(cudaMemcpy(d_output, d_h_forward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_output + BATCH_SIZE * HIDDEN_SIZE, d_h_backward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);


    // Free device memory
    cudaFree(d_input);
    cudaFree(d_h_forward);
    cudaFree(d_c_forward);
    cudaFree(d_h_backward);
    cudaFree(d_c_backward);
    cudaFree(d_W);
    cudaFree(d_U);
    cudaFree(d_b);
    cudaFree(d_output);
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
