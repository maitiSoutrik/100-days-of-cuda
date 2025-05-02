#include "bidirectional_lstm.h"
#include <ctime>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm> // For std::min

int main() {
    srand(time(NULL));
    // Allocate host memory
    std::vector<float> h_input(SEQ_LEN * BATCH_SIZE * INPUT_SIZE, 1.0f);
    std::vector<float> h_output_host(2 * BATCH_SIZE * HIDDEN_SIZE, 0.0f); // Host buffer to receive final concatenated output
    std::vector<float> h_W(4 * HIDDEN_SIZE * INPUT_SIZE);
    std::vector<float> h_U(4 * HIDDEN_SIZE * HIDDEN_SIZE);
    std::vector<float> h_b(4 * HIDDEN_SIZE);

    for (auto& w : h_W) w = ((float) rand() / RAND_MAX) * 0.1f;
    for (auto& u : h_U) u = ((float) rand() / RAND_MAX) * 0.1f;
    for (auto& b : h_b) b = 0.0f;

    // Allocate device memory
    float *d_input, *d_h_forward, *d_c_forward, *d_h_backward, *d_c_backward, *d_W, *d_U, *d_b, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, SEQ_LEN * BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_h_forward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c_forward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_h_backward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c_backward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_W, 4 * HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_U, 4 * HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, 4 * HIDDEN_SIZE * sizeof(float)));
    // d_output will hold the concatenated final hidden states (device pointers)
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, 2 * BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));


    // Initialize memory on device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), SEQ_LEN * BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_W, h_W.data(), 4 * HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_U, h_U.data(), 4 * HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b.data(), 4 * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_h_forward, 0, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_c_forward, 0, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_h_backward, 0, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_c_backward, 0, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));

    // Run bidirectional LSTM
    // Pass d_output to store the concatenated results on the device
    bidirectional_lstm(d_input, d_h_forward, d_c_forward, d_h_backward, d_c_backward, d_W, d_U, d_b, d_output);

    // After running the kernels for all timesteps in bidirectional_lstm,
    // copy the final forward and backward hidden states to the d_output buffer on the device.
    // This is done manually here based on the previous code structure.
    CHECK_CUDA_ERROR(cudaMemcpy(d_output, d_h_forward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_output + BATCH_SIZE * HIDDEN_SIZE, d_h_backward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure device copies are complete


    // Copy final concatenated result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_host.data(), d_output, 2 * BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Print output
    std::cout << "Bidirectional LSTM Output (first " << std::min((int)h_output_host.size(), 10) << " elements):\n";
    for (int i = 0; i < std::min((int)h_output_host.size(), 10); i++) {
        std::cout << h_output_host[i] << " ";
    }
    std::cout << "\n";

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_h_forward));
    CHECK_CUDA_ERROR(cudaFree(d_c_forward));
    CHECK_CUDA_ERROR(cudaFree(d_h_backward));
    CHECK_CUDA_ERROR(cudaFree(d_c_backward));
    CHECK_CUDA_ERROR(cudaFree(d_W));
    CHECK_CUDA_ERROR(cudaFree(d_U));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_output)); // Free the d_output buffer
    // No need to free h_input, h_output_host, h_W, h_U, h_b as they are std::vector and handle their memory

    return 0;
}
