#include "bidirectional_lstm.h"
#include <ctime> // Include for time

// Sigmoid activation function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Tanh activation function
__device__ float tanh_activation(float x) {
    return tanhf(x);
}

// LSTM kernel for a single timestep
__global__ void lstm_forward(float* input, float* h_prev, float* c_prev, float* W, float* U, float* b, float* h_out, float* c_out) {
    int batch_idx = blockIdx.x;
    int neuron_idx = threadIdx.x;

    if (neuron_idx >= HIDDEN_SIZE) return; // Ensure valid index range

    float x_t = input[batch_idx * INPUT_SIZE + neuron_idx];
    float h_prev_t = h_prev[batch_idx * HIDDEN_SIZE + neuron_idx];
    float c_prev_t = c_prev[batch_idx * HIDDEN_SIZE + neuron_idx];

    float i_t = sigmoid(x_t * W[neuron_idx] + h_prev_t * U[neuron_idx] + b[neuron_idx]);
    float f_t = sigmoid(x_t * W[neuron_idx + HIDDEN_SIZE] + h_prev_t * U[neuron_idx + HIDDEN_SIZE] + b[neuron_idx + HIDDEN_SIZE]);
    float o_t = sigmoid(x_t * W[neuron_idx + 2 * HIDDEN_SIZE] + h_prev_t * U[neuron_idx + 2 * HIDDEN_SIZE] + b[neuron_idx + 2 * HIDDEN_SIZE]);
    float g_t = tanh_activation(x_t * W[neuron_idx + 3 * HIDDEN_SIZE] + h_prev_t * U[neuron_idx + 3 * HIDDEN_SIZE] + b[neuron_idx + 3 * HIDDEN_SIZE]);

    float c_t = f_t * c_prev_t + i_t * g_t;
    float h_t = o_t * tanh_activation(c_t);

    h_out[batch_idx * HIDDEN_SIZE + neuron_idx] = h_t;
    c_out[batch_idx * HIDDEN_SIZE + neuron_idx] = c_t;
}

// Function to launch bidirectional LSTM
void bidirectional_lstm(float* input, float* h_forward, float* c_forward, float* h_backward, float* c_backward, float* W, float* U, float* b, float* output) {
    for (int t = 0; t < SEQ_LEN; t++) {
        lstm_forward<<<BATCH_SIZE, HIDDEN_SIZE>>>(input + t * BATCH_SIZE * INPUT_SIZE, h_forward, c_forward, W, U, b, h_forward, c_forward);
        lstm_forward<<<BATCH_SIZE, HIDDEN_SIZE>>>(input + (SEQ_LEN - 1 - t) * BATCH_SIZE * INPUT_SIZE, h_backward, c_backward, W, U, b, h_backward, c_backward);
    }
    cudaDeviceSynchronize();

    // Concatenate forward and backward outputs - This part copies back to host, should be after the loop over timesteps
    // cudaMemcpy(output, h_forward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(output + BATCH_SIZE * HIDDEN_SIZE, h_backward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // The output should store concatenated device pointers h_forward and h_backward, not host memory
}


// Simple error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(EXIT_FAILURE);
    }
}
