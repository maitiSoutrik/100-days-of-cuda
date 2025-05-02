#ifndef BIDIRECTIONAL_LSTM_H
#define BIDIRECTIONAL_LSTM_H

#include <cuda_runtime.h>
#include <iostream> // Include for std::cerr and std::endl
#include <vector>
#include <cstdlib> // Include for exit

#define HIDDEN_SIZE 128
#define INPUT_SIZE 128
#define SEQ_LEN 50
#define BATCH_SIZE 32

// Simple error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char* const func, const char* const file, const int line);

// Function to launch bidirectional LSTM
void bidirectional_lstm(float* input, float* h_forward, float* c_forward, float* h_backward, float* c_backward, float* W, float* U, float* b, float* output);

#endif // BIDIRECTIONAL_LSTM_H
