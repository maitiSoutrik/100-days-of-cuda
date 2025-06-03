// day086/hard_sigmoid.cuh
#ifndef HARD_SIGMOID_CUH
#define HARD_SIGMOID_CUH

#include <cuda_runtime.h> // For size_t

// Function declaration
// This function takes host pointers for input and output,
// and handles all CUDA operations internally (alloc, copy, kernel, copy back, free).
extern "C" void hard_sigmoid_solution(const float* input, float* output, size_t n, size_t m);

#endif // HARD_SIGMOID_CUH
