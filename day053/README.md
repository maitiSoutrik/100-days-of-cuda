# Day 53: Bidirectional LSTM

## Overview

This project implements a basic Bidirectional Long Short-Term Memory (LSTM) layer using CUDA. An LSTM is a type of Recurrent Neural Network (RNN) designed to handle sequential data and address the vanishing gradient problem, making it effective for tasks like language modeling, speech recognition, and time series analysis. A bidirectional LSTM processes the sequence in both forward and backward directions, allowing the model to capture context from both past and future timesteps.

The implementation provides a CUDA kernel for a single LSTM cell operation and a host function to orchestrate the bidirectional pass over a sequence of inputs.

## Implementation Details

The core logic resides in the `lstm_forward` CUDA kernel. This kernel is launched with a grid size equal to the batch size and a block size equal to the hidden layer size. Each thread processes the calculations for a single neuron within a batch for the current timestep.

The kernel computes the four gates (input, forget, output, candidate) and updates the cell state and hidden state using element-wise operations. The weights (`W` and `U`) and biases (`b`) for all four gates are assumed to be concatenated in contiguous memory on the device.

The `bidirectional_lstm` host function iterates through the sequence length. In each iteration, it launches the `lstm_forward` kernel twice: one for the forward pass (processing input chronologically) and one for the backward pass (processing input in reverse). After all timesteps are processed, the final hidden states from the forward and backward passes are concatenated.

## Key CUDA Features Used

*   `__global__` kernels for device execution (`lstm_forward`).
*   `__device__` functions for code reuse on the device (`sigmoid`, `tanh_activation`).
*   `cudaMalloc`, `cudaMemcpy`, `cudaMemset`, `cudaFree` for explicit device memory management.
*   `cudaDeviceSynchronize` to ensure kernel completion.
*   Basic CUDA error checking using a helper macro and function (`CHECK_CUDA_ERROR`).

## Performance Considerations

This implementation is a straightforward translation of the LSTM equations to CUDA kernels. The `lstm_forward` kernel operates element-wise across the hidden layer and batches. For better performance, especially with larger hidden sizes or batch sizes, it would be beneficial to utilize optimized libraries like cuBLAS for the matrix multiplication parts of the gate calculations (`x_t * W` and `h_prev_t * U`). The current kernel structure doesn't fully leverage the parallelism available in matrix operations. Future optimization could involve restructuring the kernel or using library calls.

The bidirectional processing is done sequentially over time steps on the host side, launching kernels for each step. While the kernel for a single timestep is parallel, the overall sequence processing is not fully parallelized across time.

Memory access patterns within the kernel for accessing `input`, `h_prev`, `c_prev`, `W`, `U`, and `b` should be considered for coalescing, although the current structure processing one neuron per thread per batch might not achieve optimal coalescing without further restructuring.

## Building and Running

To build and run the project on the target Jetson Nano environment (or a compatible system with CUDA and CMake installed):

1.  **Navigate to the root project directory:** `cd /path/to/100-days-of-cuda`
2.  **Create and navigate to the build directory:** `mkdir -p build && cd build`
3.  **Configure the project using CMake:** `cmake ..`
4.  **Build the project:** `make` (or `cmake --build .`)
5.  **Run the main executable:** `./day053/bidirectional_lstm`
6.  **Run the unit tests:** `./day053/bidirectional_lstm_test`

The CI/CD pipeline automates these build and execution steps.

## Execution Results

### Main Executable Output (`./build/day053/bidirectional_lstm`):
```
Bidirectional LSTM Output (first 10 elements):
0.000869222 0.0530395 0.0462272 0.0534633 0.0313935 0.0467185 0.0495004 0.0499889 0.0112397 0.0284454
```

### Test Executable Output (`./build/day053/bidirectional_lstm_test`):
```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from BidirectionalLSTMTest
[ RUN      ] BidirectionalLSTMTest.CudaMemoryAllocation
[       OK ] BidirectionalLSTMTest.CudaMemoryAllocation (96 ms)
[ RUN      ] BidirectionalLSTMTest.HostFunctionExecution
[       OK ] BidirectionalLSTMTest.HostFunctionExecution (27 ms)
[----------] 2 tests from BidirectionalLSTMTest (124 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (124 ms total)
[  PASSED  ] 2 tests.
```

## Learnings and Observations

Implementing the LSTM forward pass kernel highlights the challenge of mapping sequential operations to a parallel architecture. While the element-wise calculations are straightforward, the recurrent nature requires processing time steps iteratively. This basic implementation serves as a foundation, but optimizing performance for neural network layers like LSTM often necessitates leveraging highly optimized libraries that are designed for common deep learning operations. The bidirectional approach requires careful handling of input indexing for the backward pass and concatenation of results.

## References

*   [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
*   [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
