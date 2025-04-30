# Day 52: Minimal GRU (minGRU) with Parallel Scan

## Overview

This project implements the Minimal Gated Recurrent Unit (minGRU), a simplified version of the standard GRU. As described in the paper "Were RNNs All We Needed?", minGRU modifies gate computations to depend only on the current input `x_t`, removing the dependency on the previous hidden state `h_{t-1}`. This structural change allows the recurrence `h_t = a_t * h_{t-1} + b_t` to be computed efficiently using a parallel scan algorithm across the time dimension.

The goal for Day 52 is to implement minGRU in CUDA, utilizing a parallel scan for sequence processing, and compare its performance against a standard sequential CPU implementation.

## Implementation Details

The CUDA implementation consists of several key components:

1.  **Kernels:**
    *   `min_gru_extract_scan_params_kernel`: Computes the scan coefficients `a_t = (1 - z_t)` and `b_t = z_t * h_tilde` for all time steps in parallel based on the input sequence `x`.
    *   `compose_scan_ops_kernel`: Combines two scan operations `(a', b') ○ (a, b) = (a'*a, a'*b + b')`. Used within the parallel scan algorithm.
    *   `apply_scan_op_kernel`: Applies a scan operation `(a, b)` to a hidden state `h_in` to compute `h_out = a * h_in + b`.

2.  **Parallel Scan (`min_gru_parallel_scan_cuda`):**
    *   This function implements the core parallel scan logic. It allocates device memory, copies data, and uses the `compose_scan_ops_kernel` and `apply_scan_op_kernel` (potentially in a tree-based structure or similar efficient pattern adapted from the reference code) to compute all hidden states `h_t` in parallel.

3.  **Orchestration (`min_gru_process_sequence_cuda`):**
    *   This host function manages the overall CUDA execution flow: transferring the MinGRUCell weights to the GPU, allocating/copying input/output buffers, launching the `extract_scan_params` kernel, calling the `parallel_scan` function, copying results back, and freeing device memory.

4.  **CPU Baseline (`min_gru_process_sequence_cpu`):**
    *   A standard sequential implementation of minGRU is included for verification purposes. It processes the sequence step-by-step on the CPU.

5.  **Verification:**
    *   The `main` function compares the output of the CUDA parallel implementation (`h_out_cuda`) against the CPU sequential implementation (`h_out_cpu`) to ensure correctness, checking for small differences due to floating-point precision.

## Key CUDA Concepts Used

*   CUDA Kernel Programming (`__global__`)
*   Device Memory Management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)
*   Parallel Scan Algorithm Implementation on GPU
*   CUDA Error Handling (`CHECK_CUDA_ERROR` macro)
*   CUDA Events for accurate GPU timing (optional but recommended)
*   Grid-Stride Loops (potentially within kernels if handling large data)

## Performance Considerations

The parallel scan approach significantly accelerates the computation of RNN hidden states compared to the inherently sequential nature of standard GRUs/LSTMs, especially for long sequences where the parallel computation outweighs kernel launch and memory transfer overheads. The main performance factors include:
*   Efficiency of the parallel scan implementation (e.g., work efficiency of the tree reduction).
*   Latency of memory transfers between host and device.
*   GPU architecture and available parallelism.

## Building and Running

The code is built using CMake, targeting the Jetson Nano environment (Compute Capability 5.3).

1.  **Navigate to the build directory** (typically `build/` in the project root) on the Jetson Nano or CI environment.
2.  **Run CMake and Make:**
    ```bash
    cd /path/to/100-days-of-cuda/build
    cmake ..
    make day052_min_gru_scan # Or simply 'make'
    ```
3.  **Execute the compiled program:**
    ```bash
    ./bin/day052/min_gru_scan
    ```
    (Adjust the path based on your build/install configuration).

## Execution Results

**(Note: Update this section with the actual console output after running on the Jetson Nano or compatible environment.)**

The program outputs the parameters used, the execution time for both the CPU sequential version and the CUDA parallel version, and the maximum absolute difference between their results to verify correctness.

```
--- Expected Output Format ---
Parameters: input_size=X, hidden_size=Y, seq_length=Z
Processing with MinGRU (CPU Sequential)...
CPU Processing Time: A.AAAAAA seconds
Processing with MinGRU (CUDA Parallel Scan)...
CUDA Processing Time: B.BBBBBB seconds
Maximum difference between CPU and CUDA: C.CCCCCCCC
--- End Expected Output Format ---

--- ACTUAL JETSON NANO OUTPUT ---
```
drboom@JetNano ~/g/1/build> ./day052/min_gru_scan

--- MinGRU Parallel Scan CUDA Example ---
Parameters: input_size=128, hidden_size=256, seq_length=100
Host MinGRU cell initialized.
Host memory allocated.
Random host data generated.
Processing with MinGRU (CPU Sequential)...
CPU Processing Time: 0.019021 seconds
Processing with MinGRU (CUDA Parallel Scan)...
CUDA Processing Time: 0.081226 seconds
Verifying results...
Maximum absolute difference between CPU and CUDA results: 0.00000018
Results verified successfully within tolerance.
Cleaning up...
Cleanup complete.
```
--- END ACTUAL JETSON NANO OUTPUT ---

## Learnings and Observations

*   Implementing the parallel scan algorithm in CUDA requires careful management of device memory and kernel launch configurations.
*   Understanding the associative property of the scan operation `(a, b) ○ (c, d)` is key to the parallelization strategy.
*   Significant speedups are expected for the CUDA version compared to the sequential CPU version, especially with longer sequences.
*   Verification against a known correct implementation (the CPU version) is crucial for debugging CUDA code.

## References

*   "Were RNNs All We Needed?" (Paper introducing minGRU/minLSTM) - [Link to paper if available]
*   CUDA Parallel Scan Documentation/Tutorials (e.g., NVIDIA documentation, CUDPP, Thrust scan)
