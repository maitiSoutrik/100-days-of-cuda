# Day 37: Simple NN Forward Pass (GEMM + Activation)

## Overview

This project implements the forward pass of a single fully connected neural network layer using CUDA. The calculation involves a matrix multiplication (Weights * Input) performed using cuBLAS `sgemm`, followed by the addition of a bias vector and element-wise application of the ReLU activation function using a custom CUDA kernel. This demonstrates combining CUDA library calls with custom kernels for a common deep learning operation.

## Implementation Details

1.  **Input:** A batch of input vectors (M x K matrix, where M is batch size, K is input features).
2.  **Weights:** A weight matrix (K x N matrix, where N is output features/neurons).
3.  **Bias:** A bias vector (size N).
4.  **GEMM:** `cublasSgemm` is used to compute `Input * Weights`. The matrices are stored in row-major order, but the `cublasSgemm` call is structured according to the standard convention for handling row-major multiplication using the column-major API (computing `C_colmajor = B_colmajor * A_colmajor` which effectively gives `C = A * B` in row-major).
5.  **Custom Kernel (`add_bias_activate_2d`):**
    *   Launched with a 2D grid and 2D blocks to map naturally to the output matrix (M x N).
    *   Each thread calculates the index corresponding to an element in the output batch.
    *   It reads the corresponding GEMM result, adds the bias term (broadcasted across the batch dimension), and applies the ReLU activation function (`fmaxf(0.0f, value)`).
6.  **Verification:** A simple CPU implementation computes the same forward pass for comparison. The Mean Squared Error (MSE) and Maximum Absolute Difference between the GPU and CPU results are calculated to verify correctness.

## Key CUDA Features Used

*   **cuBLAS:** `cublasSgemm` for high-performance matrix multiplication.
*   **Custom Kernels:** `add_bias_activate_2d` for element-wise operations (bias addition, activation).
*   **2D Grid/Block Dim:** Used for mapping threads to the output matrix elements.
*   **Error Handling:** `CHECK_CUDA_ERROR` and `CHECK_CUBLAS_ERROR` macros.
*   **Device Memory Management:** `cudaMalloc`, `cudaMemcpy`, `cudaFree`.

## Performance Considerations

*   **cuBLAS GEMM:** Generally highly optimized for matrix multiplication on NVIDIA GPUs. Performance depends heavily on matrix dimensions and GPU architecture.
*   **Custom Kernel:** The element-wise nature of the bias addition and activation is well-suited for parallel execution. Memory access to the output matrix and bias vector is coalesced due to the 2D grid mapping matching the row-major layout.
*   **Data Transfer:** `cudaMemcpy` operations between host and device introduce overhead. For real applications, minimizing data transfers is crucial.

## Building and Running

**Note:** Compilation and execution should occur in an environment with the CUDA Toolkit and CMake installed, targeting NVIDIA compute capability 5.3 (Jetson Nano) or compatible.

1.  **Configure using CMake:**
    ```bash
    cd <path_to_100-days-of-cuda>/build
    cmake ..
    ```
2.  **Build:**
    ```bash
    cd <path_to_100-days-of-cuda>/build
    cmake --build . --target nn_forward_pass -- -j$(nproc)
    ```
3.  **Run:**
    ```bash
    cd <path_to_100-days-of-cuda>/build/day037
    ./nn_forward_pass
    ```

## Execution Results

Output from running `./nn_forward_pass` on the Jetson Nano:

```
Configuration:
 Batch Size (M): 64
 Input Features (K): 1024
 Output Features (N): 512
-----------------------------

GPU computation complete.
 GPU Execution Time: 4.488 ms
Performing CPU computation for verification...
CPU computation complete.
 CPU Execution Time: 129.627 ms

Verification Results:
 Max Absolute Difference: 5.340576e-05
 Mean Squared Error (MSE): 2.094646e-11
 Verification PASSED (Max Diff < 1.000000e-04)

Resources freed. Exiting.
```

**Analysis:** The GPU execution time (approx. 4.5 ms) is significantly faster than the CPU execution time (approx. 129.6 ms), demonstrating a speedup of roughly 29x for this forward pass with the given dimensions on the Jetson Nano. This highlights the effectiveness of using cuBLAS and custom kernels for accelerating neural network computations. The verification passed with the adjusted tolerance of 1e-4, accounting for minor floating-point differences.

## Learnings and Observations

*   Successfully combined a cuBLAS library call with a custom kernel to implement a common NN layer pattern.
*   Reinforced understanding of matrix multiplication conventions with cuBLAS (`sgemm`) when dealing with row-major data.
*   Using a 2D kernel launch configuration simplifies indexing for element-wise operations on matrices.
*   Verification against a CPU implementation is crucial for ensuring correctness, especially when dealing with floating-point operations and library nuances.

## (Optional) References

*   cuBLAS Documentation: [https://docs.nvidia.com/cuda/cublas/index.html](https://docs.nvidia.com/cuda/cublas/index.html)
*   CUDA C++ Programming Guide: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
