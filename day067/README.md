# Day 67: SwiGLU Activation and Gradient Computation

## Overview

This project implements the SwiGLU (Swish-Gated Linear Unit) activation function and its backward pass for gradient computation using CUDA C++. The SwiGLU function is a variant of Gated Linear Units that uses the Swish (or SiLU - Sigmoid Linear Unit) activation.

The forward pass computes:
`c = silu(a) * b`
where `silu(a) = a * sigmoid(a)`
and `sigmoid(a) = 1 / (1 + exp(-a))`

The backward pass computes the gradients `da` (for input `a`) and `db` (for input `b`), given the incoming gradient `dc` (gradient of the loss with respect to `c`). The necessary activations for the backward pass are recomputed from `a` and `b`.

## Implementation Details

The implementation processes 2D matrices (`a`, `b`, `c`, `dc`, `da`, `db`) in row-major order.

-   **`swiglu.cuh`**: Header file defining kernel signatures, CUDA error checking macros, and launcher function prototypes.
-   **`swiglu.cu`**: CUDA source file containing:
    -   `sigmoidf_device`: A `__device__` helper function for `sigmoid(x)`.
    -   `swiglu_forward_kernel`: CUDA kernel for the forward pass.
        -   Each element `(row, col)` of the output matrix `c` is computed by one thread.
        -   `c[row, col] = (a[row, col] * sigmoid(a[row, col])) * b[row, col]`
    -   `swiglu_backward_kernel`: CUDA kernel for the backward pass.
        -   Each element `(row, col)` of the gradient matrices `da` and `db` is computed by one thread.
        -   `s_a = sigmoid(a[row, col])` (recomputed)
        -   `da[row, col] = dc[row, col] * b[row, col] * s_a * (1 + a[row, col] * (1 - s_a))`
        -   `db[row, col] = dc[row, col] * a[row, col] * s_a`
    -   `launch_swiglu_forward` / `launch_swiglu_backward`: Wrapper functions to launch the respective kernels.
-   **Kernel Launch Configuration**:
    -   The kernels are launched with one CUDA block per row of the matrix and one CUDA thread per column of the matrix.
    -   `dim3 numBlocks(rows);`
    -   `dim3 threadsPerBlock(cols);`
    -   This configuration directly maps to the problem statement. It assumes `cols` is less than or equal to the maximum number of threads per block (e.g., 1024).

-   **`swiglu_main.cu`**: Contains the `main` function to demonstrate and verify the SwiGLU operations.
    -   Initializes sample input matrices `a`, `b`, and `dc` on the host.
    -   Allocates memory on the CUDA device for all matrices.
    -   Copies input data from host to device.
    -   Launches the forward and backward SwiGLU kernels.
    -   Copies results (`c`, `da`, `db`) from device to host.
    -   Prints GPU results.
    -   Performs the same SwiGLU forward and backward computations on the CPU for verification.
    -   Compares GPU and CPU results and reports success or mismatch.
-   **`swiglu_test.cu`**: Contains Google Test unit tests for verifying the correctness of the forward and backward passes against CPU computations for various inputs, including handling of zero-sized inputs.

## Key CUDA Features Used

-   **`__global__` kernels**: `swiglu_forward_kernel` and `swiglu_backward_kernel` for parallel computation on the GPU.
-   **`__device__` function**: `sigmoidf_device` for use within kernels.
-   **Thread Indexing**: `blockIdx.x` (for row) and `threadIdx.x` (for column) to map threads to matrix elements.
-   **Device Memory Management**:
    -   `cudaMalloc()`: To allocate memory on the GPU.
    -   `cudaMemcpy()`: To transfer data between host and device.
    -   `cudaFree()`: To deallocate GPU memory.
-   **Error Handling**: `CHECK_CUDA_ERROR` macro to check for errors from CUDA API calls and kernel launches (`cudaGetLastError()`).
-   **Synchronization**: `cudaDeviceSynchronize()` to ensure kernel completion before accessing results on the host.

## Performance Considerations

-   The chosen kernel launch configuration (one block per row, one thread per column) is a direct interpretation of the problem statement.
-   **Potential Underutilization**: If `cols` is small (e.g., much less than 32, the warp size), many threads in a warp will be inactive, leading to underutilization of GPU resources.
-   **Maximum Threads per Block**: If `cols` exceeds the maximum number of threads allowed per block (typically 1024 for most modern GPUs, specifically 1024 for sm_53 on Jetson Nano), the kernel launch will fail. The current implementation assumes `cols` is within this limit. For larger `cols`, a grid-stride loop within the kernel or a 2D block/grid configuration would be more robust.
-   **Memory Access**: Access patterns are coalesced as threads within a block (processing a row) access contiguous memory locations for `d_a[idx]`, `d_b[idx]`, etc.
-   **Recomputation vs. Storage**: The backward pass recomputes `sigmoid(a)` instead of storing it from the forward pass. This saves device memory at the cost of extra computation. For SwiGLU, this recomputation is generally acceptable as `sigmoid` is not overly complex.

## Building and Running

1.  **Prerequisites**:
    *   CUDA Toolkit (compatible with sm_53 for Jetson Nano)
    *   CMake (version 3.10 or higher)
    *   A C++ compiler (like g++)
    *   Google Test (will be fetched by CMake if not found)

2.  **Configure with CMake**:
    From the root project directory (`100-days-of-cuda`):
    ```bash
    mkdir -p build
    cd build
    cmake ..
    ```

3.  **Build**:
    Still in the `build` directory:
    ```bash
    make day067_swiglu # Or simply 'make' to build everything
    ```
    This will build the `swiglu_benchmark` executable and `swiglu_test` executable.

4.  **Run the Benchmark/Demonstration**:
    ```bash
    ./day067/swiglu_benchmark
    ```

5.  **Run Tests**:
    ```bash
    ctest --output-on-failure # From the build directory
    # or specifically
    # ./day067/swiglu_test
    ```

## Execution Results / Output

**Benchmark Output (`./build/day067/swiglu_benchmark`):**
```
Input A (Host) (4x8):
-0.5000 -0.4000 -0.3000 -0.2000 -0.1000 0.0000  0.1000  0.2000
0.3000  0.4000  -0.5000 -0.4000 -0.3000 -0.2000 -0.1000 0.0000
0.1000  0.2000  0.3000  0.4000  -0.5000 -0.4000 -0.3000 -0.2000
-0.1000 0.0000  0.1000  0.2000  0.3000  0.4000  -0.5000 -0.4000

Input B (Host) (4x8):
0.6000  0.7000  0.8000  0.9000  1.0000  0.1000  0.2000  0.3000
0.4000  0.5000  0.6000  0.7000  0.8000  0.9000  1.0000  0.1000
0.2000  0.3000  0.4000  0.5000  0.6000  0.7000  0.8000  0.9000
1.0000  0.1000  0.2000  0.3000  0.4000  0.5000  0.6000  0.7000

Input dC (Host) (4x8):
1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000
1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000
1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000
1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000

Launching SwiGLU forward kernel...
Launching SwiGLU backward kernel...
Output C (GPU) (4x8):
-0.1133 -0.1124 -0.1021 -0.0810 -0.0475 0.0000  0.0105  0.0330
0.0689  0.1197  -0.1133 -0.1124 -0.1021 -0.0810 -0.0475 0.0000
0.0105  0.0330  0.0689  0.1197  -0.1133 -0.1124 -0.1021 -0.0810
-0.0475 0.0000  0.0105  0.0330  0.0689  0.1197  -0.1133 -0.1124

Output dA (GPU) (4x8):
0.1560  0.2136  0.2818  0.3606  0.4501  0.0500  0.1100  0.1798
0.2591  0.3474  0.1560  0.2136  0.2818  0.3606  0.4501  0.0500
0.1100  0.1798  0.2591  0.3474  0.1560  0.2136  0.2818  0.3606
0.4501  0.0500  0.1100  0.1798  0.2591  0.3474  0.1560  0.2136

Output dB (GPU) (4x8):
-0.1888 -0.1605 -0.1277 -0.0900 -0.0475 0.0000  0.0525  0.1100
0.1723  0.2395  -0.1888 -0.1605 -0.1277 -0.0900 -0.0475 0.0000
0.0525  0.1100  0.1723  0.2395  -0.1888 -0.1605 -0.1277 -0.0900
-0.0475 0.0000  0.0525  0.1100  0.1723  0.2395  -0.1888 -0.1605


Performing CPU computation for verification...
Output C (CPU) (4x8):
-0.1133 -0.1124 -0.1021 -0.0810 -0.0475 0.0000  0.0105  0.0330
0.0689  0.1197  -0.1133 -0.1124 -0.1021 -0.0810 -0.0475 0.0000
0.0105  0.0330  0.0689  0.1197  -0.1133 -0.1124 -0.1021 -0.0810
-0.0475 0.0000  0.0105  0.0330  0.0689  0.1197  -0.1133 -0.1124

Output dA (CPU) (4x8):
0.1560  0.2136  0.2818  0.3606  0.4501  0.0500  0.1100  0.1798
0.2591  0.3474  0.1560  0.2136  0.2818  0.3606  0.4501  0.0500
0.1100  0.1798  0.2591  0.3474  0.1560  0.2136  0.2818  0.3606
0.4501  0.0500  0.1100  0.1798  0.2591  0.3474  0.1560  0.2136

Output dB (CPU) (4x8):
-0.1888 -0.1605 -0.1277 -0.0900 -0.0475 0.0000  0.0525  0.1100
0.1723  0.2395  -0.1888 -0.1605 -0.1277 -0.0900 -0.0475 0.0000
0.0525  0.1100  0.1723  0.2395  -0.1888 -0.1605 -0.1277 -0.0900
-0.0475 0.0000  0.0525  0.1100  0.1723  0.2395  -0.1888 -0.1605

Verification Results:
Forward pass (C) matches CPU: Yes
Backward pass (dA) matches CPU: Yes
Backward pass (dB) matches CPU: Yes

All computations verified successfully!
```

**Test Output (`./build/day067/swiglu_test`):**
```
Running main() from /home/drboom/git_repos/100-days-of-cuda/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 3 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 2 tests from SwiGLUTest
[ RUN      ] SwiGLUTest.ForwardPassCorrectness
[       OK ] SwiGLUTest.ForwardPassCorrectness (91 ms)
[ RUN      ] SwiGLUTest.BackwardPassCorrectness
[       OK ] SwiGLUTest.BackwardPassCorrectness (1 ms)
[----------] 2 tests from SwiGLUTest (92 ms total)

[----------] 1 test from SwiGLUTestEmpty
[ RUN      ] SwiGLUTestEmpty.HandlesZeroSize
[       OK ] SwiGLUTestEmpty.HandlesZeroSize (0 ms)
[----------] 1 test from SwiGLUTestEmpty (0 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 2 test suites ran. (93 ms total)
[  PASSED  ] 3 tests.
```

## Learnings and Observations

-   Implemented SwiGLU, a modern activation function used in transformers.
-   Practiced deriving and implementing gradients for a custom activation function.
-   Reinforced understanding of CUDA kernel launch configurations and memory management.
-   The "one block per row, one thread per column" mapping is simple but has performance limitations depending on matrix dimensions.
-   CPU verification is crucial for ensuring the correctness of CUDA kernels.
-   The recomputation of `sigmoid(a)` in the backward pass is a common trade-off between memory and computation.

## References
-   PaLM: Scaling Language Modeling with Pathways (contains SwiGLU): [https://arxiv.org/abs/2204.02311](https://arxiv.org/abs/2204.02311) (Though SwiGLU was introduced earlier, e.g., in "GLU Variants Improve Transformer" - [https://arxiv.org/abs/2002.05202](https://arxiv.org/abs/2002.05202))
-   NVIDIA Transformer Engine Documentation (for examples of similar activations): [https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
