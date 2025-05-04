# Day 56: Mish Activation Function Benchmark

## Overview

This project implements and benchmarks the Mish activation function on both CPU and GPU. Mish is a smooth, non-monotonic activation function defined as:

`f(x) = x * tanh(softplus(x))`

where `softplus(x) = ln(1 + exp(x))`.

It aims to combine the benefits of functions like ReLU (unbounded above) and Swish (smoothness, self-gating) while potentially offering better performance and gradient flow due to its specific mathematical properties. The goal is to compare the performance of a naive CUDA implementation against a single-threaded CPU implementation for a large number of elements.

## Implementation Details

The implementation is split into a header, a source file for the main benchmark, and a test file:

1.  **`mish_activation.cuh` (Header):**
    *   Defines the `CHECK_CUDA_ERROR` macro and `checkCuda` function declaration.
    *   Defines the `__host__ __device__ inline float mish(float x)` function using `tanhf`, `logf`, and `expf`.
    *   Declares the `mish_cpu` function.
    *   Declares the `mish_kernel` kernel.
    *   Declares the `mish_gpu_wrapper` function.
2.  **`mish_activation.cu` (Source):**
    *   Includes the header `mish_activation.cuh`.
    *   Defines the `checkCuda` error-checking function.
    *   Defines the `mish_cpu` function for the host-side implementation.
    *   Defines the `mish_gpu_wrapper` function, which launches the `mish_kernel` with appropriate grid/block sizes and handles optional CUDA event timing.
    *   Contains the `main` function for the benchmark:
        *   Initializes a large vector (`n = 1 << 24`) with random float values.
        *   Allocates memory on both host and device.
        *   Copies input data to the device.
        *   Runs and benchmarks the CPU implementation (`mish_cpu`) using `std::chrono`.
        *   Runs and benchmarks the GPU implementation (`mish_gpu_wrapper`) using CUDA events (includes a warm-up run).
        *   Copies results back from the device.
        *   Verifies the GPU results against the CPU results using a small tolerance (`1e-5`).
        *   Calculates and prints the GPU speedup over the CPU.
3.  **`mish_activation_test.cu` (Tests):**
    *   Includes `gtest/gtest.h` and `mish_activation.cuh`.
    *   Contains unit tests using Google Test:
        *   `MishActivationTest.BasicValues`: Tests the `mish` function directly with known values.
        *   `MishGpuTest.KernelExecution`: Uses a test fixture (`MishGpuTest`) to set up input data on host/device, run the `mish_gpu_wrapper`, copy results back, and verify against the expected CPU output for a smaller dataset (`n=1024`).

## Key CUDA Concepts Used

*   **CUDA Kernels:** Defining and launching a `__global__` function (`mish_kernel`).
*   **Thread Hierarchy:** Using `blockIdx`, `blockDim`, and `threadIdx` for element mapping.
*   **Device Memory Management:** `cudaMalloc`, `cudaMemcpy` (HostToDevice, DeviceToHost), `cudaFree`.
*   **CUDA Events:** Using `cudaEvent_t` for accurate GPU timing (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`).
*   **CUDA Error Handling:** Basic error checking macro (`CHECK_CUDA_ERROR`) and function (`checkCuda`).
*   **Math Functions:** Using device-compatible math functions (`tanhf`, `logf`, `expf`).
*   **Unit Testing:** Integrating CUDA code with Google Test for verification.

## Performance Considerations

*   The Mish function involves several transcendental functions (`expf`, `logf`, `tanhf`), making it computationally more expensive than simpler functions like ReLU (`max(0, x)`).
*   The operation is element-wise, making it highly parallelizable and suitable for GPU acceleration.
*   The benchmark measures only the kernel execution time for the GPU, excluding memory transfer times.
*   The CPU implementation is single-threaded, providing a baseline for comparison. A multi-threaded CPU implementation would likely perform better but is not included here.
*   Numerical stability of `logf(1.0f + expf(x))` could be a concern for very large positive or negative inputs, although the typical range for activation functions might mitigate this.

## Building and Running

**Note:** Build and run these instructions on the **target platform (Jetson Nano)** or a compatible environment with CUDA toolkit, CMake, and Google Test installed.

1.  **Navigate to the root directory** of the `100-days-of-cuda` project.
2.  **Create a build directory (if it doesn't exist):**
    ```bash
    mkdir -p build
    cd build
    ```
3.  **Configure CMake:** Make sure you are pointing to the root `CMakeLists.txt`.
    ```bash
    cmake .. -D CMAKE_BUILD_TYPE=Release # Configure for Release build
    ```
    *(CMake should automatically find and configure the `day056` subdirectory, find GTest, and set up targets)*
4.  **Build the executables (benchmark and test):**
    ```bash
    cmake --build . --parallel $(nproc)
    # Or: make -j$(nproc)
    ```
    This will build both `mish_benchmark` and `mish_benchmark_test`.
5.  **Run the benchmark:** The executable will be located in `build/day056/`.
    ```bash
    ./day056/mish_benchmark
    ```
6.  **Run the tests:** Use CTest, which was configured by CMake.
    ```bash
    ctest --output-on-failure
    # Or run the test executable directly: ./day056/mish_benchmark_test
    ```
    The tests should pass.

## Execution Results

*(Note: The following is example output. Actual times will vary based on the specific hardware, especially the Jetson Nano.)*

```text
Processing 16777216 elements (64.00 MB)
Host input data initialized.
Device memory allocated.
Input data copied to device.

--- CPU Execution ---
CPU execution time: 1234.567 ms

--- GPU Execution ---
GPU execution time: 15.890 ms
GPU results copied back to host.

--- Verification ---
Verification successful: CPU and GPU results match.

--- Performance ---
GPU Speedup over CPU: 77.70x

Cleanup complete.
```

## Learnings and Observations

*   Implementing element-wise functions like Mish in CUDA is straightforward. Each thread handles one output element independently.
*   Despite being computationally more complex than ReLU, Mish still benefits significantly from GPU parallelism, achieving substantial speedup over a single-threaded CPU implementation.
*   The use of CUDA events is crucial for accurately measuring GPU kernel execution time, excluding API overhead and memory transfer times.
*   Verification against a CPU implementation (done in both the benchmark `main` and the unit tests) is essential to ensure the correctness of the GPU kernel. Floating-point comparisons require using a tolerance.
*   Unit testing CUDA kernels, even simple ones, using a framework like Google Test helps catch errors early and ensures individual components work as expected.
*   The trade-off for Mish's potential accuracy improvements in neural networks is its higher computational cost per element compared to simpler activations like ReLU.
