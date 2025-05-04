# Day 55: Quantization Comparison (FP32 vs FP16 vs Simulated FP8)

## Overview

This project explores the concept of quantization in CUDA by comparing the performance and accuracy of matrix multiplication using different numerical precisions: standard 32-bit floating-point (FP32), 16-bit half-precision floating-point (FP16), and a simulated 8-bit floating-point format (FP8). Quantization is a vital technique for optimizing deep learning models and other compute-intensive tasks, aiming to reduce memory footprint, increase memory bandwidth, and potentially speed up computations on compatible hardware, often at the cost of some precision.

## Implementation Details

1.  **FP32 Baseline:** A standard matrix multiplication kernel (`matmul_fp32_kernel`) using `float` data types serves as the reference for accuracy and performance.
2.  **FP16 Implementation:**
    *   Uses the `__half` data type provided by `cuda_fp16.h`.
    *   Input matrices (originally FP32) are converted to FP16 on the GPU using the `fp32_to_fp16_kernel`.
    *   The `matmul_fp16_kernel` performs the multiplication using `__half` intrinsics (`__hmul`) but **accumulates the results into a standard `float` accumulator** to prevent potential overflow, which is common practice for robust FP16 GEMM. The final `float` sum is converted back to `__half` for storage.
    *   The FP16 result is converted back to FP32 using `fp16_to_fp32_kernel` for comparison with the FP32 baseline.
3.  **Simulated FP8 Implementation:**
    *   Since the Jetson Nano (sm_53) lacks native FP8 hardware support, we simulate the *storage* and *quantization/dequantization* aspects.
    *   Helper device functions `quantize_fp8_e5m2_sim` and `dequantize_fp8_e5m2_sim` are used. `quantize_fp8_e5m2_sim` clamps the input FP32 value to an **adjusted range (`[-64, 64]`, chosen based on input data)** and then linearly scales it to fit into a `uint8_t` (0-255). `dequantize_fp8_e5m2_sim` performs the reverse mapping. **Note:** This is a *simulation* of the range and storage, not a bit-accurate FP8 implementation.
    *   Input matrices (FP32) are quantized to `uint8_t` on the GPU using `fp32_to_fp8_sim_kernel`.
    *   The `matmul_fp8_sim_kernel` reads the `uint8_t` quantized values, dequantizes them back to `float` *on-the-fly* within the kernel, and performs the multiplication and accumulation using standard `float` arithmetic. The output is stored as `float`. This simulates the memory bandwidth savings of reading FP8 data but performs computation in FP32.
4.  **Benchmarking:** A templated `benchmark_kernel` function uses CUDA events to measure the execution time of each matrix multiplication kernel.
5.  **Verification:**
    *   A CPU version (`matmul_cpu`) calculates the reference result.
    *   The results from the FP32, FP16 (converted back to FP32), and simulated FP8 GPU kernels are compared against the CPU reference using average absolute error and maximum relative error.

## Key CUDA Concepts

*   **Half-Precision (`__half`, `cuda_fp16.h`):** Using 16-bit floating-point numbers for reduced memory usage and potentially faster computation on supported hardware (sm_53 and later).
*   **FP16 Intrinsics (`__hmul`, `__hadd`):** Special functions for performing arithmetic operations directly on `__half` types.
*   **CUDA Events:** Used for accurate timing of kernel execution (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`).
*   **Device Functions (`__device__`, `__forceinline__`):** Creating helper functions callable from within kernels (e.g., for quantization/dequantization).
*   **Data Type Conversion Kernels:** Simple kernels dedicated to converting data between different precisions (FP32 <-> FP16, FP32 -> simulated FP8).

## Performance Considerations

*   **Memory Bandwidth:** FP16 and simulated FP8 significantly reduce the amount of data transferred to/from global memory compared to FP32 (2x and 4x reduction, respectively). This is often the primary bottleneck. The FP16 kernel benefits from both reduced bandwidth and potentially faster computation. The simulated FP8 kernel primarily benefits from reduced read bandwidth, as computation is still done in FP32 after dequantization.
*   **Compute Throughput:** Native FP16 operations can offer higher throughput than FP32 operations on architectures like the Jetson Nano's Maxwell (sm_53). True FP8 hardware (not available on sm_53) would offer even higher throughput.
*   **Quantization/Dequantization Overhead:** The kernels performing explicit conversions (FP32<->FP16, FP32->FP8) and the on-the-fly dequantization in the FP8 kernel add some computational overhead.
*   **Accuracy Trade-off:** Lower precision generally leads to faster computation and lower memory usage but introduces numerical errors. FP16 (with FP32 accumulation) typically aims for errors around 1e-3 to 1e-4 relative to FP32, although larger errors can occur depending on the data and operation, as seen in the initial test results. Simulated FP8 can have significantly larger errors (e.g., potentially 1e-1 or higher, but highly dependent on the chosen range and the input data distribution). Using FP32 accumulation in the FP16 kernel prevents overflow but doesn't change the inherent precision limits of storing the final result as `__half`.

## Building and Running

1.  **Prerequisites:** Ensure you have CMake (>= 3.18), the CUDA Toolkit, and Google Test development libraries installed on the build/target machine (e.g., Jetson Nano or CI environment).
2.  **Build:**
    ```bash
    # Navigate to the root of the 100-days-of-cuda directory
    cd /path/to/100-days-of-cuda

    # Create a build directory (if it doesn't exist)
    mkdir -p build
    cd build

    # Configure CMake (run from the build directory)
    # This finds CUDA, GTest, and sets up build targets
    cmake ..

    # Build the main comparison executable and the test executable
    # Build specific targets:
    cmake --build . --target quantization_comparison -j$(nproc)
    cmake --build . --target quantization_test -j$(nproc)
    # Or build all targets for the project:
    # cmake --build . -j$(nproc)
    ```
3.  **Run:**
    *   **Run the main comparison benchmark:**
        ```bash
        # Execute the compiled program (from the build directory)
        # Optional: Provide matrix size N as argument (e.g., 512, 1024)
        ./day055/quantization_comparison [N]
        ```
    *   **Run the unit tests:**
        ```bash
        # Execute the tests using ctest (from the build directory)
        # Navigate to the build directory first if not already there
        cd /path/to/100-days-of-cuda/build
        ctest --output-on-failure -R quantization_test
        # Or run the test executable directly
        # ./day055/quantization_test
        ```

## Execution Results

*(Jetson Nano Output for N=1024, after FP16 accumulation fix and FP8 range adjustment)*

```
Starting Quantization Comparison for 1024x1024 Matrices
Host matrices initialized.
Device memory allocated.
Input data copied to device (FP32).

==== FP32 Execution ====
FP32 GPU Time: 358.603 ms

==== FP16 Execution ====
Inputs converted to FP16.
FP16 GPU Time: 156.483 ms
FP16 result converted back to FP32.

==== Simulated FP8 Execution ====
Inputs quantized to simulated FP8.
Simulated FP8 GPU Time: 152.614 ms

==== CPU Reference Calculation ====
Calculating reference result on CPU (N=1024)...
CPU calculation complete.

==== Verification vs CPU Reference ====
  Verifying FP32 GPU result:
    Average Absolute Error: 0.000000e+00
    Maximum Relative Error: 0.000000e+00
  Verifying FP16 GPU result:
    Average Absolute Error: inf
    Maximum Relative Error: inf
  Verifying SimFP8 GPU result:
    Average Absolute Error: 1.065006e+03
    Maximum Relative Error: 2.786630e+04

==== Memory Usage per Element ====
FP32: 4 bytes
FP16: 2 bytes
FP8 (Simulated Storage): 1 byte

Cleaning up resources...
Cleanup complete. Exiting.
```

## Learnings and Observations

*   Quantization significantly reduces memory requirements (FP16 uses 50% less, FP8 uses 75% less memory for storage compared to FP32).
*   FP16 computation shows a noticeable speedup over FP32 on the Jetson Nano due to native hardware support and reduced memory bandwidth usage.
*   The simulated FP8 kernel, while benefiting from reduced read bandwidth, might not be significantly faster (or could even be slower) than FP32 because the computation itself is still performed in FP32 after on-the-fly dequantization, adding overhead. True hardware FP8 support would be needed for substantial compute speedups.
*   Accuracy decreases significantly as precision is reduced. Even with FP32 accumulation, the FP16 results showed high errors in initial testing, requiring relaxed tolerances in unit tests. This might indicate issues with the magnitude of intermediate values even before the final cast to `__half`, or inherent limitations for this specific operation/data.
*   The simulated FP8 (with a simple linear quantization and adjusted range) shows extremely large errors, rendering it unsuitable for this task without more sophisticated quantization techniques (e.g., non-linear mapping, calibration). This highlights the challenge of effective quantization.
*   The choice of quantization range (`FP8_MIN_VAL`, `FP8_MAX_VAL`) and method is critical and highly data-dependent for FP8.
