# Day 70: Mean Squared Error (MSE) Calculation with CUDA

## Overview

This project implements the Mean Squared Error (MSE) calculation, a common metric in machine learning and statistics, using both CPU and GPU (via CUDA). The goal is to compare their performance, especially for large datasets where GPU parallelism can offer significant speedups.

MSE is calculated as:
$$ \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (Y_i - \hat{Y}_i)^2 $$
where $N$ is the number of data points, $Y_i$ are the true target values, and $\hat{Y}_i$ are the predicted values.

## Implementation Details

The implementation consists of:
1.  **`mse.cuh`**: Header file defining the `CHECK_CUDA_ERROR` macro and function prototypes for `mse_cpu` and `mse_gpu`.
2.  **`mse.cu`**:
    *   `mse_cpu()`: A straightforward C++ implementation that iterates through the input vectors, calculates squared differences, sums them, and divides by N.
    *   `mse_gpu()`: The main GPU wrapper function, utilizing a **two-step Thrust library** approach:
        *   Allocating GPU memory for predictions, targets, and an intermediate difference vector ($D$).
        *   Copying input data (predictions $P$, targets $T$) from host to device.
        *   **Step 1: Compute Differences**: Using `thrust::transform` with `thrust::minus<float>()` to calculate $D_i = P_i - T_i$ for all elements, storing the results in the `d_diff` device vector.
        *   **Step 2: Sum of Squares**: Using `thrust::transform_reduce` on the `d_diff` vector.
            *   A `square_functor` is used as the unary transform operation to compute $D_i^2$.
            *   `thrust::plus<float>()` is used as the reduction operation to sum these squared values.
        *   The sum of squared errors is returned by `thrust::transform_reduce`.
        *   Calculating the final MSE by dividing this sum by $N$.
        *   Freeing all allocated GPU memory.
    *   This two-step Thrust approach was chosen for potentially better compiler compatibility with older Thrust versions over a single, more complex `transform_reduce` call with two input iterators.
3.  **`mse_main.cu`**: The main executable.
    *   Generates large synthetic vectors for predictions and targets.
    *   Measures the execution time for both CPU and GPU MSE calculations using `std::chrono` and CUDA Events.
    *   Verifies that the results from CPU and GPU implementations are numerically close.
    *   Prints the MSE results, execution times, and speedup.
4.  **`mse_test.cu`**: Google Test suite.
    *   Tests for zero elements.
    *   Tests basic calculations with small, known inputs for both CPU and GPU.
    *   Compares CPU and GPU results for medium and larger datasets with random data.

## Key CUDA Features Used

*   **Thrust Library**: Utilized for high-level parallel algorithms:
    *   `thrust::device_ptr`: Wraps raw device pointers for use with Thrust algorithms.
    *   `thrust::transform`: Performs a parallel transformation. Used here with `thrust::minus<float>()` to compute element-wise differences $D = P - T$.
    *   `thrust::transform_reduce`: Performs a parallel map-reduce operation. Used here on the intermediate difference vector $D$ to square each element (transform) and then sum the results (reduce).
    *   `thrust::plus<float>()`, `thrust::minus<float>()`: Predefined functors for addition and subtraction.
    *   Custom Functor (`square_functor`): A user-defined structure with `__host__ __device__ operator()` to perform element-wise squaring.
*   **Device Memory Management**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
*   **Error Handling**: `CHECK_CUDA_ERROR` macro. (The `CHECK_CUBLAS_ERROR` macro is still present in `mse.cuh` but not actively used by the Thrust `mse_gpu` version).
*   **CUDA Events**: `cudaEvent_t` for accurate timing of GPU operations.

## Performance Considerations

*   **Thrust Optimizations**: Thrust is designed to generate efficient CUDA code. `thrust::transform_reduce` can often fuse the transform and reduction steps into fewer kernel launches or more optimized kernels than manually chaining operations (like separate transform and then reduce kernels, or multiple cuBLAS calls). This can lead to better performance by reducing overhead and improving memory access patterns.
*   **Data Size**: As with any GPU acceleration, sufficiently large vectors are needed to overcome memory transfer costs (Host-to-Device for input vectors) and the overhead of Thrust algorithm invocation.
*   **Memory Transfers**: The `mse_gpu` function copies prediction and target vectors to the GPU. The result of `thrust::transform_reduce` (the sum of squared errors) is returned as a value to the host, involving an implicit Device-to-Host transfer for the final scalar.
*   **Functor Efficiency**: The custom functor provided to Thrust should be efficient. Simple arithmetic operations, as in `squared_difference_functor`, are generally well-handled.
*   **Comparison to cuBLAS/Custom Kernels**: The two-step Thrust approach (`transform` then `transform_reduce`) breaks the problem down using standard, well-optimized Thrust primitives. While it introduces an intermediate device vector (`d_diff`), Thrust's internal optimizations might still make this competitive or better than manual cuBLAS chaining or simple custom kernels, especially if it avoids compiler/version issues seen with more complex Thrust calls. The single-call `transform_reduce` (with two input iterators) would theoretically be more memory efficient by avoiding the explicit intermediate vector.

## Building and Running

### Prerequisites
*   NVIDIA CUDA Toolkit (>= 10.x, tested with 11.x/12.x)
*   CMake (>= 3.18 recommended)
*   A C++ compiler compatible with CUDA (e.g., GCC, Clang, MSVC)
*   Google Test (fetched by the root `CMakeLists.txt` or installed system-wide)

### Build Steps (from the `100-days-of-cuda/build` directory)

1.  **Configure CMake** (if not already done for the whole project):
    ```bash
    cd /path/to/100-days-of-cuda/
    mkdir -p build
    cd build
    cmake .. 
    ```
2.  **Build Day 70 Project**:
    ```bash
    cmake --build . --target mse_benchmark --config Release
    cmake --build . --target mse_test_runner --config Release 
    ```
    (Or build all targets: `cmake --build . --config Release`)

### Running

1.  **Run the Benchmark**:
    ```bash
    ./day070/mse_benchmark
    ```
2.  **Run Tests**:
    ```bash
    ./day070/mse_test_runner
    # Or using CTest from the build directory
    # ctest --output-on-failure -R day070_mse # (If tests are correctly registered with CTest by root)
    # Or more specifically if tests are named as per project in CMakeLists.txt
    # ctest -R mse_test_runner 
    ```

## Execution Results

*(This section will be populated after running the code on the target platform, e.g., Jetson Nano)*

**Expected Output Format from `mse_benchmark`:**
```
Mean Squared Error (MSE) Calculation
Number of elements (N): 16777216
------------------------------------
Generating synthetic data...
Data generation complete.

Calculating MSE on CPU...
CPU MSE Result: 0.XXXXXXX
CPU Execution Time: YYY.YYY ms

Calculating MSE on GPU...
GPU MSE Result: 0.XXXXXXX
GPU Execution Time (cudaEvent): ZZZ.ZZZ ms
GPU Execution Time (chrono, incl. overhead): AAA.AAA ms

--- Verification ---
SUCCESS: CPU and GPU results are close.
Difference: E.EEEEEEE

--- Performance Comparison ---
Speedup (CPU Time / GPU Event Time): S.SSx

Execution finished.
```

**Actual Jetson Nano Output (with `sudo jetson_clocks` active, using two-step Thrust):**
```
drboom@JetNano ~/g/1/build> ./day070/mse_benchmark 
Mean Squared Error (MSE) Calculation
Number of elements (N): 16777216
------------------------------------
Generating synthetic data...
Data generation complete.

Calculating MSE on CPU...
CPU MSE Result: 0.16670118
CPU Execution Time: 46.72210300 ms

Calculating MSE on GPU...
GPU MSE Result: 0.16670120
GPU Execution Time (cudaEvent): 197.14395142 ms
GPU Execution Time (chrono, incl. overhead): 197.26404000 ms

--- Verification ---
SUCCESS: CPU and GPU results are close.
Difference: 0.00000001

--- Performance Comparison ---
Speedup (CPU Time / GPU Event Time): 0.24x

Execution finished.

drboom@JetNano ~/g/1/build> ./day070/mse_test_runner 
[==========] Running 6 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 6 tests from MSETest
[ RUN      ] MSETest.HandlesZeroElements
[       OK ] MSETest.HandlesZeroElements (0 ms)
[ RUN      ] MSETest.BasicCalculationCPU
[       OK ] MSETest.BasicCalculationCPU (0 ms)
[ RUN      ] MSETest.BasicCalculationGPU
[       OK ] MSETest.BasicCalculationGPU (80 ms)
[ RUN      ] MSETest.CPUvsGPUMediumSize
[       OK ] MSETest.CPUvsGPUMediumSize (1 ms)
[ RUN      ] MSETest.LargerRandomData
[       OK ] MSETest.LargerRandomData (4 ms)
[ RUN      ] MSETest.SingleElement
[       OK ] MSETest.SingleElement (1 ms)
[----------] 6 tests from MSETest (87 ms total)

[----------] Global test environment tear-down
[==========] 6 tests from 1 test suite ran. (87 ms total)
[  PASSED  ] 6 tests.
```

## Learnings and Observations

*   **Performance on Jetson Nano**: Even with optimized libraries like Thrust and maximized clock speeds (`sudo jetson_clocks`), the GPU implementation (0.24x speedup) was significantly slower than the CPU for this MSE calculation with N=16.7M elements. This highlights that for memory-bound operations with relatively simple arithmetic per element, the overhead of data transfers, GPU kernel launches (even those managed by Thrust), and the Jetson Nano's specific architecture (limited memory bandwidth, less powerful GPU compared to discrete desktop cards) can prevent achieving a speedup over a modern CPU.
*   **Thrust vs. cuBLAS vs. Custom Kernels**:
    *   The initial custom kernel was very slow (0.12x speedup before `jetson_clocks`).
    *   cuBLAS with `jetson_clocks` improved slightly to 0.22x.
    *   The two-step Thrust approach with `jetson_clocks` yielded a marginal improvement to 0.24x.
    *   This suggests that for this problem, the bottleneck is likely fundamental (memory bandwidth, overheads) rather than just the kernel implementation details, once a reasonably optimized library like Thrust or cuBLAS is used.
*   **Importance of `jetson_clocks`**: Running `sudo jetson_clocks` nearly halved the GPU execution time for the cuBLAS version (from ~380ms to ~200ms), emphasizing its importance for benchmarking on the Jetson platform.
*   **Numerical Precision**: CPU and GPU results were consistently close, within acceptable floating-point error tolerance.
*   **Test Durations**: The GPU tests, especially `BasicCalculationGPU` (80ms for small N), show significant overhead for small problem sizes, which is expected.

## (Optional) Future Improvements

*   Implement a fully device-side multi-level reduction for the sum to avoid copying partial sums to the host.
*   Use Thrust library (`thrust::transform_reduce`) for a more concise and potentially optimized MSE calculation.
*   Experiment with different block and grid sizes for kernel launches.
*   Investigate using half-precision (FP16) or mixed-precision for potential speedups on supported hardware, if precision requirements allow.

## (Optional) References

*   NVIDIA CUDA C++ Programming Guide
*   Mark Harris - Optimizing Parallel Reduction in CUDA: [https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
