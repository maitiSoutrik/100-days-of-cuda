# Day 54: AdaHessian Optimizer Kernel

## Overview

This project implements the core update step of the AdaHessian optimization algorithm using CUDA. AdaHessian is a second-order optimization method that incorporates an approximation of the Hessian matrix's diagonal to adapt the learning rate for each parameter, potentially improving convergence on complex loss surfaces compared to first-order methods like Adam.

## Implementation Details

The implementation is structured as follows:

1.  **`adahessian.h`**: Declares the `adaHessianUpdateKernel` CUDA kernel and includes a common error-checking macro (`CHECK_CUDA_ERROR`).
2.  **`adahessian.cu`**: Provides the implementation for the `adaHessianUpdateKernel`.
    *   **Hessian Diagonal Approximation:** Uses finite differences: `H_ii ≈ (g_perturbed_i - g_i) / delta`.
    *   **Moment Updates:** Calculates exponential moving averages for the first moment (gradient `m`) and the second moment (squared Hessian diagonal `v`).
    *   **Parameter Update:** Updates parameters `theta` using the rule: `theta = theta - lr * m / (sqrt(v) + epsilon)`.
    *   Each thread handles one parameter element in parallel.
3.  **`main_adahessian.cu`**: A demonstration program that:
    *   Initializes large host arrays with sample data.
    *   Allocates device memory.
    *   Copies data to the GPU.
    *   Launches the `adaHessianUpdateKernel`.
    *   Measures kernel execution time using CUDA events.
    *   Copies results back to the host.
    *   Prints some results and cleans up resources.
4.  **`test_adahessian.cu`**: A small verification program that:
    *   Uses a small dataset (`N=10`).
    *   Runs the GPU kernel.
    *   Runs an equivalent calculation on the CPU for the first element (`adaHessianUpdateCPU`).
    *   Compares the GPU and CPU results for the first element to verify correctness within a tolerance.
    *   Returns 0 on success (match) and 1 on failure (mismatch).

## Key CUDA Features Used

*   **CUDA Kernels (`__global__`)**: For parallel execution of the update logic on the GPU.
*   **Device Memory Management**: `cudaMalloc`, `cudaMemcpy`, `cudaFree` for handling data on the GPU.
*   **Thread Indexing**: `blockIdx`, `blockDim`, `threadIdx` for mapping threads to parameter elements.
*   **CUDA Error Handling**: Using `cudaError_t` and `cudaGetErrorString` (via the `CHECK_CUDA_ERROR` macro) for robust code.
*   **CUDA Events**: `cudaEvent_t` for timing the kernel execution accurately (`main_adahessian.cu`).
*   **Math Functions**: `sqrtf` from `math.h` (implicitly used via CUDA's math library).
*   **Separate Compilation**: Header (`.h`) and implementation (`.cu`) files linked by CMake.

## Performance Considerations

*   **Memory Access:** The kernel performs element-wise operations with independent calculations per thread. Access patterns for `theta`, `grad`, `gradPerturbed`, `m`, and `v` are coalesced as each thread accesses contiguous elements based on its global index `idx`.
*   **Computational Intensity:** The kernel involves several floating-point operations per element. Performance depends on the GPU's arithmetic throughput and memory bandwidth.
*   **Finite Difference Cost:** The main overhead compared to Adam in a real training loop is the need to compute an additional gradient (`gradPerturbed`) per step. This implementation assumes these gradients are provided. The update kernel itself is comparable in complexity to Adam.

## Building and Testing

**Note:** Compilation and execution require an environment with **CUDA Toolkit**, **CMake (>=3.18 recommended)**, and **Google Test** installed and findable by CMake. This should be performed on the Jetson Nano, the CI/CD runner, or a similarly configured build environment.

1.  **Navigate to the root directory** (`100-days-of-cuda`).
2.  **Create/update the build directory:** `mkdir -p build && cd build`
3.  **Run CMake:** `cmake ..` (This configures all days, including Day 54, and should find Google Test).
4.  **Build the Day 54 executables:** `make main_adahessian test_adahessian`
5.  **Run the main example:** `./day054/main_adahessian`
6.  **Run the tests:**
    *   Using CTest (recommended): `ctest --output-on-failure -R day054` (Runs tests specifically for day 54)
    *   Directly: `./day054/test_adahessian`

## Execution Results (Sample Output)

### Main Example (`./day054/main_adahessian`)

(Actual output from Jetson Nano)

```text
AdaHessian Main Example
Number of parameters: 1048576
Learning Rate: 0.0100, Beta1: 0.900, Beta2: 0.999, Epsilon: 1.0e-07, Delta: 1.0e-04
Initializing host data...
Host data initialization complete.
Allocating device memory...
Device memory allocation complete.
Copying data from host to device...
Data copy complete.
Launching kernel with grid size 4096 and block size 256
Kernel execution time: 14.076042 ms
Copying results from device to host...
Result copy complete.

Updated theta values (first/last 10):
0.968374 0.968374 0.968374 0.968374 0.968374 0.968374 0.968374 0.968374 0.968374 0.968374 ... 0.994202 0.994202 0.994202 0.994202 0.994202 0.994202 0.994202 0.994202 0.994202 0.994202 
Freeing device memory...
Device memory freed.
Freeing host memory...
Host memory freed.

Execution finished successfully.
```

### Google Test Executable (`./day054/test_adahessian` or via `ctest`)

(Actual output from Jetson Nano)

```text
Running main() from /home/drboom/git_repos/100-days-of-cuda/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from AdaHessianTest
[ RUN      ] AdaHessianTest.BasicUpdateVerification
[       OK ] AdaHessianTest.BasicUpdateVerification (98 ms)
[----------] 1 test from AdaHessianTest (98 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (98 ms total)
[  PASSED  ] 1 test.
```
*(The test successfully passes, verifying the kernel logic against the CPU implementation for the tested elements.)*

## Learnings and Observations

*   AdaHessian incorporates second-order information (curvature) via finite differences, potentially improving optimization.
*   Separating kernel declaration (`.h`), implementation (`.cu`), and application/test logic (`main_*.cu`, `test_*.cu`) improves code organization. CMake handles linking these components.
*   Using a standard unit testing framework like Google Test provides structured testing, clear pass/fail reporting, and integrates well with tools like CTest for automated testing.
*   The main performance cost of AdaHessian in training is the extra gradient computation, not the update step itself.
*   Comparing GPU results against a trusted CPU implementation within a test framework is crucial for verifying kernel correctness.
