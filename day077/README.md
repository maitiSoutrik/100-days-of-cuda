# Day 77: Huber Loss Implementation in CUDA

## Overview

This project implements the Huber loss function and its derivative in CUDA C++. Huber loss is a robust loss function that is less sensitive to outliers than Mean Squared Error (MSE) and smoother around zero than Mean Absolute Error (MAE). It combines the advantages of both by being quadratic for small errors and linear for large errors.

## Huber Loss Explained

Huber Loss is a hybrid loss function designed to balance sensitivity to small errors and robustness against outliers. It combines the quadratic behavior of Mean Squared Error (MSE) for minor deviations with the linear penalty of Mean Absolute Error (MAE) for larger errors, controlled by a threshold parameter denoted as \\(\delta\\) (delta). This design makes it particularly effective for regression tasks where the data might contain noise or outliers.

### Mathematical Formulation

The Huber Loss function, \\(L_\delta(y, f(x))\\), calculates the penalty for an estimation procedure \\(f(x)\\) given the true value \\(y\\). It is defined piecewise:

\\[
L_\delta(a) =
\begin{cases}
\frac{1}{2}a^2 & \text{if } |a| \leq \delta \\
\delta \cdot \left(|a| - \frac{1}{2}\delta\right) & \text{if } |a| > \delta
\end{cases}
\\]

Where:
*   \\(a = y - f(x)\\) is the residual or error.
*   \\(\delta\\) is the threshold parameter.

The derivative of the Huber loss function with respect to \\(a\\) is:

\\[
\frac{dL_\delta(a)}{da} =
\begin{cases}
a & \text{if } |a| \leq \delta \\
\delta \cdot \text{sign}(a) & \text{if } |a| > \delta
\end{cases}
\\]

### Design Rationale

The core idea behind Huber Loss is to get the "best of both worlds" from MSE and MAE:
1.  **Robustness to Outliers:** Like MAE, when an error is large (beyond \\(\delta\\)), Huber Loss increases linearly. This prevents outliers from having an overwhelming impact.
2.  **Efficiency for Small Errors:** Like MSE, for small errors (within \\(\delta\\)), Huber Loss is quadratic. This provides a stronger "pull" towards zero error when predictions are close.
3.  **Smoothness and Differentiability:** The formulation ensures the function is differentiable everywhere, crucial for gradient-based optimization.

## Implementation Details

The project includes:
-   `huber_loss.cuh`: Header file defining the CUDA kernels for Huber loss and its derivative, host wrapper functions, and CPU implementations.
-   `huber_loss.cu`: Source file with the CUDA kernel implementations (`huber_loss_kernel`, `huber_loss_derivative_kernel`), host functions to launch these kernels (`compute_huber_loss_gpu`, `compute_huber_loss_derivative_gpu`), and CPU versions for comparison (`huber_loss_cpu`, `huber_loss_derivative_cpu`).
-   `huber_loss_main.cu`: Main application to demonstrate and benchmark the CPU vs. GPU implementations. It initializes data, including some outliers, computes loss and gradients using both CPU and GPU, verifies the results, and prints performance metrics.
-   `huber_loss_test.cu`: Google Test suite to verify the correctness of both CPU and GPU implementations against known values and each other for different \\(\delta\\) values.
-   `CMakeLists.txt`: CMake build script for this day's project.

## Key CUDA Features Used

-   Basic CUDA kernel implementation (`__global__` functions).
-   Device memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`).
-   Thread indexing (`blockIdx.x`, `blockDim.x`, `threadIdx.x`).
-   Error checking (`CHECK_CUDA_ERROR` macro).
-   `fabsf` for absolute value in kernels.

## Performance Considerations

The GPU implementation is expected to outperform the CPU version significantly for large datasets due to parallel computation. The kernels are designed to perform element-wise independent calculations, which is well-suited for GPU parallelism. Memory transfers between host and device can be a bottleneck for smaller datasets, but for sufficiently large `N`, the computation speedup should dominate.

## Building and Running

### Prerequisites
- NVIDIA CUDA Toolkit (>= 10.0, tested with 11.x, 12.x)
- CMake (>= 3.10)
- A C++ compiler compatible with CUDA (e.g., g++)
- Google Test (fetched by CMake)

### Build Instructions (for target environment like Jetson Nano)
1.  Navigate to the root of the `100-days-of-cuda` project.
2.  Create a build directory and navigate into it:
    ```bash
    mkdir build
    cd build
    ```
3.  Run CMake to configure the project:
    ```bash
    cmake ..
    ```
4.  Build the project (specifically for day077):
    ```bash
    cmake --build . --target huber_loss_main --target huber_loss_test -j$(nproc)
    ```
    (Alternatively, `make huber_loss_main huber_loss_test -j$(nproc)`)

### Running the Executable
After building, the executables will be in the `build/day077/` directory.
```bash
./build/day077/huber_loss_main
```

### Running Tests
```bash
./build/day077/huber_loss_test
# Or run all tests from the build directory
# ctest . -R day077 # (or similar ctest command depending on configuration)
```

## Execution Results

```
drboom@JetNano ~/g/1/build> ./day077/huber_loss_main 
Running Huber Loss Calculation for N = 4194304 elements, Delta = 1

--- CPU Calculation ---
CPU Huber Loss time: 33.2743 ms
CPU Huber Loss Derivative time: 10.6288 ms

--- GPU Calculation ---
GPU Huber Loss time: 207.112 ms
GPU Huber Loss Derivative time: 117.078 ms

--- Verification ---
Predictions (first 5 values): -4.4570 2.4279 -15.7588 3.1109 -3.1689 
Targets     (first 5 values): -4.9716 -2.4998 -3.3746 -3.4566 1.1182 

Loss:
CPU Loss    (first 5 values): 0.1324 4.4276 11.8843 6.0675 3.7870 
GPU Loss    (first 5 values): 0.1324 4.4276 11.8843 6.0675 3.7870 

Gradients:
CPU Grads   (first 5 values): 0.5146 1.0000 -1.0000 1.0000 -1.0000 
GPU Grads   (first 5 values): 0.5146 1.0000 -1.0000 1.0000 -1.0000 

Total Loss (CPU): 17744272.0000
Total Loss (GPU): 17744272.0000
Total Gradients (CPU): 644.0033
Total Gradients (GPU): 644.0033

Loss results VERIFIED (sum comparison).
Gradient results VERIFIED (sum comparison).

Speedup (Loss): 0.1607x
Speedup (Gradient): 0.0908x
```

Test execution output:
```
drboom@JetNano ~/g/1/build> ./day077/huber_loss_test
[==========] Running 8 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 4 tests from HuberLossTest
[ RUN      ] HuberLossTest.CPULossCalculation
[       OK ] HuberLossTest.CPULossCalculation (0 ms)
[ RUN      ] HuberLossTest.CPUDerivativeCalculation
[       OK ] HuberLossTest.CPUDerivativeCalculation (0 ms)
[ RUN      ] HuberLossTest.GPULossMatchesCPU
[       OK ] HuberLossTest.GPULossMatchesCPU (82 ms)
[ RUN      ] HuberLossTest.GPUDerivativeMatchesCPU
[       OK ] HuberLossTest.GPUDerivativeMatchesCPU (1 ms)
[----------] 4 tests from HuberLossTest (84 ms total)

[----------] 4 tests from HuberLossTestDifferentDelta
[ RUN      ] HuberLossTestDifferentDelta.CPULossCalculation
[       OK ] HuberLossTestDifferentDelta.CPULossCalculation (0 ms)
[ RUN      ] HuberLossTestDifferentDelta.CPUDerivativeCalculation
[       OK ] HuberLossTestDifferentDelta.CPUDerivativeCalculation (0 ms)
[ RUN      ] HuberLossTestDifferentDelta.GPULossMatchesCPU
[       OK ] HuberLossTestDifferentDelta.GPULossMatchesCPU (1 ms)
[ RUN      ] HuberLossTestDifferentDelta.GPUDerivativeMatchesCPU
[       OK ] HuberLossTestDifferentDelta.GPUDerivativeMatchesCPU (1 ms)
[----------] 4 tests from HuberLossTestDifferentDelta (3 ms total)

[----------] Global test environment tear-down
[==========] 8 tests from 2 test suites ran. (87 ms total)
[  PASSED  ] 8 tests.
```

## Learnings and Observations

-   The Huber loss function provides a good balance for regression tasks with potential outliers.
-   Implementing element-wise operations in CUDA is straightforward.
-   For this specific test on the Jetson Nano with N=4M elements, the GPU implementation was slower than the CPU version. This is likely due to the overhead of `cudaMemcpy` operations for a relatively simple element-wise kernel. On systems with discrete GPUs and higher data transfer bandwidth, or for more computationally intensive kernels, a speedup would be more likely. The Jetson Nano's shared memory architecture might also influence this, though explicit copies were used here.
-   The \\(\delta\\) parameter is crucial and would typically be tuned as a hyperparameter in a machine learning context.
-   Ensuring correctness by comparing GPU results with a CPU implementation and using unit tests is essential.

## Future Improvements

-   Explore the impact of different \\(\delta\\) values on performance and robustness with more varied datasets.
-   Integrate this loss function into a simple gradient descent optimization loop to see its behavior in a learning task.
-   Implement a version that computes the sum of losses/gradients directly on the GPU using reduction techniques.
