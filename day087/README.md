# Day 087: Softplus Activation Function

## Overview

This project implements the Softplus activation function using CUDA C++. The Softplus function is a smooth approximation of the ReLU (Rectified Linear Unit) activation function and is defined as:

`Softplus(x) = log(1 + exp(x))`

The project includes:
- A CUDA kernel for element-wise Softplus computation on a tensor.
- A CPU implementation for verification.
- A main executable to demonstrate usage, compare performance (GPU vs. CPU), and verify results.
- Google Tests for unit testing the Softplus implementation.

## Implementation Details

### CUDA Kernel: `softplusKernel`
- Located in `softplus_activation.cu`.
- Takes an input tensor, an output tensor, and the number of elements `N` as arguments.
- Each thread computes `logf(1.0f + expf(input[idx]))` for its assigned element.
- Standard grid-stride loop pattern is used for thread indexing.

### Wrapper Function: `softplusActivation`
- Located in `softplus_activation.cu`.
- Manages CUDA kernel launch configuration (block and grid dimensions).
- Includes CUDA error checking using the `CHECK_CUDA_ERROR` macro (defined in `softplus_activation.cuh`).

### CPU Implementation: `softplusActivationCPU`
- Located in `softplus_activation.cu` (can also be in `.cuh` if preferred for header-only CPU part).
- A simple loop that iterates through the input tensor and applies `std::log(1.0f + std::exp(input[i]))`. Used for verifying the correctness of the GPU implementation.

### Error Checking
- The `CHECK_CUDA_ERROR` macro is used after CUDA API calls and kernel launches to detect and report errors.

## Key CUDA Concepts Used

- **CUDA Kernels (`__global__`)**: For parallel computation on the GPU.
- **Thread Indexing (`blockIdx.x`, `blockDim.x`, `threadIdx.x`)**: Standard method for mapping threads to data elements.
- **Device Memory Management**: `cudaMalloc` for allocating memory on the GPU, `cudaMemcpy` for transferring data between host and device, and `cudaFree` for deallocating GPU memory.
- **CUDA Events (`cudaEvent_t`)**: Used for accurately measuring the execution time of the CUDA kernel.
- **Error Handling (`cudaError_t`, `cudaGetErrorString`)**: Essential for robust CUDA programming.

## Performance Considerations (Jetson Nano - sm_53)

- **Mathematical Intensity**: The `expf` and `logf` functions are relatively expensive. For large datasets, the parallelism offered by the GPU significantly outweighs this cost compared to a serial CPU implementation.
- **Memory Bandwidth**: For element-wise operations like Softplus, performance can be memory-bandwidth bound, especially if the computation per element is not very high. The Jetson Nano has limited memory bandwidth compared to discrete GPUs.
- **Kernel Launch Overhead**: For very small `N`, the overhead of launching a kernel might dominate the execution time. The chosen `N = 1 << 20` in `softplus_activation_main.cu` is large enough to demonstrate GPU benefits.
- **Numerical Stability**: The direct computation `logf(1.0f + expf(x))` can suffer from overflow if `expf(x)` is very large, or loss of precision if `x` is very negative (where `expf(x)` becomes very small).
    - For large positive `x`, `log(1 + exp(x)) ≈ log(exp(x)) = x`.
    - For large negative `x`, `log(1 + exp(x)) ≈ exp(x)`.
    - The current implementation uses the direct form, which is generally acceptable for typical float ranges but might need specialized handling for extreme values or higher precision requirements. For `sm_53`, `float` precision is standard.

## Building and Running

### Prerequisites
- NVIDIA CUDA Toolkit (compatible with sm_53)
- CMake (version 3.18 or higher)
- A C++ compiler (like g++)
- Google Test library (for running tests)

### Build Instructions
1.  Navigate to the `day087` directory:
    ```bash
    cd day087
    ```
2.  Create a build directory and navigate into it:
    ```bash
    mkdir build && cd build
    ```
3.  Run CMake and build:
    ```bash
    cmake ..
    make
    ```
    This will build the `softplus_activation_main` executable and the `softplus_activation_test` test executable.

### Running the Main Program
Execute the main program from the `build` directory:
```bash
./softplus_activation_main
```
This will run the Softplus activation on a sample dataset, compare GPU and CPU execution times, and verify the results.

### Running Tests
Execute the tests from the `build` directory:
```bash
./softplus_activation_test
```

## Execution Results / Output

**`softplus_activation_main` Output:**
```
drboom@JetNano ~/g/1/build> ./day087/softplus_activation_main 
Softplus Activation Function Demo
Number of elements: 1048576
Data size: 4 MB
GPU Execution Time: 7.46458 ms
CPU Execution Time: 50.935 ms
Verification successful: CPU and GPU results match.

Example values (Input -> CPU Output | GPU Output):
Input:   6.764567 -> CPU:   6.765720 | GPU:   6.765720
Input:   8.413902 -> CPU:   8.414124 | GPU:   8.414124
Input:   0.317825 -> CPU:   0.864634 | GPU:   0.864634
Input:  -4.065935 -> CPU:   0.017002 | GPU:   0.017002
Input:   5.286551 -> CPU:   5.291597 | GPU:   5.291597
Input:   8.991934 -> CPU:   8.992058 | GPU:   8.992058
Input:  -4.476978 -> CPU:   0.011304 | GPU:   0.011304
Input:  -0.703960 -> CPU:   0.401874 | GPU:   0.401874
Input:  -1.943126 -> CPU:   0.133880 | GPU:   0.133880
Input:   2.606505 -> CPU:   2.677701 | GPU:   2.677701
```

**`softplus_activation_test` Output:**
```
drboom@JetNano ~/g/1/build> ./day087/softplus_activation_test
[==========] Running 5 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 5 tests from SoftplusActivationTest
[ RUN      ] SoftplusActivationTest.HandlesMixedValues
[       OK ] SoftplusActivationTest.HandlesMixedValues (94 ms)
[ RUN      ] SoftplusActivationTest.HandlesAllZeros
[       OK ] SoftplusActivationTest.HandlesAllZeros (1 ms)
[ RUN      ] SoftplusActivationTest.HandlesAllPositive
[       OK ] SoftplusActivationTest.HandlesAllPositive (1 ms)
[ RUN      ] SoftplusActivationTest.HandlesAllNegative
[       OK ] SoftplusActivationTest.HandlesAllNegative (1 ms)
[ RUN      ] SoftplusActivationTest.HandlesSingleElement
[       OK ] SoftplusActivationTest.HandlesSingleElement (1 ms)
[----------] 5 tests from SoftplusActivationTest (100 ms total)

[----------] Global test environment tear-down
[==========] 5 tests from 1 test suite ran. (101 ms total)
[  PASSED  ] 5 tests.
```

## Learnings and Observations

- The Softplus function provides a smooth alternative to ReLU, with its derivative being the sigmoid function.
- Implementing element-wise operations in CUDA is straightforward using a standard kernel structure.
- CUDA events are crucial for accurate GPU performance measurement.
- Verification against a CPU implementation is a key step in ensuring kernel correctness.
- Numerical precision and potential overflow/underflow are important considerations for functions like `exp` and `log`, especially across different hardware and input ranges. The Jetson Nano's `sm_53` primarily uses single-precision floats effectively.

## Future Improvements (Optional)

- Implement a more numerically stable version of Softplus for wider input ranges or higher precision needs.
- Explore performance with different data types (e.g., `double`, `half`).
- Benchmark against other activation functions on the Jetson Nano.
