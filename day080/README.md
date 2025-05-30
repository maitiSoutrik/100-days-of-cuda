# Day 80: Matrix Multiplication with Swish Activation and Scaling

## Overview
This project implements matrix multiplication (C = A * B) on the GPU using CUDA, followed by an element-wise Swish activation function and a scaling factor applied to the result. The goal is to demonstrate a common sequence of operations found in neural network layers, optimized for parallel execution on NVIDIA GPUs.

## Implementation Details
The core logic is encapsulated in the `matrix_mul_swish_scale_kernel` CUDA kernel.
- **Matrix Multiplication**: A tiled matrix multiplication approach is used. Each thread block calculates a tile of the output matrix C. Shared memory (`sA` and `sB`) is utilized to cache tiles of matrices A and B, reducing global memory accesses and improving performance. The tile dimension is set to 16x16.
- **Scaling**: After the matrix multiplication for an element of C is computed, a `scale` factor is multiplied with the result.
- **Swish Activation**: The Swish activation function, defined as `f(x) = x * sigmoid(beta * x)`, is then applied to the scaled result. The `beta` parameter for Swish is configurable. The `expf()` function is used for the exponential calculation with single-precision floats.
- **Host Function**: `matrix_mul_swish_scale` is the host-side C++ function that sets up the kernel launch configuration (grid and block dimensions) and calls the kernel.

The `benchmark_main.cu` file handles:
- Initialization of host matrices A and B with random floating-point values.
- Allocation of memory on the GPU for A, B, and C.
- Transfer of input matrices from host to device.
- Execution of the `matrix_mul_swish_scale` function, with timing using `std::chrono`.
- Transfer of the result matrix C from device to host.
- Printing a sample of the output matrix and the execution time.
- Memory deallocation.

## Key CUDA Concepts
- **Kernel Launch**: Launching `matrix_mul_swish_scale_kernel` with appropriate grid and block dimensions.
- **Shared Memory**: Using `__shared__` memory (`sA`, `sB`) for tiling in matrix multiplication to improve data reuse and reduce global memory latency.
- **Thread Hierarchy**: Utilizing `blockIdx`, `blockDim`, and `threadIdx` for mapping threads to matrix elements and data tiles.
- **Synchronization**: Using `__syncthreads()` to ensure correct loading and computation within shared memory tiles.
- **CUDA Error Handling**: Using the `CHECK_CUDA_ERROR` macro for robust error checking of CUDA API calls.
- **Device Memory Management**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
- **Element-wise Operations**: Applying scaling and Swish activation to each element of the intermediate result.

## Performance Considerations
- **Tiled Matrix Multiplication**: This is a standard optimization for matrix multiplication on GPUs. It significantly reduces global memory bandwidth requirements by promoting data reuse from faster shared memory.
- **Coalesced Memory Access**: While not explicitly detailed for every access, the way tiles are loaded into shared memory and accessed by threads within a warp should generally lead to coalesced global memory accesses.
- **Instruction-Level Parallelism**: The Swish activation and scaling are simple arithmetic operations that can be efficiently executed by the CUDA cores.
- **Occupancy**: The choice of `threadsPerBlock(16, 16)` (256 threads) is a common starting point. Optimal block size can depend on the specific GPU architecture and kernel resource usage.
- **`expf()` Cost**: The `expf` function in the Swish activation is computationally more intensive than simpler activations like ReLU.

## Building and Running
To build and run the project:
1.  Navigate to the `day080` directory.
2.  Create a build directory and change into it:
    ```bash
    mkdir build
    cd build
    ```
3.  Run CMake to configure the project:
    ```bash
    cmake ..
    ```
4.  Compile the project:
    ```bash
    make
    ```
5.  Run the benchmark executable:
    ```bash
    ./day080_benchmark
    ```

## Unit Testing
Unit tests are implemented using the Google Test framework to verify the correctness of the CUDA kernels.
- The test file is located at `tests/swish_matrix_ops_test.cu`.
- Tests cover:
    - The `swish_activation_kernel` with various input values.
    - The `matrix_mul_swish_scale_kernel` with small matrices, comparing results against a CPU implementation.

To run the tests, after building the project (steps 1-4 in "Building and Running"):
1.  From the `build` directory, you can run the tests directly:
    ```bash
    ./day080_tests
    ```
2.  Alternatively, use CTest to discover and run all registered tests:
    ```bash
    ctest
    ```
    (You might need `ctest --verbose` for more detailed output).

All tests passed successfully on the Jetson Nano, as shown below:
```
drboom@JetNano ~/g/1/build> ./day080/day080_tests 
[==========] Running 3 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 1 test from SwishActivationKernelTest
[ RUN      ] SwishActivationKernelTest.BasicValues
[       OK ] SwishActivationKernelTest.BasicValues (97 ms)
[----------] 1 test from SwishActivationKernelTest (97 ms total)

[----------] 2 tests from MatrixMulSwishScaleKernelTest
[ RUN      ] MatrixMulSwishScaleKernelTest.SmallMatrixIdentity
[       OK ] MatrixMulSwishScaleKernelTest.SmallMatrixIdentity (1 ms)
[ RUN      ] MatrixMulSwishScaleKernelTest.SmallMatrixSpecificValues
[       OK ] MatrixMulSwishScaleKernelTest.SmallMatrixSpecificValues (1 ms)
[----------] 2 tests from MatrixMulSwishScaleKernelTest (2 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 2 test suites ran. (100 ms total)
[  PASSED  ] 3 tests.
```

## Execution Results
The benchmark program (`./day080_benchmark`) output on a Jetson Nano is as follows:
```
drboom@JetNano ~/g/1/build> ./day080/day080_benchmark 
Day 80: Matrix Multiplication with Swish Activation and Scaling
Matrix dimensions: A(256x256), B(256x256), C(256x256)
Scale factor: 1, Swish beta: 1

CUDA Kernel Execution Time: 14.778 ms

Result Matrix C (first 5x5 sample):
-0.0002 0.5950  1.3120  2.0134  -0.0112
1.3842  -0.2386 2.0061  1.7367  -0.2733
-0.2749 3.6320  1.1430  1.4830  -0.0734
-0.0117 3.1145  12.1504 4.5041  -0.1319
-0.1849 0.4170  -0.0059 -0.2773 0.7026

Benchmark completed successfully.
```
The execution time for a 256x256 matrix multiplication with Swish activation and scaling was approximately 14.778 ms on the Jetson Nano. The output matrix sample shows the computed values.

## Learnings and Observations
- Implementing tiled matrix multiplication with shared memory is crucial for achieving good performance in CUDA.
- Integrating custom activation functions like Swish directly into compute kernels can be efficient.
- The structure of CUDA programs involves careful management of host and device memory, kernel launch parameters, and synchronization.
- The `beta` parameter in Swish allows for tuning the shape of the activation function, potentially impacting learning dynamics in a neural network.

## Future Improvements
- Expand unit tests with more edge cases and larger matrix sizes for the `matrix_mul_swish_scale_kernel`.
- Compare performance with cuBLAS for matrix multiplication and a separate kernel for Swish.
- Experiment with different tile sizes and block dimensions to optimize for specific GPU architectures.
- Add a CPU-based implementation for comparison and verification.
