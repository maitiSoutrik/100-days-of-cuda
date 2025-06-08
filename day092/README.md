# Day 092: Exponential Linear Unit (ELU) Activation Function

## Overview
This project implements the Exponential Linear Unit (ELU) activation function, a common activation function used in neural networks. ELU aims to address some of the shortcomings of the ReLU (Rectified Linear Unit) activation function, such as the "dying ReLU" problem and the non-zero mean activation issue, by allowing negative values and pushing mean activations closer to zero.

The ELU function is defined as:
- `f(x) = x` if `x > 0`
- `f(x) = α * (exp(x) - 1)` if `x <= 0`

Where `α` (alpha) is a positive hyperparameter, typically set to 1.0.

This implementation includes:
- A CUDA kernel for parallel ELU computation on the GPU.
- A CPU version for comparison and verification.
- A main executable to benchmark the GPU vs. CPU performance and verify results.
- Google Tests to ensure correctness of both implementations.

## Implementation Details

### `elu_activation.cuh`
- Declares the CUDA error checking macro (`CHECK_CUDA_ERROR`).
- Declares the GPU kernel wrapper `elu_activation_kernel_wrapper()`.
- Declares the main GPU host function `elu_activation_gpu()` which handles memory management and kernel launch.
- Declares the CPU implementation `elu_activation_cpu()`.

### `elu_activation.cu`
- Defines the `elu_kernel()`: A CUDA `__global__` function that computes ELU for each element of an input array in parallel. Each thread processes one element.
- Defines `elu_activation_kernel_wrapper()`: A C++ function that calculates grid and block dimensions and launches `elu_kernel()`.
- Defines `elu_activation_gpu()`: Manages CUDA memory allocation (`cudaMalloc`), data transfers (`cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`), kernel execution, and memory deallocation (`cudaFree`).
- Defines `elu_activation_cpu()`: A straightforward C++ loop that computes ELU on the CPU for each element.

### `elu_activation_main.cu`
- Initializes a large array of random floating-point numbers.
- Runs the ELU activation on this data using both the CPU and GPU implementations.
- Measures and prints the execution time for both versions.
- Compares the results from CPU and GPU for verification, checking for differences within a small tolerance.
- Prints a speedup factor of GPU over CPU.
- Allows optional command-line arguments for the number of elements and the alpha value.

### `elu_activation_test.cu`
- Contains Google Test cases to verify the correctness of:
    - `elu_activation_cpu()` with positive, negative, zero, and mixed input values, and different alpha values.
    - `elu_activation_gpu()` by comparing its output against `elu_activation_cpu()` for various small and large datasets and alpha values.

### `CMakeLists.txt`
- Sets the project name and language (CUDA, CXX).
- Specifies `CMAKE_CUDA_ARCHITECTURES 53` for Jetson Nano compatibility.
- Sets C++ and CUDA standards to 14.
- Includes optimization flags (`-O3`).
- Defines a static library `elu_activation_lib` containing `elu_activation.cu`.
- Defines an executable `elu_activation_main` linked against `elu_activation_lib`.
- Defines a test executable `elu_activation_test` linked against `elu_activation_lib` and GoogleTest, if found.

## Key CUDA Concepts Used
- **CUDA Kernels (`__global__`)**: `elu_kernel` is executed on the GPU by many threads in parallel.
- **Thread Hierarchy**: `blockIdx`, `blockDim`, `threadIdx` are used for thread identification and data partitioning.
- **Device Memory Management**: `cudaMalloc`, `cudaMemcpy`, `cudaFree` for managing memory on the GPU.
- **Error Handling**: `CHECK_CUDA_ERROR` macro for robust error checking of CUDA API calls.
- **Kernel Launch**: `<<<blocksPerGrid, threadsPerBlock>>>` syntax for launching kernels.
- **Device Synchronization**: `cudaDeviceSynchronize` to ensure kernel completion before accessing results.
- **`expf()`**: The single-precision exponential function available in CUDA device code.

## Performance Considerations (Jetson Nano Focus)
- The ELU function involves a conditional check and potentially an `expf()` call, which is more computationally intensive than simple arithmetic operations like those in ReLU.
- For the Jetson Nano (Compute Capability 5.3), the number of Streaming Multiprocessors (SMs) and cores per SM is limited. Efficiently utilizing these resources is key.
- The choice of `threadsPerBlock` (e.g., 256) is a common starting point but could be tuned for optimal occupancy on sm_53.
- Memory transfers between host and device (`cudaMemcpy`) are significant overheads. The benefits of GPU acceleration are typically seen when the computation is substantial enough to outweigh these transfer costs. For ELU, which is element-wise, the speedup will depend heavily on the size of the input data (`N`).
- The CPU comparison is single-threaded. A multi-threaded CPU implementation would provide a more challenging baseline.

## Building and Running

### Prerequisites
- NVIDIA CUDA Toolkit (compatible with Jetson Nano, e.g., CUDA 10.2 or later).
- CMake (version 3.10 or higher).
- A C++ compiler (like g++).
- Google Test libraries (for building and running tests).

### Building
1.  Navigate to the root of the `100-days-of-cuda` project directory.
2.  If you haven't already, add `day092` to the main `CMakeLists.txt`:
    ```cmake
    add_subdirectory(day092)
    ```
3.  Create a build directory and navigate into it:
    ```bash
    mkdir build
    cd build
    ```
4.  Run CMake and build:
    ```bash
    cmake .. 
    make -j$(nproc) # Or simply 'make'
    ```
    Ensure that CMake is configured to use the correct CUDA compiler and target architecture if building directly on the Jetson Nano or cross-compiling.

### Running the Benchmark
Navigate to the build directory for `day092` (e.g., `build/day092/`):
```bash
./elu_activation_main [num_elements] [alpha]
```
Example:
```bash
./elu_activation_main          # Uses default N and alpha
./elu_activation_main 1000000 0.9 # Runs with 1 million elements and alpha=0.9
```

### Running Tests
Navigate to the build directory for `day092` (e.g., `build/day092/`):
```bash
./elu_activation_test
```
Or run all tests from the main build directory using CTest:
```bash
ctest --output-on-failure
```

## Execution Results / Output

**`elu_activation_main` Output (Jetson Nano):**
```
drboom@JetNano ~/g/1/build> ./day092/elu_activation_main 
Day 092: ELU Activation Function Benchmark
-----------------------------------------
Number of elements (N): 16777216
Alpha (α): 1
Sample Input Data (first 5 elements): [-1.5715, -1.9032, 3.6511, -0.8037, -2.5128...]

CPU ELU execution time: 306.017 ms
Sample CPU Output (first 5 elements): [-0.7923, -0.8509, 3.6511, -0.5523, -0.9190...]
GPU ELU execution time: 175.879 ms
Sample GPU Output (first 5 elements): [-0.7923, -0.8509, 3.6511, -0.5523, -0.9190...]

Verification Results:
  PASSED: CPU and GPU results match within tolerance (0.0000).

GPU Speedup over CPU: 1.74x
-----------------------------------------
```

**`elu_activation_test` Output (Jetson Nano):**
```
drboom@JetNano ~/g/1/build> ./day092/elu_activation_test
[==========] Running 7 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 7 tests from ELUActivationTest
[ RUN      ] ELUActivationTest.CPU_PositiveValues
[       OK ] ELUActivationTest.CPU_PositiveValues (0 ms)
[ RUN      ] ELUActivationTest.CPU_NegativeValues
[       OK ] ELUActivationTest.CPU_NegativeValues (0 ms)
[ RUN      ] ELUActivationTest.CPU_ZeroValue
[       OK ] ELUActivationTest.CPU_ZeroValue (0 ms)
[ RUN      ] ELUActivationTest.CPU_MixedValues_AlphaOne
[       OK ] ELUActivationTest.CPU_MixedValues_AlphaOne (0 ms)
[ RUN      ] ELUActivationTest.CPU_MixedValues_AlphaHalf
[       OK ] ELUActivationTest.CPU_MixedValues_AlphaHalf (0 ms)
[ RUN      ] ELUActivationTest.GPU_vs_CPU_SmallRandomData
[       OK ] ELUActivationTest.GPU_vs_CPU_SmallRandomData (71 ms)
[ RUN      ] ELUActivationTest.GPU_vs_CPU_LargeRandomData_AlphaVaries
[       OK ] ELUActivationTest.GPU_vs_CPU_LargeRandomData_AlphaVaries (2 ms)
[----------] 7 tests from ELUActivationTest (73 ms total)

[----------] Global test environment tear-down
[==========] 7 tests from 1 test suite ran. (73 ms total)
[  PASSED  ] 7 tests.
```

## Learnings and Observations
- ELU provides an alternative to ReLU that can help with training dynamics by allowing negative activations.
- The `expf()` function is a key component for negative inputs in ELU and has a performance cost compared to simpler arithmetic.
- For element-wise operations like activation functions, achieving significant speedup on the GPU requires large datasets to overcome memory transfer overhead and fully utilize the GPU's parallelism.
- Proper error checking (`CHECK_CUDA_ERROR`) is crucial for debugging CUDA applications.
- Google Tests are invaluable for verifying the correctness of both CPU and GPU implementations across various scenarios.

## (Optional) Future Improvements
- Implement ELU using CUDA C++ templates to support different data types (e.g., `double`).
- Explore performance implications of different `alpha` values.
- Compare performance against other activation functions (ReLU, Leaky ReLU, SELU) on the Jetson Nano.
- Investigate shared memory usage for potential (though likely minor for this element-wise op) optimization if ELU were part of a more complex fused kernel.
