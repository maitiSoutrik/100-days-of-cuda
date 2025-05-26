# Day 78: Dynamic Tanh (DyT) Operation

## Overview
This project implements the Dynamic Tanh (DyT) activation function in CUDA C++. DyT is a variation of the standard hyperbolic tangent (Tanh) function that includes learnable parameters, typically denoted as `alpha` and `beta`. These parameters allow the activation function's shape (amplitude and slope) to be adjusted dynamically during the training of a neural network, potentially leading to better model performance.

The implementation is structured as follows:
- `dyt.cuh`: Header file defining the CUDA kernel prototypes and the `CHECK_CUDA_ERROR` macro.
- `dyt_core.cu`: Source file containing the CUDA kernel implementations for the DyT forward and backward passes.
- `dyt_main.cu`: Source file with a `main` function to demonstrate the DyT operation with sample data.
- `dyt_test.cu`: Source file containing Google Test unit tests for verifying the correctness of the forward and backward passes.

Atomic operations are used in the backward pass for safe accumulation of gradients for `alpha` and `beta` across multiple threads.

## Implementation Details
The DyT function is defined as:
`DyT(x) = alpha * tanh(beta * x)`

**Kernels (in `dyt_core.cu`):**
- **Forward Pass (`dyt_forward_kernel`):** Computes `y = alpha * tanhf(beta * x)` for each element `x`.
- **Backward Pass (`dyt_backward_kernel`):** Computes gradients:
    - `dL/dx = upstream_grad * alpha * beta * (1 - tanh^2(beta * x))`
    - `dL/d_alpha = upstream_grad * tanh(beta * x)` (accumulated with `atomicAdd`)
    - `dL/d_beta = upstream_grad * alpha * x * (1 - tanh^2(beta * x))` (accumulated with `atomicAdd`)

**Main Demo (`dyt_main.cu`):**
Initializes sample input data `x` and upstream gradients `dL/dy` (set to 1.0f for simplicity). It allocates GPU memory, copies data, launches kernels, and displays results.

**Unit Tests (`dyt_test.cu`):**
Uses Google Test to verify the numerical correctness of the forward and backward pass kernels against manually calculated expected values for a small dataset.

## Key CUDA Features Used
- **CUDA Kernels**: `__global__` functions in `dyt_core.cu`.
- **Header/Source Separation**: `dyt.cuh` for declarations, `dyt_core.cu` for implementations.
- **`tanhf`**: Single-precision hyperbolic tangent.
- **`atomicAdd`**: For safe gradient accumulation for scalar parameters `alpha` and `beta`.
- **CUDA Memory Management**: `cudaMalloc`, `cudaMemcpy`, `cudaMemset`, `cudaFree`.
- **Error Handling**: `CHECK_CUDA_ERROR` macro.
- **Google Test**: For unit testing CUDA kernels.

## Building and Running
To build and run the project (assuming you are in the `build` directory relative to the project root, and the Jetson environment is set up):

1.  **Configure CMake** (from the project root, if `day078` has been added to the main `CMakeLists.txt`):
    ```bash
    cd build  # If not already there
    cmake .. 
    ```
    Or, to build only this day's project (navigate to `day078` directory first):
    ```bash
    mkdir build
    cd build
    cmake ..
    ```

2.  **Build the executables**:
    ```bash
    make dyt_demo dyt_test
    # Or simply 'make' if building from day078/build
    ```

3.  **Run the demonstration executable**:
    ```bash
    ./day078/dyt_demo 
    # Or './dyt_demo' if in day078/build
    ```

4.  **Run the tests**:
    ```bash
    ./day078/dyt_test
    # Or 'ctest' or './dyt_test' if in day078/build
    ```

## Execution Results

**Demonstration Output (`dyt_demo`):**
The program will output:
- The parameters used (n, alpha, beta).
- Results from the forward pass (first 5 elements of `x` and `y`).
- Results from the backward pass (first 5 elements of `x_grad`, and the total `alpha_grad` and `beta_grad`).

```
Dynamic Tanh (DyT) Operation - Main Demo
Parameters: n = 1024, alpha = 1.500000, beta = 0.500000

--- Running Forward Pass ---
Forward Pass Results (first 5 elements):
x[0] = -2.000000, y[0] = -1.142391
x[1] = -1.800000, y[1] = -1.074447
x[2] = -1.600000, y[2] = -0.996055
x[3] = -1.400000, y[3] = -0.906552
x[4] = -1.200000, y[4] = -0.805574

--- Running Backward Pass ---

Backward Pass Results:
First 5 x_grad elements:
x_grad[0] = 0.314981
x_grad[1] = 0.365188
x_grad[2] = 0.419291
x_grad[3] = 0.476055
x_grad[4] = 0.533683
Total alpha_grad = -41.587597
Total beta_grad  = -69.505363

DyT operation demo completed.
```

**Test Output (`dyt_test`):**
The Google Test executable will report the status of the tests.
```
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from DyTTest
[ RUN      ] DyTTest.ForwardPass
[       OK ] DyTTest.ForwardPass (95 ms)
[ RUN      ] DyTTest.BackwardPass
[       OK ] DyTTest.BackwardPass (1 ms)
[----------] 2 tests from DyTTest (97 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (97 ms total)
[  PASSED  ] 2 tests.
```

## Learnings and Observations
- Refactoring into a library, main demo, and test files improves organization and reusability, aligning with standard C++ and CUDA project practices.
- `atomicAdd` is crucial for correct gradient accumulation for shared parameters.
- Unit testing (via Google Test) helps ensure the numerical correctness of CUDA kernels.
- The `.clinerules` provide a consistent structure that should be followed for maintainability.

## Future Improvements
- Integrate this DyT operation into a small neural network layer.
- Compare performance and convergence with standard Tanh or other activation functions.
- Implement learnable `alpha` and `beta` per-channel or per-neuron instead of global scalars.
