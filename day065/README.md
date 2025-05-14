# Day 65: GEGLU Activation Function

## Overview

GEGLU (Gaussian Error Gated Linear Unit) is an activation function that combines the **GELU (Gaussian Error Linear Unit)** with a **GLU (Gated Linear Unit)** mechanism. It's particularly effective in Transformer models.

The mathematical formulation is:
`GEGLU(x_split1, x_split2) = GELU(x_split1) ⊗ x_split2`

Where `x_split1` and `x_split2` are typically two different linear transformations of an original input `x`. For example:
`x_split1 = xW1 + b1`
`x_split2 = xW2 + b2`
And `⊗` denotes element-wise multiplication.

**GELU (Gaussian Error Linear Unit):**
GELU weights inputs by their percentile under a Gaussian distribution.
`GELU(x) = x * Φ(x)`, where `Φ(x)` is the CDF of the standard normal distribution.
A common approximation is:
`GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
GELU provides a smooth, non-monotonic activation that can help with gradient flow.

**GLU (Gated Linear Unit) Mechanism:**
The GLU mechanism allows the network to control the flow of information. In GEGLU, the `GELU(x_split1)` part acts as a gate for the `x_split2` part. This selective gating can improve model performance by allowing the network to learn which features are most relevant.

## Implementation Details

The CUDA implementation consists of:
-   A `__device__` function `gelu_approx(float x)` that computes the GELU approximation.
-   A `__global__` kernel `geglu_kernel(const float* input_a, const float* input_b, float* output, int n)` that takes two input arrays (`input_a` corresponds to `x_split1` and `input_b` to `x_split2`).
-   The kernel calculates `output[i] = gelu_approx(input_a[i]) * input_b[i]` for each element.
-   A host wrapper `launch_geglu_kernel` is provided for launching the kernel.
-   The main executable (`geglu_main.cu`) initializes two input arrays, runs the GEGLU kernel, and compares the GPU output with a CPU-based GEGLU computation for verification.
-   Unit tests (`geglu_test.cu`) are provided using Google Test to verify correctness across different input sizes and values.

## Key CUDA Features Used

-   `__global__` kernels for parallel computation.
-   `__device__` functions for helper computations within the GPU.
-   `cudaMalloc` and `cudaFree` for device memory management.
-   `cudaMemcpy` for data transfer between host and device.
-   Thread indexing (`blockIdx.x * blockDim.x + threadIdx.x`) for data parallelism.
-   CUDA error checking macros (`CHECK_CUDA_ERROR`).

## Performance Considerations

-   The GELU approximation involves `tanhf`, `sqrtf`, and several multiplications/additions, making it more computationally intensive per element than simpler activations like ReLU.
-   For element-wise operations like GEGLU, performance is typically memory-bandwidth bound for large inputs.
-   The kernel uses a standard grid-stride loop pattern, suitable for various input sizes.

## Building and Running

1.  **Prerequisites:**
    *   CUDA Toolkit (>= 10.0, compatible with sm_53)
    *   CMake (>= 3.10)
    *   Google Test (GTest) development libraries (usually handled by the root CMakeLists or CI)
    *   A C++14 compatible compiler (like g++)

2.  **Build (from the root project directory `100-days-of-cuda`):**
    ```bash
    mkdir -p build
    cd build
    cmake ..
    make day065_geglu # Or make geglu_benchmark, geglu_test_exec
    ```
    Alternatively, build all targets:
    ```bash
    make
    ```

3.  **Running the Benchmark:**
    ```bash
    ./day065/geglu_benchmark
    ```

4.  **Running Tests:**
    ```bash
    ./day065/geglu_test_exec
    # Or run all tests via ctest
    # ctest --output-on-failure -R day065_geglu # Run tests matching the project name
    ```

## Execution Results

Output from `geglu_benchmark`:
```
--- GEGLU Kernel Verification ---
Problem size (n): 1024
Input A (Host): [0.680375, 0.566198, 0.823295, -0.329554, -0.444451, ...]
Input B (Host): [-0.211234, 0.596880, -0.604897, 0.536459, 0.107940, ...]
Output (GPU): [-0.108047, 0.241407, -0.395777, -0.065569, -0.015754, ...]
Output (CPU Ref): [-0.108047, 0.241407, -0.395777, -0.065569, -0.015754, ...]

Verification Successful: GPU and CPU results match within tolerance.
GEGLU main finished.
```

Output from `ctest` (or direct execution of `geglu_test_exec`):
```
Running main() from /home/drboom/git_repos/100-days-of-cuda/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 4 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 4 tests from GegluTest
[ RUN      ] GegluTest.HandlesZeroSize
[       OK ] GegluTest.HandlesZeroSize (0 ms)
[ RUN      ] GegluTest.SmallInput
[       OK ] GegluTest.SmallInput (96 ms)
[ RUN      ] GegluTest.RandomMediumSizeInput
[       OK ] GegluTest.RandomMediumSizeInput (1 ms)
[ RUN      ] GegluTest.RandomLargeSizeInput
[       OK ] GegluTest.RandomLargeSizeInput (31 ms)
[----------] 4 tests from GegluTest (129 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 1 test suite ran. (129 ms total)
[  PASSED  ] 4 tests.
```

## Learnings and Observations

-   GEGLU combines a sophisticated activation (GELU) with a gating mechanism, offering more expressive power than simpler activations.
-   The implementation of GELU requires careful use of its approximation formula.
-   Testing involves comparing GPU results against a CPU reference, which is crucial for verifying correctness of CUDA kernels.
-   The `.clinerules` provide a good structure for organizing daily CUDA projects with libraries, executables, and tests.

## (Optional) Future Improvements

-   Benchmark performance against a naive CPU implementation for larger N.
-   Explore different GELU approximations if available or needed.
-   Integrate into a small neural network layer.

## (Optional) References

-   Noam Shazeer. "GLU Variants Improve Transformer." arXiv:2002.05202, 2020.
-   Dan Hendrycks and Kevin Gimpel. "Gaussian Error Linear Units (GELUs)." arXiv:1606.08415, 2016.
