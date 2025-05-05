# Day 57: Conjugate Gradient Method (CGM) using cuBLAS

## Overview

This project implements the Conjugate Gradient Method (CGM) to solve a system of linear equations Ax = b, where A is a symmetric positive-definite matrix. Instead of writing custom CUDA kernels for the underlying vector operations (like dot products, vector additions, and matrix-vector multiplications), this implementation leverages the highly optimized NVIDIA cuBLAS library. This approach aims to achieve better performance, especially on resource-constrained devices like the Jetson Nano, by utilizing vendor-tuned routines.

The core CGM algorithm iteratively refines an initial guess for x until the residual norm falls below a specified tolerance or a maximum number of iterations is reached.

## Implementation Details

The implementation consists of:

1.  **`cgm_cublas.cuh`**: Header file defining the `conjugateGradientMethodCuBLAS` function signature and common CUDA/cuBLAS error-checking macros (`CHECK_CUDA_ERROR`, `CHECK_CUBLAS_ERROR`).
2.  **`cgm_cublas.cu`**:
    *   Contains the implementation of `conjugateGradientMethodCuBLAS`. It allocates temporary device vectors for the residual (`r`), search direction (`p`), and the result of `A*p`.
    *   The main loop performs the following steps using cuBLAS functions:
        *   `cublasDgemv`: Computes the matrix-vector product `Ap = A * p`.
        *   `cublasDdot`: Computes dot products like `r' * r` and `p' * Ap`.
        *   `cublasDaxpy`: Performs vector updates like `x = x + alpha * p` and `r = r - alpha * Ap`.
        *   `cublasDscal`: Scales vectors, used in updating `p` (`p = beta * p`).
        *   `cudaMemcpy`: Used for initial setup (e.g., `p = r`).
    *   Includes basic checks for potential division by zero during alpha and beta calculations.
    *   The `main` function sets up a small 4x4 test system, initializes cuBLAS, copies data to the GPU, calls the solver, measures execution time, copies the result back, and performs a simple verification by calculating `||Ax_sol - b||`.
3.  **`cgm_cublas_test.cu`**: Contains a Google Test suite (`CgmCuBLASTest`) to verify the solver's correctness using a known 2x2 system.

Double precision (`double`) is used for all calculations.

## Key CUDA Features Used

*   **CUDA Runtime API**: For memory management (`cudaMalloc`, `cudaFree`, `cudaMemcpy`).
*   **cuBLAS Library (v2 API)**:
    *   `cublasCreate`, `cublasDestroy`: Handle management.
    *   `cublasDgemv`: Level 2 BLAS - Double-precision General Matrix-Vector multiplication.
    *   `cublasDdot`: Level 1 BLAS - Double-precision Dot product.
    *   `cublasDaxpy`: Level 1 BLAS - Double-precision Scalar-Vector multiplication and addition (`y = alpha*x + y`).
    *   `cublasDscal`: Level 1 BLAS - Double-precision Vector scaling (`x = alpha*x`).

## Performance Considerations

*   **cuBLAS Benefits**: Using cuBLAS avoids the need to write, tune, and debug custom kernels for standard linear algebra operations. cuBLAS routines are highly optimized by NVIDIA for various GPU architectures, including the Jetson Nano's Maxwell architecture (Compute Capability 5.3). This typically leads to significantly better performance compared to naive kernel implementations, especially for larger matrices.
*   **Overhead**: For very small matrices (like the 4x4 example), the overhead of launching multiple cuBLAS kernels might be noticeable compared to a single, fused custom kernel (if one were written). However, cuBLAS generally scales much better.
*   **Memory Transfers**: The current `main` function includes host-to-device and device-to-host memory transfers. In a real application, data might already reside on the GPU, reducing this overhead.
*   **Double Precision**: The Jetson Nano has limited double-precision performance compared to its single-precision capabilities. For applications where single precision is sufficient, using `cublasSgemv`, `cublasSdot`, etc., would likely yield higher performance.

*(Performance analysis comparing CPU vs. GPU or different matrix sizes on the Jetson Nano should be added here after execution.)*

## Building and Running

**Prerequisites:**
*   CUDA Toolkit (>= 10.2 recommended for Jetson Nano)
*   CMake (>= 3.18)
*   Google Test (installed or fetched by CMake on the build system)
*   A C++14 compatible compiler (like g++)

**Build Steps (on Jetson Nano or compatible cross-compilation environment):**

1.  Navigate to the root directory (`100-days-of-cuda`).
2.  Create a build directory: `mkdir build && cd build`
3.  Configure using CMake: `cmake ..`
4.  Build the target for Day 57: `make cgm_benchmark cgm_test` (or `make day057_cgm_cublas` if the project name was used directly, or simply `make` to build everything)

**Running:**

1.  Run the benchmark: `./day057/cgm_benchmark`
2.  Run the tests: `ctest --output-on-failure -R cgm_test` or directly `./day057/cgm_test`

## Execution Results

Output from running `./day057/cgm_benchmark` on the Jetson Nano:

```bash
Matrix A (Column-Major) = [
  4, 1, 0, 0;
  1, 4, 1, 0;
  0, 1, 4, 1;
  0, 0, 1, 4;
]
Vector b = [1, 2, 3, 4]
Initial x = [0, 0, 0, 0]

Starting Conjugate Gradient Method (cuBLAS)...
Converged in 4 iterations.
Execution Time: 2.85636 ms
Solution x = [0.162679, 0.349282, 0.440191, 0.889952]
Verification (A*x_sol) = [1, 2, 3, 4]
Norm of difference ||Ax_sol - b||: 0
```

Output from running `./day057/cgm_test` on the Jetson Nano (after CMake fix):

```bash
Running main() from /home/drboom/git_repos/100-days-of-cuda/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from CgmCuBLASTest
[ RUN      ] CgmCuBLASTest.Solves2x2System
[       OK ] CgmCuBLASTest.Solves2x2System (1242 ms)
[----------] 1 test from CgmCuBLASTest (1242 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (1243 ms total)
[  PASSED  ] 1 test.
```

## Learnings and Observations

*   cuBLAS significantly simplifies the implementation of algorithms involving standard linear algebra operations.
*   Understanding the expected data layout (column-major for Fortran-style BLAS/LAPACK) is crucial when using cuBLAS.
*   The CGM algorithm involves several sequential steps (matrix-vector multiply, dot products, vector updates), limiting parallelism compared to purely element-wise operations. Performance relies heavily on the efficiency of the underlying cuBLAS calls.
*   Error checking for both CUDA runtime and cuBLAS calls is essential for debugging.

## References

*   Wikipedia: Conjugate Gradient Method - [https://en.wikipedia.org/wiki/Conjugate_gradient_method](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
*   NVIDIA cuBLAS Documentation - [https://docs.nvidia.com/cuda/cublas/index.html](https://docs.nvidia.com/cuda/cublas/index.html)
