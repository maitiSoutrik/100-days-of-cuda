# Day 90: Frobenius Norm Calculation in CUDA

## Overview
This project implements the calculation of the Frobenius norm for a matrix using CUDA C++. The Frobenius norm, often denoted as \( \|A\|_F \), is the square root of the sum of the squares of its elements. It's equivalent to the Euclidean norm (L2 norm) of the matrix if it were treated as a single vector. This project provides both GPU and CPU implementations for comparison.

## Implementation Details
The Frobenius norm is calculated as:
\[ \|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n |a_{ij}|^2} \]

**GPU Implementation (`frobeniusNormGPU`):**
1.  **Memory Allocation**: Device memory is allocated for the input matrix and for an array to store partial sums from each thread block.
2.  **Kernel Launch (`sumOfSquaresReductionKernel`)**:
    *   A CUDA kernel is launched where each thread block processes a portion of the matrix.
    *   Each thread squares an element of the matrix.
    *   These squared values are then reduced (summed) within each thread block using shared memory to produce a partial sum of squares for that block.
    *   The `blockDim.x` is chosen as 256, a common choice that balances parallelism and resource usage. Shared memory size is `threads_per_block * sizeof(float)`.
3.  **Partial Sums Aggregation**:
    *   The partial sums from each block are copied back to the host.
    *   These partial sums are then summed up on the CPU to get the total sum of squares.
    *   *Note*: For a very large number of blocks, a second-level reduction kernel on the GPU would be more efficient than copying all partial sums to the host. This implementation uses host-side aggregation for simplicity, assuming the number of blocks isn't excessively large.
4.  **Final Calculation**: The square root of the total sum of squares is computed on the CPU to get the Frobenius norm.

**CPU Implementation (`frobeniusNormCPU`):**
*   A simple loop iterates through all elements of the matrix.
*   Each element is squared and added to a running sum (using `double` for precision).
*   The square root of the final sum is taken.

## Key CUDA Features Used
*   **CUDA Kernels**: `__global__` functions for parallel computation.
*   **Shared Memory**: `__shared__` memory for efficient intra-block reduction. Each block calculates a partial sum of squares.
*   **Atomic Operations**: `atomicAdd` could be used for a simpler (though potentially slower for this specific sum-of-squares task if not carefully managed) global sum, but a shared memory reduction is generally preferred for performance in this pattern. The current `sumOfSquaresReductionKernel` uses shared memory reduction. An alternative `sumOfSquaresKernel` using `atomicAdd` is also provided in `frobenius_norm.cu` but not used by default in `frobeniusNormGPU` for better performance.
*   **Thread Synchronization**: `__syncthreads()` to ensure correct order of operations during shared memory reduction.
*   **CUDA Events**: Used for timing the GPU execution.
*   **Error Handling**: `CHECK_CUDA_ERROR` macro for robust error checking.

## Performance Considerations
*   **Shared Memory Reduction**: The use of shared memory for reduction within each block significantly reduces global memory traffic compared to naive atomic operations on a global sum variable, especially for larger matrices.
*   **Number of Blocks**: The number of blocks is determined by `(total_elements + threads_per_block - 1) / threads_per_block`. If this number is large, copying all partial sums to the host for final aggregation can become a bottleneck. A multi-level GPU reduction would be more scalable.
*   **Data Type for Summation**: Using `double` for the sum in the CPU version helps maintain precision, especially for large matrices or matrices with large values. The GPU version uses `float` for sums, which is typical for many CUDA applications but can lead to precision differences compared to a `double`-based CPU sum for very large sums.
*   **Memory Coalescing**: The kernel accesses `d_matrix[i]` where `i` is `blockIdx.x * blockDim.x + threadIdx.x`. This ensures coalesced memory access as threads within a warp access contiguous memory locations.

## Building and Running
The project uses CMake. To build and run (on a system with CUDA Toolkit, CMake, and Google Test installed, typically the Jetson Nano or a configured build environment):

1.  **Create a build directory and navigate into it:**
    ```bash
    mkdir build
    cd build
    ```
2.  **Run CMake to configure the project (from within the `build` directory of `day090` or the root `build` if building all days):**
    ```bash
    # If in day090/build:
    cmake .. 
    # If in root_project/build:
    # cmake .. (ensure the root CMakeLists.txt adds day090)
    ```
3.  **Compile the project:**
    ```bash
    make
    ```
4.  **Run the main executable:**
    ```bash
    ./frobenius_norm_main
    ```
5.  **Run the tests:**
    ```bash
    ./frobenius_norm_test
    # Or using ctest if configured:
    # ctest
    ```

## Execution Results
The `frobenius_norm_main` executable will output:
*   The Frobenius norm calculated by the GPU.
*   The time taken for the GPU calculation.
*   The Frobenius norm calculated by the CPU.
*   The time taken for the CPU calculation.
*   A verification status (PASSED/FAILED) comparing the CPU and GPU results.

Actual output from Jetson Nano:
```
Frobenius Norm (GPU): 5914.069824
GPU Calculation Time: 3.025364 ms
Frobenius Norm (CPU): 5914.067871
CPU Calculation Time: 3.079000 ms
Verification: PASSED
```

The `frobenius_norm_test` executable also passed all 6 tests successfully on the Jetson Nano:
```
[==========] Running 6 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 6 tests from FrobeniusNormTest
[ RUN      ] FrobeniusNormTest.HandlesEmptyMatrix
[       OK ] FrobeniusNormTest.HandlesEmptyMatrix (0 ms)
[ RUN      ] FrobeniusNormTest.HandlesSingleElementMatrix
[       OK ] FrobeniusNormTest.HandlesSingleElementMatrix (79 ms)
[ RUN      ] FrobeniusNormTest.SmallMatrixVerification
[       OK ] FrobeniusNormTest.SmallMatrixVerification (1 ms)
[ RUN      ] FrobeniusNormTest.LargerMatrixVerification
[       OK ] FrobeniusNormTest.LargerMatrixVerification (3 ms)
[ RUN      ] FrobeniusNormTest.RowVector
[       OK ] FrobeniusNormTest.RowVector (1 ms)
[ RUN      ] FrobeniusNormTest.ColumnVector
[       OK ] FrobeniusNormTest.ColumnVector (1 ms)
[----------] 6 tests from FrobeniusNormTest (87 ms total)

[----------] Global test environment tear-down
[==========] 6 tests from 1 test suite ran. (87 ms total)
[  PASSED  ] 6 tests.
```

## Learnings and Observations
*   The reduction pattern using shared memory is a common and effective technique in CUDA for parallel summation or similar operations.
*   Careful management of block and grid dimensions is crucial for performance.
*   For high-precision requirements, especially with large sums, using `double` on the GPU (if supported and necessary) or careful error analysis for `float` operations would be important. This implementation prioritizes common `float` usage.
*   The overhead of copying partial sums back to the host can be significant if not managed. For truly large-scale reductions, a full GPU-based reduction (e.g., using multiple kernel launches or more advanced library functions like those in CUB or Thrust) is preferred.
*   The Jetson Nano's compute capability (5.3) supports shared memory and atomic operations effectively for this kind of task.

## Future Improvements
*   Implement a multi-level GPU reduction to avoid copying many partial sums to the host.
*   Explore using CUDA library functions (e.g., from cuBLAS `cublasSnrm2` if treating the matrix as a vector, or Thrust library for reduction) for potentially more optimized implementations.
*   Add support for `double` precision calculations on the GPU.
