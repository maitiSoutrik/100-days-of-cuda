# Day 36: Sparse Matrix-Vector Multiplication (SpMV) with cuSPARSE

## Introduction

This example revisits Sparse Matrix-Vector Multiplication (SpMV), previously implemented with custom CUDA kernels in Day 10. Here, we leverage the NVIDIA cuSPARSE library, which provides highly optimized routines for sparse linear algebra operations. The goal is to compare the performance and ease of use of the library approach against the custom implementation, especially for larger matrices.

We perform the `y = A * x` operation, where `A` is a large sparse matrix (defaulting to 20000x20000 with 1% sparsity) stored in Compressed Sparse Row (CSR) format, and `x` and `y` are dense vectors.

## Implementation Details

1.  **Sparse Matrix Generation:** A random sparse matrix `A` is generated on the host in CSR format, similar to Day 10 but with enhancements for potentially larger sizes and memory management.
2.  **cuSPARSE Initialization:**
    *   A cuSPARSE handle (`cusparseHandle_t`) is created using `cusparseCreate()`. This handle manages the library context.
    *   A matrix descriptor (`cusparseMatDescr_t`) is created using `cusparseCreateMatDescr()`. This descriptor informs cuSPARSE about the properties of the matrix being used (e.g., general type, zero-based indexing via `CUSPARSE_INDEX_BASE_ZERO`).
3.  **Memory Management:**
    *   Device memory is allocated using `cudaMalloc()` for the CSR arrays (`d_row_offsets`, `d_col_indices`, `d_values`) and the dense vectors (`d_x`, `d_y`).
    *   Host data (matrix `A` and vector `x`) is copied to the device using `cudaMemcpy()`.
    *   Error checking macros (`CHECK_CUDA_ERROR`, `CHECK_CUSPARSE_ERROR`) are used throughout.
4.  **cuSPARSE SpMV Execution:**
    *   The core SpMV operation is performed using `cusparseScsrmv()`. The function name indicates:
        *   `S`: Single precision (`float`).
        *   `csr`: Input matrix is in CSR format.
        *   `mv`: Operation is Matrix-Vector multiplication.
    *   Key parameters include the cuSPARSE handle, the operation type (`CUSPARSE_OPERATION_NON_TRANSPOSE`), matrix dimensions, number of non-zeros (nnz), scalar multipliers alpha (1.0) and beta (0.0) for `y = alpha*A*x + beta*y`, the matrix descriptor, and pointers to the device memory locations of the matrix and vectors.
5.  **Timing:** `cudaEvent`s are used to accurately measure the execution time of the `cusparseScsrmv` function over several iterations for stability.
6.  **Verification:** The result from the cuSPARSE computation (`h_y_gpu`) is copied back to the host and compared against a CPU implementation (`h_y_cpu`) using a relative error tolerance.

## Key CUDA Features Used

*   **cuSPARSE Library:** Primarily `cusparseCreate()`, `cusparseDestroy()`, `cusparseCreateMatDescr()`, `cusparseDestroyMatDescr()`, `cusparseSetMatType()`, `cusparseSetMatIndexBase()`, and `cusparseScsrmv()`.
*   **CUDA Runtime API:** Memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`), error handling (`cudaGetLastError`, `cudaGetErrorString`), and event timing (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`, `cudaEventDestroy`).

## Performance Considerations

*   **Library Optimization:** cuSPARSE routines are highly optimized by NVIDIA for various GPU architectures. They often implement sophisticated algorithms (e.g., different SpMV strategies based on matrix structure) that are difficult to match with custom kernels without significant effort.
*   **Overhead:** There is some overhead associated with setting up cuSPARSE handles and descriptors, but this is typically amortized over many operations or for large enough problems.
*   **Memory Bandwidth:** Like the custom kernels, cuSPARSE SpMV performance is often limited by memory bandwidth, especially for matrices with irregular access patterns. The library aims to maximize bandwidth utilization.
*   **Comparison:** We expect cuSPARSE to outperform the custom kernels from Day 10, especially the basic one, and potentially even the shared-memory optimized version for this larger matrix size, due to its internal optimizations. The ease of use (calling one function vs. writing/tuning kernels) is also a significant factor.

## Building and Running

Follow the standard CMake build process within the target environment (Jetson Nano or compatible system with CUDA toolkit and cuSPARSE installed):

```bash
cd 100-days-of-cuda
mkdir build
cd build
cmake ..
make day036_cusparse_spmv # Or simply 'make' if building all
./day036/cusparse_spmv [num_rows] [num_cols] [sparsity] # Optional arguments
```

Example: `./day036/cusparse_spmv 20000 20000 0.01`

## Execution Results (Jetson Nano)

*(This section will be populated with the actual output after running the code on the Jetson Nano via the CI/CD pipeline)*

```
cuSPARSE Sparse Matrix-Vector Multiplication (SpMV)
Matrix size: 20000 x 20000, Target Sparsity: 1.00%

Sparse Matrix Info:
  Dimensions: 20000 x 20000
  Non-zeros: 3996961
  Actual Sparsity: 0.999240%

Computing SpMV on CPU...
CPU Execution Time: 20.3990 ms

Computing SpMV on GPU using cuSPARSE...
cuSPARSE GPU Execution Time: 8.7848 ms

Verifying results...
Verification: PASSED

Performance Comparison:
  CPU Time:        20.3990 ms
  cuSPARSE GPU Time: 8.7848 ms
  Speedup vs CPU:  2.32x
  cuSPARSE GPU Throughput: 0.91 GFLOP/s

Cleaning up...
Day 36 Complete.
```

## Learnings and Observations

*(This section will be updated after analyzing the results)*

*   Using cuSPARSE significantly simplifies the implementation compared to writing custom SpMV kernels.
*   The performance difference between CPU and GPU for this larger matrix size is expected to be substantial.
*   Comparison with Day 10 results (for a smaller matrix) will highlight the scalability and potential benefits of using optimized libraries.

## References

*   cuSPARSE Documentation: [https://docs.nvidia.com/cuda/cusparse/index.html](https://docs.nvidia.com/cuda/cusparse/index.html)
*   `cusparse<t>csrmv()`: [https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrmv](https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrmv)
