# Day 085: Tensor-Matrix Multiplication in CUDA

## Overview

This project (Day 085) implements a CUDA kernel for performing tensor-matrix multiplication. Specifically, it computes `C = A * B`, where `A` is a 4D tensor and `B` is a 2D matrix. The operation can be described as `C[b,i,j,k] = sum_l (A[b,i,j,l] * B[l,k])`.

This is a fundamental operation in many deep learning and scientific computing applications.

## Implementation Details

### CUDA Kernel: `tensorMatrixMultKernel`

The core computation is performed by the `tensorMatrixMultKernel` CUDA kernel.
-   **Input:**
    -   `A`: A 4D tensor of shape `(B_dim, I_dim, J_dim, L_dim)` stored linearly in row-major order.
    -   `B`: A 2D matrix of shape `(L_dim, K_dim)` stored linearly in row-major order.
-   **Output:**
    -   `C`: A 4D tensor of shape `(B_dim, I_dim, J_dim, K_dim)` stored linearly in row-major order.

-   **Logic:**
    1.  Each thread in the CUDA grid is responsible for computing a single element of the output tensor `C`.
    2.  The global thread ID (`idx`) is calculated using `blockIdx.x * blockDim.x + threadIdx.x`.
    3.  This `idx` is then decomposed into the 4D indices `(b, i, j, k)` corresponding to the element `C[b,i,j,k]`.
    4.  A loop iterates over the contraction dimension `L_dim` (indexed by `l`). In each iteration, it multiplies `A[b,i,j,l]` with `B[l,k]` and accumulates the sum.
    5.  The final sum is stored in `C[idx]`.

### Host Wrapper: `tensor_matrix_multiply`

A C-style wrapper function `tensor_matrix_multiply` is provided to manage the kernel launch:
-   It calculates the total number of elements in the output tensor `C`.
-   It defines the number of threads per block (e.g., 256).
-   It calculates the required number of blocks in the grid.
-   It launches the `tensorMatrixMultKernel` with the computed grid and block dimensions.

### Error Handling

Standard CUDA error checking is implemented using `CHECK_CUDA_ERROR` and `CHECK_KERNEL_LAUNCH` macros, defined in `tensor_matrix_mult.cuh`. These macros help in identifying and reporting errors from CUDA API calls and kernel launches.

## Key CUDA Features Used

-   **`__global__` kernels:** `tensorMatrixMultKernel` for parallel execution on the GPU.
-   **Thread Indexing:** `blockIdx`, `blockDim`, `threadIdx` for calculating unique thread IDs and mapping them to data elements.
-   **Device Memory Management:**
    -   `cudaMalloc()`: To allocate memory on the GPU for tensors A, B, and C.
    -   `cudaMemcpy()`: To transfer data between host (CPU) and device (GPU).
    -   `cudaFree()`: To deallocate GPU memory.
-   **Kernel Launch Configuration:** `<<<blocksPerGrid, threadsPerBlock>>>` syntax to specify execution configuration.
-   **Error Handling:** `cudaError_t`, `cudaGetErrorString`, `cudaPeekAtLastError`, `cudaDeviceSynchronize`.

## Performance Considerations (Jetson Nano Focus)

-   **Parallelism:** The problem is highly parallelizable, as each element of the output tensor `C` can be computed independently. The CUDA implementation leverages this by assigning each computation to a separate thread.
-   **Memory Access Patterns:** The current kernel accesses global memory. For tensor `A`, access is `A[((b * I_dim + i) * J_dim + j) * L_dim + l]`. For tensor `B`, access is `B[l * K_dim + k]`.
    -   Access to `A` might be somewhat strided depending on `l`.
    -   Access to `B` is strided for threads within a warp if they map to different `k` values but consecutive `l` values in the inner loop. If threads map to the same `k` and iterate `l`, then access to `B` for a fixed `k` would be `B[0*K+k], B[1*K+k], ...` which is strided.
    -   Optimizations like using shared memory to cache parts of `B` or rearranging computation could improve memory coalescing, especially for the Jetson Nano's memory architecture. However, for this initial implementation, a straightforward global memory access pattern is used.
-   **Thread Block Size:** A block size of 256 threads is used, which is generally a good starting point. Optimal block size can depend on the specific kernel and GPU architecture (sm_53 for Jetson Nano).
-   **CPU vs. GPU:** The `tensor_matrix_mult_main.cu` includes a basic CPU implementation and times both CPU and GPU execution. For sufficiently large tensors, the GPU is expected to significantly outperform the CPU due to massive parallelism. The actual speedup will depend on the problem size and the efficiency of the kernel.

## Building and Running

### Prerequisites
-   CUDA Toolkit (compatible with Jetson Nano, e.g., CUDA 10.2)
-   CMake (version 3.18 or higher recommended for this `CMakeLists.txt`)
-   GCC/G++ (compatible with CUDA version)
-   Google Test (GTest) library for running tests. The root `CMakeLists.txt` of the 100-days-of-cuda project should handle GTest setup.

### Building
1.  Ensure the `day085` directory is added to the root `CMakeLists.txt`:
    ```cmake
    # In root CMakeLists.txt
    add_subdirectory(day085)
    ```
2.  Navigate to your build directory (e.g., `100-days-of-cuda/build`):
    ```bash
    cd path/to/100-days-of-cuda/build
    ```
3.  Run CMake and build:
    ```bash
    cmake .. # Configure the project (if not already done or if CMakeLists.txt changed)
    make # Build the project
    ```
    This will build the `tensor_matrix_mult_lib` static library, the `tensor_matrix_mult_main` executable, and the `tensor_matrix_mult_test` executable. These will be located in the `build/day085/` directory.

### Running

#### Main Executable (Demonstration & Benchmark)
To run the main demonstration program:
```bash
./day085/tensor_matrix_mult_main
```

#### Test Executable (Google Test)
To run the Google Tests:
```bash
./day085/tensor_matrix_mult_test
```

## Execution Results / Output

```text
drboom@JetNano ~/g/1/build> ./day085/tensor_matrix_mult_main 
Tensor-Matrix Multiplication
Dimensions:
  A: (2, 3, 4, 5)
  B: (5, 6)
  C: (2, 3, 4, 6)
------------------------------------
Performing GPU computation...
GPU computation time: 0.218 ms
Performing CPU computation for verification...
CPU computation time: 0.003 ms
Verification successful: GPU and CPU results match within tolerance.
Sample of GPU result (first ~16 elements of C):
247.500 262.500 277.500 292.500 307.500 322.500 560.000 600.000 640.000 680.000 720.000 760.000 872.500 937.500 1002.500 1067.500 ...
Sample of CPU result (first ~16 elements of C):
247.500 262.500 277.500 292.500 307.500 322.500 560.000 600.000 640.000 680.000 720.000 760.000 872.500 937.500 1002.500 1067.500 ...
------------------------------------
Day 085 execution finished.
```

```text
drboom@JetNano ~/g/1/build> ./day085/tensor_matrix_mult_test 
[==========] Running 12 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 12 tests from TensorMatrixMultTest
[ RUN      ] TensorMatrixMultTest.BasicCase
[       OK ] TensorMatrixMultTest.BasicCase (90 ms)
[ RUN      ] TensorMatrixMultTest.SmallDimensions
[       OK ] TensorMatrixMultTest.SmallDimensions (1 ms)
[ RUN      ] TensorMatrixMultTest.LargerLdim
[       OK ] TensorMatrixMultTest.LargerLdim (1 ms)
[ RUN      ] TensorMatrixMultTest.LargerKdim
[       OK ] TensorMatrixMultTest.LargerKdim (1 ms)
[ RUN      ] TensorMatrixMultTest.OneIDim
[       OK ] TensorMatrixMultTest.OneIDim (1 ms)
[ RUN      ] TensorMatrixMultTest.OneJDim
[       OK ] TensorMatrixMultTest.OneJDim (1 ms)
[ RUN      ] TensorMatrixMultTest.OneBDim
[       OK ] TensorMatrixMultTest.OneBDim (1 ms)
[ RUN      ] TensorMatrixMultTest.ZeroKDimResultsInZeroCSize
[       OK ] TensorMatrixMultTest.ZeroKDimResultsInZeroCSize (1 ms)
[ RUN      ] TensorMatrixMultTest.ZeroLDimResultsInZerosInC
[       OK ] TensorMatrixMultTest.ZeroLDimResultsInZerosInC (1 ms)
[ RUN      ] TensorMatrixMultTest.ZeroBDimResultsInZeroCSize
[       OK ] TensorMatrixMultTest.ZeroBDimResultsInZeroCSize (1 ms)
[ RUN      ] TensorMatrixMultTest.ZeroIDimResultsInZeroCSize
[       OK ] TensorMatrixMultTest.ZeroIDimResultsInZeroCSize (1 ms)
[ RUN      ] TensorMatrixMultTest.ZeroJDimResultsInZeroCSize
[       OK ] TensorMatrixMultTest.ZeroJDimResultsInZeroCSize (1 ms)
[----------] 12 tests from TensorMatrixMultTest (105 ms total)

[----------] Global test environment tear-down
[==========] 12 tests from 1 test suite ran. (105 ms total)
[  PASSED  ] 12 tests.
```

## Learnings and Observations

-   Understanding the mapping of multi-dimensional tensor indices to linear memory addresses is crucial.
-   The kernel design involves decomposing a global thread index into multiple tensor indices.
-   Error checking (`CHECK_CUDA_ERROR`, `CHECK_KERNEL_LAUNCH`) is vital for robust CUDA programming.
-   Comparing GPU results with a CPU implementation is a good way to verify correctness.
-   The choice of thread block size and grid size impacts performance and resource utilization.
-   For this problem, the number of threads is determined by the size of the output tensor `C`.
-   The provided execution results for the default small dimensions in `tensor_matrix_mult_main.cu` (A: (2,3,4,5), B: (5,6)) show the CPU (0.003 ms) outperforming the GPU (0.218 ms). This is expected due to kernel launch overhead and data transfer times dominating the computation time for small problem sizes. For larger tensors, the GPU's parallelism would yield significant speedups.

## Future Improvements (Optional)

-   **Shared Memory Optimization:** Parts of matrix `B` or tensor `A` could be loaded into shared memory to reduce global memory accesses and improve data reuse. This would be particularly beneficial if `L_dim` is large.
-   **Tiled Multiplication:** Implement a tiled version of the multiplication, where blocks of threads compute sub-blocks of the output tensor `C`.
-   **Further Performance Profiling:** Use NVIDIA Nsight Systems/Compute to analyze kernel performance in detail on the Jetson Nano and identify bottlenecks.
-   **Support for Different Data Types:** Extend to support `double` or `half` precision.
-   **Row-Major vs. Column-Major:** Add support or options for different memory layouts.

## References (Optional)
-   NVIDIA CUDA C++ Programming Guide
-   CUDA By Example - Sanders & Kandrot
