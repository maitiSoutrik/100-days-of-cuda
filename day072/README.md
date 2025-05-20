# Day 72: Total Variation Distance (TVD) Loss

## Overview
This project implements the Total Variation Distance (TVD) calculation between two discrete probability distributions using both CPU and CUDA-accelerated GPU approaches. TVD is a metric used to quantify the difference between two probability distributions. It is defined as half the L1 norm of the difference between the two probability mass functions (PMFs).

TVD Formula:
\[ \text{TVD}(P, Q) = \frac{1}{2} \sum_{i} |P_i - Q_i| \]
Where P and Q are two discrete probability distributions over the same finite sample space. The TVD is bounded between 0 (identical distributions) and 1 (distributions with disjoint support).

This day's work focuses on:
1.  Implementing a CUDA kernel for efficient parallel calculation of TVD.
2.  Providing a CPU version for comparison and verification.
3.  Benchmarking the performance of both implementations.
4.  Setting up unit tests using Google Test.

## Implementation Details

### CPU Implementation (`calculate_tvd_cpu`)
The CPU version is straightforward:
- It takes two `std::vector<float>` representing the PMFs \(P\) and \(Q\).
- It iterates through the elements, calculating the sum of the absolute differences: \(\sum |P_i - Q_i|\).
- Finally, it multiplies the sum by 0.5 to get the TVD.
- Basic error checking for vector sizes is included. Input vectors are assumed to be valid PMFs (non-negative elements summing to 1), though this is not strictly enforced within the function for performance.

### GPU Implementation (`calculate_tvd_gpu`)
The GPU implementation aims to parallelize the sum of absolute differences and the final reduction.
1.  **`sum_abs_diff_kernel`**:
    *   This kernel calculates the absolute difference \(|P_i - Q_i|\) for each element \(i\) in parallel.
    *   It then performs a parallel reduction within each thread block using shared memory (`sdata[]`) to sum these differences.
    *   Each block writes its partial sum to an intermediate array in global memory (`d_partial_sums`).
2.  **`sum_kernel` (Reduction of Partial Sums)**:
    *   The partial sums from `d_partial_sums` need to be further summed to get the total sum of absolute differences.
    *   If the number of partial sums (`num_blocks` from the previous kernel) is small enough to be processed by a single block (e.g., less than or equal to `threads_per_block`), this kernel is launched with one block to sum these partial sums using shared memory reduction.
    *   If `num_blocks` is larger, the current implementation includes a fallback: it copies the partial sums to the host, sums them on the CPU, and copies the result back to the device. A more robust GPU-only solution for larger `num_blocks` would involve a multi-stage reduction on the GPU.
3.  **Final TVD Calculation**:
    *   After obtaining the total sum of absolute differences on the device (in `d_total_sum_abs_diff`), this sum is copied to the host.
    *   The TVD is calculated by multiplying this sum by 0.5 on the host.
    *   The final TVD value is then copied back to the `d_tvd` pointer on the device.
    *   This final step could also be done in a small kernel for a fully on-device computation if `d_tvd` is intended to be used further on the GPU.

The `CHECK_CUDA_ERROR` macro is used for robust error handling in CUDA API calls.

## Key CUDA Features Used
-   **CUDA Kernels**: `__global__` functions (`sum_abs_diff_kernel`, `sum_kernel`) for parallel execution.
-   **Thread Hierarchy**: `blockIdx`, `threadIdx`, `blockDim` for managing thread execution and data access.
-   **Shared Memory**: `__shared__` memory (`sdata`) for efficient intra-block parallel reduction. This reduces global memory accesses and improves performance for the summation step within each block.
-   **Synchronization**: `__syncthreads()` to ensure correct ordering of operations within a block, especially during the shared memory reduction.
-   **Device Memory Management**: `cudaMalloc`, `cudaMemcpy`, `cudaFree` for allocating and managing memory on the GPU.
-   **CUDA Events**: `cudaEvent_t` for accurate timing of GPU operations.

## Performance Considerations
-   **Parallel Reduction**: The use of shared memory for reduction within blocks is a standard optimization technique that significantly speeds up summation compared to naive atomic operations or sequential summation on the GPU.
-   **Memory Coalescing**: Access patterns in the kernels (e.g., `p[i]`, `q[i]`) are designed to be coalesced as threads within a warp access contiguous memory locations.
-   **Number of Blocks vs. Reduction Strategy**: The strategy for summing partial sums from different blocks can impact performance. The current implementation uses a simpler CPU-sum fallback for a large number of blocks. A full multi-stage GPU reduction would be more efficient for very large inputs requiring many blocks.
-   **Data Transfer**: `cudaMemcpy` operations between host and device introduce overhead. For applications where data already resides on the GPU, this TVD calculation would be more efficient. The final multiplication by 0.5 is currently done on the host after a DtoH copy, then HtoD copy; this could be a tiny kernel call to avoid the round trip if the result needs to stay on device.

## Building and Running

### Prerequisites
-   NVIDIA CUDA Toolkit (>= 10.0, compatible with sm_53 for Jetson Nano)
-   CMake (>= 3.10)
-   A C++ compiler (e.g., g++)
-   Google Test (will be fetched by CMake if not found)

### Build Steps
1.  Ensure the `day072` directory is added to the root `CMakeLists.txt`:
    ```cmake
    add_subdirectory(day072)
    ```
2.  Configure and build the project from the root `build` directory:
    ```bash
    mkdir -p build
    cd build
    cmake ..
    make tvd_benchmark tvd_loss_test -j$(nproc) 
    ```
    (Or `make` to build everything)

### Running
-   **Benchmark/Main Executable**:
    ```bash
    ./day072/tvd_benchmark
    ```
-   **Tests**:
    ```bash
    ./day072/tvd_loss_test
    ```
    Or run all tests using CTest from the `build` directory:
    ```bash
    ctest --output-on-failure
    ```

## Execution Results

### Test Output (`./day072/tvd_loss_test`)
The following output was obtained by running the tests on a Jetson Nano:
```
[==========] Running 9 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 5 tests from TVD_Loss_CPU
[ RUN      ] TVD_Loss_CPU.EmptyVectors
[       OK ] TVD_Loss_CPU.EmptyVectors (0 ms)
[ RUN      ] TVD_Loss_CPU.IdenticalVectors
[       OK ] TVD_Loss_CPU.IdenticalVectors (0 ms)
[ RUN      ] TVD_Loss_CPU.SimpleDisjointVectors
[       OK ] TVD_Loss_CPU.SimpleDisjointVectors (0 ms)
[ RUN      ] TVD_Loss_CPU.SimpleMixedVectors
[       OK ] TVD_Loss_CPU.SimpleMixedVectors (0 ms)
[ RUN      ] TVD_Loss_CPU.DifferentSizes
Error: Input vectors must have the same size.
[       OK ] TVD_Loss_CPU.DifferentSizes (0 ms)
[----------] 5 tests from TVD_Loss_CPU (0 ms total)

[----------] 4 tests from TVD_Loss_GPU
[ RUN      ] TVD_Loss_GPU.IdenticalVectorsGPU
[       OK ] TVD_Loss_GPU.IdenticalVectorsGPU (87 ms)
[ RUN      ] TVD_Loss_GPU.SimpleDisjointVectorsGPU
[       OK ] TVD_Loss_GPU.SimpleDisjointVectorsGPU (1 ms)
[ RUN      ] TVD_Loss_GPU.CompareWithCPU
[       OK ] TVD_Loss_GPU.CompareWithCPU (1 ms)
[ RUN      ] TVD_Loss_GPU.CompareWithCPU_OddSize
[       OK ] TVD_Loss_GPU.CompareWithCPU_OddSize (1 ms)
[----------] 4 tests from TVD_Loss_GPU (92 ms total)

[----------] Global test environment tear-down
[==========] 9 tests from 2 test suites ran. (93 ms total)
[  PASSED  ] 9 tests.
```

### Benchmark Output (`./day072/tvd_benchmark`)
(Please provide the output of the benchmark executable to complete this section.)

```
[Expected Output Format]
TVD (GPU): 0.XXXXXXX
GPU Calculation Time: Y.YYY ms
TVD (CPU): 0.XXXXXXX
CPU Calculation Time: Z.ZZZ ms
Difference (GPU - CPU): E.EEEEEEE
Verification: PASS
```

## Learnings and Observations
-   Implementing parallel reduction is a common and crucial pattern in CUDA programming.
-   The choice of block size and grid size can affect performance and needs to be tuned.
-   Handling the reduction of results from multiple blocks efficiently is key for scaling.
-   Error checking (`CHECK_CUDA_ERROR`) is vital for debugging CUDA applications.
-   Normalizing input vectors to be valid PMFs is an important pre-processing step for TVD calculation.
-   TVD is sensitive to small changes in probability values, making floating-point precision a consideration, especially when comparing CPU and GPU results.

## Future Improvements
-   Implement a full multi-stage GPU reduction for the partial sums from blocks, removing the CPU fallback for large `num_blocks`.
-   Add a kernel to perform the final `0.5 * sum` on the GPU to avoid DtoH/HtoD copies if the result is needed on the device.
-   Explore using CUDA library functions (e.g., from CUB or Thrust) for reduction if higher performance or more robust implementations are needed.
-   Add template parameters for data types (e.g., `double`) for more flexibility.
