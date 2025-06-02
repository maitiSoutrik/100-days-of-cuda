# Day 084: Cumulative Product (Prefix Product / Scan)

## Introduction
This project implements the cumulative product (also known as prefix product or multiplicative scan) of an array using CUDA. The cumulative product is an operation where each element in the output array is the product of all preceding elements in the input array, up to and including the current element (inclusive scan).

For an input array `A = [a_0, a_1, ..., a_{n-1}]`, the inclusive cumulative product array `P = [p_0, p_1, ..., p_{n-1}]` is:
`p_i = a_0 * a_1 * ... * a_i`

This day focuses on implementing a single-block parallel scan algorithm on the GPU and comparing it with a sequential CPU version.

## Implementation Details

### CUDA Kernel: `inclusive_scan_kernel_blelloch`
The core GPU computation is performed by the `inclusive_scan_kernel_blelloch` kernel. This kernel implements a single-block version of a parallel scan algorithm, inspired by the Blelloch scan. It consists of two main phases:

1.  **Up-Sweep (Reduction Phase):**
    *   Input data is loaded from global memory into shared memory. Each thread can load up to two elements. Elements outside the actual data size `n` (but within the shared memory allocation of `2 * blockDim.x`) are padded with the multiplicative identity (1.0f).
    *   A tree-based reduction is performed in shared memory. In each step, threads compute partial products. For a stride `s`, `s_temp[idx_right]` becomes `s_temp[idx_left] * s_temp[idx_right]`. This phase effectively computes intermediate products across the block.

2.  **Down-Sweep Phase:**
    *   The last element of the array in shared memory (corresponding to `n-1`) is set to the identity (1.0f) to initiate the exclusive scan.
    *   The algorithm then sweeps back down the conceptual tree. In each step, pairs of elements are updated: `s_temp[idx_left_val_pos]` takes the value from `s_temp[idx_right_val_pos]`, and `s_temp[idx_right_val_pos]` becomes the product of its original value (held temporarily) and the new `s_temp[idx_right_val_pos]`. This phase distributes the prefix products.

3.  **Exclusive to Inclusive Conversion:**
    *   After the down-sweep, the shared memory `s_temp` contains the *exclusive* prefix product.
    *   To get the *inclusive* prefix product, each element `s_temp[i]` is multiplied by the original input element `d_data[i]` (which was preserved in global memory or could be re-read if the kernel overwrote `d_data` earlier).
    *   The final inclusive product is then written back to global memory.

The kernel is designed for a single CUDA block and assumes the input array size `n` is manageable within this constraint (typically `n <= 2 * threads_per_block`). For larger arrays, a multi-block scan algorithm would be necessary.

### CPU Implementation: `inclusive_scan_cpu`
A straightforward sequential loop calculates the cumulative product on the CPU for reference and verification:
```cpp
void inclusive_scan_cpu(float* h_data, int n) {
    if (n == 0) return;
    for (int i = 1; i < n; ++i) {
        h_data[i] = h_data[i] * h_data[i-1];
    }
}
```

### Error Handling
The `CHECK_CUDA_ERROR` macro is used for robust CUDA error checking throughout the host code.

## Key CUDA Concepts Used
*   **Shared Memory:** Used extensively within the `inclusive_scan_kernel_blelloch` kernel for fast inter-thread communication and to store intermediate products during the scan operation. This reduces reliance on slower global memory access.
*   **Thread Synchronization (`__syncthreads()`):** Essential after each step of the up-sweep and down-sweep phases to ensure that all threads in a block have completed their current stage of computation before proceeding to the next. This maintains data consistency in shared memory.
*   **Kernel Launch Configuration:** The kernel is launched with a single block (`<<<1, threads_per_block, shared_mem_size>>>`), and shared memory is dynamically allocated.
*   **Parallel Scan Algorithm (Blelloch-style):** The fundamental parallel algorithm adapted for prefix products.

## Performance Considerations
*   **Single-Block Limitation:** The current GPU implementation is limited to a single block. This means its performance benefits are most apparent for array sizes that fit within this constraint (e.g., up to 512 or 1024 elements, depending on `threads_per_block`). For larger arrays, a multi-block approach (e.g., scanning blocks, then scanning block sums, then updating block elements) would be required.
*   **Numerical Stability:** Cumulative products of floating-point numbers can quickly lead to underflow (if values are small) or overflow (if values are large). The input data in the `main` function is initialized with values close to 1.0 to mitigate this for demonstration purposes. In real-world scenarios with arbitrary data, this can be a significant issue.
*   **Shared Memory Bank Conflicts:** While not explicitly optimized for in this basic version, complex shared memory access patterns in scan algorithms can lead to bank conflicts if not carefully managed. The Blelloch scan is generally good, but specific indexing can matter.
*   **CPU vs. GPU:** For small arrays that fit in a single block, the overhead of CUDA kernel launch and memory transfers might make the GPU version slower than the CPU version. The benefits of GPU parallelism become more significant with larger datasets that can be processed by many blocks concurrently (requiring a multi-block scan).

## Building and Running

### Prerequisites
*   CUDA Toolkit (>= 10.0, tested with 11.x/12.x)
*   CMake (>= 3.10)
*   A C++ compiler compatible with CUDA (e.g., GCC, MSVC)
*   Google Test (GTest) library (CMake will try to find it)

### Build Steps
1.  Navigate to the root of the `100-days-of-cuda` project.
2.  If you haven't already, create a build directory:
    ```bash
    mkdir build
    cd build
    ```
3.  Run CMake from the build directory (assuming `day084` has been added to the root `CMakeLists.txt`):
    ```bash
    cmake ..
    ```
4.  Build the project (specifically the `day084` targets):
    ```bash
    cmake --build . --target cumulative_product_main --target cumulative_product_test 
    # Or build all targets:
    # cmake --build . 
    ```

### Running
*   **Main Executable (Demonstration):**
    ```bash
    ./day084/cumulative_product_main 
    # Or from the build directory:
    # ./bin/cumulative_product_main (if installed) or directly day084/cumulative_product_main
    ```
*   **Test Executable:**
    ```bash
    ./day084/cumulative_product_test
    # Or from the build directory:
    # ./bin/cumulative_product_test (if installed) or directly day084/cumulative_product_test
    # Alternatively, run tests via CTest from the build directory:
    # ctest -R day084_cumulative_product --verbose 
    # (Note: CTest name might vary based on root CMakeLists.txt configuration for GTest discovery)
    # The CMakeLists.txt for day084 uses gtest_discover_tests(cumulative_product_test)
    # so ctest should pick up 'CumulativeProductTest.GPU_SimpleProduct' etc.
    ```

## Execution Results

The `cumulative_product_main` executable will output:
1.  Array size.
2.  Input data (for small arrays).
3.  CPU execution time and CPU output (for small arrays).
4.  GPU execution time and GPU output (for small arrays).
5.  Verification status (SUCCESS or FAILED).
6.  Speedup factor (CPU Time / GPU Time).
7.  A note about the single-block implementation limitation.

**Example Output Snippet (Actual logs from Jetson Nano for n=256):**
```
drboom@JetNano ~/g/1/build> ./day084/cumulative_product_main 
--- Day 084: Cumulative Product (Prefix Product / Scan) ---
Array size: 256

CPU Execution Time: 0.001 ms
GPU Execution Time: 0.081 ms

Verification: SUCCESS! CPU and GPU results match.
Speedup (CPU Time / GPU Time): 0.01x

Note: The current GPU kernel is a single-block implementation.
It's primarily for demonstrating the scan logic within a block.
For larger arrays, a multi-block scan algorithm would be necessary for correctness and performance.
```

The `cumulative_product_test` executable will run various unit tests and report pass/fail status.
**Actual Test Output (Jetson Nano after fixes):**
```
drboom@JetNano ~/g/1/build> ./day084/cumulative_product_test 
[==========] Running 11 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 11 tests from CumulativeProductTest
[ RUN      ] CumulativeProductTest.CPU_EmptyArray
[       OK ] CumulativeProductTest.CPU_EmptyArray (0 ms)
[ RUN      ] CumulativeProductTest.GPU_EmptyArray
[       OK ] CumulativeProductTest.GPU_EmptyArray (75 ms)
[ RUN      ] CumulativeProductTest.CPU_SingleElement
[       OK ] CumulativeProductTest.CPU_SingleElement (0 ms)
[ RUN      ] CumulativeProductTest.GPU_SingleElement
[       OK ] CumulativeProductTest.GPU_SingleElement (1 ms)
[ RUN      ] CumulativeProductTest.CPU_SimpleProduct
[       OK ] CumulativeProductTest.CPU_SimpleProduct (0 ms)
[ RUN      ] CumulativeProductTest.GPU_SimpleProduct
[       OK ] CumulativeProductTest.GPU_SimpleProduct (1 ms)
[ RUN      ] CumulativeProductTest.CPU_WithZeros
[       OK ] CumulativeProductTest.CPU_WithZeros (0 ms)
[ RUN      ] CumulativeProductTest.GPU_WithZeros
[       OK ] CumulativeProductTest.GPU_WithZeros (0 ms)
[ RUN      ] CumulativeProductTest.CPU_WithNegativeNumbers
[       OK ] CumulativeProductTest.CPU_WithNegativeNumbers (0 ms)
[ RUN      ] CumulativeProductTest.GPU_WithNegativeNumbers
[       OK ] CumulativeProductTest.GPU_WithNegativeNumbers (1 ms)
[ RUN      ] CumulativeProductTest.GPU_LargerArraySingleBlock
[       OK ] CumulativeProductTest.GPU_LargerArraySingleBlock (1 ms)
[----------] 11 tests from CumulativeProductTest (81 ms total)

[----------] Global test environment tear-down
[==========] 11 tests from 1 test suite ran. (81 ms total)
[  PASSED  ] 11 tests.
```

## Learnings and Observations
*   Implementing parallel scan algorithms requires careful management of shared memory and thread synchronization.
*   The Blelloch scan provides a good foundation for work-efficient parallel prefix operations. Converting its typical exclusive scan output to an inclusive scan is an additional step.
*   Numerical stability is a key concern for cumulative products, especially with floating-point numbers over long sequences.
*   Single-block implementations are useful for understanding the core logic but are not scalable for large datasets without extending to multi-block strategies.
*   Testing with various edge cases (empty arrays, single elements, zeros, negative numbers) is crucial.

## Future Improvements
*   Implement a multi-block (segmented) scan algorithm to handle arbitrarily large arrays efficiently. This would involve:
    1.  Each block computes a local scan and the total product for its segment.
    2.  A separate kernel (or a recursive call) scans the array of block total products.
    3.  Each block then updates its local scan results using the prefix product from the preceding blocks.
*   Investigate and implement techniques for improving numerical stability if dealing with a wide range of input values.
*   Explore different scan primitives or CUDA library functions (e.g., from CUB or Thrust, though the goal here is a manual implementation) for comparison.
