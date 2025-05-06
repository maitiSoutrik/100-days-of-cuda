# Day 58: Bitonic Sort with Shared Memory Optimization

## Overview

This project implements the Bitonic Sort algorithm using CUDA, with a focus on optimizing performance using shared memory. Bitonic Sort is a parallel sorting algorithm that is well-suited for GPU architectures due to its regular comparison patterns. This implementation sorts an array of 1024 floating-point numbers, a size chosen to fit entirely within a single CUDA thread block's shared memory and thread limits.

## Implementation Details

The implementation consists of:
1.  `bitonic_sort.cuh`: Header file defining the `bitonic_sort_gpu` function, a helper `print_array_host` function, and the `CHECK_CUDA_ERROR` macro.
2.  `bitonic_sort.cu`:
    *   Contains the `bitonic_sort_kernel` which is the CUDA kernel performing the sort. It loads data into shared memory (`s_data`), performs the bitonic sort stages, and then writes the sorted data back to global memory.
    *   The kernel is designed for `N_CONST = 1024` elements.
    *   The `bitonic_sort_gpu` host function handles memory allocation on the GPU, data transfers between host and device, and kernel launch. It ensures the input array size matches `N_CONST`.
3.  `bitonic_sort_main.cu`: A simple `main` function that generates an array of random floats, calls the GPU sort, prints a sample of the unsorted and sorted arrays, and verifies the sort.
4.  `bitonic_sort_test.cu`: Contains Google Test unit tests to verify the correctness of the sort with various inputs (random, sorted, reverse sorted, all same elements).

### Bitonic Sort Kernel Logic:
The `bitonic_sort_kernel` works as follows:
-   Each thread in the block loads one element from global memory into a shared memory array `s_data`.
-   A `__syncthreads()` ensures all data is loaded before sorting begins.
-   The sorting proceeds in stages:
    -   The outer loop (`for k`) iterates from `k=2` up to `N_CONST`, doubling `k` each time. `k` represents the size of the bitonic sequences being merged.
    -   The inner loop (`for j`) iterates downwards from `j=k/2` by halving `j`. `j` is the comparison distance.
    -   Each thread `tid` calculates `ixj = tid ^ j`. If `ixj > tid`, the thread is responsible for a comparison.
    -   The sorting direction (ascending/descending) is determined by `(tid & k) == 0`.
    -   Elements `s_data[tid]` and `s_data[ixj]` are compared and swapped if necessary, based on the direction.
    -   A `__syncthreads()` is called after all comparisons for a given `j` are done, ensuring all threads in the block have completed their swaps before the next `j` or `k` iteration.
-   Finally, threads write their sorted element from `s_data` back to global memory.

## Key CUDA Features Used

-   **CUDA Kernels (`__global__`)**: For parallel execution on the GPU.
-   **Shared Memory (`__shared__`)**: Used to store the array segment being sorted by a block, significantly reducing global memory latency.
-   **Thread Indexing (`threadIdx.x`)**: To assign work to individual threads.
-   **Synchronization (`__syncthreads()`)**: To ensure correct ordering of operations within a thread block, especially after loading to shared memory and after each comparison pass.
-   **CUDA Memory Management**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
-   **Error Handling**: `CHECK_CUDA_ERROR` macro for robust error checking of CUDA API calls.

## Performance Considerations

-   **Shared Memory Usage**: The primary optimization here is the use of shared memory. For an array of 1024 floats (4KB), it fits comfortably within the typical shared memory capacity of modern GPUs (e.g., 48KB or more per SM). This greatly speeds up the numerous comparisons and swaps by avoiding global memory accesses for each.
-   **Single Block Limitation**: This specific implementation is limited to sorting an array that fits within a single block (1024 elements). For larger arrays, a multi-block approach would be necessary, where each block sorts a portion, and then a global merge step (which could also be a bitonic merge) combines the sorted portions.
-   **Coalesced Access**: When loading from global memory to shared memory (`s_data[tid] = d_array[tid]`) and writing back, memory accesses are coalesced as threads with consecutive `tid` access consecutive memory locations.
-   **Bank Conflicts**: The access pattern `s_data[tid]` and `s_data[ixj]` in shared memory could potentially lead to bank conflicts depending on `j` and `N_CONST`. For `N_CONST = 1024`, which is a power of 2, and `j` also being powers of 2, bank conflicts are generally minimized if `N_CONST` is not a multiple of the number of banks (typically 32) in a way that causes many threads in a warp to hit the same bank. However, the performance is generally good due to the high data reuse from shared memory.

## Building and Running

To build and run the project (assuming you are in the root `100-days-of-cuda` directory and have CMake and CUDA toolkit installed on the target system):

1.  **Configure CMake**:
    ```bash
    mkdir -p build
    cd build
    cmake ..
    ```
2.  **Build**:
    ```bash
    make day058_bitonic_sort_main # To build the main executable
    make day058_bitonic_sort_test # To build the test executable
    # Or simply 'make' to build everything
    ```
3.  **Run the main application**:
    ```bash
    ./day058/bitonic_sort_main
    ```
4.  **Run tests**:
    ```bash
    ./day058_tests/bitonic_sort_test
    # Or use ctest
    # ctest -R day058_bitonic_sort # (If tests are registered with CTest correctly)
    ```

## Execution Results

Output from `bitonic_sort_main` on Jetson Nano:
```
drboom@JetNano ~/g/1/build> ./day058/bitonic_sort_main 
Generating 1024 random float numbers...
Unsorted array (first 20 elements and last 10 elements if N > 30):
92.709999 76.010002 12.020000 10.510000 94.510002 49.660000 59.020000 37.209999 55.560001 54.240002 
29.590000 5.800000 89.230003 84.120003 25.760000 19.639999 28.450001 12.430000 37.570000 29.410000 

...
78.070000 80.889999 16.610001 54.090000 67.720001 88.529999 87.480003 94.680000 35.529999 57.060001 

Starting Bitonic Sort on GPU...
Bitonic Sort on GPU finished.

Sorted array (first 20 elements and last 10 elements if N > 30):
0.150000 0.270000 0.300000 0.420000 0.630000 0.680000 0.780000 0.970000 1.130000 1.310000 
1.350000 1.620000 1.640000 1.650000 1.710000 1.750000 1.830000 1.910000 2.080000 2.410000 

...
98.809998 98.809998 98.889999 98.900002 99.139999 99.150002 99.269997 99.650002 99.680000 99.739998 

Verification: Array is sorted correctly.
```

Output from `bitonic_sort_test` on Jetson Nano:
```
drboom@JetNano ~/g/1/build> ./day058/bitonic_sort_test 
Running main() from /home/drboom/git_repos/100-days-of-cuda/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 4 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 4 tests from BitonicSortTest
[ RUN      ] BitonicSortTest.RandomFloats
[       OK ] BitonicSortTest.RandomFloats (92 ms)
[ RUN      ] BitonicSortTest.AlreadySorted
[       OK ] BitonicSortTest.AlreadySorted (1 ms)
[ RUN      ] BitonicSortTest.ReverseSorted
[       OK ] BitonicSortTest.ReverseSorted (1 ms)
[ RUN      ] BitonicSortTest.AllSameElements
[       OK ] BitonicSortTest.AllSameElements (1 ms)
[----------] 4 tests from BitonicSortTest (97 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 1 test suite ran. (97 ms total)
[  PASSED  ] 4 tests.
```

## Learnings and Observations

-   Bitonic sort's regular structure maps well to SIMT execution on GPUs.
-   Shared memory is crucial for achieving good performance by reducing global memory latency.
-   The logic for determining comparison indices (`ixj = tid ^ j`) and sorting direction (`(tid & k) == 0`) is key to the algorithm.
-   Synchronization (`__syncthreads()`) is vital at each step of the shared memory sort to prevent race conditions.
-   This implementation is a "local" sort, effective for data fitting in one block. Scaling to larger datasets requires a more complex multi-block strategy.

## Future Improvements
-   Extend to handle array sizes larger than a single block's capacity. This would involve multiple kernel launches or more complex kernel logic to have blocks sort segments and then merge these sorted segments.
-   Investigate and optimize for shared memory bank conflicts if they become a bottleneck for different `N_CONST` values.
-   Compare performance against other GPU sorting algorithms like radix sort or merge sort available in libraries like Thrust.
