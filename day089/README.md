# Day 89: Memory Coalescing Demonstration

## Overview

Today's exercise focuses on a fundamental CUDA optimization technique: **memory coalescing**. Global memory accesses are often the bottleneck in CUDA kernels. When threads within a warp access global memory, these accesses can be "coalesced" into a single (or few) memory transaction(s) if the threads access contiguous or properly aligned memory locations. Uncoalesced accesses, where threads access scattered memory locations, lead to multiple memory transactions, significantly degrading performance.

This project demonstrates the performance difference between a kernel with coalesced memory access and one with uncoalesced memory access for a simple scaled copy operation (`output[i] = input[i] * scalar`).

## Implementation Details

1.  **`coalesced_access_kernel`**:
    *   Each thread `idx = blockIdx.x * blockDim.x + threadIdx.x` accesses `input[idx]` and writes to `output[idx]`.
    *   Since `idx` increments linearly for consecutive threads in a warp, their memory accesses to `input` and `output` arrays are perfectly contiguous, leading to coalesced memory transactions.

2.  **`uncoalesced_access_kernel`**:
    *   This kernel performs the same logical operation but accesses memory in a strided pattern designed to be uncoalesced.
    *   A `global_thread_idx` is calculated.
    *   This `global_thread_idx` is then mapped to an `uncoalesced_idx` using the formula:
        `uncoalesced_idx = (global_thread_idx % elements_per_stride_group) * stride_factor + (global_thread_idx / elements_per_stride_group)`
        where `stride_factor` determines the "distance" between accesses for threads that would normally be contiguous.
    *   For example, with `stride_factor = 32`, thread 0 might access element 0, thread 1 might access element 32, thread 2 element 64, and so on (depending on `elements_per_stride_group`). This forces threads within the same warp to access memory locations that are far apart, breaking coalescing.
    *   The kernel computes `output[uncoalesced_idx] = input[uncoalesced_idx] * scalar`.

3.  **`main()` (in `memory_coalescing_main.cu`):**
    *   Initializes a large input array.
    *   Runs the `coalesced_access_kernel` and times its execution using CUDA events.
    *   Runs the `uncoalesced_access_kernel` (with a specified `stride_factor`, e.g., 32) and times its execution.
    *   Verifies that both kernels produce the correct (though potentially permuted for the uncoalesced version) results.
    *   Prints the execution times for both kernels, highlighting the performance difference.

## Key CUDA Concepts Demonstrated

*   **Global Memory Access Patterns:** Understanding how threads access global memory.
*   **Memory Coalescing:** The GPU's ability to group memory accesses from threads in a warp into fewer transactions.
*   **Warp Execution:** How threads are grouped into warps and execute in SIMT fashion.
*   **Performance Impact:** Quantifying the significant speedup achieved by ensuring coalesced memory accesses.
*   **CUDA Events for Timing:** Accurate measurement of kernel execution time.

## Building and Running

1.  **Navigate to the build directory:**
    ```bash
    cd /path/to/100-days-of-cuda/build
    ```
2.  **Run CMake and Make:**
    ```bash
    cmake ..
    make memory_coalescing_main # Target name from day089/CMakeLists.txt
    ```
3.  **Execute the binary (on Jetson Nano or compatible environment):**
    ```bash
    ./day089/memory_coalescing_main
    ```

## Actual Execution Results / Output

Here's the output from running on the Jetson Nano:

```bash
drboom@JetNano ~/g/1/build> ./day089/memory_coalescing_main 
Number of elements: 16777216
Scalar: 2.5
Stride factor for uncoalesced kernel: 32

--- Running Coalesced Access Kernel ---
Coalesced Kernel Execution Time: 31.363 ms
Coalesced Output (first 16 elements): [0.000, 2.500, 5.000, 7.500, 10.000, 12.500, 15.000, 17.500, 20.000, 22.500, 25.000, 27.500, 30.000, 32.500, 35.000, 37.500]
Verification PASSED!

--- Running Uncoalesced Access Kernel ---
Uncoalesced Kernel Execution Time: 210.194 ms
Uncoalesced Output (first 16 elements): [0.000, 2.500, 5.000, 7.500, 10.000, 12.500, 15.000, 17.500, 20.000, 22.500, 25.000, 27.500, 30.000, 32.500, 35.000, 37.500]
Verification PASSED!
```

And the test results:
```bash
drboom@JetNano ~/g/1/build> ./day089/memory_coalescing_test 
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from MemoryCoalescingTest
[ RUN      ] MemoryCoalescingTest.CoalescedKernelCorrectness
[       OK ] MemoryCoalescingTest.CoalescedKernelCorrectness (89 ms)
[ RUN      ] MemoryCoalescingTest.UncoalescedKernelCorrectness
[       OK ] MemoryCoalescingTest.UncoalescedKernelCorrectness (2 ms)
[----------] 2 tests from MemoryCoalescingTest (91 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (92 ms total)
[  PASSED  ] 2 tests.
```

## Learnings and Observations

*   Memory coalescing is crucial for achieving good performance in CUDA.
*   Even simple changes in access patterns can lead to dramatic differences in execution speed.
*   Always analyze and optimize global memory access patterns in your CUDA kernels.
*   Using a profiler (like Nsight Systems or Nsight Compute) can help identify memory-bound kernels and uncoalesced accesses in more complex scenarios.

