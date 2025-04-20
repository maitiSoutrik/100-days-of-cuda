# Day 41: Parallel Radix Sort (Basic Single Pass)

## Overview

This project implements a single pass of a GPU-accelerated radix sort algorithm. Radix sort is a non-comparison integer sorting algorithm that sorts data with integer keys by grouping keys by the individual digits which share the same significant position and value. This implementation focuses on demonstrating the core components of one pass: histogram calculation, prefix sum (scan) of the histogram, and scattering elements to their partially sorted positions based on the least significant bits.

## Implementation Details

The goal is to sort an array of `unsigned int` values based on a specific group of bits (determined by `BITS_PER_PASS`). This involves three main steps executed on the GPU:

1.  **Histogram Calculation (`histogram_kernel`):**
    *   Each block calculates a local histogram for a portion of the input array based on the selected bits (`(value >> bit_shift) & (num_buckets - 1)`).
    *   Shared memory (`local_hist`) is used within each block to efficiently compute the block's histogram using atomic operations (`atomicAdd`).
    *   The local histograms are then atomically added to a global histogram array (`d_histogram`).

2.  **Prefix Sum (Scan):**
    *   The `thrust::exclusive_scan` function is used to compute the prefix sum of the global histogram. This converts the counts in the histogram into starting offsets for each bucket in the output array.

3.  **Scatter (`scatter_kernel`):**
    *   Each thread reads an element from the input array.
    *   It determines the target bucket based on the selected bits.
    *   Using the scanned offsets loaded into shared memory (`local_hist_offsets`), it calculates the correct position for the element in the output array.
    *   Shared memory (`local_scatter_pos`) is used with `atomicAdd` to manage the writing position within each bucket for threads in the same block, ensuring elements going to the same bucket are placed contiguously based on the original offsets.
    *   The element is written (scattered) to its calculated position in the global output array (`d_output`).

A simple CPU version (`cpu_radix_sort_pass`) implementing the same single-pass logic is included for comparison. A full `std::sort` is also timed as a baseline.

The input array size is increased to `N = 2^20` (1,048,576 elements) to better observe potential performance differences between the CPU and GPU for this single pass.

## Key CUDA Features Used

*   **Kernel Launch:** Standard `<<<...>>>` syntax.
*   **Shared Memory:** Used in both `histogram_kernel` and `scatter_kernel` for efficient intra-block communication and intermediate storage (local histograms, offsets, scatter positions).
*   **Atomic Operations:** `atomicAdd` is used for building histograms safely across threads and for managing scatter positions within buckets in the `scatter_kernel`.
*   **Bitwise Operations:** Right shift (`>>`) and bitwise AND (`&`) are used to extract the relevant bits for the current radix sort pass.
*   **Thrust Library:** `thrust::exclusive_scan` for efficient parallel prefix sum calculation on the GPU.
*   **CUDA Events:** Used for accurate timing of the GPU execution part.

## Performance Considerations

*   **Histogram:** The histogram kernel benefits from shared memory to reduce global memory atomics, although atomics are still needed to combine block results. Performance depends on the distribution of keys and potential contention on atomic operations.
*   **Scan:** Thrust's scan is highly optimized.
*   **Scatter:** The scatter kernel involves irregular memory access patterns as elements are written to potentially distant locations in the output array based on their bucket. Shared memory helps manage the write indices within a block efficiently. Memory coalescing is generally not achievable in the scatter phase due to the nature of the operation.
*   **Data Size:** The larger data size (`2^20`) should favor the GPU due to its massive parallelism, especially if memory bandwidth is the bottleneck.

## Building and Running

Follow the standard CMake build process within the `build` directory (intended for the Jetson Nano or compatible environment):

```bash
# cd build
# cmake ..
# make day041_radix_sort # Or just 'make'
# ./day041/radix_sort
```

## Execution Results

**(To be filled in after running on the target platform)**

```
Radix Sort (Single Pass Example)
Array size: 1048576 (2^20)
Bits per pass: 4
Number of buckets: 16
Generating random data...

--- GPU Radix Sort (Pass 0) ---
GPU Pass 0 Time: 34.248 ms

--- CPU Radix Sort (Pass 0) ---
CPU Pass 0 Time: 16 ms

--- Full CPU Sort (for reference) ---
std::sort Time: 190 ms
std::sort verification: Passed

Finished.
```

*(Timings collected from Jetson Nano execution via CI/CD)*

## Performance Analysis

*   **GPU vs CPU (Single Pass):** For this single pass implementation on the Jetson Nano with N=2^20, the CPU version (16 ms) was significantly faster than the GPU version (34.248 ms).
*   **Reasoning:** This result is likely due to the substantial overhead associated with launching CUDA kernels, memory management (allocations, potential implicit transfers not timed), and synchronization, especially for only one pass. The custom histogram and scatter kernels, relying on atomics and non-coalesced access, might also be less efficient than the CPU's cache-friendly sequential processing for this specific pass. The Jetson Nano's GPU, while parallel, has limited power compared to its ARM CPU cores, making the overhead more impactful.
*   **Comparison to Full Sort:** The standard `std::sort` on the CPU took 190 ms, which is much longer than either single pass. This suggests that a full multi-pass GPU radix sort *could* potentially outperform the full CPU sort by keeping data on the device and amortizing the overhead, but this single-pass test doesn't demonstrate that.
*   **Bottlenecks:** Potential bottlenecks for the GPU version include kernel launch overhead, atomic operation contention (especially in the histogram), and irregular memory access patterns in the scatter phase.

## Learnings and Observations

*   Implementing the core steps of radix sort (histogram, scan, scatter) on the GPU.
*   Using shared memory effectively for intermediate calculations (histograms, scatter positions).
*   Leveraging Thrust for optimized primitives like scan.
*   Understanding the memory access patterns involved in histogramming and scattering.
*   Observing the performance difference between CPU and GPU for a single pass on a large dataset.

## (Optional) Future Improvements

*   Implement the full multi-pass radix sort algorithm.
*   Compare performance with a full Thrust sort (`thrust::sort`).
*   Optimize kernels further (e.g., different histogram strategies, warp-level primitives if applicable).
*   Handle different data types (e.g., floats by interpreting bits).
