# Day 39: Thrust Library Basics

## Overview

This day explores the basics of the Thrust library, a powerful C++ template library for CUDA built upon the concepts of the Standard Template Library (STL). Thrust provides high-level interfaces for common parallel algorithms like reduction, scan, and sort, simplifying CUDA code development and often leveraging highly optimized underlying implementations.

The goal is to demonstrate the use of `thrust::reduce`, `thrust::inclusive_scan`, and `thrust::sort` with `thrust::device_vector` and compare the code complexity against manual kernel implementations from previous days (e.g., Day 4 for reduction, Day 8 for scan, Day 11 for sort).

## Implementation Details

The `thrust_basics.cu` file implements three core examples:

1.  **Reduction:**
    *   A `thrust::device_vector<int>` is initialized with pseudo-random integers (0-99).
    *   `thrust::reduce` is used with `thrust::plus<int>` to compute the sum of all elements on the GPU.
    *   Timing is measured using CUDA events.
    *   Verification against a CPU-based sum is performed for smaller vector sizes.

2.  **Inclusive Scan (Prefix Sum):**
    *   A `thrust::device_vector<int>` is initialized with all 1s.
    *   `thrust::inclusive_scan` is used to compute the prefix sum. The expected result for an input of all 1s is `1, 2, 3, ..., N`.
    *   Timing is measured using CUDA events.
    *   Verification checks if the last element of the output vector equals the total number of elements (`N`).

3.  **Sort:**
    *   A `thrust::device_vector<int>` is initialized in reverse sorted order (N-1 down to 0), representing a challenging case for sorting algorithms.
    *   `thrust::sort` is called to sort the vector in ascending order.
    *   Timing is measured using CUDA events.
    *   Verification uses `thrust::equal` to compare the sorted device vector against an expected sequence generated using `thrust::sequence` directly on the device.

## Key CUDA Features Used

*   **Thrust Library:** The core focus of this day.
    *   `thrust::device_vector`: Container for managing data on the GPU device.
    *   `thrust::host_vector`: Container for managing data on the CPU host (used for verification).
    *   `thrust::reduce`: Parallel reduction algorithm.
    *   `thrust::inclusive_scan`: Parallel prefix sum algorithm.
    *   `thrust::sort`: Parallel sorting algorithm.
    *   `thrust::fill`: Initialize vector elements with a constant value.
    *   `thrust::sequence`: Generate sequences (e.g., 0, 1, 2,...).
    *   `thrust::transform`: Apply a function element-wise to a range.
    *   `thrust::equal`: Compare two ranges element-wise on the device.
    *   `thrust::plus<int>`: Standard functor for addition.
*   **CUDA Events:** Used for accurate timing of GPU operations (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`).
*   **Error Handling:** `CHECK_CUDA_ERROR` macro for robust error checking.

## Performance Considerations

*   **Abstraction vs. Control:** Thrust offers high-level abstraction, simplifying code significantly compared to writing custom kernels. However, this comes at the cost of fine-grained control over execution (e.g., block/thread configuration, shared memory usage is hidden).
*   **Optimized Algorithms:** Thrust algorithms are generally highly optimized for various GPU architectures. They often select the best underlying implementation based on data size and type.
*   **Overhead:** There might be slight overhead associated with Thrust's templating and function call mechanisms compared to a bare-metal kernel, but the optimized algorithms usually compensate for this, especially for large datasets.
*   **Data Transfers:** Implicit data transfers can occur if host iterators are used with Thrust algorithms operating on device data, but using `device_vector` and device iterators (`.begin()`, `.end()`) ensures operations stay on the GPU. Explicit transfers (`thrust::copy` or vector assignment `h_vec = d_vec`) are used for setup and verification.

## Building and Running

**Note:** Compilation and execution should be performed on the target Jetson Nano platform or a compatible environment with CUDA Toolkit and CMake installed.

1.  **Navigate to the build directory:**
    ```bash
    cd /path/to/100-days-of-cuda/build
    ```
    (Create the build directory if it doesn't exist: `mkdir build && cd build`)

2.  **Configure using CMake:** From the `build` directory:
    ```bash
    cmake ..
    ```

3.  **Build the executable:**
    ```bash
    cmake --build . --target thrust_examples -- -j$(nproc)
    # Or simply 'make thrust_examples' if using Makefiles
    ```

4.  **Run the executable:**
    ```bash
    ./day039/thrust_examples
    ```

## Execution Results

**(Execution results from Jetson Nano via manual run on 2025-04-17)**

```
--- Thrust Reduction Example ---
Thrust reduction sum: 51903600
Thrust reduction time: 2.16193 ms
Skipping CPU verification for reduction due to large N.

--- Thrust Inclusive Scan Example ---
Thrust inclusive_scan time: 6.23208 ms
Last element of scan result: 1048576
Verification PASSED for inclusive scan (last element).

--- Thrust Sort Example ---
Thrust sort time: 109.267 ms
Verification PASSED for sort.

All Thrust examples completed.
```

*(Performance times recorded on Jetson Nano. Times may vary based on the specific model, clock speeds, and system state during execution.)*

## Learnings and Observations

*   Thrust significantly reduces the amount of boilerplate code required for common parallel patterns like reduction, scan, and sort compared to manual kernel implementation.
*   Using `thrust::device_vector` handles device memory allocation/deallocation automatically, reducing the risk of memory leaks.
*   Thrust algorithms often provide excellent performance out-of-the-box due to their optimized nature.
*   Verification using Thrust functions like `thrust::equal` can be performed efficiently on the device itself.
*   Understanding the difference between host and device vectors/iterators is crucial for ensuring operations execute on the GPU as intended.

## References

*   Thrust Quick Start Guide: [https://docs.nvidia.com/cuda/thrust/index.html](https://docs.nvidia.com/cuda/thrust/index.html)
*   Thrust GitHub Repository (Examples): [https://github.com/NVIDIA/thrust/tree/main/examples](https://github.com/NVIDIA/thrust/tree/main/examples)
