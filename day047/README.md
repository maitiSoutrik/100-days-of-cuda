# Day 47: Dynamic Parallelism (Simple Example)

## Overview

This example demonstrates CUDA Dynamic Parallelism, a feature allowing a GPU kernel (parent) to launch other GPU kernels (child) directly from the device. This is useful for algorithms where the workload is determined dynamically during execution, such as adaptive mesh refinement or certain recursive patterns.

This simple example uses a parent kernel where each thread launches a single child kernel. The child kernel performs a trivial task: writing a unique identifier (based on its parent thread ID) into an output array.

## Implementation Details

-   **`child_kernel`**: A simple `__global__` function that takes a pointer to the output array (`output_data`), the parent thread's ID (`parent_tid`), and a child ID (`child_tid`). It calculates an index based on the `parent_tid` and writes a value (`parent_tid * 1000 + child_tid`) to that location.
-   **`parent_kernel`**: A `__global__` function where each thread checks if it's within the bounds (`tid < size`). If so, it launches the `child_kernel` using the `<<<grid, block>>>` syntax directly within the parent kernel's code. In this example, each parent thread launches a child kernel with a 1x1 grid and 1x1 block (`child_kernel<<<1, 1>>>`).
-   **Synchronization:** `cudaDeviceSynchronize()` is called *within* the parent kernel after launching the child. This ensures that each parent thread waits for its *own* launched child kernel to complete before the parent thread finishes. While not strictly necessary here because each thread writes to a unique location, it demonstrates the synchronization mechanism required when child results are needed by the parent or when there are potential data dependencies.
-   **Host Code:** The `main` function handles:
    -   Allocating host and device memory (`h_output_data`, `d_output_data`).
    -   Initializing device memory.
    -   Launching the `parent_kernel`.
    -   Synchronizing the host with the device using `cudaDeviceSynchronize()` to wait for the parent (and its children) to complete.
    -   Copying results back to the host.
    -   Verifying the results by checking if `h_output_data[i]` contains the expected value `i * 1000 + 0`.
    -   Cleaning up device memory.

## Key CUDA Features Used

-   **Dynamic Parallelism:** Launching kernels (`child_kernel`) from within another kernel (`parent_kernel`).
-   **Kernel Launch Syntax:** Standard `<<<grid, block>>>` for host-to-device launches, and the same syntax used within the device kernel for dynamic launches.
-   **Device Synchronization (Intra-Kernel):** Using `cudaDeviceSynchronize()` within a kernel to wait for child kernels launched *by that thread* to complete.
-   **Compilation Flags:** Requires `-rdc=true` (Relocatable Device Code) flag during compilation.
-   **Linking:** Requires linking against the CUDA device runtime library (`cudadevrt`).

## Performance Considerations (Jetson Nano - CC 5.3)

-   Dynamic Parallelism introduces some overhead compared to static launches from the host. The overhead includes kernel launch latency and resource management on the device.
-   On devices with lower compute capability like the Jetson Nano (CC 5.3), this overhead might be more noticeable, especially for very small child kernels.
-   The maximum depth of nested launches is limited (see CUDA documentation), though this simple example only uses one level.
-   While supported on CC 5.3, it's generally recommended for scenarios where the dynamic nature truly simplifies the algorithm or handles unpredictable workloads, rather than for simple, predictable parallelism.
-   The `cudaDeviceSynchronize()` inside the parent kernel adds synchronization overhead for each parent thread.

## Building and Running (On Target: Jetson Nano or compatible environment)

1.  **Ensure CUDA Toolkit and CMake are installed** on the target build environment.
2.  **Navigate to the root directory** of the `100-days-of-cuda` project.
3.  **Create a build directory (if it doesn't exist) and navigate into it:**
    ```bash
    mkdir -p build
    cd build
    ```
4.  **Run CMake:** Ensure the top-level `CMakeLists.txt` includes `add_subdirectory(day047)`.
    ```bash
    cmake ..
    ```
5.  **Build the specific target for Day 47:**
    ```bash
    cmake --build . --target dynamic_parallelism_example -j$(nproc)
    ```
    (Or build all targets with `cmake --build . -j$(nproc)`)
6.  **Run the executable:**
    ```bash
    ./day047/dynamic_parallelism_example
    ```

## Execution Results / Output (Jetson Nano)

The following is the actual output from running the compiled code on a Jetson Nano:

```
Running Dynamic Parallelism Example with size: 256
Launching parent kernel with 2 blocks and 128 threads per block.
Parent kernel execution completed.
Kernel Execution Time (including children): 8.59453 ms
Results copied back to host.
Verification successful!
First few output values: 0 1000 2000 3000 4000 5000 6000 7000 8000 9000 
Device memory freed.
```

*The reported kernel execution time includes the overhead of launching child kernels and the internal `cudaDeviceSynchronize()` calls within the parent kernel.*

## Learnings and Observations

-   Dynamic parallelism provides flexibility but requires careful handling of compilation (`-rdc=true`) and linking (`cudadevrt`).
-   Synchronization within kernels (`cudaDeviceSynchronize()`) is necessary when parent threads depend on child results.
-   Understanding the overhead and limitations is crucial, especially on resource-constrained devices like the Jetson Nano.

## References

-   CUDA C Programming Guide - Dynamic Parallelism: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dynamic-parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dynamic-parallelism)
