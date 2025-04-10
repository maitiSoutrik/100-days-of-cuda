# Day 32: CUDA Streams for Overlap

## Overview

This project demonstrates the use of CUDA streams to overlap kernel execution with memory transfers between the host and device. By performing data copies and kernel launches asynchronously in different streams, we aim to hide memory transfer latency and improve overall application throughput. This example modifies the matrix multiplication code from Day 3 to use this technique.

## Implementation Details

The implementation involves the following steps:

1.  **Stream Creation:** Create multiple CUDA streams (`cudaStream_t`). For this example, we'll use three: one for Host-to-Device (H2D) copy, one for kernel execution, and one for Device-to-Host (D2H) copy.
2.  **Event Creation:** Create CUDA events (`cudaEvent_t`) to synchronize operations between streams. An event recorded after the H2D copy will signal the kernel stream to begin. Another event recorded after the kernel completes will signal the D2H copy stream.
3.  **Asynchronous Operations:**
    *   Use `cudaMemcpyAsync` to copy input matrices from host to device memory on stream 1.
    *   Record an event on stream 1 after the H2D copy completes using `cudaEventRecord`.
    *   Make stream 2 wait for the H2D completion event using `cudaStreamWaitEvent`.
    *   Launch the matrix multiplication kernel on stream 2 using the `<<<..., stream>>>` syntax.
    *   Record an event on stream 2 after the kernel completes using `cudaEventRecord`.
    *   Make stream 3 wait for the kernel completion event using `cudaStreamWaitEvent`.
    *   Use `cudaMemcpyAsync` to copy the result matrix from device to host memory on stream 3.
    *   Record an event on stream 3 after the D2H copy completes.
4.  **Synchronization:** Use `cudaEventSynchronize` or `cudaStreamSynchronize` to wait for the final operation (D2H copy) to complete before verifying the result or measuring the total time.
5.  **Timing:** Measure the execution time with and without streams to quantify the performance difference.

## Key CUDA Features Used

-   `cudaStream_t`: CUDA stream handle.
-   `cudaStreamCreate()`: Creates an asynchronous stream.
-   `cudaStreamDestroy()`: Destroys a stream.
-   `cudaMemcpyAsync()`: Performs asynchronous memory copies. Requires pinned host memory for true asynchronous behavior.
-   `<<<gridDim, blockDim, sharedMem, stream>>>`: Kernel launch syntax specifying the stream.
-   `cudaEvent_t`: CUDA event handle.
-   `cudaEventCreate()`: Creates an event object.
-   `cudaEventDestroy()`: Destroys an event object.
-   `cudaEventRecord()`: Records an event in a stream.
-   `cudaStreamWaitEvent()`: Makes a stream wait for an event recorded in another stream.
-   `cudaEventSynchronize()`: Blocks the CPU thread until a specific event completes.
-   `cudaMallocHost()` / `cudaFreeHost()`: Allocate/free pinned (page-locked) host memory for optimal asynchronous transfers.

## Performance Considerations

-   **Overlap Potential:** The effectiveness of stream overlap depends on the GPU's ability to concurrently execute data transfers and kernels (requires sufficient hardware resources, typically Copy Engines and Compute Engines).
-   **Pinned Memory:** Using pinned host memory (allocated via `cudaMallocHost`) is crucial for achieving true asynchronous `cudaMemcpyAsync` operations, as it allows the GPU's Direct Memory Access (DMA) engine to access host memory without CPU intervention. Regular pageable memory requires staging through a pinned buffer managed by the driver, potentially reducing overlap.
-   **Kernel vs. Transfer Time:** Overlap is most beneficial when kernel execution time and data transfer times are significant and relatively balanced. If one dominates heavily, the gains from overlapping the other might be minimal.
-   **Number of Streams:** Using too many streams can sometimes add overhead. The optimal number depends on the application and hardware.

## Building and Running

**(To be executed in the target Jetson Nano environment or a compatible cross-compilation setup)**

1.  **Navigate to the build directory:**
    ```bash
    cd /path/to/100-days-of-cuda/build 
    ```
    *(Create `build` directory if it doesn't exist: `mkdir build && cd build`)*

2.  **Configure using CMake:**
    ```bash
    cmake .. 
    ```

3.  **Build the executable:**
    ```bash
    cmake --build . --target stream_overlap -- -j$(nproc) 
    # Or: make stream_overlap -j$(nproc)
    ```
   *(Ensure you are in the `build` directory created by CMake)*

4.  **Run the executable:**
    ```bash
    ./day032/stream_overlap <matrix_width> <matrix_height>
    # Example: ./day032/stream_overlap 1024 1024
    ```

## Execution Results

*(Placeholder: Add actual console output after running on Jetson Nano)*
```
[Execution Output Will Go Here]
```

## Learnings and Observations

*(Placeholder: Add observations about performance gains, implementation challenges, etc.)*

## References

-   CUDA C++ Programming Guide - Streams: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
-   GPU Pro Tip: CUDA Streams Maximize GPU Utilization: [https://developer.nvidia.com/blog/gpu-pro-tip-cuda-streams-maximize-gpu-utilization/](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-streams-maximize-gpu-utilization/)
-   CUDA Runtime API - Events: [https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
