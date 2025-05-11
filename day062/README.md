# Day 62: Batched Vector L2 Norm Calculation

## Overview
This project implements a batched L2 norm calculation for a set of vectors using CUDA. The L2 norm (Euclidean norm) of a vector is the square root of the sum of the squares of its components. In a batched operation, we perform this calculation for many vectors simultaneously. This is a common pattern in various applications like machine learning, physics simulations, and robotics, where the same small operation needs to be applied to many independent data instances.

The key idea is to assign each vector in the batch to a separate CUDA thread block. Within each block, threads cooperatively compute the L2 norm of their assigned vector. This involves a parallel sum of squares of the vector components, followed by a square root. Shared memory is used within each block to perform an efficient parallel reduction for the sum of squares.

## Implementation Details

### CUDA Kernel: `batched_l2_norm_kernel`
-   **Input:**
    -   `d_vectors`: A device pointer to a flat array of floats representing `num_batches` vectors, each of `vector_dim` dimensions. Vectors are laid out contiguously.
    -   `d_norms`: A device pointer to an array where the computed L2 norms will be stored (one norm per batch).
    -   `num_batches`: The total number of vectors to process.
    -   `vector_dim`: The dimension of each input vector.
-   **Execution Configuration:**
    -   Grid Dimension: `num_batches` (each block processes one batch/vector).
    -   Block Dimension: A fixed size (e.g., 256 threads). The choice of block size can affect performance and should ideally be tuned.
    -   Shared Memory: Each block allocates shared memory of size `blockDim.x * sizeof(float)` to store partial sums of squares.
-   **Logic per Block (for one vector):**
    1.  **Identify Batch:** `blockIdx.x` determines which vector in the batch this block is responsible for.
    2.  **Partial Sum of Squares:** Each thread in the block computes the sum of squares for a subset of the elements in its assigned vector. This is done using a grid-stride loop: `for (int i = threadIdx.x; i < vector_dim; i += blockDim.x)`.
    3.  **Store in Shared Memory:** The partial sum computed by each thread is stored in an element of the shared memory array `s_data[threadIdx.x]`.
    4.  **Synchronization:** `__syncthreads()` ensures all threads in the block have completed their partial sum calculations before proceeding to the reduction.
    5.  **Shared Memory Reduction:** A standard parallel reduction algorithm is performed on the `s_data` array. In each step of the reduction, half of the active threads add a value from another part of the array to their own. `__syncthreads()` is called after each step to ensure correct synchronization. This continues until `s_data[0]` holds the total sum of squares for the vector.
    6.  **Final Calculation & Store:** Thread 0 of the block calculates the square root of `s_data[0]` and writes the result to the output array `d_norms[batch_idx]`.

### Host Code (`main.cu`)
-   Initializes a batch of vectors with random floating-point numbers.
-   Allocates memory on the host and device for the input vectors and output norms.
-   Copies input vectors from host to device.
-   Launches the `batched_l2_norm_kernel`.
    -   The number of blocks in the grid is `num_batches`.
    -   The number of threads per block is set (e.g., 256).
    -   The amount of dynamic shared memory is `blockDim.x * sizeof(float)`.
-   Copies the computed norms from device to host.
-   Performs the same L2 norm calculation on the CPU for verification.
-   Compares GPU and CPU results for correctness.
-   Prints timing information for both GPU and CPU computations and calculates speedup.
-   Frees device memory.

### Error Checking
-   The `CHECK_CUDA_ERROR` macro is used to check the return status of all CUDA API calls and kernel launches.

## Key CUDA Features Used
-   **Kernel Launch Configuration:** Defining grid and block dimensions (`<<<grid_size, block_size, shared_mem_size>>>`).
-   **Thread Indexing:** `blockIdx.x`, `threadIdx.x`, `blockDim.x`.
-   **Shared Memory:** `extern __shared__ float s_data[]` for intra-block communication and reduction.
-   **Synchronization:** `__syncthreads()` to coordinate threads within a block.
-   **Device Memory Management:** `cudaMalloc()`, `cudaMemcpy()`, `cudaFree()`.
-   **Error Handling:** `cudaGetLastError()` and `cudaGetErrorString()`.

## Performance Considerations
-   **Memory Coalescing:** Input vectors are processed by threads within a block. If `vector_dim` is large, threads access consecutive memory locations, leading to coalesced memory access.
-   **Shared Memory Usage:** Shared memory is used for fast reduction, reducing global memory traffic. The size of shared memory per SM is limited, which can constrain block size.
-   **Occupancy:** The choice of `block_size` affects SM occupancy. A balance is needed; too few threads per block might underutilize the SM, while too many might lead to resource limitations (registers, shared memory).
-   **Kernel Launch Overhead:** By batching many small operations into one kernel launch, the relative overhead of launching the kernel is significantly reduced compared to launching one kernel per vector.
-   **Data Transfer:** For very small vectors or few batches, the overhead of `cudaMemcpy` might dominate. Batched operations are most beneficial when the computation per batch is significant enough relative to data transfer and kernel launch overhead.
-   **Vector Dimension vs. Block Size:**
    -   If `vector_dim` is much larger than `block_size`, each thread processes multiple elements in a loop, which is efficient.
    -   If `vector_dim` is smaller than `block_size`, some threads in the block might do no work for the initial sum-of-squares part. The reduction step still utilizes the threads.
    -   The current implementation uses a fixed `block_size` (e.g., 256). For very small `vector_dim`, a smaller `block_size` might be more optimal, but the reduction benefits from power-of-2 block sizes.

## Building and Running
This project uses CMake. Ensure you have the CUDA Toolkit installed and CMake (version 3.18 or higher). These instructions are intended for the target build environment (e.g., Jetson Nano or a compatible Linux system with CUDA).

1.  **Navigate to the `day062` directory:**
    ```bash
    cd day062
    ```
2.  **Create a build directory and navigate into it:**
    ```bash
    mkdir build
    cd build
    ```
3.  **Run CMake to configure the project:**
    ```bash
    cmake ..
    ```
    (This will also fetch Google Test if not already present from a previous build in the root build directory).
4.  **Compile the project:**
    ```bash
    make
    ```
5.  **Run the main executable:**
    ```bash
    ./batched_l2_norm_main
    ```
6.  **Run the tests:**
    ```bash
    ctest
    # Or, run the test executable directly:
    # ./batched_l2_norm_test
    ```

## Execution Results
Output from `./batched_l2_norm_main` on Jetson Nano:
```
Batched L2 Norm Calculation
Number of batches: 1024
Vector dimension: 512
------------------------------------

Running GPU computation...
GPU computation finished.
GPU Norms (first 10 elements):
13.1257 12.9357 12.7825 13.3471 12.7641 13.0822 12.7237 13.2249 12.7215 12.9049 
GPU Time: 109.9983 ms

Running CPU computation for verification...
CPU computation finished.
CPU Norms (first 10 elements):
13.1257 12.9357 12.7825 13.3471 12.7641 13.0822 12.7237 13.2249 12.7215 12.9049 
CPU Time: 3.4390 ms

Verifying results...
Verification PASSED: GPU and CPU results match.
```

**Test Output:**
Output from `./day062/batched_l2_norm_test` on Jetson Nano:
```
[==========] Running 7 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 7 tests from BatchedL2NormTest
[ RUN      ] BatchedL2NormTest.HandlesEmptyInput
[       OK ] BatchedL2NormTest.HandlesEmptyInput (0 ms)
[ RUN      ] BatchedL2NormTest.HandlesZeroDimension
[       OK ] BatchedL2NormTest.HandlesZeroDimension (88 ms)
[ RUN      ] BatchedL2NormTest.SingleBatchSingleDimension
[       OK ] BatchedL2NormTest.SingleBatchSingleDimension (1 ms)
[ RUN      ] BatchedL2NormTest.MultipleBatchesSmallDimension
[       OK ] BatchedL2NormTest.MultipleBatchesSmallDimension (1 ms)
[ RUN      ] BatchedL2NormTest.LargerBatchesAndDimensions
[       OK ] BatchedL2NormTest.LargerBatchesAndDimensions (1 ms)
[ RUN      ] BatchedL2NormTest.DimensionLargerThanBlockSize
[       OK ] BatchedL2NormTest.DimensionLargerThanBlockSize (1 ms)
[ RUN      ] BatchedL2NormTest.AllZeroVectors
[       OK ] BatchedL2NormTest.AllZeroVectors (1 ms)
[----------] 7 tests from BatchedL2NormTest (95 ms total)

[----------] Global test environment tear-down
[==========] 7 tests from 1 test suite ran. (96 ms total)
[  PASSED  ] 7 tests.
```

## Learnings and Observations
-   Batched operations are effective for processing many small, independent tasks on the GPU by amortizing kernel launch overhead and improving hardware utilization.
-   Shared memory provides a significant speedup for reduction operations within a thread block by avoiding slow global memory accesses *compared to global memory reductions*.
-   The choice of block size is crucial for performance and depends on the problem size (vector dimension) and GPU architecture.
-   Careful synchronization (`__syncthreads()`) is essential when using shared memory to ensure correct data dependencies.
-   Comparing GPU results with a CPU implementation is vital for verifying correctness.
-   **Performance Anomaly (CPU vs. GPU):** The initial run of the main executable (`num_batches = 1024`, `vector_dim = 512`) on the Jetson Nano showed the CPU computation (3.4ms) to be significantly faster than the GPU computation (110ms). This can occur for several reasons:
    -   **Overhead Dominance:** For the given problem size, the overhead of CUDA kernel launch, memory transfers (even on unified memory systems like Jetson Nano, there are costs), and GPU-CPU synchronization might outweigh the benefits of parallel computation.
    -   **Small Work per Thread/Block:** If `vector_dim` is relatively small, the amount of parallel work per vector might not be enough to fully saturate the GPU's capabilities.
    -   **CPU Optimization:** Modern CPUs are highly optimized for sequential tasks and may have efficient SIMD execution for operations like sum-of-squares.
    -   **Jetson Nano Architecture:** The Jetson Nano has an integrated GPU that shares memory with the CPU. While this reduces explicit `cudaMemcpy` latency, the GPU itself is less powerful than discrete GPUs. The CPU is also relatively capable.
    -   The GPU implementation might benefit from further tuning (e.g., block size, launch configuration) or larger problem sizes (more batches, larger vector dimensions) to demonstrate a speedup over the CPU.
-   **Kernel Launch Configuration:** It's critical to ensure valid launch parameters (e.g., grid size > 0). The initial unit test failure highlighted this; a grid size of 0 is an invalid configuration.

## (Optional) Future Improvements
-   Experiment with different block sizes to find the optimal configuration for the Jetson Nano.
-   Implement the batched 2x2 or 3x3 matrix inversion (Option B from the problem description).
-   Explore using CUDA streams if there were multiple independent batches or other concurrent operations.
-   Investigate warp-level primitives (`__shfl_down_sync`, etc.) for the reduction step as an alternative to shared memory for certain architectures or problem sizes, though shared memory is generally robust.
