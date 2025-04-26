# Day 48: Parallel AABB Collision Detection

## Overview

This project implements a parallel algorithm to detect collisions between multiple 3D objects represented by Axis-Aligned Bounding Boxes (AABBs) using CUDA. Each thread in the CUDA kernel is responsible for checking a unique pair of AABBs for intersection.

## Implementation Details

1.  **AABB Structure:** A simple `struct AABB` is defined using `float3` to store the minimum and maximum corner coordinates of the bounding box.
2.  **Collision Check Logic:** The core logic checks for overlap between two AABBs. Two AABBs collide if and only if they overlap on all three axes (X, Y, and Z). A helper `__device__` function `checkOverlap` determines overlap along a single axis.
3.  **Kernel (`checkAABBCollisionKernel`):**
    *   The kernel takes an array of `AABB` structs and an output boolean array `collisionResults`.
    *   The total number of unique pairs to check for `N` boxes is `N * (N - 1) / 2`.
    *   Each thread calculates a unique global index `idx`.
    *   A mapping function converts the linear `idx` into a unique pair of box indices `(i, j)` such that `i < j`. This ensures each pair is checked exactly once. The mapping uses the formula derived from solving the triangular number summation for `i` and then finding `j`.
    *   Threads outside the valid range of pair indices exit early.
    *   Each active thread fetches the AABBs for its assigned pair `(i, j)`.
    *   It performs the collision check using `checkOverlap` for X, Y, and Z axes.
    *   The result (true if collision, false otherwise) is written to `collisionResults[idx]`.
4.  **Host Code (`checkAABBCollisionGPU`, `main`):**
    *   The `main` function initializes a set of random AABBs.
    *   It allocates host and device memory for the AABBs and the boolean collision results.
    *   `checkAABBCollisionGPU` copies the AABB data to the GPU, calculates the necessary grid and block dimensions to cover all pairs, launches the kernel, synchronizes, and copies the results back to the host.
    *   CUDA events are used to measure GPU execution time.
5.  **CPU Verification (`checkAABBCollisionCPU`):**
    *   A simple nested loop iterates through all unique pairs `(i, j)` on the CPU.
    *   It performs the same AABB collision logic for each pair.
    *   The results are compared against the GPU results to verify correctness. CPU time is measured using `clock()`.

## Key CUDA Features Used

*   **Kernel Launch:** Launching a 1D grid of threads where each thread handles one task (checking one pair).
*   **Global Memory:** Reading AABB data from global memory and writing collision results back.
*   **`__device__` Functions:** Using `checkOverlap` for modularity within the kernel.
*   **`float3`:** Using the built-in vector type for coordinates.
*   **Thread Indexing:** Calculating global thread ID (`blockIdx.x * blockDim.x + threadIdx.x`).
*   **Index Mapping:** Converting a 1D thread index to a 2D pair index `(i, j)` for pairwise comparison.
*   **CUDA Error Handling:** Using `CHECK_CUDA_ERROR` macro.
*   **CUDA Events:** Measuring kernel execution time accurately.

## Performance Considerations

*   **Memory Access:** The kernel reads AABB data for each pair. Access patterns might not be fully coalesced depending on the `(i, j)` mapping and how AABBs are laid out in memory. However, each AABB is relatively small (2 `float3`).
*   **Branching:** The primary branching is the check `if (idx >= numPairs)` and potentially inside `checkOverlap` (though it's simple). The `if (i >= n || j >= n || i >= j)` check is a safeguard but shouldn't be hit frequently with the correct mapping.
*   **Computational Load:** Each thread performs a fixed number of comparisons and logical AND operations. The workload per thread is low.
*   **Scalability:** The approach scales well as the number of threads directly maps to the number of pairs. The number of pairs grows quadratically (`O(N^2)`) with the number of boxes (`N`). For very large `N`, the total number of pairs might exceed practical GPU limits or memory, requiring tiling or other strategies. The current implementation assumes all pairs fit within the launch parameters.
*   **Mapping Overhead:** The calculation to map `idx` to `(i, j)` involves `sqrt` and floating-point operations within the kernel, adding some overhead per thread.

## Building and Running

**Note:** Compilation and execution should be done in the target environment (Jetson Nano, CI/CD pipeline, or compatible cross-compilation setup).

1.  **Navigate to the Build Directory:**
    ```bash
    cd /path/to/100-days-of-cuda/build
    ```
2.  **Configure using CMake:** (Run from the build directory)
    ```bash
    cmake ..
    ```
    *(Ensure the `day048` subdirectory is added to the root `CMakeLists.txt`)*
3.  **Build the Executable:**
    ```bash
    cmake --build . --target aabb_collision -j$(nproc)
    ```
    *(Or use `make aabb_collision`)*
4.  **Run the Executable:**
    ```bash
    ./day048/aabb_collision
    ```

## Execution Results

The following output was obtained by running the code with `numBoxes = 2000` on an NVIDIA Jetson Nano:

```
Starting AABB Collision Check for 2000 boxes.
Total unique pairs to check: 1999000
Generating 2000 random AABBs...
AABB generation complete.
Running GPU collision check...
GPU execution time: 195.532 ms
Running CPU collision check for verification...
CPU execution time: 39.794 ms
Verifying results...
Verification complete.
Total Collisions (GPU): 13574 / 1999000
Total Collisions (CPU): 13574 / 1999000
SUCCESS: GPU and CPU results match!
Day 48 execution finished.
```

## Learnings and Observations

*   **Surprising Performance:** Contrary to initial expectations, the GPU implementation (approx. 196 ms) was significantly slower than the single-threaded CPU implementation (approx. 40 ms) on the Jetson Nano for 2000 boxes (~2 million pairs).
*   **Potential Bottlenecks on Nano:**
    *   **Kernel Overhead:** The overhead of launching the kernel, copying data (~1.9MB results, ~47KB boxes), and thread synchronization might be substantial relative to the computation.
    *   **Index Mapping Cost:** The calculation to map the linear thread index `idx` to the pair `(i, j)` involves `sqrt` and floating-point division. This per-thread cost might be relatively high on the Nano's sm_53 architecture GPU compared to its ARM CPU cores.
    *   **Memory Access:** While the computation per pair is low, reading two AABB structs from global memory per thread might lead to suboptimal memory access patterns (lack of coalescing) that limit performance on the Nano's memory subsystem.
    *   **Efficient CPU:** The sequential CPU code consists of tight loops performing simple comparisons, likely benefiting from CPU cache and efficient execution on the Nano's ARM cores.
*   **Workload Characteristics:** This specific problem has relatively low computational intensity per pair compared to the overhead of managing parallelism and memory access on the GPU, especially on an embedded platform like the Jetson Nano.
*   **Verification:** The CPU implementation remains crucial for verifying the correctness of the GPU kernel's logic and the index mapping, which was successful.

## (Optional) Future Improvements

*   Explore alternative index mapping strategies that might be computationally cheaper.
*   Implement spatial partitioning (e.g., octrees, grids) on the host or GPU to reduce the number of pairs that actually need checking, especially for sparse collision scenarios. This would change the kernel significantly.
*   Use shared memory if checking collisions within localized spatial regions (not applicable to this all-pairs approach).
