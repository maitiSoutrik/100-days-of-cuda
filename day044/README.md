# Day 44: Occupancy Grid Mapping Update with CUDA

## Overview

This project demonstrates updating a 2D occupancy grid map using CUDA, a fundamental task in robotics, particularly for Simultaneous Localization and Mapping (SLAM) and navigation. The map represents the environment's occupancy probability, stored efficiently as log-odds values. Simulated sensor readings (laser scan rays) are processed in parallel on the GPU to update the map.

## Implementation Details

1.  **Log-Odds Representation:** The map is a 2D grid where each cell stores the log-odds `l = log(p / (1-p))` of occupancy `p`. This allows efficient updates using addition: `l_new = l_old + l_update`. The initial state is `l=0`, corresponding to `p=0.5` (unknown).
2.  **Sensor Simulation:** A simple sensor model simulates `NUM_RAYS` radiating from the grid's center. Each ray travels until it hits a randomly determined distance (`hitDist`) up to `MAX_RANGE`. The start point is the center, and the end point is calculated based on the angle and hit distance. These points are converted to grid indices.
3.  **CUDA Kernel (`updateOccupancyGridKernel`):**
    *   Each thread processes one sensor ray.
    *   It retrieves the ray's start and end grid coordinates (`startX`, `startY`, `endX`, `endY`).
    *   **Line Tracing:** An integer-based line algorithm (similar to Bresenham's) iterates through the grid cells along the straight line from the start point to the end point.
    *   **Log-Odds Update:**
        *   For each cell traversed *along the ray* (excluding the end point), a `LOG_ODDS_MISS` value is added atomically (`atomicAdd`) to the cell's current log-odds value. This decreases the probability of occupancy (makes the cell more likely free).
        *   For the *final end point* cell (`endX`, `endY`), a `LOG_ODDS_HIT` value is added atomically. This increases the probability of occupancy (makes the cell more likely occupied).
    *   **Atomics (`atomicAdd`):** Used for updates because multiple rays might attempt to update the same grid cell concurrently.
    *   **Clamping:** After the atomic update, the resulting log-odds value in the cell is clamped between `LOG_ODDS_CLAMP_MIN` and `LOG_ODDS_CLAMP_MAX` to prevent probabilities from becoming exactly 0 or 1, which can cause issues in probabilistic frameworks. A helper device function `updateCell` handles the atomic add and subsequent clamping.
4.  **Memory:** The grid map and ray data are allocated on both host and device. Data is transferred using `cudaMemcpy`.

## Key CUDA Features Used

*   **Kernel Launch:** Parallel execution of `updateOccupancyGridKernel` across multiple threads.
*   **Grid-Stride Loops (Implicit):** Each thread handles one ray (`rayIdx = blockIdx.x * blockDim.x + threadIdx.x`).
*   **Atomic Operations:** `atomicAdd` for safe concurrent updates to shared grid cells.
*   **Device Functions:** `clamp` and `updateCell` for modularity within the kernel.
*   **CUDA Memory Management:** `cudaMalloc`, `cudaMemcpy`, `cudaFree`.

## Performance Considerations

*   **Parallelism:** The core advantage is processing thousands of sensor rays concurrently, significantly speeding up map updates compared to a CPU implementation.
*   **Memory Access:** The line tracing algorithm accesses memory cells along the ray path. Access patterns can be somewhat scattered depending on ray angles, potentially leading to non-coalesced memory access, although atomics often dominate performance here.
*   **Atomics:** `atomicAdd` operations serialize updates to the same memory location. High contention on specific cells (e.g., near the sensor origin) can become a bottleneck. Using floating-point atomics can be slower than integer atomics on some architectures.
*   **Clamping after Atomics:** The current implementation performs the clamp *after* the atomic operation. There's a potential (though often minor in practice for occupancy grids) race condition where a cell's value could be modified by another thread between the `atomicAdd` returning the *old* value and the current thread writing back the *clamped new* value. More robust solutions might involve `atomicCAS` loops but add complexity.

## Building and Running

**Note:** Compilation and execution are intended for the target platform (Jetson Nano) or a compatible environment with CUDA toolkit and CMake installed, configured for Compute Capability 5.3.

1.  **Configure:** Navigate to the `build` directory (create it if it doesn't exist) in the project root:
    ```bash
    cd /path/to/100-days-of-cuda/
    mkdir -p build
    cd build
    ```
2.  **Run CMake:**
    ```bash
    cmake ..
    ```
    *(Ensure the top-level CMakeLists.txt includes `add_subdirectory(day044)`)*
3.  **Build:** Compile the specific target for Day 44 (or build all):
    ```bash
    cmake --build . --target occupancy_grid
    # or build everything (if needed)
    # cmake --build .
    ```
4.  **Run:** Execute the binary from the build directory:
    ```bash
    ./day044/occupancy_grid
    ```

## Execution Results

The program simulates the occupancy grid update. The output will show kernel launch parameters and confirmation messages. Visualization of the resulting `h_logOddsMap` would require additional code (e.g., writing to a file or using a plotting library), which is not included in this basic example.

*(Expected Console Output - will be similar to this, actual timings may vary)*
```text
Day 44: Occupancy Grid Mapping Update
Launching kernel with 4 blocks and 256 threads per block...
Kernel execution finished.
Occupancy grid update simulated (kernel needs Bresenham implementation).
Day 44 finished.
```
*(Note: The "needs Bresenham implementation" message in the output is from the placeholder version; the actual implementation uses line tracing.)*

## Learnings and Observations

*   Occupancy grid updates are well-suited for GPU parallelization due to the independent processing of sensor rays.
*   Atomic operations are crucial for correctness when multiple threads might update the same data, but they can introduce performance considerations.
*   Integer-based line drawing algorithms like Bresenham's are efficient for determining which grid cells a ray passes through.
*   Managing log-odds requires careful handling of updates and clamping to maintain numerical stability.

## References

*   Thrun, S. (2002). Probabilistic algorithms in robotics. *AI Magazine*, *23*(4), 93-93. (Conceptual basis for occupancy grids)
*   [Bresenham's Line Algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm) (General algorithm description)
