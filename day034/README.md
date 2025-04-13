# Day 34: Point Cloud Voxel Grid Filter

## Overview

This project implements a Voxel Grid filter for downsampling point clouds using CUDA. The filter divides the 3D space containing the point cloud into a grid of uniform voxels. All points falling within the same voxel are approximated by their centroid. This is a common preprocessing step for large point clouds to reduce computational load while retaining the overall shape characteristics.

## Implementation Details

The core idea is to:
1.  Determine the bounding box of the point cloud.
2.  Define a voxel grid resolution (voxel size).
3.  Calculate the 3D grid dimensions based on the bounding box and voxel size.
4.  For each point in the input cloud:
    - Calculate the 3D index of the voxel it belongs to (using integer division based on voxel size and minimum bounds).
    - Convert the 3D voxel index to a unique 1D linear index.
5.  Use atomic operations (`atomicAdd`) to accumulate the coordinates (X, Y, Z) and the count of points for each voxel index in separate device arrays.
6.  A second kernel (or a post-processing step on the host) computes the centroid for each non-empty voxel by dividing the summed coordinates by the point count for that voxel.
7.  The output is the set of these centroids, representing the downsampled point cloud.

The input data used is `table_scene_ascii.pcd` from Day 25, which requires host-side parsing before transferring data to the GPU.

## Key CUDA Features Used

-   `atomicAdd()`: For thread-safe accumulation of coordinates and counts per voxel.
-   Kernel Launch Configuration: Determining appropriate grid and block dimensions.
-   Device Memory Management: `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
-   3D Grid Indexing: Mapping 3D spatial coordinates to voxel indices.

## Performance Considerations

-   **Atomic Contention:** High contention on atomics can become a bottleneck if many points map to the same voxel frequently. The spatial distribution of points affects this.
-   **Memory Access:** Accessing the global memory arrays for atomic updates.
-   **Grid Size:** The number of voxels directly impacts the size of the atomic arrays and potential memory usage. A very fine grid can lead to large memory requirements.
-   **Kernel Fusion:** The centroid calculation could potentially be integrated or optimized depending on the approach.

## Building and Running

**Note:** Compilation and execution are intended for the target platform (Jetson Nano, Compute Capability 5.3) or a compatible environment, following the `.clinerules`.

1.  **Prerequisites:** CUDA Toolkit (>= 10.2 recommended), CMake (>= 3.10). The `table_scene_ascii.pcd` file needs to be accessible by the executable (e.g., copied to the build directory or accessed via an absolute path adjusted for the target system).
2.  **Configure:** Create a build directory and run CMake:
    ```bash
    mkdir build
    cd build
    cmake ..
    ```
3.  **Build:** Compile the project:
    ```bash
    make
    ```
4.  **Run:** Execute the compiled program (specify the path to the PCD file):
    ```bash
    ./day034_voxel_filter /path/to/table_scene_ascii.pcd <voxel_size_x> <voxel_size_y> <voxel_size_z>
    ```
    (Replace `/path/to/` with the actual path on the Jetson Nano, e.g., `/home/drboom/cuda-data-sets/`). Example voxel size: `0.05 0.05 0.05`.

## Execution Results / Output

*(To be filled in after running the code on the target platform)*

```
[Console output will go here]
```

*(Description of any generated output files/images if applicable)*

## Learnings and Observations

*(To be filled in after implementation and execution)*

## References

-   PCL VoxelGrid Filter Concept: <http://pointclouds.org/documentation/classpcl_1_1_voxel_grid.html>
-   CUDA Atomic Functions: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions>
