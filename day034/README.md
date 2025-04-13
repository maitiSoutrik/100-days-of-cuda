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

The code was compiled and executed on the Jetson Nano using the following command:
```bash
./voxel_filter /home/drboom/cuda-data-sets/table_scene_ascii.pcd 0.05 0.05 0.05
```

The console output was:
```
Using Voxel Size: (0.05, 0.05, 0.05)
Warning: Number of points read (460399) does not match header (460400).
Successfully loaded 460399 points from /home/drboom/cuda-data-sets/table_scene_ascii.pcd
Bounding Box:
  Min: (-1.1263, -0.6922, -1.9211)
  Max: (0.92967, 0.53329, -1.0252)
PCD Loading time: 2529.62 ms
Grid Dimensions: (42, 25, 18)
Total Voxels: 18900

--- Voxel Grid Filter Results ---
Input points: 460399
Output points (centroids): 2194

--- Performance Timings ---
Host->Device Transfer: 5.19892 ms
Voxel Hash Kernel:     37.7115 ms
Centroid Kernel:       0.402093 ms
Device->Host Transfer: 0.403811 ms
Total GPU processing (Kernels + D2H): 38.5174 ms

Voxel grid filtering completed successfully.
```
The filter successfully downsampled the input point cloud from 460,399 points to 2,194 centroid points using a 5cm voxel grid.

## Learnings and Observations

- The host-side ASCII PCD parsing is quite slow (~2.5 seconds) compared to the GPU processing time (~38.5 ms). For performance-critical applications, using a binary PCD format or a more optimized parser would be beneficial.
- The warning about reading one fewer point than the header indicates might suggest a trailing newline or slight formatting issue in the input PCD file, but the process completed successfully with the points read.
- The Voxel Hashing kernel, which involves atomic operations across the grid, takes the majority of the GPU time (37.7 ms). Atomic contention could be a factor, depending on point distribution.
- The Centroid Calculation kernel is very fast (0.4 ms) as it processes a much smaller number of voxels.
- Memory usage scales with the number of voxels. The chosen 0.05m voxel size resulted in 18,900 voxels, which is manageable. A much smaller voxel size could lead to significantly higher memory requirements for the atomic accumulation arrays.

## References

-   PCL VoxelGrid Filter Concept: <http://pointclouds.org/documentation/classpcl_1_1_voxel_grid.html>
-   CUDA Atomic Functions: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions>
