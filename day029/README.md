# Day 29: K-Means Assignment Step (with File Input)

## Overview

This program implements the parallel assignment step of the K-Means clustering algorithm using CUDA. It assigns each data point to the nearest centroid. The program supports two modes for input data:
1.  **Synthetic Data:** Generates random 2D points clustered around predefined centers.
2.  **File Input:** Reads 2D point data from a specified text file (e.g., `Aggregation.txt`).

The program takes command-line arguments to control the number of clusters, input mode, data source/size, and optional output file for assignments.

## Implementation Details

-   **Data Structures:** `float2` is used for 2D points and centroids. `std::vector` is used on the host.
-   **Argument Parsing:** Uses `getopt_long` to parse command-line arguments (`--clusters`, `--mode`, `--points`, `--input`, `--output`, `--help`).
-   **Data Loading/Generation:**
    -   `load_points_from_file`: Reads X, Y coordinates line by line from the input file. Handles file and parsing errors.
    -   `generate_synthetic_data`: Creates synthetic points with Gaussian noise around randomly generated true centers.
-   **Centroid Initialization:** `initialize_centroids` randomly places initial centroids within the data bounds.
-   **CUDA Kernel (`assign_points_kernel`):** Each thread processes one point, calculating the squared Euclidean distance to all centroids and assigning the point to the closest one.
-   **CPU Verification:** `assign_points_cpu` performs the same assignment logic sequentially for comparison.
-   **Output:** Results (assignments) can be optionally saved to a file using `save_assignments`.

## Key CUDA Features Used

-   Basic CUDA kernel launch (`<<<...>>>`).
-   Device memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`).
-   `float2` vector type.
-   Error checking macros (`CHECK_CUDA_ERROR`).

## Performance Considerations

The kernel parallelizes the distance calculations and minimum finding across all points. Performance depends on the number of points and clusters. Using squared Euclidean distance avoids costly `sqrt` operations within the kernel's inner loop.

## Building and Running

**Note:** Compilation and execution are intended for the target platform (Jetson Nano or compatible environment with CUDA toolkit and CMake >= 3.10).

1.  **Navigate to the build directory:**
    ```bash
    cd build
    ```
2.  **Run CMake:**
    ```bash
    cmake ..
    ```
3.  **Build the executable:**
    ```bash
    make day029_k_means_assignment
    ```
4.  **Run the executable:**

    *   **Synthetic Data Example:**
        ```bash
        ./day029/k_means_assignment --clusters 10 --points 10000
        ```
    *   **File Input Example (using Aggregation.txt):**
        *(Assuming `Aggregation.txt` is available at `/home/drboom/cuda-data-sets/kmeans_data/Aggregation.txt` on the Jetson)*
        ```bash
        ./day029/k_means_assignment --clusters 7 --mode file --input /home/drboom/cuda-data-sets/kmeans_data/Aggregation.txt --output assignments.txt
        ```
    *   **Help:**
        ```bash
        ./day029/k_means_assignment --help
        ```

## Execution Results

*(Placeholder: Console output from running on Jetson Nano with Aggregation.txt will be added here after implementation and testing.)*

```
[Expected Output Format]
Parsed Arguments:
  Mode: file
  Clusters: 7
  Input File: /home/drboom/cuda-data-sets/kmeans_data/Aggregation.txt
  Output File: assignments.txt
Loading points from file: /home/drboom/cuda-data-sets/kmeans_data/Aggregation.txt
Loaded 788 points.
Initializing 7 centroids randomly within data bounds...
Allocating memory...
Copying data to device...
Launching CUDA kernel...
Copying results from device...
Running CPU verification...
Comparing GPU and CPU results...
Verification Successful! All 788 assignments match.
Saving assignments to assignments.txt...
Cleaning up memory...
Done.
```

## Learnings and Observations

*(Placeholder: Reflections on the implementation and results will be added here.)*

## Input File Format (`--input`)

The input file should be a text file where each line contains at least two space-separated floating-point numbers representing the X and Y coordinates of a point. Additional columns on a line are ignored.

Example:
```
5.1 3.5 1.4 0.2
4.9 3.0 1.4 0.2
...
```

## Output File Format (`--output`)

If an output file is specified, it will contain the 0-based index of the cluster assigned to each point, one index per line, corresponding to the order of points in the input.

Example:
```
3
3
1
...
