# Day 63: Parallel Markov Chain Clustering for Robot Localization

## Overview

This project explores the use of a simplified Markov Chain Clustering (MCL) algorithm, accelerated with CUDA, for a simulated robot localization task. The goal is to identify high-probability regions (clusters) in a 2D grid world where a robot might be located. This approach differs from traditional Monte Carlo Localization (MCL, which is a particle filter method) by using graph clustering principles on a state transition matrix.

The simulation involves:
1.  Representing a 2D grid as a set of states.
2.  Initializing a transition probability matrix where entries `M_ij` represent the probability of transitioning from state `j` to state `i`.
3.  Iteratively applying the MCL algorithm's core operations (Expansion and Inflation) to this matrix using CUDA kernels.
4.  Extracting potential robot locations by identifying states with high "belief" or "attraction strength" after several iterations.

## Implementation Details

The core logic is encapsulated in the `mcl_localization_lib` static library.

### Data Structures

*   `State`: A simple struct `{float x, float y, float probability}` to represent a robot's state and its associated probability.
*   `TransitionMatrix`: A struct managing a square matrix (`float* data`) on the GPU, representing state transition probabilities. It includes methods for memory management (`cudaMalloc`, `cudaFree`), data transfer (`cudaMemcpy`), and printing. For this demonstration, a dense matrix is used.

### CUDA Kernels

1.  **`expansion_kernel`**:
    *   Performs matrix multiplication `C = A * B`. In our MCL's expansion step, `A` and `B` are both the current transition matrix, so it computes `M = M * M`.
    *   This is a naive implementation where each thread computes one element of the output matrix.
    *   Grid/block dimensions are chosen to cover the entire matrix.

2.  **`inflation_power_kernel`**:
    *   Performs the first part of the inflation step: element-wise exponentiation `M_ij = M_ij ^ gamma`.
    *   `gamma` is the `inflation_factor`.
    *   Each thread processes one element of the matrix.

3.  **`inflation_column_sum_kernel`**:
    *   Calculates the sum of each column of the (exponentiated) matrix.
    *   Each CUDA block is responsible for summing one column.
    *   Uses shared memory (`__shared__ float s_col_sum[]`) for parallel reduction within the block to compute the column sum efficiently.

4.  **`inflation_normalize_kernel`**:
    *   Performs the second part of the inflation step: normalizing each column. `M_ij = M_ij / sum_k(M_kj)`.
    *   Uses the column sums computed by `inflation_column_sum_kernel`.
    *   Each thread processes one element of the matrix, dividing it by the corresponding column sum.

### Host-Side Logic

*   `initialize_synthetic_grid_world()`: Creates an initial transition matrix for a `grid_dim` x `grid_dim` world. It assigns higher probabilities for a state to transition to itself or adjacent states, then normalizes each column to sum to 1.
*   `mcl_iteration_cuda()`: Orchestrates one full MCL iteration:
    1.  Calls `expand_matrix_cuda()` (expansion). A temporary device matrix stores the result.
    2.  Copies the result from the temporary matrix back to the main matrix.
    3.  Calls `inflate_matrix_cuda()` (inflation, which includes power and normalization steps).
*   `extract_clusters_from_probabilities()`: A simplified method to identify "localized" states. In this version, it considers the diagonal elements `M_ii` of the final matrix as a proxy for the belief or attraction strength of state `i`. States whose diagonal probability exceeds a `probability_threshold` are returned. This is a simplification; a true MCL cluster extraction would be more involved.

## Key CUDA Features Used

*   **CUDA Kernels**: `__global__` functions for parallel execution on the GPU.
*   **Device Memory Management**: `cudaMalloc`, `cudaFree`, `cudaMemset` for allocating and managing GPU memory.
*   **Data Transfer**: `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost` for moving data between CPU and GPU.
*   **Thread Hierarchy**: `blockIdx`, `blockDim`, `threadIdx` for thread identification and work distribution.
*   **Shared Memory**: `__shared__` memory used in `inflation_column_sum_kernel` for efficient parallel reduction within a block.
*   **Error Handling**: `CHECK_CUDA_ERROR` macro for robustly checking CUDA API call statuses.
*   **Device Synchronization**: `cudaDeviceSynchronize()` to ensure kernel completion before proceeding with dependent CPU tasks or further CUDA calls.

## Performance Considerations

*   **Dense vs. Sparse Matrices**: This implementation uses dense matrices for simplicity. Real-world MCL applications, especially on large graphs, benefit significantly from sparse matrix representations (e.g., ELLPACK-R, CSR) and corresponding sparse matrix operations (SpMV, SpMM) provided by libraries like cuSPARSE. This would drastically reduce memory footprint and computational load as the matrix becomes sparse during MCL iterations.
*   **Expansion Kernel**: The naive matrix multiplication kernel is `O(N^3)` (conceptually, for N states, though parallelized). Optimized matrix multiplication algorithms (e.g., using shared memory tiling) would improve performance. For `M = M*M`, specialized symmetric matrix multiplication could also be considered if applicable.
*   **Inflation Kernel**: The column sum reduction in `inflation_column_sum_kernel` is a common parallel primitive. The shared memory implementation is more efficient than naive global memory reductions.
*   **Data Transfers**: Minimizing data transfers between host and device is crucial. In this example, the matrix stays on the device during iterations. Only initial setup and final result extraction involve transfers.

## Building and Running

### Prerequisites
*   CUDA Toolkit (>= 10.0, tested with versions compatible with compute capability 5.3)
*   CMake (>= 3.10)
*   A C++ compiler compatible with CUDA (e.g., g++)
*   Google Test (fetched by CMake)

### Build Instructions (Target Environment - e.g., Jetson Nano)
1.  Navigate to the root of the `100-days-of-cuda` project.
2.  Create a build directory (if it doesn't exist) and navigate into it:
    ```bash
    mkdir -p build
    cd build
    ```
3.  Run CMake to configure the project (from the build directory):
    ```bash
    cmake ..
    ```
4.  Build the specific target for Day 63:
    ```bash
    cmake --build . --target mcl_main --config Release
    # To build tests:
    # cmake --build . --target mcl_localization_test --config Release
    ```
    Alternatively, build all targets:
    ```bash
    cmake --build . --config Release
    ```

### Running the Simulation
The main executable will be located in the `build/day063/` directory (or `build/bin` if installation paths are standard).
```bash
./day063/mcl_main [grid_dim] [num_iterations] [inflation_factor] [prob_threshold]
```
Example:
```bash
./day063/mcl_main 10 10 2.0 0.01
```
This will run a simulation on a 10x10 grid for 10 iterations with an inflation factor of 2.0, and extract states with probability > 0.01.

### Running Tests
Tests can be run using CTest from the build directory after building the test target:
```bash
ctest --output-on-failure -C Release -R mcl_localization_test
```
Or by directly running the test executable:
```bash
./day063/mcl_localization_test
```

## Execution Results

Output from running `./day063/mcl_main` (default parameters: 10x10 grid, 10 iterations, inflation 2.0, threshold 0.01) on Jetson Nano:
```
Starting MCL Localization Simulation...
Grid Dimensions: 10x10 (100 states)
Number of Iterations: 10
Inflation Factor: 2
Probability Threshold for Cluster Extraction: 0.01
-------------------------------------------------
Initializing synthetic grid world...
Initial Transition Matrix (sample):
Transition Matrix (first 10x10):
  0.6579   0.1136   0.0222   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.1316   0.5682   0.1111   0.0222   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0263   0.1136   0.5556   0.1111   0.0222   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0227   0.1111   0.5556   0.1111   0.0222   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0222   0.1111   0.5556   0.1111   0.0222   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0222   0.1111   0.5556   0.1111   0.0222   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0222   0.1111   0.5556   0.1111   0.0227   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0222   0.1111   0.5556   0.1136   0.0263 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0222   0.1111   0.5682   0.1316 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0222   0.1136   0.6579 
-------------------------------------------------
Running MCL iterations...
Completed Iteration 1/10
Matrix after iteration 1 (sample):
Transition Matrix (first 10x10):
  0.7650   0.1084   0.0089   0.0001   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.1023   0.6656   0.1000   0.0079   0.0001   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0080   0.0954   0.6638   0.0978   0.0078   0.0001   0.0000   0.0000   0.0000   0.0000 
  0.0001   0.0075   0.0973   0.6661   0.0978   0.0078   0.0001   0.0000   0.0000   0.0000 
  0.0000   0.0001   0.0078   0.0978   0.6662   0.0978   0.0078   0.0001   0.0000   0.0000 
  0.0000   0.0000   0.0001   0.0078   0.0978   0.6662   0.0978   0.0078   0.0001   0.0000 
  0.0000   0.0000   0.0000   0.0001   0.0078   0.0978   0.6661   0.0973   0.0075   0.0001 
  0.0000   0.0000   0.0000   0.0000   0.0001   0.0078   0.0978   0.6638   0.0954   0.0080 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0001   0.0079   0.1000   0.6656   0.1023 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0001   0.0089   0.1084   0.7650 
Completed Iteration 2/10
Matrix after iteration 2 (sample):
Transition Matrix (first 10x10):
  0.8867   0.0893   0.0021   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0539   0.7911   0.0689   0.0015   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0011   0.0606   0.8049   0.0653   0.0015   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0014   0.0651   0.8087   0.0654   0.0015   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0015   0.0654   0.8086   0.0654   0.0015   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0015   0.0654   0.8086   0.0654   0.0015   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0015   0.0654   0.8087   0.0651   0.0014   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0015   0.0653   0.8049   0.0606   0.0011 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0015   0.0689   0.7911   0.0539 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0021   0.0893   0.8867 
Completed Iteration 3/10
Matrix after iteration 3 (sample):
Transition Matrix (first 10x10):
  0.9743   0.0511   0.0002   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0127   0.9148   0.0264   0.0001   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0213   0.9365   0.0238   0.0001   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0001   0.0241   0.9391   0.0240   0.0001   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0001   0.0240   0.9390   0.0240   0.0001   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0001   0.0240   0.9390   0.0240   0.0001   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0001   0.0240   0.9391   0.0241   0.0001   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0001   0.0238   0.9365   0.0213   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0001   0.0264   0.9148   0.0127 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0002   0.0511   0.9743 
Completed Iteration 4/10
Matrix after iteration 4 (sample):
Transition Matrix (first 10x10):
  0.9987   0.0130   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0006   0.9842   0.0031   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0022   0.9937   0.0025   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0026   0.9942   0.0026   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0026   0.9942   0.0026   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0026   0.9942   0.0026   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0026   0.9942   0.0026   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0025   0.9937   0.0022   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0031   0.9842   0.0006 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0130   0.9987 
Completed Iteration 5/10
Matrix after iteration 5 (sample):
Transition Matrix (first 10x10):
  1.0000   0.0007   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.9993   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.9999   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.9999   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.9999   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.9999   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.9999   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.9999   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.9993   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0007   1.0000 
Completed Iteration 6/10
Matrix after iteration 6 (sample):
Transition Matrix (first 10x10):
  1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000 
Completed Iteration 7/10
Matrix after iteration 7 (sample):
Transition Matrix (first 10x10):
  1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000 
Completed Iteration 8/10
Matrix after iteration 8 (sample):
Transition Matrix (first 10x10):
  1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000 
Completed Iteration 9/10
Matrix after iteration 9 (sample):
Transition Matrix (first 10x10):
  1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000 
Completed Iteration 10/10
Matrix after iteration 10 (sample):
Transition Matrix (first 10x10):
  1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000 
-------------------------------------------------
Final Transition Matrix (sample):
Transition Matrix (first 10x10):
  1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000   0.0000 
  0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   1.0000 
-------------------------------------------------
Extracting clusters (states with probability > 0.0100):
Found 100 significant state(s):
  State (0.0000, 0.0000) - Probability: 1.0000
  State (1.0000, 0.0000) - Probability: 1.0000
  State (2.0000, 0.0000) - Probability: 1.0000
  State (3.0000, 0.0000) - Probability: 1.0000
  State (4.0000, 0.0000) - Probability: 1.0000
  State (5.0000, 0.0000) - Probability: 1.0000
  State (6.0000, 0.0000) - Probability: 1.0000
  State (7.0000, 0.0000) - Probability: 1.0000
  State (8.0000, 0.0000) - Probability: 1.0000
  State (9.0000, 0.0000) - Probability: 1.0000
  State (0.0000, 1.0000) - Probability: 1.0000
  State (1.0000, 1.0000) - Probability: 1.0000
  State (2.0000, 1.0000) - Probability: 1.0000
  State (3.0000, 1.0000) - Probability: 1.0000
  State (4.0000, 1.0000) - Probability: 1.0000
  State (5.0000, 1.0000) - Probability: 1.0000
  State (6.0000, 1.0000) - Probability: 1.0000
  State (7.0000, 1.0000) - Probability: 1.0000
  State (8.0000, 1.0000) - Probability: 1.0000
  State (9.0000, 1.0000) - Probability: 1.0000
  State (0.0000, 2.0000) - Probability: 1.0000
  State (1.0000, 2.0000) - Probability: 1.0000
  State (2.0000, 2.0000) - Probability: 1.0000
  State (3.0000, 2.0000) - Probability: 1.0000
  State (4.0000, 2.0000) - Probability: 1.0000
  State (5.0000, 2.0000) - Probability: 1.0000
  State (6.0000, 2.0000) - Probability: 1.0000
  State (7.0000, 2.0000) - Probability: 1.0000
  State (8.0000, 2.0000) - Probability: 1.0000
  State (9.0000, 2.0000) - Probability: 1.0000
  State (0.0000, 3.0000) - Probability: 1.0000
  State (1.0000, 3.0000) - Probability: 1.0000
  State (2.0000, 3.0000) - Probability: 1.0000
  State (3.0000, 3.0000) - Probability: 1.0000
  State (4.0000, 3.0000) - Probability: 1.0000
  State (5.0000, 3.0000) - Probability: 1.0000
  State (6.0000, 3.0000) - Probability: 1.0000
  State (7.0000, 3.0000) - Probability: 1.0000
  State (8.0000, 3.0000) - Probability: 1.0000
  State (9.0000, 3.0000) - Probability: 1.0000
  State (0.0000, 4.0000) - Probability: 1.0000
  State (1.0000, 4.0000) - Probability: 1.0000
  State (2.0000, 4.0000) - Probability: 1.0000
  State (3.0000, 4.0000) - Probability: 1.0000
  State (4.0000, 4.0000) - Probability: 1.0000
  State (5.0000, 4.0000) - Probability: 1.0000
  State (6.0000, 4.0000) - Probability: 1.0000
  State (7.0000, 4.0000) - Probability: 1.0000
  State (8.0000, 4.0000) - Probability: 1.0000
  State (9.0000, 4.0000) - Probability: 1.0000
  State (0.0000, 5.0000) - Probability: 1.0000
  State (1.0000, 5.0000) - Probability: 1.0000
  State (2.0000, 5.0000) - Probability: 1.0000
  State (3.0000, 5.0000) - Probability: 1.0000
  State (4.0000, 5.0000) - Probability: 1.0000
  State (5.0000, 5.0000) - Probability: 1.0000
  State (6.0000, 5.0000) - Probability: 1.0000
  State (7.0000, 5.0000) - Probability: 1.0000
  State (8.0000, 5.0000) - Probability: 1.0000
  State (9.0000, 5.0000) - Probability: 1.0000
  State (0.0000, 6.0000) - Probability: 1.0000
  State (1.0000, 6.0000) - Probability: 1.0000
  State (2.0000, 6.0000) - Probability: 1.0000
  State (3.0000, 6.0000) - Probability: 1.0000
  State (4.0000, 6.0000) - Probability: 1.0000
  State (5.0000, 6.0000) - Probability: 1.0000
  State (6.0000, 6.0000) - Probability: 1.0000
  State (7.0000, 6.0000) - Probability: 1.0000
  State (8.0000, 6.0000) - Probability: 1.0000
  State (9.0000, 6.0000) - Probability: 1.0000
  State (0.0000, 7.0000) - Probability: 1.0000
  State (1.0000, 7.0000) - Probability: 1.0000
  State (2.0000, 7.0000) - Probability: 1.0000
  State (3.0000, 7.0000) - Probability: 1.0000
  State (4.0000, 7.0000) - Probability: 1.0000
  State (5.0000, 7.0000) - Probability: 1.0000
  State (6.0000, 7.0000) - Probability: 1.0000
  State (7.0000, 7.0000) - Probability: 1.0000
  State (8.0000, 7.0000) - Probability: 1.0000
  State (9.0000, 7.0000) - Probability: 1.0000
  State (0.0000, 8.0000) - Probability: 1.0000
  State (1.0000, 8.0000) - Probability: 1.0000
  State (2.0000, 8.0000) - Probability: 1.0000
  State (3.0000, 8.0000) - Probability: 1.0000
  State (4.0000, 8.0000) - Probability: 1.0000
  State (5.0000, 8.0000) - Probability: 1.0000
  State (6.0000, 8.0000) - Probability: 1.0000
  State (7.0000, 8.0000) - Probability: 1.0000
  State (8.0000, 8.0000) - Probability: 1.0000
  State (9.0000, 8.0000) - Probability: 1.0000
  State (0.0000, 9.0000) - Probability: 1.0000
  State (1.0000, 9.0000) - Probability: 1.0000
  State (2.0000, 9.0000) - Probability: 1.0000
  State (3.0000, 9.0000) - Probability: 1.0000
  State (4.0000, 9.0000) - Probability: 1.0000
  State (5.0000, 9.0000) - Probability: 1.0000
  State (6.0000, 9.0000) - Probability: 1.0000
  State (7.0000, 9.0000) - Probability: 1.0000
  State (8.0000, 9.0000) - Probability: 1.0000
  State (9.0000, 9.0000) - Probability: 1.0000
-------------------------------------------------
MCL Localization Simulation Finished.
```
Output from `./day063/mcl_localization_test`:
```
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from MCLTest
[ RUN      ] MCLTest.MatrixInitialization
[       OK ] MCLTest.MatrixInitialization (101 ms)
[ RUN      ] MCLTest.SingleMCLIteration
[       OK ] MCLTest.SingleMCLIteration (1 ms)
[ RUN      ] MCLTest.ClusterExtraction
[       OK ] MCLTest.ClusterExtraction (5 ms)
[----------] 3 tests from MCLTest (109 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (109 ms total)
[  PASSED  ] 3 tests.
```

## Learnings and Observations

*   The MCL algorithm naturally lends itself to parallelization due to its reliance on matrix operations (multiplication, element-wise power, column-wise normalization).
*   Shared memory is effective for parallel reductions like column summation.
*   The choice of inflation factor (`gamma`) significantly impacts convergence and cluster granularity.
*   Interpreting the resulting matrix for "localization" requires careful consideration. The diagonal approach used here is a simplification. True MCL clustering identifies attractor columns.
*   For practical robot localization, this method would need to be integrated with sensor updates (e.g., re-weighting probabilities based on sensor readings) and a motion model that dynamically updates the transition matrix or influences the random walk.
*   The current synthetic data initialization is very basic. A more realistic setup would involve a map and a motion model.
*   **Build Issue Resolution**: A persistent build error (`namespace "std" is not a type name`) on the Jetson Nano related to `std::vector` in header function signatures was resolved by refactoring the `TransitionMatrix` copy methods (`copy_to_device`, `copy_to_host`) to use raw C-style pointers in the header (`.cuh`) and managing `std::vector` conversions internally within the implementation file (`.cu`). This suggests a specific sensitivity in the Jetson Nano's `nvcc` toolchain to complex C++ types in headers when processed as part of CUDA compilation units.

## Future Improvements

*   Implement sparse matrix operations using cuSPARSE for scalability.
*   Use optimized matrix multiplication kernels (e.g., with shared memory tiling).
*   Develop a more sophisticated cluster extraction method aligned with MCL principles (identifying attractor columns and the states belonging to them).
*   Integrate a sensor model: after MCL iterations, re-weight state probabilities based on simulated sensor readings.
*   Incorporate a robot motion model to update the base transition probabilities or to bias the random walks.
*   Compare performance and localization accuracy with traditional particle filters (Monte Carlo Localization).

## References
*   Stijn van Dongen, "Graph Clustering by Flow Simulation" (PhD thesis, University of Utrecht, 2000) - The original MCL algorithm.
*   Papers on parallel MCL implementations (as found by Tavily/Perplexity search, e.g., those mentioning ELLPACK-R).
