# Day 26: Parallel Kernel Density Estimation (KDE)

## Overview

This project implements a CUDA kernel to compute Kernel Density Estimation (KDE) for a set of 1D query points based on a set of 1D input values, using a Gaussian kernel. KDE is a non-parametric method for estimating the probability density function (PDF) of a random variable.

## Implementation Details

The CUDA kernel `compute_kde_kernel` calculates the KDE. Each thread is responsible for computing the density estimate for one query point. The kernel iterates through all input values to calculate the Gaussian kernel contribution for each query point.

Key components:

*   **Input Data:** `d_values` (input data points), `d_query_points` (query points), `h` (bandwidth).
*   **Kernel Calculation:**  Gaussian kernel is used: `K(u) = (1 / sqrt(2*PI)) * exp(-0.5 * u^2)`.
*   **Parallelism:** Each thread calculates the density for one query point.
*   **Normalization:** The final density is normalized by `N * h * sqrtf(2.0f * PI)`.

## Key CUDA Features Used

*   CUDA Kernels (`__global__`)
*   Device memory allocation (`cudaMalloc`) and deallocation (`cudaFree`)
*   Data transfer between host and device (`cudaMemcpy`)
*   Device math functions (`expf`, `sqrtf`)
*   Grid and block configuration

## Performance Considerations

*   The kernel has a complexity of O(N\*M), where N is the number of input values and M is the number of query points.
*   Memory access is coalesced as each thread accesses a different query point.
*   The performance is limited by the number of threads and the memory bandwidth.

## Building and Running

1.  Create a `build` directory: `mkdir build`
2.  Navigate to the `build` directory: `cd build`
3.  Run CMake: `cmake ..`
4.  Build the project: `make`
5.  Run the executable: `./kde`

## Execution Results

```
KDE Results (first 10):
Query Point 0: 0.00011959
Query Point 0.001001001: 0.00012002
Query Point 0.002002002: 0.00012045
Query Point 0.003003003: 0.00012088
Query Point 0.004004004: 0.00012131
Query Point 0.005005005: 0.00012174
Query Point 0.006006006: 0.00012217
Query Point 0.007007007: 0.0001226
Query Point 0.008008008: 0.00012303
Query Point 0.009009009: 0.00012346
```

(Note: The results will vary slightly due to the random data generation.)

## Learnings and Observations

*   Implemented a parallel KDE calculation using a CUDA kernel.
*   Demonstrated the use of device memory, kernel launches, and data transfer.
*   Observed the impact of kernel configuration (block size, grid size) on performance.
*   The code is structured to be easily adaptable to different kernel functions and data distributions.

## Future Improvements

*   Implement a 2D KDE.
*   Optimize memory access patterns.
*   Compare performance with CPU implementation.
*   Experiment with different kernel functions.
