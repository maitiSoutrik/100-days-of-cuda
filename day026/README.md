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
drboom@JetNano ~/g/1/build> ./day026/kde
KDE Results (first 10):
Query Point 0: 0.503697
Query Point 0.001001: 0.507743
Query Point 0.002002: 0.511792
Query Point 0.003003: 0.515839
Query Point 0.004004: 0.519886
Query Point 0.005005: 0.523931
Query Point 0.00600601: 0.527975
Query Point 0.00700701: 0.532018
Query Point 0.00800801: 0.536056
Query Point 0.00900901: 0.540093
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
