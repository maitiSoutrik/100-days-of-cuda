# Day 081: Generalized Jensen-Shannon Divergence Loss in CUDA

## Overview

This project implements a CUDA C++ kernel for computing the generalized Jensen-Shannon Divergence (JSD) loss, including both forward and backward passes. The implementation focuses on numerical stability using shared memory for reductions within the JSD kernel and a separate reduction kernel for summing per-element losses to a scalar. It handles different interpretations of the generalization parameter `beta` (affecting forward KL, reverse KL, or mixed/symmetric JSD calculations). A CPU version of the forward pass is also provided for benchmarking, and performance is compared using CUDA events (GPU) and C++ chrono (CPU).

## Implementation Details

The core of the JSD computation involves two probability distribution matrices, P and Q, and a generalization parameter `beta`. The JSD is defined based on Kullback-Leibler (KL) divergences.

The mixture distribution M is calculated as `M = 0.5 * (P + Q)`.
The generalized JSD is then computed based on `beta`:
- If `beta` is close to 0.5 (symmetric JSD): `0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)`
- If `beta` is close to 1.0 (forward KL-like): `D_KL(P || M)`
- If `beta` is close to 0.0 (reverse KL-like): `D_KL(Q || M)`
- Otherwise (general weighted case): `beta * D_KL(P || M) + (1 - beta) * D_KL(Q || M)`

**CUDA Kernels:**
1.  `jsd_loss_kernel`:
    *   Launched with one block per distribution (row in P and Q).
    *   Each thread within a block processes one or more elements of the distribution.
    *   Computes `p_val`, `q_val`, and `m_val` for each element.
    *   Calculates the JSD contribution for that element based on `beta`.
    *   Sums these contributions for the row using shared memory reduction.
    *   Simultaneously calculates the gradients `d_grad_P` and `d_grad_Q` for each element.
    *   Stores the per-row sum of JSD contributions into `d_per_row_loss`.

2.  `sum_reduction_kernel`:
    *   A generic sum reduction kernel.
    *   Takes `d_per_row_loss` (an array of losses, one for each distribution) as input.
    *   Reduces this array to a single scalar value, `d_loss`.
    *   Handles cases where the number of per-row losses might require multiple stages of reduction (though the current `jsd_loss_gpu` implements a two-stage reduction if needed).

**CPU Implementation:**
-   `jsd_loss_forward_cpu`:
    *   Iterates through each distribution and each element using nested loops.
    *   Calculates `m_val`, `kl_p_m`, `kl_q_m`, and the JSD contribution similarly to the GPU kernel's forward pass logic.
    *   Sums all contributions to get the total JSD loss.

**Error Handling:**
-   A `CHECK_CUDA_ERROR` macro is used for robust CUDA error checking.
-   An `epsilon` value is used throughout calculations to prevent `log(0)` and division by zero, ensuring numerical stability.

## Key CUDA Features Used

-   **CUDA Kernels**: `__global__` functions for parallel execution on the GPU.
-   **Shared Memory**: `__shared__` memory is used within `jsd_loss_kernel` and `sum_reduction_kernel` for efficient parallel reduction of sums.
-   **Thread Hierarchy**: `blockIdx`, `blockDim`, `threadIdx` are used to manage parallel computation across distributions and elements.
-   **CUDA Runtime API**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaEvent_t` for memory management and timing.
-   **Device Functions**: `__device__` function `kl_divergence_element` for modularity within the kernel.
-   **CUDA Events**: Used for accurate measurement of GPU kernel execution time.
-   **Atomic Operations**: Not explicitly used in this version of reduction, but an alternative for more complex reduction patterns. The current reduction relies on block-level synchronization and careful sum accumulation.

## Performance Considerations

-   **Shared Memory Reduction**: Using shared memory for summing up losses within a block (`jsd_loss_kernel`) and for the final reduction (`sum_reduction_kernel`) is significantly faster than global memory operations for these tasks.
-   **Coalesced Memory Access**: While not explicitly optimized for in this description (as P, Q are processed row-wise, which is generally good if rows are contiguous), memory access patterns are crucial. The current implementation processes elements `P[row_idx * num_elements + elem_idx]`, which should lead to coalesced access if `num_elements` is reasonably large and data is stored row-major.
-   **Kernel Launch Overhead**: For a very small number of distributions or elements, CPU computation might be faster due to kernel launch overhead. The benefits of GPU parallelism shine with larger datasets.
-   **Multi-stage Reduction**: The `sum_reduction_kernel` and its invocation in `jsd_loss_gpu` are designed to handle a potentially large number of per-row losses by performing a two-stage reduction if necessary. This is more scalable than a single-pass reduction that might exceed resource limits for one block.
-   **Numerical Stability**: The use of `epsilon` is critical. Without it, `log(0)` or division by very small numbers can lead to `NaN` or `Inf` values, corrupting the loss and gradients.

## Building and Running

**Prerequisites:**
-   NVIDIA CUDA Toolkit (>= 10.0 recommended)
-   CMake (>= 3.18)
-   A C++ compiler compatible with CUDA (e.g., g++)
-   Google Test (will be fetched by CMake if not found)

**Build Steps (from the root `100-days-of-cuda` directory):**
1.  Create a build directory: `mkdir build && cd build`
2.  Configure CMake: `cmake ..`
3.  Build the project: `make day081_jsd_loss` (or `make jsd_benchmark` and `make jsd_loss_test`)

**Running the Benchmark:**
After building, the benchmark executable will be in the `build/day081/` directory (or `build/bin` if installed).
```bash
./day081/jsd_benchmark
```

**Running Tests:**
Tests can be run via CTest from the build directory:
```bash
ctest --output-on-failure -R day081_jsd_loss # Or specify the test name directly
```
Or by running the test executable directly:
```bash
./day081/jsd_loss_test
```

## Execution Results

The `jsd_benchmark` executable will output the computed JSD loss values for different `beta` parameters from both GPU and CPU (forward pass only for CPU), along with their respective execution times.

```
Running JSD Loss Computations:
Num Distributions: 1024, Num Elements per Distribution: 512
-----------------------------------------------------------
      Beta            GPU Loss      CPU Loss (Fwd)  GPU Time (ms)  CPU Time (ms)
-----------------------------------------------------------
    0.0000            104.7552            104.7552        19.7534        38.1220
    0.5000            104.8143            104.8143        18.8581        42.8310
    1.0000            104.8734            104.8734        19.8452        35.3610
-----------------------------------------------------------
```

## Learnings and Observations

-   The generalized JSD provides flexibility in defining a loss function that can behave like forward KL, reverse KL, or symmetric JSD based on the `beta` parameter.
-   Implementing numerically stable KL divergence and JSD requires careful handling of potential `log(0)` or division-by-zero issues, typically by adding a small epsilon.
-   Shared memory reductions are a powerful technique for parallel summation on the GPU.
-   The backward pass (gradient calculation) for JSD involves derivatives of log functions and ratios, which also need careful handling for stability. The chain rule is applied considering `M`'s dependency on `P` and `Q`.
-   Comparing GPU and CPU execution times clearly demonstrates the acceleration provided by CUDA for data-parallel tasks like this, especially as the number of distributions and elements increases.

## Future Improvements

-   Implement more advanced, multi-pass reduction kernels for `sum_reduction_kernel` to ensure optimal performance for very large `num_distributions`.
-   Explore the impact of different `epsilon` values on accuracy and stability.
-   Add more comprehensive gradient checks in the unit tests, possibly using numerical differentiation on the CPU side for comparison.
-   Benchmark with varying `num_distributions` and `num_elements` to analyze performance scaling.
