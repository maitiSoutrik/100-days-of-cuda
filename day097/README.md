# Day 97: Fused RMSNorm + SwiGLU Activation

## Overview

This project implements a fused CUDA kernel that combines Root Mean Square Normalization (RMSNorm) and the SwiGLU activation function. Fusing these operations aims to reduce memory bandwidth bottlenecks and improve performance, which is crucial in modern machine learning models, especially transformers. This builds upon concepts from Day 93 (RMS Normalization) and Day 67 (SwiGLU Activation).

## Implementation Details

The core of this project is a single CUDA kernel, `fused_rmsnorm_swiglu_kernel`, which performs the following steps for each input row (e.g., a token embedding):

1.  **RMS Normalization**:
    *   Calculates the sum of squares of the input elements in the row. This is done using a parallel reduction within the CUDA block, leveraging shared memory for efficiency.
    *   Computes the reciprocal of the square root of the mean square (rrms).
    *   Normalizes each element of the input row: `normalized_x = (x / sqrt(mean_square + epsilon)) * weight`. The `weight` (gamma) is a learnable parameter.

2.  **SwiGLU Activation**:
    *   The SwiGLU activation function is defined as `SwiGLU(x, gate) = Silu(x) * gate`, where `Silu(x) = x * sigmoid(x)`.
    *   The input `hidden_dim` is conceptually split into two halves. The first half is treated as `x` and the second half as `gate`.
    *   After RMS normalization, the normalized `x` and `gate` components are used.
    *   The Silu function is applied to the normalized `x` component.
    *   The result of `Silu(normalized_x)` is then multiplied element-wise by the `normalized_gate` component.
    *   The output dimension is `hidden_dim / 2`.

The kernel is designed such that each CUDA block processes one input row. Threads within the block collaborate on the RMSNorm reduction and then individually compute elements of the SwiGLU output.

### Kernel Launch Configuration

-   **Grid Dimensions**: `num_rows` (one block per input row/token).
-   **Block Dimensions**: `block_size` threads per block. `block_size` is chosen to be efficient for the reduction (e.g., a power of 2 like 128, 256) and must be sufficient to cover the output computations (ideally `block_size >= hidden_dim / 2`, or the kernel would need internal looping for threads to cover all outputs). The current implementation in `fused_rmsnorm_swiglu.cu` assumes `block_size` threads compute `block_size` output elements if `hidden_dim/2 >= block_size`, or all `hidden_dim/2` elements if `hidden_dim/2 < block_size`. The `main.cu` logic attempts to set a reasonable `block_size`.
-   **Shared Memory**: Dynamically allocated shared memory of size `block_size * sizeof(float)` is used for the parallel reduction in RMSNorm.

## Key CUDA Features Used

-   **Shared Memory**: Utilized for efficient parallel reduction within a block to calculate the sum of squares for RMSNorm. This reduces global memory accesses.
-   **Parallel Reduction**: Implemented within the kernel for the RMSNorm statistics.
-   **Kernel Fusion**: Combining RMSNorm and SwiGLU into a single kernel call reduces kernel launch overhead and data movement between global memory and registers/shared memory.
-   **Device Functions**: `sigmoidf_device` and `silu_device` are used for clarity and modularity within the kernel.
-   **Dynamic Shared Memory**: The kernel is declared with `extern __shared__ float s_sum_sq[]` and the size is specified at launch.

## Performance Considerations

-   **Memory Bandwidth**: The primary motivation for fusion is to reduce reads and writes to global memory. Instead of writing the RMSNorm output to global memory and then reading it back for SwiGLU, the intermediate result stays in registers or shared memory.
-   **Kernel Launch Overhead**: Reducing two kernel launches to one saves on launch latency.
-   **Occupancy**: The choice of `block_size` and shared memory usage can affect kernel occupancy. The current `block_size` selection in `main.cu` tries to balance the needs of reduction and output computation.
-   **Instruction Mix**: The fused kernel combines arithmetic-intensive parts (normalization, sigmoid, multiplications) which can be efficiently handled by CUDA cores.
-   **Data Reuse**: The `weight` (gamma) values and the `rrms` value are reused across the `hidden_dim` elements within a row.

## Building and Running

The project uses CMake for building. Ensure you have the CUDA Toolkit installed.

**Build Steps (on Jetson Nano or compatible cross-compilation environment):**

1.  Navigate to the root of the `100-days-of-cuda` project.
2.  Create a build directory (if it doesn't exist) and navigate into it:
    ```bash
    mkdir -p build
    cd build
    ```
3.  Run CMake to configure the project (from the `build` directory):
    ```bash
    cmake ..
    ```
4.  Build the specific target for Day 97:
    ```bash
    cmake --build . --target fused_rmsnorm_swiglu_main --config Release
    # To build tests:
    cmake --build . --target fused_rmsnorm_swiglu_test --config Release
    ```
5.  Run the executable:
    ```bash
    ./day097/fused_rmsnorm_swiglu_main
    ```
6.  Run tests:
    ```bash
    ./day097/fused_rmsnorm_swiglu_test
    # Or using ctest from the build directory
    # ctest -R day097_fused_rmsnorm_swiglu_test --verbose
    ```

## Execution Results

The `fused_rmsnorm_swiglu_main` executable will output:
-   Parameters used for the benchmark.
-   CUDA block size selected.
-   Execution time for the GPU kernel.
-   Execution time for the CPU reference implementation.
-   Sample input, weights, GPU output, and CPU output.
-   A verification message indicating whether GPU and CPU results match.

```
Running Fused RMSNorm + SwiGLU Benchmark
Parameters: Batch Size=4, Seq Len=64, Hidden Dim=256
Total rows (tokens): 256
Output dimension after SwiGLU: 128
Using CUDA block size: 256
GPU Kernel Execution Time: 1.98229 ms
Performing CPU computation for verification...
CPU Execution Time: 1.1222 ms
Input (Host) (sample 2x8):
-0.0820 0.4567  0.7454  0.9915  0.2577  0.3906  -0.0717 0.3848
0.4164  0.2180  -0.6020 -0.5723 0.8202  -0.9892 -0.5964 -0.0048

Weights (Host) (sample 1x8):
0.6534  0.6967  0.5460  1.4411  1.1862  0.5979  0.9066  0.8689

Output (GPU) (sample 2x8):
0.0270  0.1344  -0.0320 -0.8807 0.1536  -0.0490 -0.0754 -0.5365
0.5052  -0.0589 0.0011  -0.3072 1.4818  -0.0310 -0.3850 0.0076

Output (CPU) (sample 2x8):
0.0270  0.1344  -0.0320 -0.8807 0.1536  -0.0490 -0.0754 -0.5365
0.5052  -0.0589 0.0011  -0.3072 1.4818  -0.0310 -0.3850 0.0076

Verification Successful: GPU and CPU results match.
```

The `fused_rmsnorm_swiglu_test` executable will run several test cases with different input sizes and report pass/fail status.

```
[==========] Running 6 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 6 tests from FusedRMSNormSwiGLUTest
[ RUN      ] FusedRMSNormSwiGLUTest.SmallInput
[       OK ] FusedRMSNormSwiGLUTest.SmallInput (78 ms)
[ RUN      ] FusedRMSNormSwiGLUTest.MediumInput
[       OK ] FusedRMSNormSwiGLUTest.MediumInput (2 ms)
[ RUN      ] FusedRMSNormSwiGLUTest.LargerHiddenDim
[       OK ] FusedRMSNormSwiGLUTest.LargerHiddenDim (2 ms)
[ RUN      ] FusedRMSNormSwiGLUTest.HiddenDimEqualsBlockSize
[       OK ] FusedRMSNormSwiGLUTest.HiddenDimEqualsBlockSize (2 ms)
[ RUN      ] FusedRMSNormSwiGLUTest.MinimalHiddenDim
[       OK ] FusedRMSNormSwiGLUTest.MinimalHiddenDim (2 ms)
[ RUN      ] FusedRMSNormSwiGLUTest.OddNumRows
[       OK ] FusedRMSNormSwiGLUTest.OddNumRows (2 ms)
[----------] 6 tests from FusedRMSNormSwiGLUTest (91 ms total)

[----------] Global test environment tear-down
[==========] 6 tests from 1 test suite ran. (91 ms total)
[  PASSED  ] 6 tests.
```

## Learnings and Observations

-   Kernel fusion can significantly reduce memory operations and improve performance by keeping intermediate data in faster on-chip memory (registers, shared memory).
-   Designing a fused kernel requires careful consideration of how threads will cooperate and share data for different stages of the fused operation. The parallel reduction for RMSNorm needs a different thread cooperation pattern than the element-wise SwiGLU computation.
-   The choice of `block_size` is critical. It needs to be suitable for the reduction part of RMSNorm (often a power of 2, e.g., 128, 256) and also for efficiently computing the `hidden_dim / 2` outputs of SwiGLU. The current kernel implementation has a slight simplification where if `hidden_dim / 2` is larger than `block_size`, it won't compute all outputs without modification. The `main.cu` attempts to select a `block_size` that is the smallest power of two greater than or equal to `hidden_dim / 2` (capped at 512), which should work correctly with the current kernel.
-   Shared memory usage for reduction is effective but limited by the amount of shared memory per SM. For very large `hidden_dim` values where `block_size` would also need to be large for a single-pass reduction, multi-pass reduction or different strategies might be needed.
-   Comparing against a CPU implementation is vital for verifying correctness, especially with complex fused operations.

## Future Improvements

-   Implement a more robust kernel version where threads loop to handle cases where `hidden_dim / 2` is larger than `block_size`, allowing for more flexible `block_size` choices.
-   Explore using half-precision (`__half` or `bfloat16`) for further performance gains, common in modern LLMs.
-   Benchmark against separate RMSNorm and SwiGLU kernels to quantify the fusion benefit.
-   Optimize the shared memory reduction further (e.g., using warp-level primitives if applicable and beneficial).

## References

-   [RMS Normalization Paper (Layer Normalization without Mean-Centering)](https://arxiv.org/abs/1910.07467)
-   [GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202)
-   NVIDIA CUDA C++ Programming Guide
