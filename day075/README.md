# Day 75: Fused Linear Transformation and Softmax Cross-Entropy Loss

## Overview

This day focuses on implementing a fused CUDA kernel that performs three operations common in neural networks in a single pass:
1.  **Linear Transformation**: Matrix multiplication of input features with a weight matrix, followed by adding a bias term, to produce logits. (`Z = XW^T + b`)
2.  **Softmax Activation**: Applied to the logits to convert them into probabilities.
3.  **Cross-Entropy Loss Computation**: Calculated between the predicted probabilities and the true labels.

The primary motivation for fusing these operations is to improve performance by reducing memory bandwidth usage (avoiding writing intermediate logits to global memory) and minimizing kernel launch overhead. This is particularly beneficial on GPUs where memory access can be a bottleneck.

## Implementation Details

The core of this day's work is the `fused_linear_softmax_loss_kernel` CUDA kernel.

### Kernel Logic:

The kernel is launched with `M` blocks, where `M` is the batch size. Each block is responsible for processing one input sample and has `THREADS_PER_BLOCK` threads (e.g., 256). Dynamic shared memory is utilized to store intermediate results for the current sample being processed by the block.

1.  **Shared Memory Allocation**:
    *   `s_logits_dyn`: Stores the `N` computed logits for the current sample.
    *   `s_reduction_scratch_dyn`: Used as scratch space for block-wide parallel reductions (sum and max).
    The total dynamic shared memory per block is `(N + THREADS_PER_BLOCK) * sizeof(float)`.

2.  **Linear Transformation (Logit Computation)**:
    *   Each thread within a block cooperatively computes a subset of the `N` logits for the current sample (`m = blockIdx.x`).
    *   The computation for each logit `n_idx` is: `logit = sum(input_features[m][k] * weights[n_idx][k] for k in K) + bias[n_idx]`.
    *   The computed logits are stored in `s_logits_dyn`.
    *   A `__syncthreads()` barrier ensures all logits are computed before proceeding.

3.  **Softmax and Cross-Entropy Loss (Fused)**:
    *   **Find `max_logit`**: A block-wide parallel reduction (implemented in `blockReduceMax`) is performed on `s_logits_dyn` to find the maximum logit value for the current sample. This is crucial for numerical stability in the softmax calculation.
    *   **Get `logit_true_class`**: The logit corresponding to the true class label (`d_true_labels[m]`) is read from `s_logits_dyn`.
    *   **Compute `sum_exp_shifted_logits`**: Each thread calculates `exp(s_logits_dyn[i] - max_logit)` for its assigned portion of logits. Another block-wide parallel reduction (implemented in `blockReduceSum`) sums these exponentiated values.
    *   **Compute Loss**: The cross-entropy loss for the sample `m` is calculated using the numerically stable formula:
        `loss_m = logf(sum_exp_shifted) - (logit_true_class - max_logit)`.
    *   **Store Loss**: `threadIdx.x == 0` writes `loss_m` to `d_output_loss_per_sample[m]`.

### Helper Device Functions:

*   `blockReduceSum(float val, float* s_reduction_scratch)`: Performs a block-wide sum reduction using shared memory.
*   `blockReduceMax(float val, float* s_reduction_scratch)`: Performs a block-wide max reduction using shared memory.

### Host Wrapper Function:

*   `compute_fused_linear_softmax_loss_gpu()`: Manages CUDA memory allocations (inputs, outputs), data transfers (host-to-device and device-to-host), configures kernel launch parameters (grid/block dimensions, dynamic shared memory size), launches the kernel, and synchronizes. It returns the average loss over the batch.

## Key CUDA Features Used

*   **Dynamic Shared Memory**: Efficiently stores intermediate logits and reduction scratch space per block, localizing memory access.
*   **Block-Wide Parallel Reductions**: Custom `__device__` functions (`blockReduceSum`, `blockReduceMax`) are implemented for finding the sum and maximum values across all threads in a block. These are essential for the softmax calculation.
*   **Kernel Fusion**: Combining multiple logical operations (linear layer, softmax, cross-entropy) into a single kernel to reduce global memory traffic and kernel launch overhead.
*   **Thread Cooperation**: Threads within a block work together to compute logits and perform reductions.

## Performance Considerations

*   **Reduced Memory Bandwidth**: By not materializing the full `M x N` logit matrix in global memory, significant memory bandwidth is saved. Logits are computed and consumed within shared memory for each sample.
*   **Fewer Kernel Launches**: A single kernel performs what might otherwise be three or more separate kernel calls (e.g., one for MatMul, one for Softmax (possibly involving multiple steps like max, exp, sum), one for loss indexing). This reduces CPU-GPU synchronization overhead.
*   **Numerical Stability**: The softmax calculation incorporates the max-logit subtraction trick (`exp(logit - max_logit)`) to prevent overflow/underflow with large logit values. The loss is also calculated using the log-sum-exp formulation.
*   **MatMul Optimization**: The current MatMul part within the kernel is basic (`current_logit += d_input_features[m * K + k_idx] * d_weights[n_idx * K + k_idx];`). For larger `K`, this could be further optimized by loading parts of `d_input_features[m]` or `d_weights` into shared memory to improve data reuse, similar to tiled matrix multiplication.

## Building and Running

1.  **Navigate to the build directory:**
    ```bash
    cd build
    ```
2.  **Compile using CMake and Make:**
    Ensure the `day075` subdirectory is added to the root `CMakeLists.txt`.
    ```bash
    cmake ..
    make day075_fused_linear_softmax_loss_app # For the main application
    make day075_fused_linear_softmax_loss_test_app # For the tests
    ```
3.  **Run the application:**
    ```bash
    ./day075/day075_fused_linear_softmax_loss_app
    ```
4.  **Run the tests:**
    ```bash
    ./day075/day075_fused_linear_softmax_loss_test_app
    # Or run all tests using CTest from the build directory
    # ctest --output-on-failure -R day075 # Assuming CTest is configured
    ```

## Execution Results

Output from running `./day075/day075_fused_linear_softmax_loss_app` on the Jetson Nano:
```
Problem Dimensions:
Batch Size (M): 4
Input Features (K): 8
Number of Classes (N): 5

GPU Average Loss: 1.452645
CPU Average Loss: 1.452645
Mean Squared Error between GPU and CPU per-sample losses: 0.000000
Verification PASSED!
```

The unit tests (`./day075/day075_fused_linear_softmax_loss_test_app`) also passed successfully:
```
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from FusedLinearSoftmaxLossTest
[ RUN      ] FusedLinearSoftmaxLossTest.BasicCorrectness
[       OK ] FusedLinearSoftmaxLossTest.BasicCorrectness (99 ms)
[ RUN      ] FusedLinearSoftmaxLossTest.SingleSampleSingleClass
[       OK ] FusedLinearSoftmaxLossTest.SingleSampleSingleClass (1 ms)
[ RUN      ] FusedLinearSoftmaxLossTest.LargerRandomCase
[       OK ] FusedLinearSoftmaxLossTest.LargerRandomCase (1 ms)
[----------] 3 tests from FusedLinearSoftmaxLossTest (103 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (103 ms total)
[  PASSED  ] 3 tests.
```

## Learnings and Observations

*   Implementing fused kernels requires careful management of shared memory and synchronization (`__syncthreads()`).
*   Parallel reduction algorithms are fundamental building blocks for many complex CUDA kernels, including those for softmax.
*   The log-sum-exp trick is essential for numerically stable softmax and cross-entropy loss calculations.
*   Verification against a CPU implementation is crucial for ensuring the correctness of complex CUDA kernels.
*   The design of how threads within a block cooperate to process data (e.g., how logits are computed, how reductions are performed) significantly impacts kernel efficiency.

(Further observations after testing and potential profiling will be added here.)
