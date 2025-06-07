# Day 91: Hinge Loss Implementation in CUDA

## Overview

This project implements the Hinge Loss function using CUDA. Hinge Loss is commonly used in machine learning, particularly for training Support Vector Machines (SVMs) and other maximum margin classifiers. The goal is to penalize predictions that are not only incorrect but also those that are correct but too close to the decision boundary (i.e., within the margin).

The Hinge Loss for a true label `t` (either +1 or -1) and a raw prediction score `y_score` is defined as:
`L(t, y_score) = max(0, 1 - t * y_score)`

This implementation provides:
1.  A CUDA kernel to compute Hinge Loss for an array of predictions.
2.  A CUDA function to compute the sum of Hinge Losses over an array (useful for calculating total loss for a batch).
3.  A demonstration program (`hinge_loss_main.cu`) that generates sample data, computes Hinge Loss on both CPU and GPU, and compares the results.
4.  Unit tests (`hinge_loss_test.cu`) using Google Test to verify the correctness of the CUDA implementation.

## Implementation Details

### `hinge_loss.cuh`
-   Defines the `CHECK_CUDA_ERROR` macro for robust error handling.
-   Declares two main functions:
    -   `void hinge_loss_cuda(const int* d_true_labels, const float* d_pred_scores, float* d_loss, int num_elements);`
        -   Computes element-wise Hinge Loss.
    -   `void sum_hinge_loss_cuda(const int* d_true_labels, const float* d_pred_scores, float* d_total_loss, int num_elements, float* d_temp_storage);`
        -   Computes the sum of Hinge Losses. `d_temp_storage` is used internally to hold individual losses before summation.

### `hinge_loss.cu`
-   **`hinge_loss_kernel`**: A CUDA kernel that takes true labels, predicted scores, and an output array for losses. Each thread computes the Hinge Loss for one element: `max(0.0f, 1.0f - (float)true_labels[idx] * pred_scores[idx])`.
-   **`hinge_loss_cuda`**: A host function that allocates device memory (if not already allocated by caller for output), copies input data to the device, launches `hinge_loss_kernel`, and copies results back to the host (if needed by caller). It primarily orchestrates the kernel launch.
-   **`sum_hinge_loss_cuda`**: This function first calls `hinge_loss_kernel` to compute individual losses into a temporary device buffer (`d_temp_storage`). Then, for simplicity and to ensure correctness without implementing a complex parallel reduction for this example, it copies these individual losses back to the host and performs the summation on the CPU. The final sum is then copied back to the `d_total_loss` device pointer.
    *Note: A production-grade sum would use an optimized parallel reduction algorithm entirely on the GPU (e.g., using shared memory, warp shuffles, or libraries like CUB/Thrust).*

### `hinge_loss_main.cu`
-   Generates random true labels (`-1` or `+1`) and predicted scores.
-   Calls `hinge_loss_cuda` to compute individual losses on the GPU.
-   Calls `sum_hinge_loss_cuda` to compute the total Hinge Loss on the GPU.
-   Implements CPU versions (`hinge_loss_cpu`, `sum_hinge_loss_cpu`) for verification.
-   Prints both GPU and CPU results and checks if they match within a small tolerance.

### `hinge_loss_test.cu`
-   Contains Google Test cases:
    -   `IndividualLossesSmall`: Tests element-wise loss computation with a small, predefined dataset.
    -   `SumLossSmall`: Tests total loss computation with the same small dataset.
    -   `AllCorrectOutsideMargin`: Tests a scenario where all predictions are correct and outside the margin (expected total loss = 0).
    -   `AllMisclassifiedOrInMargin`: Tests a scenario where all predictions are either misclassified or correct but within the margin, ensuring non-zero loss is correctly calculated.

## Key CUDA Features Used
-   Basic CUDA kernel launch (`<<<...>>>`).
-   Device memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`).
-   Error checking (`cudaGetLastError`, `cudaGetErrorString` via `CHECK_CUDA_ERROR` macro).
-   Thread indexing (`blockIdx.x`, `blockDim.x`, `threadIdx.x`).

## Performance Considerations
-   The `hinge_loss_kernel` is a simple element-wise operation and should perform well, being memory-bound by the reads from `d_true_labels` and `d_pred_scores` and writes to `d_loss`.
-   The `sum_hinge_loss_cuda` function, in its current form, involves a device-to-host copy of all individual losses for CPU summation. This is inefficient for large datasets due to the DtoH transfer overhead and serial CPU summation. For optimal performance, a full parallel reduction on the GPU would be necessary. This was simplified to focus on the Hinge Loss computation itself and ensure correctness for the example.

## Building and Running

This project uses CMake. Ensure you have the CUDA Toolkit and CMake installed and configured for your Jetson Nano environment (or a compatible cross-compilation setup targeting `sm_53`).

1.  **Configure CMake** (from the `build` directory, assuming you are in the root of the `100-days-of-cuda` project):
    ```bash
    cd build # Or create it if it doesn't exist: mkdir build && cd build
    cmake .. 
    ```
    If you are building for a specific day directly (e.g., inside `day091/build`):
    ```bash
    cd day091
    mkdir build && cd build
    cmake ..
    ```

2.  **Build the project**:
    ```bash
    make -j$(nproc) 
    ```
    This will build the `hinge_loss_main` executable and the `hinge_loss_test` executable. They will be located in `build/day091/` (if building from the root `build` dir) or `day091/build/` (if building from `day091/build`).

3.  **Run the main demonstration**:
    ```bash
    ./build/day091/hinge_loss_main 
    ```
    (Adjust path if building from within `day091/build`)

4.  **Run the tests**:
    ```bash
    cd build # (Or your build directory)
    ctest --output-on-failure -R day091_hinge_loss # Run tests for day091
    # Or directly:
    # ./build/day091/hinge_loss_test
    ```

## Execution Results

The code was compiled and run on a Jetson Nano.

### Output from `hinge_loss_main`:
```
drboom@JetNano ~/g/1/build> ./day091/hinge_loss_main 
Generating input data...
True Labels (Host): [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1]
Predicted Scores (Host): [0.851821, -0.286116, 0.763539, 0.876601, -0.0355242, 1.12011, -0.356302, 0.318777, -1.4402, -0.39593, 0.509268, -0.703396, -1.02096, 0.779021, 0.37561, 0.527168]

--- Testing Individual Hinge Loss Computation ---
Individual Losses (GPU): [0.1482, 0.7139, 1.7635, 0.1234, 1.0355, 2.1201, 1.3563, 0.6812, 0.0000, 0.6041, 1.5093, 1.7034, 0.0000, 1.7790, 1.3756, 0.4728]
Individual Losses (CPU - Verification): [0.1482, 0.7139, 1.7635, 0.1234, 1.0355, 2.1201, 1.3563, 0.6812, 0.0000, 0.6041, 1.5093, 1.7034, 0.0000, 1.7790, 1.3756, 0.4728]
Individual losses match CPU: Yes

--- Testing Sum of Hinge Losses ---
Total Hinge Loss (GPU): 15.3864
Total Hinge Loss (CPU - Verification): 15.3864
Total loss matches CPU: Yes

Demonstration finished.
```
The `hinge_loss_main` executable returns 0, indicating all internal checks passed.

### Output from `hinge_loss_test` (Google Test):
```
drboom@JetNano ~/g/1/build> ./day091/hinge_loss_test
Running main() from /home/drboom/git_repos/100-days-of-cuda/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 4 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 4 tests from HingeLossTest
[ RUN      ] HingeLossTest.IndividualLossesSmall
[       OK ] HingeLossTest.IndividualLossesSmall (79 ms)
[ RUN      ] HingeLossTest.SumLossSmall
[       OK ] HingeLossTest.SumLossSmall (1 ms)
[ RUN      ] HingeLossTest.AllCorrectOutsideMargin
[       OK ] HingeLossTest.AllCorrectOutsideMargin (1 ms)
[ RUN      ] HingeLossTest.AllMisclassifiedOrInMargin
[       OK ] HingeLossTest.AllMisclassifiedOrInMargin (1 ms)
[----------] 4 tests from HingeLossTest (83 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 1 test suite ran. (83 ms total)
[  PASSED  ] 4 tests.
```
All 4 Google Tests passed successfully.

## Learnings and Observations
-   Implementing Hinge Loss element-wise in CUDA is straightforward.
-   The main challenge in such computations often lies in efficient parallel reductions (like sum) if needed. This example simplified the sum for clarity but highlighted where optimizations would go.
-   Error checking (`CHECK_CUDA_ERROR`) is crucial for debugging CUDA applications.
-   Comparing GPU results with a CPU implementation is a vital step for verification.

## Future Improvements
-   Implement a fully parallel sum reduction on the GPU for `sum_hinge_loss_cuda` to avoid the DtoH copy and CPU sum, making it efficient for large datasets.
-   Benchmark the CUDA implementation against the CPU version for various data sizes.
-   Extend to support different data types (e.g., double precision).
