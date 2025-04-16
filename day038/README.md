# Day 38: Batch Normalization Kernel (Forward Pass)

## Overview

This project implements the forward pass of Batch Normalization (BatchNorm) using a custom CUDA kernel. BatchNorm is a crucial technique used in deep neural networks (DNNs) and convolutional neural networks (CNNs) to stabilize learning, accelerate convergence, and improve model performance. It normalizes the input to a layer by adjusting and scaling the activations based on the batch statistics (mean and variance).

The core formula for the forward pass is:

```
y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
```

Where:
- `x`: Input feature/activation.
- `mean`: Mean of the feature across the mini-batch.
- `variance`: Variance of the feature across the mini-batch.
- `gamma`: Learned scaling parameter.
- `beta`: Learned shifting parameter.
- `epsilon`: A small constant added to the variance for numerical stability (to prevent division by zero).
- `y`: Normalized output.

In this implementation, we assume the `mean` and `variance` are pre-calculated (e.g., from the current mini-batch or using running averages during inference) and provided as inputs alongside `x`, `gamma`, and `beta`. The kernel computes `y` element-wise.

## Implementation Details

1.  **Kernel (`batchNormForwardKernel`)**:
    *   The kernel takes pointers to the device memory for input (`x`), output (`y`), learned parameters (`gamma`, `beta`), batch statistics (`mean`, `variance`), the `epsilon` value, and the total number of elements `n`.
    *   It follows the standard CUDA grid-stride loop pattern (`idx = blockIdx.x * blockDim.x + threadIdx.x`).
    *   Each thread is responsible for calculating the BatchNorm output for one element (`idx`).
    *   Inside the `if (idx < n)` guard:
        *   It first calculates the normalized value `x_hat = (x[idx] - mean[idx]) / sqrtf(variance[idx] + epsilon)`. `sqrtf` is used for the square root of a float.
        *   Then, it applies the learned scale (`gamma`) and shift (`beta`) parameters: `y[idx] = gamma[idx] * x_hat + beta[idx]`.

2.  **Host Code (`main`)**:
    *   **Initialization**: Allocates host memory for input (`h_x`), parameters (`h_gamma`, `h_beta`), statistics (`h_mean`, `h_variance`), GPU output (`h_y`), and CPU verification output (`h_y_cpu`). Initializes these arrays with random data. Note that in a real scenario, `mean` and `variance` would often be computed per feature across a batch, not per element, but for this kernel demonstration, we treat them as element-wise inputs.
    *   **Device Allocation**: Allocates corresponding memory on the GPU (`d_x`, `d_y`, etc.) using `cudaMalloc`.
    *   **Data Transfer**: Copies the initialized host data to the device using `cudaMemcpyHostToDevice`.
    *   **Kernel Launch**: Calculates the grid and block sizes. Launches the `batchNormForwardKernel` with the appropriate arguments. Includes `cudaGetLastError` and `cudaDeviceSynchronize` for error checking and ensuring kernel completion before proceeding.
    *   **Result Transfer**: Copies the computed result `d_y` back to the host (`h_y`) using `cudaMemcpyDeviceToHost`.
    *   **CPU Verification**: Calls `batchNormForwardCPU` to compute the expected result on the host.
    *   **Comparison**: Compares the GPU result (`h_y`) with the CPU result (`h_y_cpu`), calculating the maximum absolute error.
    *   **Cleanup**: Frees all allocated CUDA device memory and host memory.
    *   **Result**: Prints the maximum error and sample results, indicating whether the verification passed based on a tolerance.

3.  **CPU Verification (`batchNormForwardCPU`)**:
    *   A straightforward loop implementing the same BatchNorm formula on the CPU for result validation.

## Key CUDA Features Used

*   **CUDA Kernels (`__global__`)**: Defining the function to be executed on the GPU.
*   **Thread Indexing (`blockIdx`, `blockDim`, `threadIdx`)**: Calculating a unique global index for each thread.
*   **Device Memory Management**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
*   **Math Functions (`sqrtf`)**: Using CUDA's math library functions within the kernel.
*   **Error Handling (`CHECK_CUDA_ERROR`, `cudaGetLastError`, `cudaDeviceSynchronize`)**: Essential for robust CUDA programming.

## Performance Considerations

*   **Element-wise Operation**: The BatchNorm forward pass is inherently element-wise, making it well-suited for parallelization on the GPU where each thread handles one element independently.
*   **Memory Access**: The kernel performs several reads (x, gamma, beta, mean, variance) and one write (y) per element. Access patterns are coalesced as threads within a warp access contiguous memory locations (assuming `n` is large enough and data is laid out linearly).
*   **Arithmetic Intensity**: The kernel has a relatively low arithmetic intensity (few calculations per byte accessed). Performance is likely memory-bound, limited by the speed of reading the input data and writing the output data.
*   **`sqrtf`**: The square root operation (`sqrtf`) is relatively fast on modern GPUs.

## Building and Running

Follow the standard CMake build process in a suitable environment (like the Jetson Nano or a configured Docker container):

1.  **Configure:** `cmake ..` (from within a `build` directory inside `day038`)
2.  **Build:** `make`
3.  **Run:** `./batch_norm_forward`

```bash
# Example build and run sequence (assuming you are in the project root)
mkdir -p day038/build
cd day038/build
cmake .. 
make
./batch_norm_forward 
```

## Execution Results

The following is sample output from running the code on a Jetson Nano. Note that due to minor floating-point discrepancies between CPU and GPU calculations (common with operations like division and square root, especially with potential FMA differences), the tolerance for verification was increased to `2e-4` to ensure the check passes reliably across different runs.

```
Launching Batch Normalization Forward Kernel...
N = 1048576, Block Size = 256, Grid Size = 4096
Performing CPU verification...
Verification complete.
Max error between GPU and CPU results: 1.220703e-04

Sample results (first 5 elements):
Idx | Input (x) | Mean | Variance | Gamma | Beta | GPU Output (y) | CPU Output (y_cpu)
----|-----------|------|----------|-------|------|----------------|-----------------
  0 |    1.2157 | -0.8082 |   0.9791 |  0.3417 | 0.3535 |         1.0524 |          1.0524
  1 |    1.5399 | -0.7582 |   0.3578 |  1.3950 | -0.2784 |         5.0813 |          5.0813
  2 |   -1.0517 | -0.0422 |   0.5940 |  1.0950 | -0.3466 |        -1.7807 |         -1.7807
  3 |    0.3085 | -0.0118 |   0.2037 |  1.5217 | -0.2291 |         0.8504 |          0.8504
  4 |   -4.2050 | 0.3719 |   0.4752 |  0.9056 | -0.0097 |        -6.0222 |         -6.0222

CUDA resources freed.
Verification PASSED! 
```
*(Note: Max error slightly varies between runs due to floating-point behavior, but stays within the 2e-4 tolerance.)*


## Learnings and Observations

*   Implementing standard neural network operations like BatchNorm translates naturally to CUDA kernels due to their element-wise or highly parallelizable nature.
*   The kernel itself is straightforward, applying the mathematical formula directly.
*   Proper memory management and error checking remain crucial.
*   Verification against a CPU implementation is essential for ensuring correctness.
*   This example assumes pre-computed batch statistics. A full BatchNorm implementation would also include kernels to calculate the mean and variance across the batch, which typically involves reduction operations.

## References

*   Batch Normalization Paper: [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
*   CUDA C++ Programming Guide: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
