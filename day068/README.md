# Day 068: LoRA (Low-Rank Adaptation) Implementation and Benchmarking

## Overview

This project implements Low-Rank Adaptation (LoRA) for a linear layer and benchmarks its forward pass on both CPU and GPU. LoRA is a technique used to efficiently fine-tune large pre-trained models by injecting trainable low-rank matrices into the existing weights. Instead of updating the entire weight matrix `W` (size `d x d`), LoRA adds an update `BA`, where `A` (size `r x d`) is a random Gaussian matrix and `B` (size `d x r`) is a zero matrix. Here, `r` is the rank and is much smaller than `d`. The forward pass modification is `h = Wx + (alpha/r)BAx`. This project focuses on computing the `(alpha/r)BAx` part.

## Implementation Details

The core LoRA logic involves two matrix-vector multiplications and a scaling step:
1.  **Down-projection:** `temp = A * x`
    *   `A`: `rank x d_model` (randomly initialized, e.g., Kaiming normal)
    *   `x`: `d_model x 1` (input vector)
    *   `temp`: `rank x 1`
2.  **Up-projection:** `lora_output = B * temp`
    *   `B`: `d_model x rank` (initialized to zeros, then trained)
    *   `temp`: `rank x 1`
    *   `lora_output`: `d_model x 1`
3.  **Scaling:** `lora_output = lora_output * (alpha / rank)`
    *   `alpha`: A scaling constant, often set to `1.0` or similar to the rank.

Both CPU and GPU versions of this forward pass are implemented.
-   `lora.cuh`: Declares the `LoRAParameters` struct, initialization/free functions, and forward pass functions (`loraForwardCPU`, `loraForwardGPU`), along with CUDA kernel declarations.
-   `lora.cu`: Implements the functions for LoRA.
    -   `initializeLoRAParameters`: Allocates and initializes LoRA matrices `A` (random normal) and `B` (zeros) on both host and device. It also initializes a `cublasHandle_t`.
    -   `loraForwardGPU`: Orchestrates the GPU computation using cuBLAS functions (`cublasSgemv` for matrix-vector products and `cublasSscal` for scaling). Custom kernels for these operations are present in the file for reference but are no longer used by `loraForwardGPU`.
    -   `loraForwardCPU`: Implements the same logic sequentially on the CPU.
-   `lora_main.cu`: Contains the main function for benchmarking. It generates synthetic input data, runs the CPU and GPU forward passes multiple times, measures execution time, and verifies that the outputs match. The benchmark parameters were adjusted to `d_model=4096` and `rank=64`.
-   `lora_test.cu`: Contains Google Tests to verify the correctness of the LoRA forward pass by comparing CPU and GPU outputs and checking parameter initialization.

## Key CUDA Features Used

-   **cuBLAS Library:** The GPU implementation (`loraForwardGPU`) now leverages the cuBLAS library for optimized Level 2 BLAS operations:
    -   `cublasSgemv`: Used for the two matrix-vector multiplications (`A*x` and `B*temp_vec`). Row-major matrices are handled by using `CUBLAS_OP_T` and adjusting `m, n, lda` parameters appropriately.
    -   `cublasSscal`: Used for scaling the final LoRA output vector.
    -   `cublasHandle_t`: A cuBLAS context handle is created and used for library calls.
-   **Device Memory Management:** `cudaMalloc` for allocating memory on the GPU for matrices `A`, `B`, input vector, and output vectors. `cudaMemcpy` for transferring data between host and device. `cudaFree` for deallocating GPU memory.
-   **Error Handling:** `CHECK_CUDA_ERROR` and `CHECK_CUBLAS_ERROR` macros are used to check for errors from CUDA API and cuBLAS calls respectively.
-   **CUDA Events:** `cudaEvent_t` is used for accurate timing of GPU execution.

## Performance Considerations

-   **Matrix-Vector Multiplication with cuBLAS:** Using `cublasSgemv` significantly improves the performance and reliability of the matrix-vector multiplications compared to basic custom kernels, especially as problem sizes grow. cuBLAS routines are highly optimized for NVIDIA GPU architectures.
-   **Kernel Launch Overhead vs. Library Calls:** While cuBLAS calls are efficient, there's still some overhead. For very small problems, this might be noticeable. The current implementation makes three separate cuBLAS calls.
-   **Data Transfers:** Data for matrices `A`, `B`, and the input vector `x` are transferred to the GPU once. The output is transferred back after computation. For iterative processes (like model training), `A` and `B` would reside on the GPU.
-   **CPU vs. GPU:** The GPU is expected to outperform the CPU significantly for larger `d_model` values due to its parallel processing capabilities, especially when the data and LoRA parameters are already on the device.

## Building and Running

### Prerequisites
- NVIDIA CUDA Toolkit (>= 10.0, tested with 11.x, 12.x)
- CMake (>= 3.10)
- A C++ compiler compatible with CUDA (e.g., g++)
- Google Test (will be fetched by CMake if not found, or assumed pre-installed on the build system)

### Build Instructions (from the root `100-days-of-cuda` directory)
1.  Ensure `day068` is added to the root `CMakeLists.txt`:
    ```cmake
    # ... other days ...
    add_subdirectory(day068)
    ```
2.  Create a build directory and navigate into it:
    ```bash
    mkdir build
    cd build
    ```
3.  Run CMake and build:
    ```bash
    cmake ..
    make -j # The -j flag enables parallel compilation
    ```
    On the Jetson Nano or CI environment, the CMake command might be:
    ```bash
    cmake .. -DCMAKE_CUDA_ARCHITECTURES=53
    # or the architecture might be set globally
    ```

### Running the Benchmark
The benchmark executable will be located at `build/day068/lora_benchmark`. Ensure you have compiled with the latest changes (using cuBLAS).
```bash
./build/day068/lora_benchmark
```

### Running Tests
Tests can be run using CTest from the build directory:
```bash
cd build # if not already there
ctest --output-on-failure -R day068_lora # Run tests specifically for day068
```
Or by directly running the test executable:
```bash
./build/day068/lora_test_exec
```

## Execution Results

The following results were obtained on a Jetson Nano with `d_model = 4096`, `rank = 64`, and 1000 iterations, using the cuBLAS implementation for the GPU:

```
--- LoRA Implementation Benchmark ---
Model Dimension (d_model): 4096
LoRA Rank (rank):          64
LoRA Alpha (alpha):        1
Benchmark Iterations:    1000

Running CPU LoRA Forward Pass...
CPU LoRA Forward Pass (avg per iteration): 1.538770 ms

Running GPU LoRA Forward Pass...
GPU LoRA Forward Pass (avg per iteration): 1.442278 ms

Verifying CPU and GPU results...
SUCCESS: CPU and GPU results match.

Cleaning up resources...

Benchmark complete.
```

**Test Output (should remain passing):**
```
[Insert console output of ctest or lora_test_exec here, e.g.:]
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from LoRATest
[ RUN      ] LoRATest.ForwardPassComparison
[       OK ] LoRATest.ForwardPassComparison (XXX ms)
[ RUN      ] LoRATest.ParameterInitialization
[       OK ] LoRATest.ParameterInitialization (X ms)
[----------] 2 tests from LoRATest (XXX ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (XXX ms total)
[  PASSED  ] 2 tests.
```

## Learnings and Observations

-   Initial custom CUDA kernels for matrix-vector multiplication were significantly slower than the CPU implementation on the Jetson Nano, even for moderately large problem sizes (`d_model=4096, rank=64`). This highlights that naive kernel implementations can be outperformed by optimized CPU code due to factors like kernel launch overhead, memory access patterns, and the CPU's own SIMD capabilities.
-   Switching to cuBLAS (`cublasSgemv` and `cublasSscal`) for the GPU implementation resulted in correct execution and a performance improvement, making the GPU slightly faster than the CPU for the tested dimensions on the Jetson Nano.
-   Correctly using cuBLAS with row-major C-style matrices requires careful handling of the `trans` operation parameter and the `m`, `n`, and `lda` arguments to `cublasSgemv`. The common approach is to use `CUBLAS_OP_T` and adjust `m`, `n`, and `lda` based on the original row-major dimensions.
-   The Jetson Nano's GPU, while capable, provides a modest speedup for this LoRA workload compared to its CPU. On more powerful dGPUs, the speedup from cuBLAS would likely be much more substantial.
-   Error checking for both CUDA API calls and cuBLAS library calls (`CHECK_CUDA_ERROR`, `CHECK_CUBLAS_ERROR`) is crucial for debugging.

## (Optional) Future Improvements
-   Implement a fused kernel for the entire `(alpha/r)BAx` operation to potentially reduce overhead from multiple cuBLAS calls and intermediate memory (d_temp_vec). This would be a good exercise in advanced kernel writing.
-   Extend to support batch processing of input vectors.
-   Explore different initialization strategies for matrices A and B.

## (Optional) References
-   Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.
