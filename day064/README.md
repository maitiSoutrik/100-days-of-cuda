# Day 064: Spectral Normalization in GANs

## Overview

Spectral Normalization is a weight normalization technique primarily used to stabilize the training of Generative Adversarial Networks (GANs). It was introduced by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida in their 2018 paper "Spectral Normalization for Generative Adversarial Networks."

The core idea is to constrain the Lipschitz constant of the discriminator network by normalizing the spectral norm of each weight matrix in the network. The spectral norm of a matrix is its largest singular value. By ensuring that the spectral norm of each layer's weight matrix is 1 (or close to 1), the overall Lipschitz constant of the discriminator is controlled, preventing it from becoming too sensitive to input changes. This leads to more stable training dynamics and can improve the quality of generated samples.

## Implementation Details

The implementation involves two main parts:
1.  **Estimating the Spectral Norm:** The spectral norm `σ(W)` of a weight matrix `W` is estimated using the power iteration method. This is an iterative algorithm that converges to the largest singular value (and corresponding singular vectors).
    *   Initialize a random vector `v_0`.
    *   Iteratively compute:
        *   `u_k = W * v_{k-1}`
        *   `u_k = u_k / ||u_k||_2` (normalize `u_k`)
        *   `v_k = W^T * u_k`
        *   `v_k = v_k / ||v_k||_2` (normalize `v_k`)
    *   After a sufficient number of iterations, `σ(W) ≈ ||W * v_k||_2`.
    The implementation uses cuBLAS for matrix-vector multiplications (`cublasSgemv`) and vector L2-norm calculations (`cublasSnrm2`). Vector normalization is done using `cublasSscal` or a custom kernel.

2.  **Normalizing the Weight Matrix:** Once the spectral norm `σ(W)` is estimated, the weight matrix `W` is normalized by dividing it by `σ(W)`:
    `W_normalized = W / σ(W)`
    This operation is performed element-wise on the matrix `W`. A custom CUDA kernel (`scale_matrix_kernel`) is used for this division.

The `spectral_norm.cuh` header declares the functions, and `spectral_norm.cu` provides their CUDA implementations. `spectral_norm_main.cu` demonstrates the normalization on sample matrices, and `spectral_norm_test.cu` contains Google Tests for verification.

## Key CUDA Features Used

*   **cuBLAS Library:**
    *   `cublasCreate()`, `cublasDestroy()`: For cuBLAS handle management.
    *   `cublasSgemv()`: For matrix-vector multiplication (essential for power iteration).
    *   `cublasSnrm2()`: For calculating the L2 norm of vectors (used for normalizing `u` and `v` in power iteration and for the final spectral norm value).
    *   `cublasSscal()`: For scaling vectors (used for normalizing `u` and `v`).
*   **cuRAND Library:**
    *   `curandCreateGenerator()`, `curandSetPseudoRandomGeneratorSeed()`, `curandGenerateUniform()`, `curandDestroyGenerator()`: For initializing the vector `v` in the power iteration method with random numbers.
*   **Custom CUDA Kernels:**
    *   `normalize_vector_kernel()`: (Initially considered, but `cublasSscal` is more efficient for dense vectors) A kernel to normalize a vector by its L2 norm. The `initialize_random_vector` function uses this for the initial normalization.
    *   `scale_matrix_kernel()`: A kernel to perform element-wise division of a matrix by a scalar (the estimated spectral norm).
*   **CUDA Error Handling:** `CHECK_CUDA_ERROR`, `CHECK_CUBLAS_ERROR`, `CHECK_CURAND_ERROR` macros are used for robust error checking.
*   **Device Memory Management:** `cudaMalloc()`, `cudaFree()`, `cudaMemcpy()`.

## Performance Considerations

*   **Power Iteration:** The number of iterations in the power iteration method affects the accuracy of the spectral norm estimation and the computational cost. More iterations lead to better accuracy but higher cost. For GAN training, a small number of iterations (e.g., 1-10) is often sufficient.
*   **cuBLAS vs. Custom Kernels:** cuBLAS functions are highly optimized. For operations like matrix-vector multiplication and vector scaling/norm, they are generally preferred over custom kernels unless very specific optimizations are needed for particular matrix structures or sizes.
*   **Overhead:** For very small matrices, the overhead of CUDA calls and kernel launches might be significant. Spectral normalization is typically applied to layers in neural networks, which can have substantial weight matrices.

## Building and Running

**Prerequisites:**
*   NVIDIA CUDA Toolkit (>= 10.0, tested with 11.x, 12.x)
*   CMake (>= 3.10)
*   A C++ compiler compatible with CUDA (e.g., g++)
*   Google Test (will be fetched by CMake)

**Build Instructions (from the root `100-days-of-cuda` directory):**
1.  Ensure the `day064` subdirectory is added to the root `CMakeLists.txt`:
    ```cmake
    # In root CMakeLists.txt
    add_subdirectory(day064)
    ```
2.  Create a build directory and navigate into it:
    ```bash
    mkdir build
    cd build
    ```
3.  Run CMake and build:
    ```bash
    cmake ..
    make -j # The -j flag builds in parallel
    ```
    On the Jetson Nano, `make` without `-j` might be more stable if memory is constrained.

**Running the Executables:**
*   **Main Application:**
    ```bash
    ./day064/spectral_norm_app
    ```
*   **Tests:**
    ```bash
    ./day064/spectral_norm_test_app
    # Or run all tests using CTest from the build directory
    # ctest --output-on-failure
    ```

## Execution Results (Expected Output from `spectral_norm_app`)

The main application will output:
1.  For a 2x2 matrix `W1 = [1 0; 0 2]`:
    *   Original matrix.
    *   Estimated spectral norm (should be close to 2.0).
    *   Normalized matrix (should be `[0.5 0; 0 1]`).
    *   Estimated spectral norm of the normalized matrix (should be close to 1.0).
2.  For a 3x2 matrix `W2 = [1 4; 2 5; 3 6]`:
    *   Original matrix.
    *   Estimated spectral norm (around 9.5080).
    *   Normalized matrix.
    *   Estimated spectral norm of the normalized matrix (should be close to 1.0).
3.  For a 2x3 matrix `W3 = [1 3 5; 2 4 6]`:
    *   Original matrix.
    *   Estimated spectral norm (around 9.5080, same as W2 due to singular value properties).
    *   Normalized matrix.
    *   Estimated spectral norm of the normalized matrix (should be close to 1.0).

The exact floating-point values might vary slightly due to the nature of power iteration and floating-point arithmetic. The following output includes results from both `spectral_norm_app` and `spectral_norm_test_app`.

```
--- Test Case 1: 2x2 Matrix ---
Original Matrix W1 (2x2):
1.0000  0.0000
0.0000  2.0000

Estimated Spectral Norm (before normalization) for W1: 2.0000

Normalized Matrix W1_norm (2x2):
0.5000  0.0000
0.0000  1.0000

Estimated Spectral Norm (after normalization) for W1_norm: 1.0000


--- Test Case 2: 3x2 Matrix ---
Original Matrix W2 (3x2):
1.0000  4.0000
2.0000  5.0000
3.0000  6.0000

Estimated Spectral Norm (before normalization) for W2: 9.5080

Normalized Matrix W2_norm (3x2):
0.1052  0.4207
0.2103  0.5259
0.3155  0.6310

Estimated Spectral Norm (after normalization) for W2_norm: 1.0000


--- Test Case 3: 2x3 Matrix ---
Original Matrix W3 (2x3):
1.0000  3.0000  5.0000
2.0000  4.0000  6.0000

Estimated Spectral Norm (before normalization) for W3: 9.5255

Normalized Matrix W3_norm (2x3):
0.1050  0.3149  0.5249
0.2100  0.4199  0.6299

Estimated Spectral Norm (after normalization) for W3_norm: 1.0000

[==========] Running 5 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 5 tests from SpectralNormTest
[ RUN      ] SpectralNormTest.EstimateSpectralNorm_Identity2x2
[       OK ] SpectralNormTest.EstimateSpectralNorm_Identity2x2 (1140 ms)
[ RUN      ] SpectralNormTest.EstimateSpectralNorm_Simple2x2
[       OK ] SpectralNormTest.EstimateSpectralNorm_Simple2x2 (75 ms)
[ RUN      ] SpectralNormTest.EstimateSpectralNorm_3x2Matrix
[       OK ] SpectralNormTest.EstimateSpectralNorm_3x2Matrix (81 ms)
[ RUN      ] SpectralNormTest.SpectralNormalizeMatrix_Simple2x2
[       OK ] SpectralNormTest.SpectralNormalizeMatrix_Simple2x2 (150 ms)
[ RUN      ] SpectralNormTest.SpectralNormalizeMatrix_ZeroMatrix
[       OK ] SpectralNormTest.SpectralNormalizeMatrix_ZeroMatrix (202 ms)
[----------] 5 tests from SpectralNormTest (1650 ms total)

[----------] Global test environment tear-down
[==========] 5 tests from 1 test suite ran. (1650 ms total)
[  PASSED  ] 5 tests.
```

## Learnings and Observations

*   Spectral normalization effectively constrains the largest singular value of a weight matrix to 1.
*   The power iteration method is a practical way to estimate the spectral norm without computing the full SVD. The number of iterations is a trade-off between accuracy and speed.
*   Using cuBLAS for linear algebra operations is crucial for performance in CUDA.
*   Careful management of device memory and cuBLAS/cuRAND handles is necessary.
*   This technique is a key component in improving the stability and performance of modern GAN architectures.

## Future Improvements
*   Implement spectral normalization for convolutional layers, which involves reshaping the kernel tensor into a matrix.
*   Compare the performance with different numbers of power iterations.
*   Integrate this into a small GAN discriminator layer to see its effect in a network context (though this is beyond a single-day scope for this project).
