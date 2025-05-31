# Day 82: Negative Cosine Similarity (Cosine Distance)

## Date: 2025-05-31

## Overview

This project implements the calculation of "Negative Cosine Similarity," more accurately termed Cosine Distance, using CUDA. Cosine similarity measures the cosine of the angle between two non-zero vectors. It ranges from -1 (perfectly opposite) through 0 (orthogonal) to +1 (perfectly similar).

When cosine similarity is used in contexts like loss functions for machine learning (e.g., triplet loss, contrastive loss), it's often transformed into a distance metric where 0 indicates perfect similarity. A common transformation is `1.0 - cosine_similarity`. This project implements this transformation:

-   If `cosine_similarity` = +1 (vectors are identical), `output` = `1.0 - 1.0 = 0.0`.
-   If `cosine_similarity` = 0 (vectors are orthogonal), `output` = `1.0 - 0.0 = 1.0`.
-   If `cosine_similarity` = -1 (vectors are opposite), `output` = `1.0 - (-1.0) = 2.0`.

The resulting "cosine distance" ranges from 0 (most similar) to 2 (most dissimilar/opposite). A negative raw cosine similarity (e.g., -0.8) would result in a large cosine distance (e.g., `1 - (-0.8) = 1.8`), indicating high dissimilarity.

## Implementation Details

The core logic is implemented in `negative_cosine_similarity.cu` and its header `negative_cosine_similarity.cuh`.

-   **`cosine_similarity_kernel` (__global__):** This CUDA kernel is launched with one thread per vector pair. Each thread calculates:
    1.  The dot product of the two input vectors (`predictions[i]` and `targets[i]`).
    2.  The L2 norm (magnitude) of each vector.
    3.  The cosine similarity: `dot_product / (norm_prediction * norm_target)`.
        -   An epsilon (`1e-8f`) is used with `fmaxf` to prevent division by zero if a vector's norm is zero. If norms are extremely small, cosine similarity defaults to 0.
    4.  The final output: `1.0f - cosine_similarity`.
-   **`launch_cosine_similarity_kernel` (extern "C"):** This host function calculates the necessary grid and block dimensions and launches the `cosine_similarity_kernel`. It handles basic error checking using `cudaGetLastError()`.

A demonstration program (`negative_cosine_similarity_main.cu`) initializes sample vector data, calls the CUDA implementation, and verifies the results against a CPU version. Unit tests (`negative_cosine_similarity_test.cu`) provide further verification using Google Test.

## Key CUDA Features Used

-   Basic CUDA kernel launch (`<<<...>>>`).
-   Thread indexing using `blockIdx.x`, `blockDim.x`, and `threadIdx.x`.
-   Device math functions: `sqrtf` (for norms), `fmaxf` (for robust division).
-   CUDA memory management: `cudaMalloc`, `cudaMemcpy`, `cudaFree`.
-   Error handling: `cudaGetLastError`, `cudaGetErrorString`, `cudaDeviceSynchronize`.

## Performance Considerations

-   **Memory Access:** For each vector pair, the kernel reads `2 * d` float values (where `d` is the vector dimension). If `d` is small and data is contiguous, reads can be somewhat coalesced within a warp.
-   **Parallelism:** The problem is inherently parallel, with each vector pair's cosine distance calculation being independent. The implementation assigns one thread per pair.
-   **Computational Load:** The main computation per thread involves a loop of `d` iterations for dot products and norm calculations, followed by a few arithmetic operations.
-   **Robustness:** Using `fmaxf` with a small epsilon for the denominator in the cosine similarity calculation prevents division by zero and handles zero-magnitude vectors gracefully, typically resulting in a cosine similarity of 0 for such cases.

## Building and Running

The project uses CMake. Ensure you have the CUDA Toolkit and CMake installed. Google Test is fetched automatically by CMake.

**Build Instructions (from the root `100-days-of-cuda` directory):**

1.  **Configure CMake:**
    ```bash
    mkdir -p build
    cd build
    cmake ..
    ```
2.  **Build the project (and specifically Day 82 targets):**
    ```bash
    cmake --build . --target neg_cosine_sim_main --target neg_cosine_sim_test -j $(nproc) 
    ```
    (Replace `$(nproc)` with the number of cores you want to use, e.g., `4` or `8`).

**Running the Executables:**

-   **Main demonstration:**
    ```bash
    ./build/day082/neg_cosine_sim_main 
    ```
-   **Tests:**
    ```bash
    cd build # (if not already there)
    ctest --output-on-failure -R day082_negative_cosine_similarity # Run tests for day082
    # Or directly: ./day082/neg_cosine_sim_test
    ```

## Execution Results

The `neg_cosine_sim_main` executable outputs the input vectors, the GPU-computed `1.0 - cosine_similarity` values, and the CPU-computed values for verification. The tests from `neg_cosine_sim_test` verify various scenarios.

**Output from `neg_cosine_sim_main`:**
```text
Predictions (Host) (Rows: 5, Dim: 3):
  Vec 0: [1.0000, 2.0000, 3.0000]
  Vec 1: [1.0000, 0.0000, 0.0000]
  Vec 2: [1.0000, 1.0000, 1.0000]
  Vec 3: [0.5000, -0.5000, 1.0000]
  Vec 4: [0.0000, 0.0000, 0.0000]
Targets (Host) (Rows: 5, Dim: 3):
  Vec 0: [1.0000, 2.0000, 3.0000]
  Vec 1: [0.0000, 1.0000, 0.0000]
  Vec 2: [-1.0000, -1.0000, -1.0000]
  Vec 3: [0.2000, 0.8000, -0.3000]
  Vec 4: [1.0000, 2.0000, 3.0000]
GPU Output (1.0 - Cosine Similarity):
  Output for pair 0: 0.000000
  Output for pair 1: 1.000000
  Output for pair 2: 2.000000
  Output for pair 3: 1.558290
  Output for pair 4: 1.000000
CPU Output (1.0 - Cosine Similarity) for Verification:
  Output for pair 0: 0.000000
  Output for pair 1: 1.000000
  Output for pair 2: 2.000000
  Output for pair 3: 1.558290
  Output for pair 4: 1.000000
Verification Successful: GPU and CPU results match within tolerance.
```

**Output from `neg_cosine_sim_test`:**
```text
Running main() from /home/drboom/git_repos/100-days-of-cuda/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 6 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 6 tests from NegativeCosineSimilarityTest
[ RUN      ] NegativeCosineSimilarityTest.HandlesIdenticalVectors
[       OK ] NegativeCosineSimilarityTest.HandlesIdenticalVectors (92 ms)
[ RUN      ] NegativeCosineSimilarityTest.HandlesOrthogonalVectors
[       OK ] NegativeCosineSimilarityTest.HandlesOrthogonalVectors (1 ms)
[ RUN      ] NegativeCosineSimilarityTest.HandlesOppositeVectors
[       OK ] NegativeCosineSimilarityTest.HandlesOppositeVectors (1 ms)
[ RUN      ] NegativeCosineSimilarityTest.HandlesZeroVectorCases
[       OK ] NegativeCosineSimilarityTest.HandlesZeroVectorCases (1 ms)
[ RUN      ] NegativeCosineSimilarityTest.MultipleVectorsBatch
[       OK ] NegativeCosineSimilarityTest.MultipleVectorsBatch (1 ms)
[ RUN      ] NegativeCosineSimilarityTest.LargerNAndD
[       OK ] NegativeCosineSimilarityTest.LargerNAndD (2 ms)
[----------] 6 tests from NegativeCosineSimilarityTest (101 ms total)

[----------] Global test environment tear-down
[==========] 6 tests from 1 test suite ran. (101 ms total)
[  PASSED  ] 6 tests.
```
*(Note: The exact float precision for pair 3 in `neg_cosine_sim_main` might vary slightly but should be close to 1.558290 based on the provided logs.)*

## Learnings and Observations

-   The `1.0 - cosine_similarity` transformation is a straightforward way to convert similarity into a distance-like metric suitable for minimization in loss functions.
-   Handling zero vectors or vectors with very small magnitudes requires careful use of epsilon values to ensure numerical stability in the cosine similarity calculation. The `fmaxf(eps, norm)` approach for the denominator components is effective.
-   The CUDA kernel structure for this type of element-wise operation (one thread per vector pair) is simple and maps well to the GPU architecture.
-   Consistent CPU verification and unit testing are crucial for ensuring the correctness of CUDA kernels, especially with floating-point arithmetic.

## (Optional) References
- Cosine Similarity: [https://en.wikipedia.org/wiki/Cosine_similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
