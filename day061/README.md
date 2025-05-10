# Day 61: Fisher Information Matrix

## Overview
This project explores the concept of the Fisher Information Matrix (FIM) and its applications. The FIM is a way of measuring the amount of information that an observable random variable X carries about an unknown parameter θ upon which the probability of X depends.

## Implementation Details
The current implementation focuses on:
- Taking pre-computed "log probabilities" (interpreted as score vectors per sample) as input.
- Computing the Fisher Information Matrix (FIM) by averaging the outer product of these score vectors across all samples.
- Providing both CPU and GPU (CUDA) implementations for this computation.
- The FIM element \( F_{ij} \) is computed as \( \frac{1}{N_{samples}} \sum_{k=1}^{N_{samples}} s_{ki} s_{kj} \), where \( s_{ki} \) is the i-th component of the score vector for the k-th sample.
- Demonstrating its use with randomly generated score vectors and benchmarking CPU vs GPU performance.

The calculation of the score function itself (gradient of a specific log-likelihood model) is not part of this specific implementation but is a prerequisite for generating the input `log_probs`.

## Key CUDA Features Used
- CUDA kernel for parallel computation of FIM elements. Each element \( F_{ij} \) is computed by a separate thread.
- Summation over samples is performed sequentially within each thread. This part could be further parallelized using reduction for very large `n_samples`, but for moderately sized `n_samples` and many parameters, the current approach is effective.
- Standard CUDA memory management (cudaMalloc, cudaMemcpy, cudaFree).
- Kernel launch using a 2D grid of thread blocks, mapping each thread to an element of the resulting Fisher matrix.

## Performance Considerations
The GPU implementation demonstrates significant speedup over the CPU for larger problem sizes (i.e., a larger number of parameters and samples).
- For small problem sizes (e.g., 1000 samples, 16 parameters), the overhead of CUDA kernel launch and memory transfers can make the GPU version slower than the CPU version.
- As the number of parameters (`n_params`) and samples (`n_samples`) increases, the parallelism offered by the GPU becomes highly beneficial. The computation for each element of the Fisher matrix involves a sum over `n_samples`, and these \( \text{n\_params}^2 \) sums are computed in parallel by GPU threads.
- The provided benchmarks show a speedup of up to ~250x for 50,000 samples and 64 parameters on the Jetson Nano. This highlights the suitability of CUDA for this type of computation when data parallelism can be effectively exploited.

## Building and Running
The project is built using CMake. Ensure you have the CUDA Toolkit installed.

1.  Create a build directory:
    ```bash
    mkdir build
    cd build
    ```
2.  Run CMake and build:
    ```bash
    cmake ..
    make
    ```
3.  Run the main executable:
    ```bash
    ./day061/fisher_main
    ```
4.  Run the tests:
    ```bash
    ctest --output-on-failure -R day061_fisher_matrix
    # Or directly: ./day061/fisher_matrix_test
    ```

## Execution Results

Output from `./day061/fisher_main` on Jetson Nano:
```
Day 61: Fisher Information Matrix - Main Program
--- Testing and Benchmarking for N_SAMPLES = 1000, N_PARAMS = 16 ---
Verification: Max difference CPU vs GPU = 0.000000
Benchmark: CPU Time = 1.37 ms, GPU Time = 71.28 ms
Speedup (CPU/GPU) = 0.02x
----------------------------------------------------------
--- Testing and Benchmarking for N_SAMPLES = 10000, N_PARAMS = 32 ---
Verification: Max difference CPU vs GPU = 0.000000
Benchmark: CPU Time = 78.97 ms, GPU Time = 3.03 ms
Speedup (CPU/GPU) = 26.06x
----------------------------------------------------------
--- Testing and Benchmarking for N_SAMPLES = 50000, N_PARAMS = 64 ---
Verification: Max difference CPU vs GPU = 0.000000
Benchmark: CPU Time = 5018.75 ms, GPU Time = 20.05 ms
Speedup (CPU/GPU) = 250.35x
----------------------------------------------------------

All tests and benchmarks completed.
```

Output from `./day061/fisher_matrix_test`:
```
Running main() from /home/drboom/git_repos/100-days-of-cuda/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 2 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 2 tests from FisherMatrixTest
[ RUN      ] FisherMatrixTest.GpuVsCpuComparison
[       OK ] FisherMatrixTest.GpuVsCpuComparison (92 ms)
[ RUN      ] FisherMatrixTest.GpuVsCpuComparisonLargerParams
[       OK ] FisherMatrixTest.GpuVsCpuComparisonLargerParams (1 ms)
[----------] 2 tests from FisherMatrixTest (94 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test suite ran. (94 ms total)
[  PASSED  ] 2 tests.
```

## Learnings and Observations
- The Fisher Information Matrix calculation, particularly the summation of outer products of score vectors, is well-suited for parallelization on a GPU.
- The CUDA kernel assigns each thread to compute one element of the FIM. The main parallelism comes from computing all \( \text{n\_params}^2 \) elements concurrently.
- For smaller datasets or fewer parameters, the overhead associated with CUDA (kernel launch, data transfers between host and device) can outweigh the benefits of parallelism, leading to the GPU version being slower than a simple CPU loop. This was observed for the (1000 samples, 16 params) case.
- As the problem size scales (more samples and/or more parameters), the GPU's parallel processing capability leads to substantial speedups. The benchmark with 50,000 samples and 64 parameters showed a ~250x speedup.
- The input to this implementation is assumed to be pre-computed score vectors (gradients of log-likelihood per sample). In a full application, generating these scores (which might involve evaluating a complex model and its derivatives for each data point) could also be a target for CUDA acceleration.
- Correctness was verified by comparing GPU results against a CPU implementation, with negligible differences.

## References
- Wikipedia: Fisher Information
- A Tutorial on Fisher Information (arXiv:1705.01064)
- Wittman, D. (N.D.). Fisher Matrix Guide. UC Davis Physics.
