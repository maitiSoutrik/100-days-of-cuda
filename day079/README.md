# Day 79: Upper Triangular Matrix Multiplication

## Overview
This CUDA kernel performs matrix multiplication of two upper triangular matrices, exploiting the sparsity pattern to optimize performance.

## Implementation Details
- Uses shared memory to cache tiles of the input matrices, reducing global memory accesses
- Each thread block computes one tile of the output matrix
- Threads only compute the upper triangle of the result matrix
- Breaks out of inner loop early when i < j to skip unnecessary work

## Key CUDA Features Used
- Shared memory
- Thread block tiling
- Synchronization with `__syncthreads()`

## Performance Considerations
- The kernel avoids unnecessary computation and memory accesses by only computing the upper triangle
- Shared memory usage reduces global memory bandwidth requirements
- Different block sizes can be experimented with to find the optimal configuration

## Building and Running
From the `day079` directory:
```bash
mkdir build
cd build
cmake ..
make
./upper_tri_gemm
```

## Execution Results
The following output is from the `upper_tri_gemm` executable when run with default parameters (typically a 4x4 or 5x5 matrix, depending on `upper_tri_gemm_main.cu`). The test suite (`upper_tri_gemm_test`) also passes, confirming correctness for float and double types with a 4x4 matrix.

Output from `./day079/upper_tri_gemm`:
```
0 0 -1 -4 -10 
0 -2 -7 -16 
0 -4 -13 
0 -6 
0 
```

Output from `./day079/upper_tri_gemm_test`:
```
Running main() from /home/drboom/git_repos/100-days-of-cuda/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 2 tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 1 test from UpperTriGemmTest/0, where TypeParam = float
[ RUN      ] UpperTriGemmTest/0.Correctness
[       OK ] UpperTriGemmTest/0.Correctness (83 ms)
[----------] 1 test from UpperTriGemmTest/0 (83 ms total)

[----------] 1 test from UpperTriGemmTest/1, where TypeParam = double
[ RUN      ] UpperTriGemmTest/1.Correctness
[       OK ] UpperTriGemmTest/1.Correctness (1 ms)
[----------] 1 test from UpperTriGemmTest/1 (1 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 2 test suites ran. (85 ms total)
[  PASSED  ] 2 tests.
```

## Learnings and Observations
Initial implementation had a bug where the GPU output was all zeros. This was resolved by:
1.  **Robust Shared Memory Handling:** Ensuring that elements loaded into shared memory tiles (`A_tile`, `B_tile`) are explicitly set to zero if they correspond to out-of-bounds accesses in the global matrices. This prevents uninitialized or stale data in shared memory from corrupting the calculation.
2.  **CUDA Error Checking:** Adding a `CHECK_CUDA_ERROR` macro and using it after CUDA API calls (especially kernel launches and memory copies) in the test file. This is crucial for diagnosing silent CUDA errors.

Exploiting the sparsity pattern of upper triangular matrices can lead to significant performance improvements by reducing unnecessary computation and memory accesses. Shared memory tiling is an effective technique to further optimize memory bandwidth usage.
Exploiting the sparsity pattern of upper triangular matrices can lead to significant performance improvements by reducing unnecessary computation and memory accesses. Shared memory tiling is an effective technique to further optimize memory bandwidth usage.

## Future Improvements
- Experiment with using vectorized loads for the shared memory tiles
- Investigate using warp-level primitives like WMMA to utilize Tensor Cores on supported architectures
- Explore more advanced matrix tiling strategies
