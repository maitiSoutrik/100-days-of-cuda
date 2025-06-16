# Day 096 – Product Reduction over a Dimension

## Overview
This day implements a CUDA kernel that computes the **product** of elements along a specified tensor dimension (axis). The routine mirrors the API introduced for Day 083’s minimum–reduction, but performs multiplicative reduction instead of a min-operation.

Target platform: **NVIDIA Jetson Nano (SM 53)**.

## Implementation Details
* `product_reduction.cuh` – public C-style wrapper `product_reduction_dimension_cuda`.
* `product_reduction.cu` –
  * error-checked host wrapper that validates arguments, determines launch configuration, and invokes the kernel
  * `productReduceKernel` which splits the flattened tensor into `before × dim × after` regions and multiplies the `dim` elements.
* Host code follows the same memory-layout assumptions (row-major, contiguous).
* All device memory operations are checked with the project-wide `CHECK_CUDA_ERROR` macro.

## Key CUDA Concepts
1. **Thread-level parallelism:** each output element (resulting from the reduction) is computed by a single thread.
2. **Grid & block configuration:** 1-D grid where `gridSize = ceil(outElems / 256)`.
3. **Memory coalescing:** contiguous thread access within the innermost (after) dimension.
4. **Synchronization:** kernel is followed by `cudaDeviceSynchronize()` in the host wrapper to guarantee completion before use.

## Performance Considerations
* This simple version accumulates the product sequentially inside each thread. Shared-memory tree reductions could accelerate very large `dim_size`, but typical ML shapes keep `dim` modest.
* Jetson Nano’s limited SM count favours lighter kernels; the 256-thread block keeps occupancy reasonable.

## Building and Running (Jetson Nano)
```bash
# From project root on Jetson (or via CI job)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Demo executable
./build/day096/product_reduction_main

# Google tests
ctest --test-dir build --output-on-failure -R product_reduction_test
```

## Execution Results

Output from `product_reduction_main` on Jetson Nano:
```
GPU result vs CPU reference:
  45  (ref=45)
  120  (ref=120)
  231  (ref=231)
  384  (ref=384)
  4641  (ref=4641)
  5544  (ref=5544)
  6555  (ref=6555)
  7680  (ref=7680)
SUCCESS
```

Output from `product_reduction_test` on Jetson Nano:
```
[==========] Running 5 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 5 tests from ProductReductionTests/ProductReductionTest
[ RUN      ] ProductReductionTests/ProductReductionTest.MatchesCPU/0
[       OK ] ProductReductionTests/ProductReductionTest.MatchesCPU/0 (86 ms)
[ RUN      ] ProductReductionTests/ProductReductionTest.MatchesCPU/1
[       OK ] ProductReductionTests/ProductReductionTest.MatchesCPU/1 (1 ms)
[ RUN      ] ProductReductionTests/ProductReductionTest.MatchesCPU/2
[       OK ] ProductReductionTests/ProductReductionTest.MatchesCPU/2 (1 ms)
[ RUN      ] ProductReductionTests/ProductReductionTest.MatchesCPU/3
[       OK ] ProductReductionTests/ProductReductionTest.MatchesCPU/3 (1 ms)
[ RUN      ] ProductReductionTests/ProductReductionTest.MatchesCPU/4
[       OK ] ProductReductionTests/ProductReductionTest.MatchesCPU/4 (2 ms)
[----------] 5 tests from ProductReductionTests/ProductReductionTest (93 ms total)

[----------] Global test environment tear-down
[==========] 5 tests from 1 test suite ran. (93 ms total)
[  PASSED  ] 5 tests.
```

## Learnings and Observations
* Re-using the Day 083 design accelerated development.
* Validating edge cases (zero-length dimensions) on CPU ensures robustness.
* **Bug Fix:** Corrected a `SIGSEGV` error caused by attempting to dereference a device pointer (`shape`) in host code within the `product_reduction_dimension_cuda` function. The fix involved passing the `shape` array as a host pointer and removing the unnecessary device-side copy of `shape`, as the kernel itself did not require it directly. This highlights the importance of careful pointer management between host and device code.

## Future Improvements
* Replace per-thread inner loop with warp-level intrinsics for large reduction axes.
* Consider half-precision support for speed-critical paths.

## References
* CUDA Programming Guide – Reduction patterns
* Day 083 – Minimum Reduction implementation
