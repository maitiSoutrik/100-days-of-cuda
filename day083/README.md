# Day 083: Minimum Reduction Over a Specific Dimension

## Overview
This project implements a CUDA kernel to perform a minimum reduction operation on a multi-dimensional array (tensor) along a user-specified dimension. For an N-dimensional input tensor, the output will be an (N-1)-dimensional tensor where each element is the minimum of the values along the reduced dimension of the input.

## Implementation Details
The core logic resides in `min_reduce.cu` and is exposed via `min_reduce.cuh`.

**Kernel: `minReduceKernel`**
- Each thread in the CUDA grid is responsible for calculating one element in the output (reduced) tensor.
- The kernel takes the input tensor, output tensor, shape of the input, number of dimensions, the dimension to reduce, and pre-calculated `before_size`, `dim_size`, and `after_size` values.
    - `before_size`: The product of the sizes of dimensions *before* the one being reduced.
    - `dim_size`: The size of the dimension being reduced.
    - `after_size`: The product of the sizes of dimensions *after* the one being reduced.
- Each thread calculates its corresponding `before_idx` (index in the "meta-dimension" before the reduced one) and `after_idx` (index in the "meta-dimension" after the reduced one).
- It then iterates `dim_size` times, accessing the appropriate elements in the input tensor using a calculated `input_idx`, and finds the minimum value.
- This minimum value is written to the corresponding position in the `output` tensor.

**Host Function: `min_reduction_dimension_cuda`**
- This C-style extern function serves as the entry point.
- It performs input validation (checks for null pointers, valid dimension to reduce, non-zero dimensions).
- It calculates `before_size`, `dim_size`, and `after_size` based on the input `shape` and `dim` to reduce.
- It determines the `output_size` and configures the CUDA kernel launch parameters (`blockSize`, `numBlocks`).
- It launches `minReduceKernel` and includes CUDA error checking and device synchronization.

## Key CUDA Concepts
- **Kernel Launch Configuration**: Dynamically calculating `numBlocks` based on output size.
- **Thread Indexing**: Mapping 1D `blockIdx.x * blockDim.x + threadIdx.x` to multi-dimensional access patterns.
- **Global Memory Access**: Reading from the input tensor and writing to the output tensor.
- **Parallel Reduction (Simplified)**: Each output element is computed independently by a thread, which performs a local sequential reduction along the specified dimension. This is not a full parallel reduction in the typical sense (like reducing an entire array to a single value using shared memory optimizations within blocks), but rather a parallel application of many independent 1D reductions.

## Performance Considerations
- **Memory Access Patterns**: The kernel reads elements strided by `after_size` within the inner loop. If `after_size` is large, this could lead to non-coalesced memory access for threads within a warp if they are processing adjacent `before_idx` values. However, if threads in a warp process adjacent `after_idx` values for the same `before_idx` and `d`, access would be coalesced. The current mapping of `idx` to `before_idx` and `after_idx` means threads with consecutive `idx` values will likely have different `before_idx` or `after_idx`, making coalescing dependent on the relationship between `blockDim.x` and `after_size`.
- **Workload per Thread**: Each thread iterates `dim_size` times. If `dim_size` is very large, this increases the work per thread. If very small, kernel launch overhead might dominate.
- **No Shared Memory Usage**: For this specific approach where each thread computes one output element independently by iterating along the reduction dimension, shared memory is not directly used for inter-thread communication within the reduction itself. A more advanced reduction for very large `dim_size` might involve multiple threads within a block collaborating to reduce a single slice, which would then use shared memory.

## Building and Running
The code is built using CMake. From the root project directory:
1.  Configure CMake: `cmake -B build`
2.  Build the project: `cmake --build build`
    (This will build all days. To build only day083: `cmake --build build --target min_reduce_main --target min_reduce_test`)

To run the main executable (demonstration and benchmark):
```bash
./build/day083/min_reduce_main
```

To run the tests:
```bash
./build/day083/min_reduce_test
```
Alternatively, run all tests using CTest from the `build` directory:
```bash
cd build
ctest --output-on-failure -R day083_min_reduction_dim # Run tests for day083
```

## Execution Results
The `min_reduce_main` executable runs several test cases with different tensor shapes and reduction dimensions. It compares the GPU result with a CPU implementation and reports PASS/FAIL status along with timings.

Example output from `min_reduce_main`:
```
-----------------------------------------------------
Testing with shape: 2x3x4, reducing dimension 0
Verification PASSED!
GPU Time: 31.1368 ms
CPU Time: 0.000677 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 2x3x4, reducing dimension 1
Verification PASSED!
GPU Time: 30.8228 ms
CPU Time: 0.000677 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 2x3x4, reducing dimension 2
Verification PASSED!
GPU Time: 30.874 ms
CPU Time: 0.000573 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 512x512, reducing dimension 0
Verification PASSED!
GPU Time: 58.7724 ms
CPU Time: 2.02515 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 512x512, reducing dimension 1
Verification PASSED!
GPU Time: 56.9795 ms
CPU Time: 0.592825 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 100, reducing dimension 0
Verification PASSED!
GPU Time: 30.0472 ms
CPU Time: 0.00099 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 10x1x10, reducing dimension 1
Verification PASSED!
GPU Time: 30.8152 ms
CPU Time: 0.001093 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 10x0x10, reducing dimension 0
Input tensor is empty. Skipping test.
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 10x5x10, reducing dimension 1
Verification PASSED!
GPU Time: 30.5832 ms
CPU Time: 0.006562 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 2x2x2x2, reducing dimension 0
Verification PASSED!
GPU Time: 30.8152 ms
CPU Time: 0.000469 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 2x2x2x2, reducing dimension 1
Verification PASSED!
GPU Time: 30.8492 ms
CPU Time: 0.000469 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 2x2x2x2, reducing dimension 2
Verification PASSED!
GPU Time: 30.8489 ms
CPU Time: 0.000625 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 2x2x2x2, reducing dimension 3
Verification PASSED!
GPU Time: 30.874 ms
CPU Time: 0.000521 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 5x1x6, reducing dimension 1
Verification PASSED!
GPU Time: 30.6675 ms
CPU Time: 0.000885 ms
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 5x0x6, reducing dimension 0
Input tensor is empty. Skipping test.
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 5x0x6, reducing dimension 2
Input tensor is empty. Skipping test.
-----------------------------------------------------

-----------------------------------------------------
Testing with shape: 5x0x6, reducing dimension 1
Input tensor is empty. Skipping test.
-----------------------------------------------------

All tests completed.
```

Example output from `min_reduce_test` (Google Test):
```
[==========] Running 20 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 20 tests from MinReductionTests/MinReductionTest
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/0
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/0 (8505 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/1
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/1 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/2
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/2 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/3
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/3 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/4
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/4 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/5
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/5 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/6
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/6 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/7
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/7 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/8
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/8 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/9
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/9 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/10
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/10 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/11
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/11 (60 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/12
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/12 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/13
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/13 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/14
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/14 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/15
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/15 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/16
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/16 (63 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/17
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/17 (31 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/18
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/18 (31 ms)
[ RUN      ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/19
[       OK ] MinReductionTests/MinReductionTest.HandlesVariousShapesAndDims/19 (31 ms)
[----------] 20 tests from MinReductionTests/MinReductionTest (9619 ms total)

[----------] Global test environment tear-down
[==========] 20 tests from 1 test suite ran. (9620 ms total)
[  PASSED  ] 20 tests.
```

## Learnings and Observations
- The provided kernel structure is a straightforward way to achieve reduction along a specific dimension when each output element can be computed independently.
- Calculating `before_size`, `dim_size`, and `after_size` correctly is crucial for mapping threads to the correct input and output elements.
- Handling edge cases like zero-sized dimensions (either the one being reduced or other dimensions leading to an empty output) is important for robustness. The current implementation includes checks for these in the host wrapper and the kernel.
- For very large reduction dimensions (`dim_size`), more optimized intra-block reduction strategies (using shared memory and warp-level primitives) could be beneficial, but that would change the kernel structure significantly (e.g., a block might compute one output element, or a portion of it). The current approach is simpler and effective when `dim_size` is moderate.

## Future Improvements
- **Optimized Kernel for Large `dim_size`**: Implement a version where multiple threads within a block collaborate to reduce a single slice along `dim_size`, using shared memory and parallel reduction patterns.
- **Template for Data Type**: Generalize the kernel to work with different data types (e.g., `int`, `double`) using templates.
- **Support for other Reduction Operations**: Extend to support `sum`, `max`, `product`, etc., possibly via a template parameter or function pointer.
