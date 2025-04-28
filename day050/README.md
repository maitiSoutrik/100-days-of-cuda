# Day 50: Unit Testing CUDA Kernels with Google Test

## Overview

This day explores how to integrate CUDA kernel testing into a standard C++ unit testing framework. We use Google Test (gtest) to write and run tests for the simple Vector Addition kernel developed in Day 1. The goal is to demonstrate best practices for ensuring CUDA code correctness through automated testing. Tests are executed on the host, invoking CUDA kernels and verifying their results against CPU computations.

## Implementation Details

1.  **Kernel Isolation:** The `vectorAddKernel` and the `checkCudaError` utility function were extracted from `day001/vector_add.cu` into a separate file (`vector_add_kernel.cu`) to be compiled as a CUDA source.
2.  **Test Fixture (`VectorAddTest`):** A Google Test fixture (`VectorAddTest` in `vector_add_test.cpp`) was created to manage the setup and teardown for each test case:
    *   `SetUp()`: Allocates host vectors (`std::vector`) and device memory (`cudaMalloc`) needed for the tests. Initializes host input vectors (A and B) with random data.
    *   `TearDown()`: Frees device memory (`cudaFree`). Host memory is managed automatically by `std::vector`.
3.  **Test Cases (`TEST_F`):**
    *   `BasicVerification`: The core test case. It copies host input data (A, B) to the device, launches the `vectorAddKernel`, copies the result (C) back to the host, computes the expected result on the CPU (`vectorAddCPU`), and compares the GPU and CPU results using `ASSERT_NEAR` for floating-point values.
    *   `ZeroVectors`: Tests the kernel with input vectors containing all zeros.
    *   `NegativeNumbers`: Tests the kernel with input vectors containing negative numbers.
4.  **CPU Verification:** A simple host-side `vectorAddCPU` function is used within the test file to compute the expected results for comparison.
5.  **Build Integration (CMake):**
    *   The `day050/CMakeLists.txt` uses `FetchContent` to download and configure Google Test automatically during the CMake configuration step.
    *   `cuda_add_executable` is used to compile both the `.cu` kernel file and the `.cpp` test file into a single test executable (`vector_add_test`).
    *   The executable is linked against the CUDA runtime (`${CUDA_LIBRARIES}`), Google Test (`gtest`), and the Google Test main library (`gtest_main`, which provides the `main` function).
    *   `enable_testing()` and `gtest_discover_tests()` are used to integrate the executable with CTest, allowing tests to be run via `ctest`.

## Key CUDA Features Used

*   Basic CUDA Runtime API: `cudaMalloc`, `cudaMemcpy` (HostToDevice, DeviceToHost), `cudaFree`, `cudaGetLastError`, `cudaDeviceSynchronize`.
*   Kernel Launch Syntax: `kernel<<<...>>>()`.
*   Error Handling: `checkCudaError` utility function (integrated with test fixture setup/teardown).

## Performance Considerations

The primary focus here is correctness, not performance benchmarking. However, the test fixture sets up reasonably sized vectors (1M elements) to ensure the kernel operates on non-trivial data. Test execution time will include data transfers and kernel execution. In a production scenario, separate performance tests might be designed.

## Building and Running (Target Environment - e.g., Jetson Nano or CI Runner)

1.  **Navigate to Build Directory:**
    ```bash
    cd /path/to/100-days-of-cuda/build # Or create it if it doesn't exist
    ```
2.  **Configure with CMake:** (Run from the build directory)
    ```bash
    cmake .. 
    ```
    *(CMake should detect Day 50, download Google Test, and configure the build)*
3.  **Build the Test Executable:**
    ```bash
    cmake --build . --target vector_add_test -j$(nproc) 
    ```
    *(Or use `make vector_add_test`)*
4.  **Run Tests using CTest:**
    ```bash
    # Ensure you are in the build directory
    ctest --output-on-failure -R vector_add_test 
    ```
    *(This command specifically runs tests associated with the `vector_add_test` target)*
    
    Alternatively, run the executable directly to see Google Test output:
    ```bash
    ./day050/vector_add_test
    ```

## Execution Results / Output

The following output was obtained by running the test executable directly (`./day050/vector_add_test`) in the build directory on the Jetson Nano:

```
Running main() from /home/drboom/git_repos/100-days-of-cuda/build/_deps/googletest-src/googletest/src/gtest_main.cc
[==========] Running 3 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 3 tests from VectorAddTest
[ RUN      ] VectorAddTest.BasicVerification
[       OK ] VectorAddTest.BasicVerification (524 ms)
[ RUN      ] VectorAddTest.ZeroVectors
[       OK ] VectorAddTest.ZeroVectors (93 ms)
[ RUN      ] VectorAddTest.NegativeNumbers
[       OK ] VectorAddTest.NegativeNumbers (118 ms)
[----------] 3 tests from VectorAddTest (736 ms total)

[----------] Global test environment tear-down
[==========] 3 tests from 1 test suite ran. (736 ms total)
[  PASSED  ] 3 tests.
```

*(Note: CTest output would show a summary line indicating the test passed.)*

## Learnings and Observations

*   Integrating CUDA code with Google Test is straightforward using CMake and `FetchContent`.
*   Test fixtures are crucial for managing CUDA resource allocation (device memory) and ensuring cleanup (`cudaFree`).
*   Forward declarations or header files are needed to make CUDA kernels and helper functions callable from C++ test files. Using `extern "C"` is important if the CUDA code is compiled as C.
*   Testing various input conditions (random data, zeros, negative numbers) increases confidence in the kernel's robustness.
*   CTest provides a convenient way to run tests as part of the build process.

## Future Improvements

*   Create a common header file (`.cuh`) for the kernel signature and helper functions instead of using `extern "C"`.
*   Abstract the CUDA error checking into the test framework's assertion mechanism (e.g., custom `ASSERT_CUDA_SUCCESS`).
*   Test edge cases like empty vectors or vectors of size 1.
*   Parameterize tests to run with different vector sizes or data types.
