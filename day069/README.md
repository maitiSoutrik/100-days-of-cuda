# Day 69: Parallel Password Cracking (FNV-1a)

## Overview

This project implements a parallel brute-force password cracker using CUDA. Given a target FNV-1a hash value, the original password (composed of lowercase English letters), and a number of hashing rounds (R), the program searches for the password. The FNV-1a hash is applied R times to a candidate password; if the result matches the target hash, the password is considered found.

The core task is to efficiently search the space of all possible passwords of a given length, leveraging GPU parallelism.

## Implementation Details

### Password Generation
Candidate passwords are generated directly on the GPU by each thread. Each thread calculates a unique global index, which is then converted into a base-26 representation, where each digit corresponds to a character ('a' through 'z'). For example, index 0 for length 3 is "aaa", index 1 is "aab", and so on.

### Hashing
The FNV-1a hash function is implemented as a `__device__` function:
```cuda
__device__ unsigned int fnv1a_hash_bytes(const unsigned char* data, int length);
```
This function takes the password candidate (or the byte representation of a previous hash output) and its length.

For `R` rounds of hashing:
1.  The first round hashes the `password_length` bytes of the candidate password string.
2.  For subsequent `R-1` rounds, the 4-byte unsigned integer output of the previous hash is treated as the input. These 4 bytes are hashed to produce the next hash in the chain. This process repeats until `R` hashes have been computed.

### Parallel Search
The CUDA kernel `password_cracker_kernel` assigns each thread a range of password candidates to check:
- Each thread generates its assigned password candidate.
- It performs `R` rounds of FNV-1a hashing on the candidate.
- If the resulting hash matches the `target_hash`, the thread attempts to write the found password to the `output_password_device` (a device memory buffer) and set a global `d_found_flag` using `atomicCAS`.
- The `atomicCAS` ensures that only the first thread to find the password writes the result and sets the flag.
- Other threads check this flag periodically (at the start of their execution) and exit early if the password has already been found, saving computation.

### Host Orchestration (`solve` function)
The `solve` function on the host:
1.  Calculates the total number of possible passwords for the given `password_length`.
2.  Allocates device memory for `output_password_device` (passed in) and `d_found_flag`.
3.  Initializes `d_found_flag` to 0 (false).
4.  Determines kernel launch parameters (number of blocks and threads per block).
5.  Launches the `password_cracker_kernel`.
6.  Synchronizes the device and cleans up allocated memory for `d_found_flag`.

## Key CUDA Features Used

-   **CUDA Kernels**: `password_cracker_kernel` for parallel execution.
-   **Thread Indexing**: `blockIdx.x`, `blockDim.x`, `threadIdx.x` for unique candidate generation per thread.
-   **Device Functions**: `fnv1a_hash_bytes` for in-kernel hashing.
-   **Atomic Operations**: `atomicCAS` (Compare-And-Swap) is used to ensure only one thread writes the found password and signals completion. This helps in efficiently stopping other threads once a solution is found.
-   **Global Memory**: For `output_password_device` and `d_found_flag`.
-   **Error Handling**: `CHECK_CUDA_ERROR` macro for robust error checking.

## Performance Considerations

-   **Search Space**: The search space grows exponentially with `password_length` (26^length). Length 6 is already ~308 million candidates.
-   **Hashing Rounds (R)**: More rounds increase the computation per candidate linearly.
-   **GPU Occupancy**: Kernel launch parameters (threads per block, number of blocks) should be tuned for the target GPU (Jetson Nano, sm_53). A block size of 256 threads is a common starting point.
-   **Early Exit**: The `d_found_flag` allows threads to terminate early if another thread has already found the password, significantly reducing unnecessary work, especially if the password is found early in the search space.
-   **Memory Access**: Password candidates are generated on-the-fly by each thread, minimizing global memory reads for candidate data. The main global memory accesses are for the `d_found_flag` and writing the `output_password_device`.

## Building and Running

### Prerequisites
- CUDA Toolkit
- CMake (version 3.18 or higher)
- A C++ compiler compatible with CUDA (e.g., g++)
- Google Test (will be fetched by CMake via FetchContent)

### Build Steps (from the root `100-days-of-cuda` directory)
1.  Ensure the `day069` directory is added to the root `CMakeLists.txt`:
    ```cmake
    add_subdirectory(day069)
    ```
2.  Create a build directory and navigate into it:
    ```bash
    mkdir build
    cd build
    ```
3.  Run CMake and build:
    ```bash
    cmake ..
    make password_cracker_main password_cracker_test 
    # Or 'make' to build everything
    ```
    This will compile the main executable and the test executable.

### Running the Main Program
The main executable `password_cracker_main` runs a series of predefined test cases.
```bash
./day069/password_cracker_main 
```

### Running Tests
The test executable `password_cracker_test` uses Google Test.
```bash
./day069/password_cracker_test
```

## Execution Results

Actual output from running on a Jetson Nano:

### `password_cracker_main` Output:
```
----------------------------------------
Test Case:
  Expected Password: "abc"
  Password Length  : 3
  Hashing Rounds (R): 2
  Target Hash      : 537089824 (0x20035720)
  Found Password   : "abc"
  Execution Time   : 0.390113 ms
  Result: PASSED
----------------------------------------

----------------------------------------
Test Case:
  Expected Password: "abc"
  Password Length  : 3
  Hashing Rounds (R): 1
  Target Hash      : 440920331 (0x1a47e90b)
  Found Password   : "abc"
  Execution Time   : 0.429384 ms
  Result: PASSED
----------------------------------------

----------------------------------------
Test Case:
  Expected Password: "a"
  Password Length  : 1
  Hashing Rounds (R): 1
  Target Hash      : 3826002220 (0xe40c292c)
  Found Password   : "a"
  Execution Time   : 0.340737 ms
  Result: PASSED
----------------------------------------

----------------------------------------
Test Case:
  Expected Password: "z"
  Password Length  : 1
  Hashing Rounds (R): 1
  Target Hash      : 4278997933 (0xff0c53ad)
  Found Password   : "z"
  Execution Time   : 0.24662 ms
  Result: PASSED
----------------------------------------

----------------------------------------
Test Case:
  Expected Password: "aa"
  Password Length  : 2
  Hashing Rounds (R): 1
  Target Hash      : 1277494327 (0x4c250437)
  Found Password   : "aa"
  Execution Time   : 0.229068 ms
  Result: PASSED
----------------------------------------

----------------------------------------
Test Case:
  Expected Password: "az"
  Password Length  : 2
  Hashing Rounds (R): 2
  Target Hash      : 165781343 (0x9e19f5f)
  Found Password   : "az"
  Execution Time   : 0.300996 ms
  Result: PASSED
----------------------------------------

----------------------------------------
Test Case:
  Expected Password: "zy"
  Password Length  : 2
  Hashing Rounds (R): 3
  Target Hash      : 2426205364 (0x909cf4b4)
  Found Password   : "zy"
  Execution Time   : 0.249745 ms
  Result: PASSED
----------------------------------------

----------------------------------------
Test Case:
  Expected Password: "cuda"
  Password Length  : 4
  Hashing Rounds (R): 1
  Target Hash      : 2340616438 (0x8b82f8f6)
  Found Password   : "cuda"
  Execution Time   : 3.63534 ms
  Result: PASSED
----------------------------------------

----------------------------------------
Test Case:
  Expected Password: "test"
  Password Length  : 4
  Hashing Rounds (R): 5
  Target Hash      : 209436132 (0xc7bbde4)
  Found Password   : "test"
  Execution Time   : 18.6837 ms
  Result: PASSED
----------------------------------------

----------------------------------------
Test Case:
  Expected Password: "hello"
  Password Length  : 5
  Hashing Rounds (R): 1
  Target Hash      : 1335831723 (0x4f9f2cab)
  Found Password   : "hello"
  Execution Time   : 69.7002 ms
  Result: PASSED
----------------------------------------

----------------------------------------
Test Case:
  Expected Password: "world"
  Password Length  : 5
  Hashing Rounds (R): 10
  Target Hash      : 922584772 (0x36fd86c4)
  Found Password   : "world"
  Execution Time   : 120.788 ms
  Result: PASSED
----------------------------------------

All tests completed.
```

### `password_cracker_test` (Google Test) Output:
```
[==========] Running 15 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 15 tests from CrackerTests/PasswordCrackerTest
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/0
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/0 (97 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/1
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/1 (1 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/2
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/2 (1 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/3
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/3 (1 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/4
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/4 (1 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/5
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/5 (1 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/6
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/6 (1 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/7
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/7 (1 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/8
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/8 (1 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/9
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/9 (1 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/10
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/10 (1 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/11
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/11 (4 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/12
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/12 (19 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/13
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/13 (1 ms)
[ RUN      ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/14
[       OK ] CrackerTests/PasswordCrackerTest.FindsCorrectPassword/14 (20 ms)
[----------] 15 tests from CrackerTests/PasswordCrackerTest (156 ms total)

[----------] Global test environment tear-down
[==========] 15 tests from 1 test suite ran. (156 ms total)
[  PASSED  ] 15 tests.
```

## Learnings and Observations

-   The FNV-1a hash function is simple and fast, suitable for brute-force attacks where many hashes need to be computed.
-   Mapping a global index to a unique password candidate (base-26 conversion) is a common technique in GPU-based brute-forcing.
-   The `atomicCAS` operation is crucial for efficiently managing the "found" state across many parallel threads, preventing redundant work and ensuring only one thread writes the final result.
-   The performance is heavily dependent on `password_length` and `R`. Longer passwords or more hash rounds significantly increase the workload.
-   For very large search spaces (e.g., password_length > 5 or 6), the execution time can become substantial even on a GPU.
-   The process of hashing the output of a previous hash (iterative hashing) requires careful handling of the data type conversion (integer hash to byte array for the next hash input).

## Future Improvements
-   **Dynamic Block/Grid Sizing**: Adjust kernel launch parameters based on GPU properties and input size for better performance.
-   **Shared Memory**: For very small `R` values, the benefit might be minimal. If `R` was extremely large and intermediate hash results could be shared or reused within a block (unlikely for this specific problem structure), shared memory could be explored.
-   **Character Set Flexibility**: Allow different character sets beyond lowercase English letters.
-   **Benchmark Mode**: Add a dedicated mode to test performance for specific password lengths and R values without verifying correctness against known passwords.
