# Day 086: Hard Sigmoid Activation Function

## Overview

This project implements the Hard Sigmoid activation function using CUDA C++. The Hard Sigmoid function is a computationally cheaper approximation of the standard sigmoid function, often used in neural networks.

## Implementation Details

The Hard Sigmoid function is defined as:

- `output = 0` if `input <= -3`
- `output = 1` if `input >= 3`
- `output = (input + 3) / 6` if `-3 < input < 3`

The CUDA implementation consists of:

- `hard_sigmoid_kernel.cu`: Contains the `__global__` kernel that applies the Hard Sigmoid function element-wise to an input array.
- `hard_sigmoid.cuh`: Header file declaring the main solution function.
- `hard_sigmoid.cu`: Contains the `hard_sigmoid_solution` wrapper function which handles CUDA memory allocation, data transfers between host and device, kernel launch, and deallocation.
- `common_utils.h`: Provides common CUDA error-checking macros.
- `hard_sigmoid_main.cu`: A standalone executable to demonstrate the Hard Sigmoid function with sample data and verify its output against a CPU computation.
- `hard_sigmoid_test.cu`: Google Test suite for unit testing the Hard Sigmoid implementation with various input scenarios.

## Key CUDA Concepts Used

- **CUDA Kernels (`__global__`)**: The `hard_sigmoid_kernel` is a CUDA kernel that runs in parallel on the GPU, processing each element of the input array.
- **Thread Indexing**: `blockIdx.x * blockDim.x + threadIdx.x` is used to calculate a unique global index for each thread.
- **Memory Management**: `cudaMalloc`, `cudaMemcpy` (HostToDevice and DeviceToHost), and `cudaFree` are used for managing memory on the GPU device.
- **Error Handling**: `CHECK_CUDA_ERROR` and `CHECK_LAST_CUDA_ERROR` macros (from `common_utils.h`) are used for robust error checking of CUDA API calls and kernel launches.
- **Device Synchronization**: `cudaDeviceSynchronize` is used to ensure kernel completion before results are copied back or further operations are performed.

## Performance Considerations (Jetson Nano - sm_53)

- **Simplicity**: The Hard Sigmoid function involves simple arithmetic operations (comparisons, addition, division), which are efficient on the GPU.
- **Memory Access**: The kernel performs element-wise operations, leading to coalesced memory access patterns if the input data is contiguous, which is generally good for performance.
- **Computational Cost**: Compared to the standard sigmoid (which involves an exponential), Hard Sigmoid is much faster due to its piecewise linear nature. This makes it attractive for resource-constrained devices like the Jetson Nano.
- **Overhead**: For very small arrays, the overhead of CUDA memory transfers and kernel launch might outweigh the benefits of GPU parallelism. However, for larger arrays typical in neural network layers, the GPU acceleration becomes significant.

## Building and Running

### Prerequisites

- NVIDIA CUDA Toolkit (compatible with sm_53 for Jetson Nano)
- CMake (version 3.18 or higher)
- A C++ compiler (like g++)
- Google Test (for running unit tests)

### Building

1. Navigate to the `day086` directory.
2. Create a build directory: `mkdir build && cd build`
3. Run CMake and build:

   ```bash
   cmake ..
   make
   ```

This will build the `hard_sigmoid_lib` static library, the `hard_sigmoid_main` executable, and the `hard_sigmoid_test` executable.

### Running the Main Executable

```bash
./hard_sigmoid_main
```

This will execute the Hard Sigmoid function on sample data and print the input, GPU output, and a verification status.

### Running Unit Tests

```bash
./hard_sigmoid_test
```

This will run the Google Test suite to verify the correctness of the Hard Sigmoid implementation under various conditions.

## Execution Results / Output

```console
drboom@JetNano ~/g/1/build> ./day086/hard_sigmoid_main 
Input Matrix:
-10.000 -9.000  -8.000  -7.000  -6.000
-5.000  -4.000  -3.000  -2.000  -1.000
0.000   1.000   2.000   3.000   4.000
5.000   6.000   7.000   8.000   9.000

Output Matrix (Hard Sigmoid):
0.000   0.000   0.000   0.000   0.000
0.000   0.000   0.000   0.167   0.333
0.500   0.667   0.833   1.000   1.000
1.000   1.000   1.000   1.000   1.000

Verification successful!
drboom@JetNano ~/g/1/build> ./day086/hard_sigmoid_test
[==========] Running 5 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 5 tests from HardSigmoidTest
[ RUN      ] HardSigmoidTest.HandlesNegativeValues
[       OK ] HardSigmoidTest.HandlesNegativeValues (87 ms)
[ RUN      ] HardSigmoidTest.HandlesPositiveValues
[       OK ] HardSigmoidTest.HandlesPositiveValues (1 ms)
[ RUN      ] HardSigmoidTest.HandlesMixedValues
[       OK ] HardSigmoidTest.HandlesMixedValues (1 ms)
[ RUN      ] HardSigmoidTest.HandlesZeroElements
[       OK ] HardSigmoidTest.HandlesZeroElements (0 ms)
[ RUN      ] HardSigmoidTest.HandlesSingleElement
[       OK ] HardSigmoidTest.HandlesSingleElement (1 ms)
[----------] 5 tests from HardSigmoidTest (91 ms total)

[----------] Global test environment tear-down
[==========] 5 tests from 1 test suite ran. (91 ms total)
[  PASSED  ] 5 tests.
```

## Learnings and Observations

- The Hard Sigmoid function is straightforward to implement in CUDA.
- Proper error handling and memory management are crucial for stable CUDA applications.
- Unit testing helps ensure the correctness of the CUDA kernel across different input ranges.
- The current `hard_sigmoid_solution` encapsulates all CUDA operations, making it easy to call from C++ code that is unaware of CUDA specifics, but this might not be ideal for integration into larger CUDA pipelines where data might already reside on the device.

## (Optional) Future Improvements

- Modify `hard_sigmoid_solution` to accept device pointers directly, allowing integration into larger CUDA workflows where data is already on the GPU.
- Benchmark performance against a CPU implementation for various input sizes on the Jetson Nano.

## (Optional) References

- [Hard Sigmoid (PapersWithCode)](https://paperswithcode.com/method/hard-sigmoid)
