# Day 3: Matrix Multiplication in CUDA

After implementing matrix addition in Day 2, today we're taking on matrix multiplication using CUDA. This is a fundamental operation in linear algebra with numerous applications in scientific computing, machine learning, computer graphics, and more.

## Matrix Multiplication

Matrix multiplication is an operation that produces a single matrix from two input matrices. For two matrices A (m×n) and B (n×p), the resulting matrix C (m×p) is defined as:

C[i,j] = ∑(k=0 to n-1) A[i,k] × B[k,j]

Where i ranges from 0 to m-1, j ranges from 0 to p-1, and k ranges from 0 to n-1.

## CUDA Implementation

In our CUDA implementation:

1. We use a 2D grid of thread blocks to match the 2D structure of the output matrix
2. Each thread computes one element of the output matrix

### Key Components

- **Thread Indexing**: We calculate the global row and column indices using blockIdx, blockDim, and threadIdx
- **Boundary Checking**: We ensure threads only operate within the matrix dimensions
- **Memory Management**: We allocate and transfer memory between host and device

## Implementation Approach

Our implementation assigns each thread to compute one element of the output matrix. Each thread:

1. Calculates its global row and column indices
2. Checks if it's within the output matrix boundaries
3. Computes the dot product of the corresponding row of A and column of B
4. Stores the result in the output matrix C

## Performance Considerations

- **Thread Block Size**: We use a 16×16 thread block size as a common efficient choice for many GPUs
- **Memory Access Patterns**: Matrix multiplication involves accessing rows of A and columns of B, which can lead to different memory access patterns
- **Parallelism**: Each thread operates independently, allowing for massive parallelism on the GPU

## Compilation and Execution

Compile the program using the top-level CMake build:

```bash
# From the root directory of the repository
mkdir -p build
cd build
cmake ..
make
```

Run the executable:

```bash
# From the build directory
./day003/matrix_multiply
```

Alternatively, you can also build just this day's project:

```bash
# From the day003 directory
mkdir -p build
cd build
cmake ..
make
./matrix_multiply
```

## Output from Jetson Nano

```bash
drboom@JetNano ~/g/1/build> ./day003/matrix_multiply 
Device name: NVIDIA Tegra X1
Compute capability: 5.3
Total global memory: 3.87 GB
Matrix dimensions: A(1024x1024) * B(1024x1024) = C(1024x1024)
Total elements: A=1048576, B=1048576, C=1048576

----- CPU Execution -----
CPU execution time: 32491.79 ms

----- GPU Implementation -----
CUDA kernel launch with grid of 64 x 64 blocks, each with 16 x 16 threads
GPU kernel execution time: 255.86 ms
GPU implementation verification: PASSED

----- Performance Comparison -----
CPU execution time: 32491.79 ms
GPU kernel execution time: 255.86 ms
Speedup (GPU vs CPU): 126.99x

Matrix multiplication completed successfully!
```

The program:

1. Generated two random 1024×1024 matrices
2. Performed matrix multiplication on the CPU as a reference
3. Performed matrix multiplication on the GPU
4. Verified that the GPU results matched the CPU results
5. Displayed performance metrics showing a dramatic 127x speedup

## Performance Analysis

Matrix multiplication is a compute-bound operation with O(n³) complexity, making it an ideal candidate for GPU acceleration.

The performance results from the Jetson Nano demonstrate the power of GPU parallelism for this type of computation:

1. **CPU Execution Time**: 32491.79 ms (about 32.5 seconds)
2. **GPU Execution Time**: 255.86 ms (about 0.26 seconds)
3. **Speedup**: 126.99x

This impressive 127x speedup highlights why matrix multiplication is considered a perfect fit for GPU computing. Unlike the matrix addition we saw in Day 2 (which was memory-bound and showed modest speedup), matrix multiplication is compute-bound with a high arithmetic intensity, allowing the GPU's massive parallelism to shine.

Even with the Jetson Nano's relatively modest GPU (compared to desktop GPUs), we see dramatic performance improvements because:

1. The computation-to-memory-access ratio is much higher than in matrix addition
2. The O(n³) complexity means the CPU has to perform over a billion operations for our 1024×1024 matrices
3. The GPU can distribute these operations across thousands of threads

## Next Steps

Possible extensions and optimizations for future exploration:

- Using shared memory to optimize memory access patterns
- Using CUDA libraries like cuBLAS for better performance
- Implementing matrix multiplication for non-square matrices
- Exploring double-precision vs. single-precision performance
