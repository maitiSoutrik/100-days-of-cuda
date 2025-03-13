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

## Expected Output

The program will:

1. Generate two random matrices of compatible dimensions
2. Perform matrix multiplication on the CPU as a reference
3. Perform matrix multiplication on the GPU using both the basic and optimized kernels
4. Compare the results for correctness
5. Display performance metrics

## Performance Analysis

Matrix multiplication is a compute-bound operation with O(n³) complexity, making it an ideal candidate for GPU acceleration.

Expected performance characteristics:

1. **GPU vs. CPU**: The GPU implementation should outperform the CPU for large matrices due to massive parallelism
2. **Memory Transfer Overhead**: For small matrices, memory transfer overhead may dominate, but for large matrices, computation time will dominate

## Next Steps

Possible extensions and optimizations for future exploration:

- Using shared memory to optimize memory access patterns
- Using CUDA libraries like cuBLAS for better performance
- Implementing matrix multiplication for non-square matrices
- Exploring double-precision vs. single-precision performance
