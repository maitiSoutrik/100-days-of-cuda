# Day 2: Matrix Addition in CUDA

After implementing vector addition in Day 1, today we're taking the next logical step by implementing matrix addition using CUDA. This example demonstrates how to use a 2D grid of threads to perform element-wise addition of two matrices.

## Matrix Addition

Matrix addition is a fundamental operation in linear algebra where corresponding elements of two matrices are added together to form a new matrix. For two matrices A and B of the same dimensions, the resulting matrix C is defined as:

C[i,j] = A[i,j] + B[i,j]

Where i and j are the row and column indices, respectively.

## CUDA Implementation

In our CUDA implementation:

1. We use a 2D grid of thread blocks to match the 2D structure of matrices
2. Each thread computes one element of the output matrix
3. We use row-major ordering for storing matrices in memory

### Key Components

- **Thread Indexing**: We calculate the global row and column indices using blockIdx, blockDim, and threadIdx
- **Boundary Checking**: We ensure threads only operate within the matrix dimensions
- **Memory Management**: We allocate and transfer memory between host and device

## Code Explanation

### Kernel Function

```cuda
__global__ void matrixAdd(const float *A, const float *B, float *C, int rows, int cols) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within matrix bounds
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}
```

This kernel function:

1. Calculates the global row and column indices for each thread
2. Checks if the thread is within the matrix boundaries
3. Computes the linear memory index from 2D coordinates
4. Performs the addition operation

### Kernel Launch

```cuda
// Define block and grid dimensions
dim3 blockDim(16, 16);
dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, 
             (rows + blockDim.y - 1) / blockDim.y);

// Launch the kernel
matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);
```

We use:

- 16×16 thread blocks (256 threads per block)
- A grid size calculated to cover the entire matrix, with ceiling division to handle non-multiple dimensions

## Performance Considerations

- **Thread Block Size**: We chose 16×16 as it's a common efficient size for many GPUs
- **Memory Coalescing**: Row-major order helps with coalesced memory access patterns
- **Shared Memory**: For larger matrices, using shared memory could improve performance (not implemented in this basic example)

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
./day002/matrix_add
```

Alternatively, you can also build just this day's project:

```bash
# From the day002 directory
mkdir -p build
cd build
cmake ..
make
./matrix_add
```

## Output

The program will:

1. Generate two random 4×4 matrices
2. Display the input matrices
3. Perform the addition on the GPU
4. Display the resulting matrix


## Output from Jetson Nano

```bash
drboom@JetNano ~/g/1/build> ./day002/matrix_add 
Device name: NVIDIA Tegra X1
Compute capability: 5.3
Total global memory: 3.87 GB
Matrix dimensions: 1024 x 1024 (Total elements: 1048576)

----- CPU Execution -----
CPU execution time: 3.90 ms

----- GPU Execution -----
CUDA kernel launch with grid of 64 x 64 blocks, each with 16 x 16 threads
GPU kernel execution time: 7.60 ms
GPU total time (with memory transfers): 27.39 ms

All tests PASSED

----- Performance Comparison -----
CPU execution time: 3.90 ms
GPU kernel execution time: 7.60 ms
GPU total time (with memory transfers): 27.39 ms
Speedup (kernel only): 0.51x
Speedup (with transfers): 0.14x

Matrix addition completed successfully!
```

## Performance Analysis

Interestingly, the performance results for matrix addition show that the CPU implementation outperforms the GPU implementation in this case. This is contrary to what we might expect, but there are several factors that explain this result:

1. **Problem Size**: The 1024×1024 matrix size (approximately 1 million elements) is relatively small for GPU computation. GPUs excel at massive parallelism with much larger datasets.

2. **Memory Transfer Overhead**: The time required to transfer data between the host and device (23.79 ms) dominates the total execution time, accounting for about 87% of the total GPU time.

3. **Operation Complexity**: Matrix addition is a memory-bound operation with a very simple computation (just addition). The ratio of computation to memory access is low, which doesn't play to the GPU's strengths.

4. **Tegra X1 Architecture**: The Jetson Nano's Tegra X1 GPU is designed for embedded applications and has fewer CUDA cores compared to desktop GPUs, limiting its parallel processing capability.

These results highlight an important lesson in GPU programming: not all problems benefit from GPU acceleration, especially when the computation is simple and the data transfer overhead is significant.

## Next Steps

Based on these results, some interesting next steps would be:

- Experimenting with larger matrix sizes to find the crossover point where GPU becomes more efficient
- Implementing matrix multiplication, which has higher computational intensity and should benefit more from GPU parallelism
- Exploring techniques to reduce memory transfer overhead, such as using pinned memory or CUDA streams for asynchronous operations
- Using shared memory to optimize the matrix operations
- Implementing other matrix operations (transpose, etc.)
