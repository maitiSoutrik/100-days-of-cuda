# Day 6: Matrix Transpose in CUDA

This example demonstrates how to implement a basic matrix transpose operation using CUDA and compares its performance with a CPU implementation. Matrix transposition is a fundamental operation in linear algebra where the rows and columns of a matrix are swapped.

## Implementation Details

### GPU Implementation

The CUDA kernel for matrix transposition is implemented as follows:

```cuda
__global__ void transposeMatrix(const float* input, float* output, int width, int height) {
    // Calculate the row and column index of the element
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Perform the transposition if within bounds
    if (x < width && y < height) {
        int inputIndex = y * width + x;
        int outputIndex = x * height + y;
        output[outputIndex] = input[inputIndex];
    }
}
```

This kernel maps each thread to one element of the input matrix. Each thread reads an element from the input matrix and writes it to the appropriate location in the output matrix.

### CPU Implementation

For comparison, a CPU implementation is also provided:

```cpp
void transposeMatrixCPU(const float* input, float* output, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int inputIndex = y * width + x;
            int outputIndex = x * height + y;
            output[outputIndex] = input[inputIndex];
        }
    }
}
```

### Memory Access Pattern

The simple transpose implementation shown here has a straightforward memory access pattern:

1. Reading from the input matrix: `input[y * width + x]` (coalesced access in GPU)
2. Writing to the output matrix: `output[x * height + y]` (strided access in GPU)

This basic implementation doesn't optimize for memory coalescing in the output, which can lead to performance issues due to strided memory access patterns.

## Performance Benchmarking

The implementation includes timing measurements for both CPU and GPU versions:

- For the CPU implementation, we use C++'s `std::chrono` high-resolution clock
- For the GPU implementation, we use CUDA events to measure time including memory transfers

The program calculates and reports the speedup achieved by the GPU implementation compared to the CPU implementation.

## Building and Running

To build the example:

```bash
# Navigate to the build directory
cd build

# Build the project
cmake ..
make

# Run the matrix transpose example
./day006/matrix_transpose
```

## Expected Output

The program will output:

1. A small portion of the input matrix (for verification)
2. The time taken by the CPU implementation
3. The time taken by the GPU implementation (including memory transfers)
4. The speedup achieved by the GPU implementation
5. Verification result (success or failure)
6. A small portion of the output matrix (for verification)

## Performance Considerations

This implementation is a basic version of matrix transpose. For larger matrices or performance-critical applications, several optimizations can be applied:

1. **Shared Memory**: Using shared memory to improve memory access patterns
2. **Memory Coalescing**: Reorganizing the algorithm to ensure coalesced memory access
3. **Bank Conflicts**: Avoiding shared memory bank conflicts
4. **Padding**: Adding padding to avoid partition camping

These optimizations would be explored in more advanced CUDA examples.
