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

## Performance Results

When run on a Jetson Nano, the following results were observed:

```text
Matrix size: 1024x1024
Input Matrix (top-left corner):
     0      1      2      3      4      5      6      7 
    24     25     26     27     28     29     30     31 
    48     49     50     51     52     53     54     55 
    72     73     74     75     76     77     78     79 
    96     97     98     99      0      1      2      3 
    20     21     22     23     24     25     26     27 
    44     45     46     47     48     49     50     51 
    68     69     70     71     72     73     74     75 

CPU Transpose Time: 30.4393 ms
GPU Transpose Time (including memory transfers): 45.5097 ms
Speedup: 0.668853x
Matrix transposition succeeded!
Output Matrix (GPU, top-left corner):
     0     24     48     72     96     20     44     68 
     1     25     49     73     97     21     45     69 
     2     26     50     74     98     22     46     70 
     3     27     51     75     99     23     47     71 
     4     28     52     76      0     24     48     72 
     5     29     53     77      1     25     49     73 
     6     30     54     78      2     26     50     74 
     7     31     55     79      3     27     51     75 
```

Interestingly, for this basic implementation on the Jetson Nano, the CPU outperforms the GPU when including memory transfer times. This is because:

1. The memory transfer overhead is significant for this operation
2. The basic implementation doesn't optimize for memory access patterns
3. The Jetson Nano's GPU architecture has different performance characteristics compared to desktop GPUs

This highlights the importance of optimizing CUDA code specifically for the target hardware and considering memory transfer costs in real-world applications.

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
