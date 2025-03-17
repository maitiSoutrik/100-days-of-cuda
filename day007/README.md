# Day 7: 1D & 2D Convolution in CUDA

This example demonstrates how to implement 1D and 2D convolution operations using CUDA. Convolution is a fundamental operation in signal processing, image processing, and deep learning, particularly in convolutional neural networks (CNNs).

## Convolution: First Principles

### What is Convolution?

Convolution is a mathematical operation that combines two functions to produce a third function. In the context of signal processing and image processing, convolution applies a filter (also called a kernel) to an input signal or image to produce an output.

### 1D Convolution

In 1D convolution, we slide a kernel (a small array of weights) over a 1D input signal. At each position, we compute the sum of element-wise multiplications between the kernel and the overlapping portion of the input.

Mathematically, for a 1D input signal `x` of length `n` and a kernel `k` of length `m`, the 1D convolution `y` is defined as:

```math
y[i] = Σ(j=0 to m-1) k[j] * x[i+j-m/2]
```

Where appropriate padding is applied to handle boundary conditions.

### 2D Convolution

2D convolution extends this concept to two dimensions. It's commonly used in image processing where the input is a 2D image and the kernel is a 2D filter. We slide the 2D kernel over the 2D input, computing the sum of element-wise multiplications at each position.

Mathematically, for a 2D input `x` of size `n×n` and a kernel `k` of size `m×m`, the 2D convolution `y` is defined as:

```math
y[i,j] = Σ(p=0 to m-1) Σ(q=0 to m-1) k[p,q] * x[i+p-m/2, j+q-m/2]
```

Again, appropriate padding is applied to handle boundary conditions.

## Implementation Details

### 1D Convolution CUDA Kernel

The CUDA kernel for 1D convolution is implemented as follows:

```cuda
__global__ void convolution1D(const float* input, float* output, const float* kernel, 
                            int inputSize, int kernelSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes one output element
    if (idx < inputSize) {
        float result = 0.0f;
        int radius = kernelSize / 2;
        
        // Apply the kernel
        for (int k = 0; k < kernelSize; k++) {
            int pos = idx + (k - radius);
            
            // Handle boundary conditions (zero padding)
            if (pos >= 0 && pos < inputSize) {
                result += input[pos] * kernel[k];
            }
        }
        
        output[idx] = result;
    }
}
```

### 2D Convolution CUDA Kernel

The CUDA kernel for 2D convolution is implemented as follows:

```cuda
__global__ void convolution2D(const float* input, float* output, const float* kernel,
                            int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Each thread computes one output element
    if (x < width && y < height) {
        float result = 0.0f;
        int radius = kernelSize / 2;
        
        // Apply the 2D kernel
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                int imgX = x + (kx - radius);
                int imgY = y + (ky - radius);
                
                // Handle boundary conditions (zero padding)
                if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
                    result += input[imgY * width + imgX] * 
                              kernel[ky * kernelSize + kx];
                }
            }
        }
        
        output[y * width + x] = result;
    }
}
```

### CPU Implementation

For comparison, CPU implementations of both 1D and 2D convolution are provided to verify the results and measure performance improvements.

## Performance Considerations

### Memory Access Patterns

Convolution operations involve multiple memory accesses for each output element. In the naive implementation:

1. For 1D convolution, each thread reads `kernelSize` elements from the input array.
2. For 2D convolution, each thread reads `kernelSize × kernelSize` elements from the input array.

This can lead to inefficient memory access patterns, especially for 2D convolution where memory accesses may not be coalesced.

### Optimization Strategies

Several optimization strategies can be applied to improve the performance of convolution operations in CUDA:

1. **Shared Memory**: Load blocks of the input into shared memory to reduce global memory accesses.
2. **Memory Coalescing**: Organize memory access patterns to ensure coalesced access to global memory.
3. **Loop Unrolling**: Unroll loops to reduce loop overhead and increase instruction-level parallelism.
4. **Constant Memory**: Store the kernel in constant memory for faster access.

## Building and Running

To build the example:

```bash
# Navigate to the build directory
cd build

# Build the project
cmake ..
make

# Run the convolution example
./day007/convolution
```

## Expected Output

The program will output:

1. The input signal/image and kernel (for verification)
2. The time taken by the CPU implementation
3. The time taken by the GPU implementation (including memory transfers)
4. The speedup achieved by the GPU implementation
5. Verification result (success or failure)
6. The output signal/image (for verification)

## Applications of Convolution

Convolution operations have numerous applications:

1. **Image Processing**: Blurring, sharpening, edge detection, and other image filters.
2. **Signal Processing**: Filtering, smoothing, and feature extraction from signals.
3. **Deep Learning**: Convolutional layers in neural networks for feature extraction.
4. **Computer Vision**: Feature detection, image recognition, and object detection.

## Conclusion

This implementation demonstrates the basic principles of 1D and 2D convolution operations in CUDA. While the naive implementation provides a good starting point, real-world applications would benefit from the optimization strategies mentioned above to achieve better performance.
