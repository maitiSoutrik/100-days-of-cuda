# Day 7: 1D & 2D Convolution in CUDA

This example demonstrates how to implement 1D and 2D convolution operations using CUDA. Convolution is a fundamental operation in signal processing, image processing, and deep learning, particularly in convolutional neural networks (CNNs).

## Convolution: First Principles

### What is Convolution?

Convolution is a mathematical operation that combines two functions to produce a third function. In the context of signal processing and image processing, convolution applies a filter (also called a kernel) to an input signal or image to produce an output.

### 1D Convolution

In 1D convolution, we slide a kernel (a small array of weights) over a 1D input signal. At each position, we compute the sum of element-wise multiplications between the kernel and the overlapping portion of the input.

Mathematically, for a 1D input signal `x` of length `n` and a kernel `k` of length `m`, the 1D convolution `y` is defined as:

```math
y[i] = Σ(j=0 to m-1) k[j] * x[i + j - m / 2]
```

Where appropriate padding is applied to handle boundary conditions.

### 2D Convolution

2D convolution extends this concept to two dimensions. It's commonly used in image processing where the input is a 2D image and the kernel is a 2D filter. We slide the 2D kernel over the 2D input, computing the sum of element-wise multiplications at each position.

Mathematically, for a 2D input `x` of size `n×n` and a kernel `k` of size `m×m`, the 2D convolution `y` is defined as:

```math
y[i,j] = Σ(p=0 to m-1) Σ(q=0 to m-1) k[p,q] * x[i + p - m / 2, j + q - m / 2]
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

## Performance Results

When run on a Jetson Nano, the following results were observed:

```text
========== 1D Convolution ==========
Input Signal (first 16 elements):
  0.0000   0.0998   0.1987   0.2955   0.3894   0.4794   0.5646   0.6442   0.7174   0.7833   0.8415   0.8912   0.9320   0.9636   0.9854   0.9975 

1D Kernel (first 5 elements):
  0.0545   0.2442   0.4026   0.2442   0.0545 

1D CPU Convolution Time: 0.0162 ms
1D GPU Convolution Time (including memory transfers): 0.5236 ms
1D Speedup: 0.0309x
1D Convolution succeeded!
1D Output Signal (GPU) (first 16 elements):
  0.0352   0.1048   0.1978   0.2942   0.3876   0.4772   0.5620   0.6412   0.7140   0.7797   0.8376   0.8871   0.9277   0.9591   0.9809   0.9929 


========== 2D Convolution ==========
Input Image (top-left 8x8 corner):
  0.5000   0.5499   0.5993   0.6478   0.6947   0.7397   0.7823   0.8221 
  0.4975   0.5474   0.5968   0.6453   0.6922   0.7372   0.7798   0.8196 
  0.4900   0.5400   0.5894   0.6378   0.6847   0.7297   0.7724   0.8121 
  0.4777   0.5276   0.5770   0.6254   0.6724   0.7174   0.7600   0.7998 
  0.4605   0.5104   0.5599   0.6083   0.6552   0.7002   0.7429   0.7826 
  0.4388   0.4887   0.5381   0.5866   0.6335   0.6785   0.7211   0.7609 
  0.4127   0.4626   0.5120   0.5604   0.6074   0.6524   0.6950   0.7348 
  0.3824   0.4323   0.4818   0.5302   0.5771   0.6221   0.6647   0.7045 

2D Kernel (top-left 5x5 corner):
  0.0030   0.0133   0.0219   0.0133   0.0030 
  0.0133   0.0596   0.0983   0.0596   0.0133 
  0.0219   0.0983   0.1621   0.0983   0.0219 
  0.0133   0.0596   0.0983   0.0596   0.0133 
  0.0030   0.0133   0.0219   0.0133   0.0030 

2D CPU Convolution Time: 28.8878 ms
2D GPU Convolution Time (including memory transfers): 21.4876 ms
2D Speedup: 1.3444x
2D Convolution succeeded!
2D Output Image (GPU) (top-left 8x8 corner):
  0.2575   0.3672   0.4188   0.4526   0.4854   0.5168   0.5466   0.5744 
  0.3449   0.4921   0.5616   0.6072   0.6513   0.6937   0.7338   0.7713 
  0.3597   0.5136   0.5866   0.6349   0.6816   0.7264   0.7688   0.8084 
  0.3511   0.5020   0.5743   0.6225   0.6693   0.7141   0.7565   0.7961 
  0.3391   0.4858   0.5573   0.6055   0.6522   0.6970   0.7394   0.7790 
  0.3239   0.4654   0.5356   0.5838   0.6306   0.6754   0.7178   0.7574 
  0.3057   0.4408   0.5096   0.5578   0.6046   0.6494   0.6918   0.7314 
  0.2846   0.4123   0.4795   0.5277   0.5745   0.6193   0.6617   0.7013 
```

### Performance Analysis

1. **1D Convolution**: The CPU implementation outperforms the GPU implementation for 1D convolution (speedup of 0.0309x). This is expected for small 1D operations where the overhead of memory transfers to and from the GPU exceeds the computational benefits.

2. **2D Convolution**: The GPU implementation shows a modest speedup (1.3444x) over the CPU for 2D convolution. This demonstrates that as the computational complexity increases (from 1D to 2D), the GPU's parallel processing capabilities begin to outweigh the memory transfer overhead.

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
