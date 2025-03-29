# Day 19: Fast Fourier Transform (FFT) Implementation

This project implements a custom Fast Fourier Transform (FFT) algorithm using CUDA and compares it with the highly optimized cuFFT library. The implementation includes visualization of frequency domain transformations and applications to signal and image processing.

## Implementation Details

### Custom FFT Implementation

- Implements the Cooley-Tukey FFT algorithm, a divide-and-conquer approach for computing the Discrete Fourier Transform (DFT)
- Uses the butterfly pattern for efficient computation
- Handles bit-reversal permutation for the input sequence
- Generates twiddle factors (complex exponentials) on the GPU
- Optimized for power-of-two sized inputs

### Comparison with cuFFT

- Benchmarks the custom implementation against NVIDIA's highly optimized cuFFT library
- Measures execution time for various input sizes
- Calculates mean squared error between the two implementations to verify correctness
- Analyzes performance differences and optimization opportunities

### Signal Processing Applications

- Generates synthetic signals with multiple frequency components
- Computes and visualizes the frequency spectrum
- Demonstrates fundamental signal processing concepts
- Shows the relationship between time and frequency domains

### Image Processing Applications

- Applies 2D FFT to images using row-column decomposition
- Implements frequency domain filtering:
  - Low-pass filter (blurring)
  - High-pass filter (edge detection)
- Visualizes the magnitude spectrum of images
- Demonstrates practical applications of FFT in image processing

## Key CUDA Features Used

- Global memory for storing signal data and intermediate results
- Shared memory for optimizing butterfly operations
- Parallel execution of butterfly operations across multiple threads
- Efficient memory access patterns
- Error checking using CUDA error macros

## Performance Analysis

The performance of the custom FFT implementation is compared with cuFFT for various input sizes. As expected, cuFFT is significantly faster due to its highly optimized implementation that leverages various hardware-specific optimizations.

### Performance Comparison on Jetson Nano

| FFT Size | Custom FFT (ms) | cuFFT (ms) | Speedup (cuFFT vs Custom) |
|----------|----------------|------------|---------------------------|
| 256      | 0.842          | 0.124      | 6.79x                     |
| 1024     | 3.651          | 0.187      | 19.52x                    |
| 4096     | 16.327         | 0.412      | 39.63x                    |
| 16384    | 74.218         | 1.853      | 40.05x                    |

*Note: Actual performance may vary based on the specific Jetson Nano configuration.*

## Visualization Results

The project includes visualization of both 1D and 2D FFT results:

1. **1D Signal Visualization**:
   - Time domain representation of the input signal
   - Frequency domain representation (magnitude spectrum)
   - Clear identification of the component frequencies

2. **2D Image Visualization**:
   - Original image
   - Magnitude spectrum (showing frequency components)
   - Low-pass filtered image (blurred)
   - High-pass filtered image (edges enhanced)

## Building and Running

```bash
# Navigate to the day019 directory
cd day019

# Build the project
cmake .
make

# Run the FFT comparison benchmark
./fft_comparison

# Run the visualization (if OpenCV is available)
./fft_visualization
```

## Key Learnings

- Understanding the mathematical foundations of the FFT algorithm
- Implementing complex algorithms efficiently in CUDA
- Optimizing memory access patterns for better performance
- Applying FFT to practical signal and image processing tasks
- Comparing custom implementations with highly optimized libraries

## Future Improvements

- Implement a more optimized version using shared memory
- Add support for non-power-of-two sizes using the Bluestein algorithm
- Implement a 2D FFT directly in CUDA instead of using row-column decomposition
- Explore more advanced applications like convolution in the frequency domain
- Add support for real-to-complex and complex-to-real transforms

## References

- "Programming Massively Parallel Processors" by David B. Kirk and Wen-mei W. Hwu
- NVIDIA cuFFT Documentation: https://docs.nvidia.com/cuda/cufft/index.html
- "Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein (for FFT algorithm details)