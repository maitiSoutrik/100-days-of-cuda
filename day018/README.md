# Day 18: Matrix Multiplication with CUBLAS

This example demonstrates how to perform matrix multiplication using the CUBLAS library, which provides highly optimized linear algebra operations for CUDA-enabled GPUs.

## Implementation Details

- Uses CUBLAS for efficient matrix multiplication (C = A × B)
- Demonstrates proper initialization and cleanup of CUBLAS handles
- Shows how to convert between row-major (C/C++) and column-major (CUBLAS) formats
- Compares performance with a CPU implementation for reference
- Includes error checking and validation of results

## Key CUBLAS Functions Used

- `cublasCreate()` - Initialize the CUBLAS library
- `cublasSgemm()` - Single-precision general matrix multiplication
- `cublasDestroy()` - Clean up CUBLAS resources

## Performance Notes

- CUBLAS operations are highly optimized for GPU execution
- For small matrices, the overhead of data transfer might outweigh computation benefits
- For large matrices, the performance gain can be significant (10-100x faster than CPU)
- The implementation handles matrix layout differences between C/C++ and CUBLAS

## Sample Output

```cuda
Matrix A (2x3):
1.00 2.00 3.00 
4.00 5.00 6.00 

Matrix B (3x2):
7.00 8.00 
9.00 10.00 
11.00 12.00 

Matrix C = A × B (2x2):
58.00 64.00 
139.00 154.00 

CPU Execution Time: X.XXX ms
GPU Execution Time: X.XXX ms
Speedup: X.XX
```
