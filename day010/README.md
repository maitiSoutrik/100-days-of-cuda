# Day 10: Sparse Matrix-Vector Multiplication (SpMV)

## Introduction to Sparse Matrices

Sparse matrices are matrices in which most elements are zero. They appear frequently in scientific computing, machine learning, graph algorithms, and many other domains. Efficiently handling sparse matrices is crucial for performance in applications where data sparsity is common.

The key insight with sparse matrices is that we can save both memory and computation by storing and processing only the non-zero elements.

## Compressed Sparse Row (CSR) Format

The Compressed Sparse Row (CSR) format is one of the most common representations for sparse matrices. It consists of three arrays:

1. **Values**: Stores all non-zero values of the matrix
2. **Column Indices**: Stores the column index for each non-zero value
3. **Row Offsets**: Stores the starting position of each row in the values array

For a matrix with `m` rows and `nnz` non-zero elements:

- Values array has length `nnz`
- Column indices array has length `nnz`
- Row offsets array has length `m+1` (the extra element points to the end of the data)

## Sparse Matrix-Vector Multiplication (SpMV)

SpMV is a fundamental operation in many algorithms. The operation computes `y = A * x` where:
- `A` is a sparse matrix
- `x` is a dense vector
- `y` is the resulting dense vector

For a matrix in CSR format, the sequential algorithm is:

```c
for (int i = 0; i < num_rows; i++) {
    float sum = 0;
    for (int j = row_offsets[i]; j < row_offsets[i+1]; j++) {
        sum += values[j] * x[column_indices[j]];
    }
    y[i] = sum;
}
```

## CUDA Implementation

This implementation provides two CUDA kernels for SpMV:

1. **Basic SpMV Kernel**: Each thread computes one element of the output vector
2. **Optimized SpMV Kernel**: Uses shared memory to cache the input vector for better memory access patterns

### Key Optimization Techniques

1. **Thread Mapping**: One thread per output element (row of the sparse matrix)
2. **Shared Memory**: The optimized kernel uses shared memory to cache the input vector, reducing global memory accesses
3. **Coalesced Memory Access**: The CSR format naturally enables coalesced access to the values and column indices arrays
4. **Workload Distribution**: Each thread processes a variable number of non-zero elements based on the sparsity pattern

## Performance Considerations

- **Memory-Bound Operation**: SpMV is typically memory-bound rather than compute-bound
- **Irregular Memory Access**: The column indices create irregular access patterns to the input vector
- **Load Imbalance**: Different rows may have different numbers of non-zero elements, leading to thread divergence
- **Sparsity Pattern**: The performance is highly dependent on the sparsity pattern of the matrix

## Execution Results

Below are the execution results from running the implementation on a Jetson Nano:

```bash
Sparse Matrix-Vector Multiplication (SpMV)
Matrix size: 10000 x 10000, Sparsity: 1.00%

Sparse Matrix Info:
  Dimensions: 10000 x 10000
  Non-zeros: 1000008
  Sparsity: 1.0000%

Performance Results:
Optimized GPU SpMV (Shared Memory): PASSED
  Execution time: 3.6543 ms
  Speedup vs CPU: 4.06x
  Speedup vs Basic GPU: 2.92x
```

These results from the Jetson Nano demonstrate how the CUDA implementation efficiently processes sparse matrices on embedded GPU hardware. The shared memory optimization provides substantial performance improvements, achieving a 2.92x speedup over the basic GPU implementation. This highlights the importance of memory access optimization on memory-constrained platforms like the Jetson Nano.

The overall 4.06x speedup over CPU execution confirms that even on embedded GPU platforms, CUDA can deliver significant acceleration for sparse matrix operations. The matrix sample shows a typical sparse pattern with approximately 100 non-zero elements per row in a 10,000 × 10,000 matrix, maintaining the target 1% sparsity level.

## Applications

SpMV operations are fundamental to many algorithms and applications:

1. **Iterative Solvers**: Conjugate Gradient, GMRES, and other iterative methods for solving linear systems
2. **Graph Algorithms**: PageRank, breadth-first search, and other graph traversal algorithms
3. **Machine Learning**: Sparse neural networks, recommendation systems, and natural language processing
4. **Scientific Computing**: Finite element methods, computational fluid dynamics, and other simulation techniques

## References

- Bell, N., & Garland, M. (2009). Implementing sparse matrix-vector multiplication on throughput-oriented processors. In Proceedings of the Conference on High Performance Computing Networking, Storage and Analysis.
- Gremse, F., Höfter, A., Schwen, L. O., Kiessling, F., & Naumann, U. (2015). GPU-Accelerated Sparse Matrix-Matrix Multiplication by Iterative Row Merging. SIAM Journal on Scientific Computing.
- Liu, W., & Vinter, B. (2015). CSR5: An efficient storage format for cross-platform sparse matrix-vector multiplication. In Proceedings of the 29th ACM on International Conference on Supercomputing.
