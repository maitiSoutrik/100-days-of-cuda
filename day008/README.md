# Day 8: Brent-Kung Algorithm for Prefix Sum

## Introduction to Prefix Sum

A prefix sum (also known as a scan) is an operation that calculates the running total of a sequence of numbers. For an input array `[a₀, a₁, a₂, ..., aₙ₋₁]`, the prefix sum produces an output array `[a₀, (a₀+a₁), (a₀+a₁+a₂), ..., (a₀+a₁+...+aₙ₋₁)]`.

Prefix sums are fundamental operations in parallel computing and have numerous applications including:

- Sorting algorithms
- Graph algorithms
- Dynamic programming
- Image processing
- Signal processing

## The Brent-Kung Algorithm

The Brent-Kung algorithm, developed by Richard P. Brent and H. T. Kung in 1980, is an efficient parallel algorithm for computing prefix sums. It's particularly well-suited for GPU implementation due to its work-efficient nature and good utilization of parallel resources.

### Algorithm Overview

The Brent-Kung algorithm works in three phases:

1. **Up-sweep (Reduction) Phase**: Build a binary tree by combining pairs of elements.
2. **Down-sweep Phase**: Traverse back down the tree to compute the final prefix sums.

The algorithm requires O(n) work and O(log n) time with n processors, making it work-efficient.

### Key Characteristics

- **Work-Efficient**: Performs O(n) total operations, which is optimal.
- **Logarithmic Time**: Completes in O(log n) steps with sufficient parallelism.
- **Balanced Workload**: Distributes work evenly across threads.
- **Reduced Bank Conflicts**: Minimizes memory access conflicts in GPU implementation.

## Implementation Details

The implementation in this directory demonstrates the Brent-Kung algorithm using CUDA. The key aspects include:

1. **Shared Memory Usage**: Utilizes fast shared memory for efficient data access.
2. **Thread Synchronization**: Uses barriers to ensure correct execution order.
3. **Work Distribution**: Assigns work to threads in a way that maximizes parallelism.

Check out the `brent_kung.cu` file for the complete implementation.

## Performance Considerations

- The Brent-Kung algorithm performs better than sequential algorithms for large arrays.
- It uses fewer operations than the Kogge-Stone algorithm (another parallel prefix sum algorithm), making it more work-efficient.
- However, it may have slightly higher latency due to its tree-based approach.

## Execution Results

Below are the actual execution results from running the implementation on a Jetson Nano:

```bash
Demonstrating Brent-Kung Algorithm Steps:
Input array: 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 

Up-sweep (Reduction) Phase:
Step (d=4): 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 
Step (d=2): 1.0 3.0 3.0 7.0 5.0 11.0 7.0 15.0 
Step (d=1): 1.0 3.0 3.0 10.0 5.0 11.0 7.0 26.0 

Down-sweep Phase:
Initial: 1.0 3.0 3.0 10.0 5.0 11.0 7.0 0.0 
Step (d=1): 1.0 3.0 3.0 0.0 5.0 11.0 7.0 10.0 
Step (d=2): 1.0 0.0 3.0 3.0 5.0 10.0 7.0 21.0 
Step (d=4): 0.0 1.0 3.0 6.0 10.0 15.0 21.0 28.0 

Final Result (Exclusive Scan):
Input:  1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 
Output: 0.0 1.0 3.0 6.0 10.0 15.0 21.0 28.0 
```

These results demonstrate how the algorithm transforms the input array `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]` into the exclusive prefix sum `[0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]`, where each element at position i contains the sum of all elements from positions 0 to i-1.

## References

- Brent, R. P., & Kung, H. T. (1980). A regular layout for parallel adders. IEEE Transactions on Computers, C-31, 260-264.
- Harris, M., Sengupta, S., & Owens, J. D. (2007). Parallel prefix sum (scan) with CUDA. GPU Gems 3, 39(1), 851-876.
