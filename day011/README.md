# Day 011: CUDA Merge Sort

This implementation demonstrates a parallel merge sort algorithm using CUDA. The approach combines recursive CPU-based division of the array with GPU-accelerated merging of sorted subarrays.

## Implementation Details

1. **Co-Rank Function**: A device function that efficiently finds the position where elements from two sorted arrays should be placed in the merged result.

2. **Parallel Merge Kernel**: Each thread computes where an element in the final merged array should come from (either the first or second input array).

3. **Recursive Division**: The algorithm recursively divides the array into smaller subarrays until reaching the base case.

4. **GPU-Accelerated Merging**: The merging of sorted subarrays is performed on the GPU using the parallel merge kernel.

5. **Performance Comparison**: The implementation includes both parallel (GPU) and sequential (CPU) versions for performance comparison.

## Performance Characteristics

- The parallel merge operation has a time complexity of O(log(n+m)) per thread, where n and m are the sizes of the two arrays being merged.
- The overall algorithm has a time complexity of O(n log²(n)) due to the recursive nature of merge sort combined with the parallel merge.
- For small arrays on the Jetson Nano, the sequential CPU implementation is actually faster than the GPU implementation due to the overhead of data transfers and kernel launches.

## Compilation and Execution

```bash
cmake ..
make
./day011/merge_sort
```

## Output

Execution results on Jetson Nano:

```text
Array size: 1024
First 10 elements of unsorted array:
260 7158 1855 9827 9200 2391 1293 5866 4014 9853 

First 10 elements after parallel merge sort:
29 50 61 94 96 100 120 121 132 135 

First 10 elements after sequential merge sort:
29 50 61 94 96 100 120 121 132 135 

Parallel merge sort successful
Sequential merge sort successful
Parallel merge sort time: 1.112032 seconds
Sequential merge sort time: 0.000360 seconds
Speedup: 0.000324
```

### Performance Analysis

Interestingly, the sequential implementation outperformed the parallel implementation for this array size (1024 elements) on the Jetson Nano. This is likely due to:

1. The overhead of data transfers between CPU and GPU memory
2. The cost of kernel launches for relatively small workloads
3. The recursive nature of the algorithm requiring multiple kernel launches

For larger arrays and more computationally intensive sorting tasks, the GPU implementation would likely show better performance. This demonstrates an important principle in GPU programming: not all problems benefit from parallelization, especially when the problem size is small.
