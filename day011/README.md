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
- For large arrays, the GPU implementation can provide significant speedup compared to the sequential CPU version.

## Compilation and Execution

```bash
nvcc -o merge_sort merge_sort.cu
./merge_sort
```

## Output

The program outputs:
- First few elements of the unsorted array
- First few elements after parallel and sequential sorting
- Verification of sorting success
- Execution time for both parallel and sequential implementations
- Speedup achieved by the parallel implementation
