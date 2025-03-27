# Day 17: Naive Bayes Classifier Training with CUDA

This example demonstrates how to implement the training phase of a Naive Bayes classifier using CUDA. The program calculates class-wise means and variances for each feature, which are essential parameters for the Gaussian Naive Bayes model.

## Implementation Details

- The program processes a dataset of 2D points classified into two classes (binary classification)
- For each class and feature combination, it calculates:
  - Mean value
  - Variance
  - Class counts (number of samples in each class)
- Uses a single CUDA kernel to parallelize computations across feature-class combinations
- Each thread handles calculations for a specific feature-class pair

## Sample Output
```
Class Counts:
Class 0: 3
Class 1: 3

Means:
Class 0: 1.00 2.00 
Class 1: 5.00 6.00 

Variances:
Class 0: 0.67 0.67 
Class 1: 0.67 0.67 
```

## Key CUDA Features Used
- Global memory for storing feature data and results
- Grid-stride loop pattern for processing multiple samples
- Atomic operations avoided by having each thread handle independent calculations
- Error checking using CUDA error macros

## Performance Notes
- The implementation uses coalesced memory access patterns
- Each thread processes all samples for its assigned feature-class combination
- Memory transfers are minimized by computing all statistics in a single kernel launch
