# Day 76: Contrastive Loss Implementation

## Overview

Contrastive loss is a crucial component in deep learning, particularly for tasks involving similarity learning or representation learning, such as Siamese networks or self-supervised learning. The fundamental idea is to learn an embedding space where similar (positive) input pairs are pulled closer together, while dissimilar (negative) input pairs are pushed further apart.

The loss function typically involves a margin parameter. For positive pairs, the loss aims to minimize the distance between their embeddings. For negative pairs, the loss penalizes them only if their distance is within this predefined margin; if they are already sufficiently far apart (beyond the margin), no loss is incurred for that pair.

This day's task is to implement the contrastive loss function in CUDA, including both the forward pass (calculating the loss) and the backward pass (calculating gradients with respect to the input embeddings).

The mathematical formulation for a single pair of embeddings \((x_i, x_j)\) with label \(y_{ij}\) (1 for positive, 0 for negative) and margin \(m\) is:

\[ \mathcal{L}(x_i, x_j) = y_{ij} \cdot d(x_i, x_j)^2 + (1 - y_{ij}) \cdot \max(0, m - d(x_i, x_j))^2 \]

Where \(d(x_i, x_j)\) is the Euclidean distance between the embeddings.

## Implementation Details

The implementation consists of:
1.  **`contrastive_loss.cuh`**: Header file defining the CUDA error checking macro and the function signatures for the forward and backward passes.
2.  **`contrastive_loss.cu`**: CUDA source file containing:
    *   `contrastiveLossForwardKernel`: A CUDA kernel that computes the loss for each pair in a batch. Each thread typically handles one pair. It first calculates the squared Euclidean distance between the feature vectors of a pair. Then, based on the label (similar or dissimilar) and the margin, it computes the contrastive loss.
    *   `contrastiveLossBackwardKernel`: A CUDA kernel that computes the gradients of the loss with respect to the input embeddings (`input1` and `input2`). Each thread in a 2D grid computes the gradient for a single feature dimension of a single pair. The gradient calculation differs for similar and dissimilar pairs and also depends on whether a dissimilar pair's distance is within the margin. An epsilon is used in the denominator during gradient calculation for dissimilar pairs to prevent division by zero.
    *   Wrapper functions `contrastiveLossForward` and `contrastiveLossBackward` that manage CUDA memory (though allocation/deallocation is done in `main` for this example) and launch the respective kernels.
3.  **`contrastive_loss_main.cu`**: A C++/CUDA source file with a `main` function that:
    *   Initializes sample input feature vectors and labels on the host.
    *   Allocates memory on the CUDA device for inputs, labels, loss, and gradients.
    *   Copies host data to the device.
    *   Calls the `contrastiveLossForward` function.
    *   Copies the computed loss back to the host and prints it.
    *   Calls the `contrastiveLossBackward` function.
    *   Copies the computed gradients back to the host and prints them.
    *   Frees all allocated device memory.
4.  **`contrastive_loss_test.cu`**: Google Test cases to verify the correctness of the forward and backward pass implementations using known input values and expected outputs.

### Forward Pass Kernel (`contrastiveLossForwardKernel`)
- Each thread processes one pair of input vectors.
- It computes the squared Euclidean distance \(d^2\).
- If the pair is similar (label=1), loss = \(d^2\).
- If the pair is dissimilar (label=0), loss = \((\max(0, m - d))^2\), where \(d = \sqrt{d^2}\).

### Backward Pass Kernel (`contrastiveLossBackwardKernel`)
- Each thread (in a 2D grid) computes the gradient for one feature of one input vector in a pair.
- For a similar pair (\(y=1\)):
    - \(\frac{\partial \mathcal{L}}{\partial x_{1k}} = 2 (x_{1k} - x_{2k})\)
    - \(\frac{\partial \mathcal{L}}{\partial x_{2k}} = -2 (x_{1k} - x_{2k})\)
- For a dissimilar pair (\(y=0\)) and \(d < m\):
    - \(\frac{\partial \mathcal{L}}{\partial x_{1k}} = -2 (m - d) \frac{x_{1k} - x_{2k}}{d + \epsilon}\)
    - \(\frac{\partial \mathcal{L}}{\partial x_{2k}} = 2 (m - d) \frac{x_{1k} - x_{2k}}{d + \epsilon}\)
- For a dissimilar pair (\(y=0\)) and \(d \ge m\), gradients are 0.
- An epsilon (\(1e-8f\)) is added to \(d\) in the denominator to prevent division by zero.

## Key CUDA Features Used

*   **CUDA Kernels**: `__global__` functions for parallel computation on the GPU.
*   **Device Memory Management**: `cudaMalloc`, `cudaMemcpy`, `cudaFree` for managing memory on the GPU. `cudaMemsetAsync` for initializing gradient buffers.
*   **Thread Indexing**: `blockIdx`, `blockDim`, `threadIdx` for identifying and managing threads.
*   **CUDA Error Handling**: A custom `CHECK_CUDA_ERROR` macro is used for robust error checking.
*   **CUDA Streams**: Function signatures include `cudaStream_t` parameter for potential asynchronous execution, though the main example uses synchronous calls for simplicity.
*   **Math Functions**: `sqrtf`, `fmaxf` used within the kernels.

## Performance Considerations

*   **Kernel Launch Configuration**:
    *   The forward kernel uses a 1D grid where each thread handles one pair and iterates through feature dimensions.
    *   The backward kernel uses a 2D grid where each thread handles one feature of one pair. This avoids loops over features within the kernel but requires recalculating the distance `d` for each feature thread of the same pair. This could be optimized by computing `d` once per pair using a block-level reduction or shared memory if multiple threads in a block work on the same pair's features. For simplicity, this optimization was not implemented.
*   **Memory Access Patterns**: Input data is accessed linearly within the feature dimension loop in the forward kernel. In the backward kernel, `input1` and `input2` are accessed based on `pair_idx` and `feature_k`.
*   **Shared Memory**: For the backward pass, if multiple threads within a block were to process different features of the *same* pair, the pair's feature vectors and their distance `d` could be loaded into shared memory to reduce global memory reads. The current 2D grid launch (one thread per feature of a pair) doesn't lend itself as directly to this without restructuring, as threads in a block might span different pairs.
*   **Numerical Stability**: An epsilon is added in the denominator of the gradient calculation for dissimilar pairs to prevent division by zero.

## Building and Running

1.  **Prerequisites**:
    *   NVIDIA CUDA Toolkit (>= 10.0, tested with 11.x, 12.x)
    *   CMake (>= 3.18 for `gtest_discover_tests`)
    *   A C++ compiler (g++ or clang compatible with CUDA)
    *   Google Test (will be fetched by the root `CMakeLists.txt` if not found)

2.  **Building**:
    Navigate to the root directory of the `100-days-of-cuda` project.
    ```bash
    mkdir build
    cd build
    cmake ..
    make day076_contrastive_loss_main day076_contrastive_loss_test_exe # Or simply 'make'
    ```
    This will build the main executable `contrastive_loss_main` and the test executable `contrastive_loss_test_exe` in the `build/day076/` directory.

3.  **Running the Main Executable**:
    ```bash
    ./day076/contrastive_loss_main
    ```

4.  **Running Tests**:
    ```bash
    ./day076/contrastive_loss_test_exe
    # Or using CTest from the build directory
    # ctest -R day076_contrastive_loss_test_exe # Or ctest --verbose
    ```

## Execution Results

The output of `./day076/contrastive_loss_main` will show the initialized input vectors, labels, the computed loss for each pair, and the computed gradients for each input vector.

Actual console output from Jetson Nano:
```
Contrastive Loss CUDA Implementation Test
Batch Size: 8, Feature Dim: 4, Margin: 1

h_input1 (first 32 elements, 8 samples):
  Sample 0: [-0.9993, 0.4708, -0.2476, 0.9517]
  Sample 1: [-0.7858, 0.8011, -0.5092, -0.6235]
  Sample 2: [-0.7931, 0.0688, 0.4740, 0.0780]
  Sample 3: [-0.0561, -0.5331, -0.9539, 0.8309]
  Sample 4: [0.1416, 0.2471, -0.2487, -0.2965]
  Sample 5: [0.0573, -0.0924, -0.6144, 0.9731]
  Sample 6: [-0.8217, -0.1325, -0.6265, 0.9909]
  Sample 7: [-0.1221, -0.6439, 0.6122, -0.9406]

h_input2 (first 32 elements, 8 samples):
  Sample 0: [-0.9933, 0.4223, -0.6074, 0.0246]
  Sample 1: [0.6310, -0.0959, -0.5052, -0.3534]
  Sample 2: [-0.7206, 0.0617, -0.8433, -0.7380]
  Sample 3: [-0.1000, -0.9754, 0.9870, 0.5717]
  Sample 4: [0.1144, 0.1577, -0.4658, 0.2819]
  Sample 5: [-0.6846, -0.3414, -0.8705, 0.0922]
  Sample 6: [-0.7849, -0.1527, 0.0874, -0.6201]
  Sample 7: [0.4940, 0.6216, -0.8051, -0.7321]

h_labels (first 8 elements):
  [1, 0, 1, 0, 1, 0, 1, 0]

--- Running Forward Pass ---
h_loss (Forward) (first 8 elements):
  [0.9914, 0.0000, 2.4066, 0.0000, 0.3904, 0.0000, 3.1066, 0.0000]

--- Running Backward Pass ---
h_grad_input1 (Backward) (first 32 elements, 8 samples):
  Sample 0: [-0.0122, 0.0972, 0.7198, 1.8542]
  Sample 1: [0.0000, 0.0000, 0.0000, 0.0000]
  Sample 2: [-0.1450, 0.0142, 2.6347, 1.6320]
  Sample 3: [0.0000, 0.0000, 0.0000, 0.0000]
  Sample 4: [0.0545, 0.1789, 0.4341, -1.1568]
  Sample 5: [0.0000, 0.0000, 0.0000, 0.0000]
  Sample 6: [-0.0737, 0.0402, -1.4278, 3.2219]
  Sample 7: [0.0000, 0.0000, 0.0000, 0.0000]

h_grad_input2 (Backward) (first 32 elements, 8 samples):
  Sample 0: [0.0122, -0.0972, -0.7198, -1.8542]
  Sample 1: [-0.0000, -0.0000, -0.0000, -0.0000]
  Sample 2: [0.1450, -0.0142, -2.6347, -1.6320]
  Sample 3: [-0.0000, -0.0000, -0.0000, -0.0000]
  Sample 4: [-0.0545, -0.1789, -0.4341, 1.1568]
  Sample 5: [-0.0000, -0.0000, -0.0000, -0.0000]
  Sample 6: [0.0737, -0.0402, 1.4278, -3.2219]
  Sample 7: [-0.0000, -0.0000, -0.0000, -0.0000]

Test finished.
```

The Google Test output:
```
[==========] Running 7 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 7 tests from ContrastiveLossTest
[ RUN      ] ContrastiveLossTest.ForwardPassSimilarPair
[       OK ] ContrastiveLossTest.ForwardPassSimilarPair (91 ms)
[ RUN      ] ContrastiveLossTest.ForwardPassDissimilarPairWithinMargin
[       OK ] ContrastiveLossTest.ForwardPassDissimilarPairWithinMargin (1 ms)
[ RUN      ] ContrastiveLossTest.ForwardPassDissimilarPairOutsideMargin
[       OK ] ContrastiveLossTest.ForwardPassDissimilarPairOutsideMargin (1 ms)
[ RUN      ] ContrastiveLossTest.BackwardPassSimilarPair
[       OK ] ContrastiveLossTest.BackwardPassSimilarPair (1 ms)
[ RUN      ] ContrastiveLossTest.BackwardPassDissimilarPairWithinMargin
[       OK ] ContrastiveLossTest.BackwardPassDissimilarPairWithinMargin (1 ms)
[ RUN      ] ContrastiveLossTest.BackwardPassDissimilarPairOutsideMargin
[       OK ] ContrastiveLossTest.BackwardPassDissimilarPairOutsideMargin (1 ms)
[ RUN      ] ContrastiveLossTest.ForwardPassBatch
[       OK ] ContrastiveLossTest.ForwardPassBatch (1 ms)
[----------] 7 tests from ContrastiveLossTest (101 ms total)

[----------] Global test environment tear-down
[==========] 7 tests from 1 test suite ran. (101 ms total)
[  PASSED  ] 7 tests.
```

## Learnings and Observations

*   Implementing contrastive loss involves careful handling of two cases: similar pairs and dissimilar pairs.
*   The margin parameter is crucial for defining the boundary for dissimilar pairs.
*   The backward pass (gradient calculation) requires deriving the partial derivatives of the loss function with respect to each element of the input embeddings. Special care is needed for the `max(0, ...)` part of the loss, as the gradient is zero if the term inside `max` is negative.
*   Numerical stability (e.g., adding epsilon to prevent division by zero) is an important consideration in gradient calculations involving distances.
*   Testing with simple, hand-crafted inputs is essential for verifying the correctness of both forward and backward passes.
*   The choice of kernel launch configuration (1D vs. 2D grid) can impact implementation complexity and potential for further optimization (e.g., using shared memory).

## References
*   Original Contrastive Loss concept: Chopra, S., Hadsell, R., & LeCun, Y. (2005). Learning a similarity metric discriminatively, with application to face verification. In CVPR.
*   Explanations from Tavily and Perplexity AI were used to solidify understanding before implementation.
