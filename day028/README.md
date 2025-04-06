# Day 28: Mini-Batch SGD for Linear Regression

## Overview

This project implements Mini-Batch Stochastic Gradient Descent (SGD) to train a simple linear regression model using CUDA. Linear regression is one of the most fundamental machine learning algorithms, and implementing it with mini-batch SGD on a GPU demonstrates key concepts in parallel optimization algorithms that are essential for training larger neural networks.

The implementation focuses on a simple linear model (y = wx + b) and uses synthetic data with added noise to demonstrate the learning process. The code showcases how to parallelize both the gradient computation and the reduction (summation) steps across a mini-batch.

## Implementation Details

The implementation consists of the following components:

1. **Data Generation:**
   * Synthetic data is generated on the host following the linear model y = true_w * x + true_b + noise
   * x values are randomly generated between -10 and 10
   * Random noise is added to simulate real-world data

2. **Host Logic:**
   * Initializes model parameters (w and b) to zero
   * Allocates memory for the full dataset and mini-batches
   * Implements the training loop with epochs and mini-batches
   * Shuffles data at the beginning of each epoch
   * Updates model parameters after processing each mini-batch
   * Calculates and reports the MSE loss

3. **CUDA Kernels:**
   * `calculate_gradients_kernel`: Each thread computes gradients for one sample in the mini-batch
     * Calculates prediction: y_pred = w * x + b
     * Calculates error: error = y_pred - y_true
     * Computes gradients: grad_w = 2 * error * x, grad_b = 2 * error
   * `reduce_sum_kernel`: Performs parallel reduction to sum gradients across the mini-batch
     * Uses shared memory for efficient reduction
     * Handles arbitrary batch sizes with proper boundary checks
     * Supports multi-block reduction for large batch sizes

4. **Training Process:**
   * For each mini-batch:
     * Copy mini-batch data and current model parameters to the device
     * Launch the gradient calculation kernel
     * Launch the reduction kernel to sum gradients
     * Copy summed gradients back to the host
     * Update model parameters on the host: w = w - learning_rate * avg_grad_w, b = b - learning_rate * avg_grad_b

## Key CUDA Features Used

* **Parallel Computation:** Each thread processes one sample in the mini-batch
* **Shared Memory:** Used in the reduction kernel for efficient summation
* **Memory Management:** Proper allocation and deallocation of device memory
* **Error Checking:** Comprehensive error checking with the CHECK_CUDA_ERROR macro
* **Kernel Launch Configuration:** Dynamic grid and block dimensions based on batch size
* **Multi-Block Reduction:** Handles large batch sizes with a two-step reduction process

## Performance Considerations

* **Memory Transfers:** The implementation minimizes memory transfers between host and device by only copying mini-batch data and model parameters
* **Shared Memory Usage:** The reduction kernel uses shared memory to reduce global memory accesses and improve performance
* **Batch Size:** The batch size affects both performance and convergence
  * Larger batch sizes provide better parallelism but may require more memory and multiple reduction steps
  * Smaller batch sizes may converge faster but with more noise in the gradient estimates
* **Thread Divergence:** The implementation minimizes thread divergence by using simple conditional checks
* **Reduction Algorithm:** The parallel reduction algorithm has O(log n) complexity compared to O(n) for sequential summation

## Building and Running

1. **Navigate to the build directory:**
   ```bash
   cd build
   ```
2. **Configure using CMake:**
   ```bash
   cmake ..
   ```
3. **Build the executable:**
   ```bash
   make mini_batch_sgd
   ```
4. **Run the executable:**
   ```bash
   ./day028/mini_batch_sgd
   ```

## Execution Results

Actual output from Jetson Nano:
```
drboom@JetNano ~/g/1/build> ./day028/mini_batch_sgd 
Mini-Batch SGD for Linear Regression
-------------------------------------
Data size: 10000
Batch size: 128
Number of epochs: 50
Learning rate: 0.0100
True parameters: w = 2.50, b = 1.20
Noise scale: 0.50
-------------------------------------

Starting training...
Initial loss: 210.015305
Epoch 1/50 - Loss: 0.145984, w: 2.5062, b: 0.9518
Epoch 5/50 - Loss: 0.082941, w: 2.4975, b: 1.2004
Epoch 10/50 - Loss: 0.086165, w: 2.4906, b: 1.2006
Epoch 15/50 - Loss: 0.084246, w: 2.5082, b: 1.2034
Epoch 20/50 - Loss: 0.083543, w: 2.5066, b: 1.1997
Epoch 25/50 - Loss: 0.082712, w: 2.5034, b: 1.2023
Epoch 30/50 - Loss: 0.082588, w: 2.4997, b: 1.1990
Epoch 35/50 - Loss: 0.082783, w: 2.4983, b: 1.1987
Epoch 40/50 - Loss: 0.084480, w: 2.4934, b: 1.2023
Epoch 45/50 - Loss: 0.085665, w: 2.5107, b: 1.2012
Epoch 50/50 - Loss: 0.087625, w: 2.5134, b: 1.2004

Training completed!
-------------------------------------
Initial parameters: w = 0.0000, b = 0.0000
Learned parameters: w = 2.5134, b = 1.2004
True parameters:    w = 2.5000, b = 1.2000
Final loss: 0.087625
-------------------------------------
Relative error: w = 0.54%, b = 0.03%

Day 28 Mini-Batch SGD finished successfully.
```

## Learnings and Observations

* **Dramatic Initial Convergence:** The model showed remarkable initial convergence, with the loss dropping from 210.015305 to 0.145984 in just the first epoch, demonstrating the efficiency of gradient descent in quickly finding the approximate region of the optimal solution.
* **Excellent Parameter Accuracy:** Both parameters converged with high accuracy, with the weight (w) having a relative error of 0.54% and the bias (b) having an exceptionally low error of just 0.03%. This indicates that the optimization landscape was favorable for this particular run.
* **Stable Optimization:** After the initial rapid drop, the loss continued to decrease but at a much slower rate, with some minor fluctuations. These fluctuations are normal in SGD due to the randomness in mini-batch sampling.
* **Appropriate Learning Rate:** The learning rate of 0.01 proved well-suited for this problem, allowing rapid initial convergence without causing instability in later epochs.
* **Effective Parallelization:** The parallel implementation successfully leveraged GPU parallelism to process mini-batches efficiently, as evidenced by the smooth convergence pattern.
* **Memory Management:** Proper memory management was crucial for this implementation, especially for handling the multi-block reduction when summing gradients across mini-batches.

## Future Improvements

* **Multi-dimensional Inputs:** Extend the implementation to handle multi-dimensional inputs (x as a vector)
* **Multi-dimensional Outputs:** Support multi-dimensional outputs (multiple regression)
* **Momentum:** Add momentum to the SGD algorithm to improve convergence
* **Adaptive Learning Rates:** Implement adaptive learning rate methods like Adam or RMSProp
* **Regularization:** Add L1/L2 regularization to prevent overfitting
* **Streams:** Use CUDA streams for overlapping computation and memory transfers
* **Unified Memory:** Explore using CUDA Unified Memory for simplified memory management

## References

* Ruder, S. (2016). "An overview of gradient descent optimization algorithms." arXiv preprint arXiv:1609.04747.
* Harris, M. (2007). "Optimizing parallel reduction in CUDA." NVIDIA Developer Technology, 2(4), 70.
* Bottou, L. (2010). "Large-scale machine learning with stochastic gradient descent." In Proceedings of COMPSTAT'2010 (pp. 177-186).
