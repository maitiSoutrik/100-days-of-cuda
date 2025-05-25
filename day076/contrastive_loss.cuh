#ifndef CONTRASTIVE_LOSS_CUH
#define CONTRASTIVE_LOSS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath> // For std::sqrt, std::pow

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

/**
 * @brief Computes the contrastive loss forward pass.
 *
 * @param d_input1 Device pointer to the first set of feature vectors (batch_size x feature_dim).
 * @param d_input2 Device pointer to the second set of feature vectors (batch_size x feature_dim).
 * @param d_labels Device pointer to the labels (0 or 1) for each pair (batch_size).
 * @param d_loss Device pointer to store the computed loss for each pair (batch_size).
 * @param batch_size Number of pairs.
 * @param feature_dim Dimensionality of the feature vectors.
 * @param margin The margin for dissimilar pairs.
 * @param stream CUDA stream for asynchronous execution.
 */
void contrastiveLossForward(const float* d_input1,
                            const float* d_input2,
                            const int* d_labels,
                            float* d_loss,
                            int batch_size,
                            int feature_dim,
                            float margin,
                            cudaStream_t stream = 0);

/**
 * @brief Computes the contrastive loss backward pass (gradients).
 *
 * @param d_input1 Device pointer to the first set of feature vectors.
 * @param d_input2 Device pointer to the second set of feature vectors.
 * @param d_labels Device pointer to the labels (0 or 1) for each pair.
 * @param d_grad_input1 Device pointer to store the gradients for d_input1.
 * @param d_grad_input2 Device pointer to store the gradients for d_input2.
 * @param batch_size Number of pairs.
 * @param feature_dim Dimensionality of the feature vectors.
 * @param margin The margin for dissimilar pairs.
 * @param stream CUDA stream for asynchronous execution.
 */
void contrastiveLossBackward(const float* d_input1,
                             const float* d_input2,
                             const int* d_labels,
                             float* d_grad_input1,
                             float* d_grad_input2,
                             int batch_size,
                             int feature_dim,
                             float margin,
                             cudaStream_t stream = 0);

#endif // CONTRASTIVE_LOSS_CUH
