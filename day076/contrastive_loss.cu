#include "contrastive_loss.cuh"
#include <cuda_fp16.h> // For __half if needed, though using float for now

// Kernel to compute contrastive loss (forward pass)
__global__ void contrastiveLossForwardKernel(const float* input1,
                                             const float* input2,
                                             const int* labels,
                                             float* loss,
                                             int batch_size,
                                             int feature_dim,
                                             float margin) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        float dist_sq = 0.0f;
        // Calculate squared Euclidean distance
        for (int i = 0; i < feature_dim; ++i) {
            float diff = input1[idx * feature_dim + i] - input2[idx * feature_dim + i];
            dist_sq += diff * diff;
        }
        //float distance = sqrtf(dist_sq); // distance is sqrt(dist_sq)

        if (labels[idx] == 1) { // Similar pair
            loss[idx] = dist_sq;
        } else { // Dissimilar pair
            // loss[idx] = fmaxf(0.0f, margin - distance); // This is for margin - d
            // loss[idx] = loss[idx] * loss[idx];         // Then square it: (max(0, m-d))^2
            // More numerically stable to work with dist_sq if possible, but formula uses d.
            // Let's stick to the formula: max(0, m - d)^2
            float distance = sqrtf(dist_sq);
            float term = margin - distance;
            loss[idx] = fmaxf(0.0f, term) * fmaxf(0.0f, term);
        }
    }
}

// Kernel to compute gradients for contrastive loss (backward pass)
__global__ void contrastiveLossBackwardKernel(const float* input1,
                                              const float* input2,
                                              const int* labels,
                                              float* grad_input1,
                                              float* grad_input2,
                                              int batch_size,
                                              int feature_dim,
                                              float margin) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x; // Index for the pair
    int feature_k = blockIdx.y * blockDim.y + threadIdx.y; // Index for the feature dimension

    if (pair_idx < batch_size && feature_k < feature_dim) {
        float dist_sq = 0.0f;
        // Recalculate squared Euclidean distance for the pair
        // This is somewhat inefficient as it's recalculated per feature,
        // could be optimized by calculating once per pair and storing in shared memory if threads in a block handle same pair.
        // For simplicity now, recalculate.
        for (int i = 0; i < feature_dim; ++i) {
            float diff_val = input1[pair_idx * feature_dim + i] - input2[pair_idx * feature_dim + i];
            dist_sq += diff_val * diff_val;
        }
        float distance = sqrtf(dist_sq);
        float epsilon = 1e-8f; // To prevent division by zero

        float grad_val = 0.0f;
        float x1_k = input1[pair_idx * feature_dim + feature_k];
        float x2_k = input2[pair_idx * feature_dim + feature_k];

        if (labels[pair_idx] == 1) { // Similar pair
            // dL/dx1_k = 2 * (x1_k - x2_k)
            grad_val = 2.0f * (x1_k - x2_k);
        } else { // Dissimilar pair
            if (distance < margin) {
                // dL/dx1_k = 2 * (margin - d) * (-1) * (x1_k - x2_k) / d
                // dL/dx1_k = -2 * (margin - d) * (x1_k - x2_k) / (d + epsilon)
                grad_val = -2.0f * (margin - distance) * (x1_k - x2_k) / (distance + epsilon);
            } else {
                grad_val = 0.0f; // Gradient is 0 if d >= m
            }
        }
        
        grad_input1[pair_idx * feature_dim + feature_k] = grad_val;
        // dL/dx2_k = - dL/dx1_k (for the (x1_k - x2_k) part)
        // For similar: dL/dx2_k = -2 * (x1_k - x2_k)
        // For dissimilar (d < m): dL/dx2_k = 2 * (margin - d) * (x1_k - x2_k) / (d + epsilon)
        grad_input2[pair_idx * feature_dim + feature_k] = -grad_val; // This is incorrect for dissimilar.
                                                                    // Let's re-derive dL/dx2_k properly.
                                                                    // dL/dx2_k = 2 * (x1_k - x2_k) for similar (correct, -(-2*(x1_k-x2_k)))
                                                                    // dL/dx2_k = 2 * (margin - d) * (x1_k - x2_k) / (d + epsilon) for dissimilar (d < m)
                                                                    // So grad_input2 is -grad_input1 if grad_val is dL/dx1.
                                                                    // If grad_val = 2 * (x1-x2) for similar, then grad_input2 = -grad_val. Correct.
                                                                    // If grad_val = -2 * (m-d) * (x1-x2)/(d+eps) for dissimilar,
                                                                    // then grad_input2 = -grad_val = 2 * (m-d) * (x1-x2)/(d+eps). Correct.
        grad_input2[pair_idx * feature_dim + feature_k] = -grad_val;
    }
}


// Wrapper for forward pass
void contrastiveLossForward(const float* d_input1,
                            const float* d_input2,
                            const int* d_labels,
                            float* d_loss,
                            int batch_size,
                            int feature_dim,
                            float margin,
                            cudaStream_t stream) {
    dim3 threadsPerBlock(256);
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    contrastiveLossForwardKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_input1, d_input2, d_labels, d_loss, batch_size, feature_dim, margin);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Wrapper for backward pass
void contrastiveLossBackward(const float* d_input1,
                             const float* d_input2,
                             const int* d_labels,
                             float* d_grad_input1,
                             float* d_grad_input2,
                             int batch_size,
                             int feature_dim,
                             float margin,
                             cudaStream_t stream) {
    // Initialize gradients to zero
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_grad_input1, 0, batch_size * feature_dim * sizeof(float), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_grad_input2, 0, batch_size * feature_dim * sizeof(float), stream));

    // For the backward kernel, each thread can compute the gradient for one feature of one pair.
    // So, gridDim.x for batch_size, gridDim.y for feature_dim
    // Or, a 1D grid for batch_size, and loop over features (like forward)
    // Or, a 2D grid: (batch_size, feature_dim)
    // Let's try a 2D grid.
    dim3 threadsPerBlock(16, 16); // 256 threads, suitable for feature_dim and batch_size
    dim3 numBlocks((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (feature_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);

    contrastiveLossBackwardKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        d_input1, d_input2, d_labels, d_grad_input1, d_grad_input2,
        batch_size, feature_dim, margin);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
