#ifndef HINGE_LOSS_CUH
#define HINGE_LOSS_CUH

#include <cuda_runtime.h>
#include <cstdio> // For printf in error macro

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief Computes Hinge Loss on the GPU.
 *
 * Calculates L(t, y_score) = max(0, 1 - t * y_score) for each element.
 *
 * @param d_true_labels Pointer to an array of true labels on the device (+1 or -1).
 * @param d_pred_scores Pointer to an array of predicted scores on the device.
 * @param d_loss Pointer to an array on the device where the computed loss for each element will be stored.
 * @param num_elements The number of elements in the arrays.
 */
void hinge_loss_cuda(const int* d_true_labels, const float* d_pred_scores, float* d_loss, int num_elements);

/**
 * @brief Computes the sum of Hinge Loss values on the GPU.
 * 
 * This function first computes individual hinge losses and then sums them up.
 * This is a common operation when calculating the total loss for a batch.
 *
 * @param d_true_labels Pointer to an array of true labels on the device (+1 or -1).
 * @param d_pred_scores Pointer to an array of predicted scores on the device.
 * @param d_total_loss Pointer to a single float on the device where the sum of losses will be stored.
 * @param num_elements The number of elements in the arrays.
 * @param d_temp_storage Temporary device storage for reduction. Size should be num_elements.
 */
void sum_hinge_loss_cuda(const int* d_true_labels, const float* d_pred_scores, float* d_total_loss, int num_elements, float* d_temp_storage);

#endif // HINGE_LOSS_CUH
