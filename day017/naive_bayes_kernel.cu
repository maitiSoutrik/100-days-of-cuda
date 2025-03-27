#include <cuda_runtime.h>
#include <thrust/device_vector.h>

__global__ void calculate_class_stats(
    const float* features,
    const int* labels,
    float* class_means,
    float* class_variances,
    int* class_counts,
    int num_samples,
    int num_features,
    int num_classes) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int class_idx = blockIdx.y;
    
    if (feature_idx >= num_features || class_idx >= num_classes) return;

    float sum = 0.0f;
    float sq_sum = 0.0f;
    int count = 0;

    for (int i = 0; i < num_samples; i++) {
        if (labels[i] == class_idx) {
            float val = features[i * num_features + feature_idx];
            sum += val;
            sq_sum += val * val;
            count++;
        }
    }

    if (count > 0) {
        float mean = sum / count;
        class_means[class_idx * num_features + feature_idx] = mean;
        class_variances[class_idx * num_features + feature_idx] = 
            (sq_sum - (sum * sum) / count) / (count - 1);
        class_counts[class_idx] = count;
    }
}
