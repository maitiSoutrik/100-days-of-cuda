#include <iostream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void calculate_class_stats(float* features, int* labels, float* means, float* variances, int* class_counts, int num_samples, int num_features, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features * num_classes) return;

    int feature_idx = idx % num_features;
    int class_idx = idx / num_features;

    float sum = 0.0f;
    float sum_squares = 0.0f;
    int count = 0;

    for (int i = 0; i < num_samples; i++) {
        if (labels[i] == class_idx) {
            sum += features[i * num_features + feature_idx];
            sum_squares += features[i * num_features + feature_idx] * features[i * num_features + feature_idx];
            count++;
        }
    }

    if (count > 0) {
        means[idx] = sum / count;
        variances[idx] = sum_squares / count - (sum / count) * (sum / count);
        class_counts[class_idx] = count;
    } else {
        means[idx] = 0.0f;
        variances[idx] = 0.0f;
        class_counts[class_idx] = 0;
    }
}

void run_naive_bayes(const std::vector<float>& features,
                     const std::vector<int>& labels,
                     int num_classes, int num_features) {
    thrust::device_vector<float> d_features(features);
    thrust::device_vector<int> d_labels(labels);
    
    thrust::device_vector<float> d_means(num_classes * num_features);
    thrust::device_vector<float> d_variances(num_classes * num_features);
    thrust::device_vector<int> d_class_counts(num_classes);

    dim3 block(256);
    dim3 grid((num_features * num_classes + block.x - 1) / block.x);
    
    calculate_class_stats<<<grid, block>>>(
        thrust::raw_pointer_cast(d_features.data()),
        thrust::raw_pointer_cast(d_labels.data()),
        thrust::raw_pointer_cast(d_means.data()),
        thrust::raw_pointer_cast(d_variances.data()),
        thrust::raw_pointer_cast(d_class_counts.data()),
        labels.size(),
        num_features,
        num_classes
    );
    
    thrust::host_vector<float> h_means = d_means;
    thrust::host_vector<float> h_variances = d_variances;
    thrust::host_vector<int> h_counts = d_class_counts;
    
    std::cout << "Class statistics:\n";
    for (int c = 0; c < num_classes; c++) {
        std::cout << "Class " << c << " (count: " << h_counts[c] << ")\n";
        for (int f = 0; f < num_features; f++) {
            std::cout << "  Feature " << f 
                     << ": μ=" << h_means[c * num_features + f]
                     << ", σ²=" << h_variances[c * num_features + f] << "\n";
        }
    }
}

int main() {
    std::vector<float> features = {
        5.1, 3.5, 1.4, 0.2,
        4.9, 3.0, 1.4, 0.2,
        6.0, 3.0, 4.8, 1.8,
        6.7, 3.1, 5.6, 2.4
    };
    std::vector<int> labels = {0, 0, 1, 1};
    
    run_naive_bayes(features, labels, 2, 4);
    return 0;
}
