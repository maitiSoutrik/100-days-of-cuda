#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(error), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void calculate_class_stats(
    const float* features,
    const int* labels,
    float* means,
    float* variances,
    int* class_counts,
    int num_samples,
    int num_features,
    int num_classes) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features * num_classes) return;

    int feature_idx = idx % num_features;
    int class_idx = idx / num_features;

    float sum = 0.0f;
    float sum_squares = 0.0f;
    int count = 0;

    for (int i = 0; i < num_samples; i++) {
        if (labels[i] == class_idx) {
            float val = features[i * num_features + feature_idx];
            sum += val;
            sum_squares += val * val;
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

int main() {
    // Example data
    const int num_samples = 6;
    const int num_features = 2;
    const int num_classes = 2;

    // Sample features (2D points)
    float h_features[] = {
        1.0f, 2.0f,  // Sample 1
        2.0f, 3.0f,  // Sample 2
        0.0f, 1.0f,  // Sample 3
        5.0f, 6.0f,  // Sample 4
        4.0f, 5.0f,  // Sample 5
        6.0f, 7.0f   // Sample 6
    };

    // Labels (0 or 1)
    int h_labels[] = {0, 0, 0, 1, 1, 1};

    // Allocate device memory
    float *d_features, *d_means, *d_variances;
    int *d_labels, *d_class_counts;

    CHECK_CUDA_ERROR(cudaMalloc(&d_features, num_samples * num_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, num_samples * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_means, num_classes * num_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_variances, num_classes * num_features * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_class_counts, num_classes * sizeof(int)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_features, h_features, num_samples * num_features * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels, num_samples * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 block(256);
    dim3 grid((num_features * num_classes + block.x - 1) / block.x);
    
    calculate_class_stats<<<grid, block>>>(
        d_features,
        d_labels,
        d_means,
        d_variances,
        d_class_counts,
        num_samples,
        num_features,
        num_classes
    );

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy results back to host
    float h_means[num_classes * num_features];
    float h_variances[num_classes * num_features];
    int h_class_counts[num_classes];

    CHECK_CUDA_ERROR(cudaMemcpy(h_means, d_means, num_classes * num_features * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_variances, d_variances, num_classes * num_features * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_class_counts, d_class_counts, num_classes * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    printf("Class Counts:\n");
    for (int i = 0; i < num_classes; i++) {
        printf("Class %d: %d\n", i, h_class_counts[i]);
    }

    printf("\nMeans:\n");
    for (int c = 0; c < num_classes; c++) {
        printf("Class %d: ", c);
        for (int f = 0; f < num_features; f++) {
            printf("%.2f ", h_means[c * num_features + f]);
        }
        printf("\n");
    }

    printf("\nVariances:\n");
    for (int c = 0; c < num_classes; c++) {
        printf("Class %d: ", c);
        for (int f = 0; f < num_features; f++) {
            printf("%.2f ", h_variances[c * num_features + f]);
        }
        printf("\n");
    }

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_features));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_means));
    CHECK_CUDA_ERROR(cudaFree(d_variances));
    CHECK_CUDA_ERROR(cudaFree(d_class_counts));

    return 0;
}

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
