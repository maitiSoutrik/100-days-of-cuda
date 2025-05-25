#include "contrastive_loss.cuh"
#include <vector>
#include <iostream>
#include <iomanip> // For std::fixed and std::setprecision
#include <random>   // For std::default_random_engine and std::uniform_real_distribution

// Helper function to initialize data
void initialize_data(std::vector<float>& h_input1,
                     std::vector<float>& h_input2,
                     std::vector<int>& h_labels,
                     int batch_size,
                     int feature_dim) {
    std::default_random_engine generator(42); // Seed for reproducibility
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            h_input1[i * feature_dim + j] = distribution(generator);
            h_input2[i * feature_dim + j] = distribution(generator);
        }
        // Make some pairs similar and some dissimilar for testing
        if (i % 2 == 0) {
            h_labels[i] = 1; // Similar
            // For similar pairs, make input2 somewhat close to input1 for more interesting gradients
            for (int j = 0; j < feature_dim / 2; ++j) { // Modify half the features
                 h_input2[i * feature_dim + j] = h_input1[i * feature_dim + j] + distribution(generator) * 0.1f;
            }
        } else {
            h_labels[i] = 0; // Dissimilar
        }
    }
}

void print_vector(const std::string& name, const std::vector<float>& vec, int batch_size, int feature_dim, int count = 5) {
    std::cout << name << " (first " << count * feature_dim << " elements, " << count << " samples):" << std::endl;
    for (int i = 0; i < std::min(count, batch_size); ++i) {
        std::cout << "  Sample " << i << ": [";
        for (int j = 0; j < feature_dim; ++j) {
            std::cout << std::fixed << std::setprecision(4) << vec[i * feature_dim + j] << (j == feature_dim - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

void print_labels(const std::string& name, const std::vector<int>& vec, int count = 5) {
    std::cout << name << " (first " << count << " elements):" << std::endl;
    std::cout << "  [";
    for (int i = 0; i < std::min((int)vec.size(), count); ++i) {
        std::cout << vec[i] << (i == std::min((int)vec.size(), count) - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl << std::endl;
}

void print_loss(const std::string& name, const std::vector<float>& vec, int count = 5) {
    std::cout << name << " (first " << count << " elements):" << std::endl;
    std::cout << "  [";
    for (int i = 0; i < std::min((int)vec.size(), count); ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i] << (i == std::min((int)vec.size(), count) - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl << std::endl;
}


int main() {
    int batch_size = 8; // Number of pairs
    int feature_dim = 4; // Dimensionality of feature vectors
    float margin = 1.0f;   // Margin for contrastive loss

    std::cout << "Contrastive Loss CUDA Implementation Test" << std::endl;
    std::cout << "Batch Size: " << batch_size << ", Feature Dim: " << feature_dim << ", Margin: " << margin << std::endl << std::endl;

    // Host data
    std::vector<float> h_input1(batch_size * feature_dim);
    std::vector<float> h_input2(batch_size * feature_dim);
    std::vector<int> h_labels(batch_size);
    std::vector<float> h_loss(batch_size);
    std::vector<float> h_grad_input1(batch_size * feature_dim);
    std::vector<float> h_grad_input2(batch_size * feature_dim);

    initialize_data(h_input1, h_input2, h_labels, batch_size, feature_dim);

    print_vector("h_input1", h_input1, batch_size, feature_dim, batch_size);
    print_vector("h_input2", h_input2, batch_size, feature_dim, batch_size);
    print_labels("h_labels", h_labels, batch_size);

    // Device pointers
    float *d_input1, *d_input2, *d_loss, *d_grad_input1, *d_grad_input2;
    int *d_labels;

    // Allocate memory on device
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input1, batch_size * feature_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input2, batch_size * feature_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_labels, batch_size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_loss, batch_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad_input1, batch_size * feature_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad_input2, batch_size * feature_dim * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input1, h_input1.data(), batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input2, h_input2.data(), batch_size * feature_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_labels, h_labels.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice));

    // --- Forward Pass ---
    std::cout << "--- Running Forward Pass ---" << std::endl;
    contrastiveLossForward(d_input1, d_input2, d_labels, d_loss, batch_size, feature_dim, margin);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure kernel completion

    // Copy loss from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_loss.data(), d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
    print_loss("h_loss (Forward)", h_loss, batch_size);

    // --- Backward Pass ---
    std::cout << "--- Running Backward Pass ---" << std::endl;
    contrastiveLossBackward(d_input1, d_input2, d_labels, d_grad_input1, d_grad_input2, batch_size, feature_dim, margin);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure kernel completion

    // Copy gradients from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_grad_input1.data(), d_grad_input1, batch_size * feature_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_grad_input2.data(), d_grad_input2, batch_size * feature_dim * sizeof(float), cudaMemcpyDeviceToHost));

    print_vector("h_grad_input1 (Backward)", h_grad_input1, batch_size, feature_dim, batch_size);
    print_vector("h_grad_input2 (Backward)", h_grad_input2, batch_size, feature_dim, batch_size);
    
    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_input1));
    CHECK_CUDA_ERROR(cudaFree(d_input2));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_loss));
    CHECK_CUDA_ERROR(cudaFree(d_grad_input1));
    CHECK_CUDA_ERROR(cudaFree(d_grad_input2));

    std::cout << "Test finished." << std::endl;
    return 0;
}
