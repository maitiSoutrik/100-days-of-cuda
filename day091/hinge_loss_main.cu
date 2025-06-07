#include "hinge_loss.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip> // For std::fixed and std::setprecision

// Helper function to print arrays (for debugging/verification)
template<typename T>
void print_array(const T* arr, int size, const std::string& label) {
    std::cout << label << ": [";
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << (i == size - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

// CPU implementation of Hinge Loss for verification
void hinge_loss_cpu(const int* true_labels, const float* pred_scores, float* loss, int num_elements) {
    for (int i = 0; i < num_elements; ++i) {
        float t_y = (float)true_labels[i] * pred_scores[i];
        loss[i] = std::max(0.0f, 1.0f - t_y);
    }
}

float sum_hinge_loss_cpu(const int* true_labels, const float* pred_scores, int num_elements) {
    double total_loss = 0.0; // Use double for precision
    for (int i = 0; i < num_elements; ++i) {
        float t_y = (float)true_labels[i] * pred_scores[i];
        total_loss += std::max(0.0f, 1.0f - t_y);
    }
    return (float)total_loss;
}


int main() {
    const int num_elements = 16; // Example size

    // Host data
    std::vector<int> h_true_labels(num_elements);
    std::vector<float> h_pred_scores(num_elements);
    std::vector<float> h_loss_gpu(num_elements);
    std::vector<float> h_loss_cpu(num_elements);
    float h_total_loss_gpu = 0.0f;
    float h_total_loss_cpu = 0.0f;

    // Initialize random data
    std::mt19937 rng(123); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> label_dist(0, 1); // For 0 or 1, then map to -1 or 1
    std::uniform_real_distribution<float> score_dist(-2.0f, 2.0f);

    std::cout << "Generating input data..." << std::endl;
    for (int i = 0; i < num_elements; ++i) {
        h_true_labels[i] = (label_dist(rng) == 0) ? -1 : 1;
        h_pred_scores[i] = score_dist(rng);
    }

    print_array(h_true_labels.data(), num_elements, "True Labels (Host)");
    print_array(h_pred_scores.data(), num_elements, "Predicted Scores (Host)");

    // Device data pointers
    int* d_true_labels = nullptr;
    float* d_pred_scores = nullptr;
    float* d_loss = nullptr;
    float* d_total_loss = nullptr;
    float* d_temp_storage_for_sum = nullptr; // For sum_hinge_loss_cuda

    // Allocate memory on device
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_true_labels, num_elements * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pred_scores, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_loss, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_total_loss, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_temp_storage_for_sum, num_elements * sizeof(float))); // Used to store individual losses before CPU sum

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_true_labels, h_true_labels.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pred_scores, h_pred_scores.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

    // --- Test 1: Compute individual Hinge Losses ---
    std::cout << "\n--- Testing Individual Hinge Loss Computation ---" << std::endl;
    hinge_loss_cuda(d_true_labels, d_pred_scores, d_loss, num_elements);
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_loss_gpu.data(), d_loss, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute on CPU for verification
    hinge_loss_cpu(h_true_labels.data(), h_pred_scores.data(), h_loss_cpu.data(), num_elements);

    // Print and compare results
    std::cout << std::fixed << std::setprecision(4);
    print_array(h_loss_gpu.data(), num_elements, "Individual Losses (GPU)");
    print_array(h_loss_cpu.data(), num_elements, "Individual Losses (CPU - Verification)");

    bool individual_loss_match = true;
    for (int i = 0; i < num_elements; ++i) {
        if (std::abs(h_loss_gpu[i] - h_loss_cpu[i]) > 1e-5) {
            individual_loss_match = false;
            break;
        }
    }
    std::cout << "Individual losses match CPU: " << (individual_loss_match ? "Yes" : "No") << std::endl;


    // --- Test 2: Compute Sum of Hinge Losses ---
    std::cout << "\n--- Testing Sum of Hinge Losses ---" << std::endl;
    sum_hinge_loss_cuda(d_true_labels, d_pred_scores, d_total_loss, num_elements, d_temp_storage_for_sum);

    // Copy total loss back to host
    CHECK_CUDA_ERROR(cudaMemcpy(&h_total_loss_gpu, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost));

    // Compute sum on CPU for verification
    h_total_loss_cpu = sum_hinge_loss_cpu(h_true_labels.data(), h_pred_scores.data(), num_elements);
    
    std::cout << "Total Hinge Loss (GPU): " << h_total_loss_gpu << std::endl;
    std::cout << "Total Hinge Loss (CPU - Verification): " << h_total_loss_cpu << std::endl;
    
    bool total_loss_match = std::abs(h_total_loss_gpu - h_total_loss_cpu) < 1e-4; // Allow slightly larger tolerance for sum
    std::cout << "Total loss matches CPU: " << (total_loss_match ? "Yes" : "No") << std::endl;


    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_true_labels));
    CHECK_CUDA_ERROR(cudaFree(d_pred_scores));
    CHECK_CUDA_ERROR(cudaFree(d_loss));
    CHECK_CUDA_ERROR(cudaFree(d_total_loss));
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage_for_sum));

    std::cout << "\nDemonstration finished." << std::endl;

    if (individual_loss_match && total_loss_match) {
        return 0; // Success
    } else {
        return 1; // Failure
    }
}
