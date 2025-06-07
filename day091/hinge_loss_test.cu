#include "gtest/gtest.h"
#include "hinge_loss.cuh"
#include <vector>
#include <random>
#include <algorithm> // For std::max

// CPU implementation of Hinge Loss for verification within tests
void hinge_loss_cpu_test_version(const int* true_labels, const float* pred_scores, float* loss, int num_elements) {
    for (int i = 0; i < num_elements; ++i) {
        float t_y = (float)true_labels[i] * pred_scores[i];
        loss[i] = std::max(0.0f, 1.0f - t_y);
    }
}

float sum_hinge_loss_cpu_test_version(const int* true_labels, const float* pred_scores, int num_elements) {
    double total_loss = 0.0;
    for (int i = 0; i < num_elements; ++i) {
        float t_y = (float)true_labels[i] * pred_scores[i];
        total_loss += std::max(0.0f, 1.0f - t_y);
    }
    return (float)total_loss;
}

TEST(HingeLossTest, IndividualLossesSmall) {
    const int num_elements = 8;
    std::vector<int> h_true_labels = {-1, 1, -1, 1, -1, 1, 1, -1};
    std::vector<float> h_pred_scores = {-0.5f, 1.2f, 0.3f, -0.8f, 2.0f, 0.1f, -1.5f, -1.2f};
    std::vector<float> h_loss_gpu(num_elements);
    std::vector<float> h_loss_cpu(num_elements);

    int* d_true_labels;
    float* d_pred_scores;
    float* d_loss;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_true_labels, num_elements * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pred_scores, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_loss, num_elements * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_true_labels, h_true_labels.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pred_scores, h_pred_scores.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

    hinge_loss_cuda(d_true_labels, d_pred_scores, d_loss, num_elements);
    CHECK_CUDA_ERROR(cudaMemcpy(h_loss_gpu.data(), d_loss, num_elements * sizeof(float), cudaMemcpyDeviceToHost));

    hinge_loss_cpu_test_version(h_true_labels.data(), h_pred_scores.data(), h_loss_cpu.data(), num_elements);

    for (int i = 0; i < num_elements; ++i) {
        ASSERT_NEAR(h_loss_gpu[i], h_loss_cpu[i], 1e-5);
    }

    CHECK_CUDA_ERROR(cudaFree(d_true_labels));
    CHECK_CUDA_ERROR(cudaFree(d_pred_scores));
    CHECK_CUDA_ERROR(cudaFree(d_loss));
}

TEST(HingeLossTest, SumLossSmall) {
    const int num_elements = 8;
    std::vector<int> h_true_labels = {-1, 1, -1, 1, -1, 1, 1, -1};
    std::vector<float> h_pred_scores = {-0.5f, 1.2f, 0.3f, -0.8f, 2.0f, 0.1f, -1.5f, -1.2f};
    float h_total_loss_gpu;
    float h_total_loss_cpu;

    int* d_true_labels;
    float* d_pred_scores;
    float* d_total_loss;
    float* d_temp_storage;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_true_labels, num_elements * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pred_scores, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_total_loss, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_temp_storage, num_elements * sizeof(float)));


    CHECK_CUDA_ERROR(cudaMemcpy(d_true_labels, h_true_labels.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pred_scores, h_pred_scores.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

    sum_hinge_loss_cuda(d_true_labels, d_pred_scores, d_total_loss, num_elements, d_temp_storage);
    CHECK_CUDA_ERROR(cudaMemcpy(&h_total_loss_gpu, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost));

    h_total_loss_cpu = sum_hinge_loss_cpu_test_version(h_true_labels.data(), h_pred_scores.data(), num_elements);

    ASSERT_NEAR(h_total_loss_gpu, h_total_loss_cpu, 1e-4);

    CHECK_CUDA_ERROR(cudaFree(d_true_labels));
    CHECK_CUDA_ERROR(cudaFree(d_pred_scores));
    CHECK_CUDA_ERROR(cudaFree(d_total_loss));
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage));
}

TEST(HingeLossTest, AllCorrectOutsideMargin) {
    const int num_elements = 4;
    std::vector<int> h_true_labels = {1, -1, 1, -1};
    std::vector<float> h_pred_scores = {1.5f, -1.2f, 2.0f, -1.8f}; // All t*y >= 1
    float h_total_loss_gpu;
    float h_total_loss_cpu;

    int* d_true_labels;
    float* d_pred_scores;
    float* d_total_loss;
    float* d_temp_storage;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_true_labels, num_elements * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pred_scores, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_total_loss, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_temp_storage, num_elements * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_true_labels, h_true_labels.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pred_scores, h_pred_scores.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

    sum_hinge_loss_cuda(d_true_labels, d_pred_scores, d_total_loss, num_elements, d_temp_storage);
    CHECK_CUDA_ERROR(cudaMemcpy(&h_total_loss_gpu, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    h_total_loss_cpu = sum_hinge_loss_cpu_test_version(h_true_labels.data(), h_pred_scores.data(), num_elements);

    ASSERT_NEAR(h_total_loss_gpu, 0.0f, 1e-5);
    ASSERT_NEAR(h_total_loss_cpu, 0.0f, 1e-5);


    CHECK_CUDA_ERROR(cudaFree(d_true_labels));
    CHECK_CUDA_ERROR(cudaFree(d_pred_scores));
    CHECK_CUDA_ERROR(cudaFree(d_total_loss));
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage));
}


TEST(HingeLossTest, AllMisclassifiedOrInMargin) {
    const int num_elements = 4;
    std::vector<int> h_true_labels = {1, -1, 1, -1};
    // t*y values: 1*0.2=0.2, (-1)*0.5 = -0.5, 1*(-0.1)=-0.1, (-1)*(-0.3)=0.3
    // losses: 1-0.2=0.8, 1-(-0.5)=1.5, 1-(-0.1)=1.1, 1-0.3=0.7
    // sum = 0.8 + 1.5 + 1.1 + 0.7 = 4.1
    std::vector<float> h_pred_scores = {0.2f, 0.5f, -0.1f, -0.3f}; 
    float h_total_loss_gpu;
    float h_total_loss_cpu;

    int* d_true_labels;
    float* d_pred_scores;
    float* d_total_loss;
    float* d_temp_storage;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_true_labels, num_elements * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pred_scores, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_total_loss, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_temp_storage, num_elements * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_true_labels, h_true_labels.data(), num_elements * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pred_scores, h_pred_scores.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

    sum_hinge_loss_cuda(d_true_labels, d_pred_scores, d_total_loss, num_elements, d_temp_storage);
    CHECK_CUDA_ERROR(cudaMemcpy(&h_total_loss_gpu, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    h_total_loss_cpu = sum_hinge_loss_cpu_test_version(h_true_labels.data(), h_pred_scores.data(), num_elements);
    
    ASSERT_NEAR(h_total_loss_gpu, 4.1f, 1e-4);
    ASSERT_NEAR(h_total_loss_cpu, 4.1f, 1e-5);

    CHECK_CUDA_ERROR(cudaFree(d_true_labels));
    CHECK_CUDA_ERROR(cudaFree(d_pred_scores));
    CHECK_CUDA_ERROR(cudaFree(d_total_loss));
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage));
}

// Entry point for Google Test
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
// The main for gtest is usually linked by GTest::gtest_main
