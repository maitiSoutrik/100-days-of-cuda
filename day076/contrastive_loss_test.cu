#include "gtest/gtest.h"
#include "contrastive_loss.cuh"
#include <vector>
#include <cmath> // For sqrtf, powf
#include <numeric> // For std::iota
#include <algorithm> // For std::fill

// Helper to calculate Euclidean distance squared
float euclidean_dist_sq(const std::vector<float>& v1, const std::vector<float>& v2, int start_idx1, int start_idx2, int feature_dim) {
    float dist_sq = 0.0f;
    for (int i = 0; i < feature_dim; ++i) {
        float diff = v1[start_idx1 + i] - v2[start_idx2 + i];
        dist_sq += diff * diff;
    }
    return dist_sq;
}

// Test fixture for contrastive loss tests
class ContrastiveLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for tests if needed
    }

    void TearDown() override {
        // Common teardown for tests if needed
    }

    // Helper to run forward pass and get loss
    std::vector<float> run_forward(const std::vector<float>& h_input1,
                                   const std::vector<float>& h_input2,
                                   const std::vector<int>& h_labels,
                                   int batch_size, int feature_dim, float margin) {
        float *d_input1, *d_input2, *d_loss;
        int *d_labels_dev;

        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input1, h_input1.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input2, h_input2.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_labels_dev, h_labels.size() * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_loss, batch_size * sizeof(float)));

        CHECK_CUDA_ERROR(cudaMemcpy(d_input1, h_input1.data(), h_input1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_input2, h_input2.data(), h_input2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_labels_dev, h_labels.data(), h_labels.size() * sizeof(int), cudaMemcpyHostToDevice));

        contrastiveLossForward(d_input1, d_input2, d_labels_dev, d_loss, batch_size, feature_dim, margin);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        std::vector<float> h_loss_output(batch_size);
        CHECK_CUDA_ERROR(cudaMemcpy(h_loss_output.data(), d_loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

        CHECK_CUDA_ERROR(cudaFree(d_input1));
        CHECK_CUDA_ERROR(cudaFree(d_input2));
        CHECK_CUDA_ERROR(cudaFree(d_labels_dev));
        CHECK_CUDA_ERROR(cudaFree(d_loss));
        return h_loss_output;
    }

    // Helper to run backward pass and get gradients
    std::pair<std::vector<float>, std::vector<float>> run_backward(
        const std::vector<float>& h_input1,
        const std::vector<float>& h_input2,
        const std::vector<int>& h_labels,
        int batch_size, int feature_dim, float margin) {

        float *d_input1, *d_input2, *d_grad1, *d_grad2;
        int *d_labels_dev;

        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input1, h_input1.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input2, h_input2.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_labels_dev, h_labels.size() * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad1, h_input1.size() * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad2, h_input2.size() * sizeof(float)));
        
        CHECK_CUDA_ERROR(cudaMemcpy(d_input1, h_input1.data(), h_input1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_input2, h_input2.data(), h_input2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_labels_dev, h_labels.data(), h_labels.size() * sizeof(int), cudaMemcpyHostToDevice));

        contrastiveLossBackward(d_input1, d_input2, d_labels_dev, d_grad1, d_grad2, batch_size, feature_dim, margin);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        std::vector<float> h_grad1_output(h_input1.size());
        std::vector<float> h_grad2_output(h_input2.size());
        CHECK_CUDA_ERROR(cudaMemcpy(h_grad1_output.data(), d_grad1, h_input1.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_grad2_output.data(), d_grad2, h_input2.size() * sizeof(float), cudaMemcpyDeviceToHost));

        CHECK_CUDA_ERROR(cudaFree(d_input1));
        CHECK_CUDA_ERROR(cudaFree(d_input2));
        CHECK_CUDA_ERROR(cudaFree(d_labels_dev));
        CHECK_CUDA_ERROR(cudaFree(d_grad1));
        CHECK_CUDA_ERROR(cudaFree(d_grad2));
        return {h_grad1_output, h_grad2_output};
    }
};

TEST_F(ContrastiveLossTest, ForwardPassSimilarPair) {
    int batch_size = 1;
    int feature_dim = 2;
    float margin = 1.0f;
    std::vector<float> h_input1 = {1.0f, 2.0f};
    std::vector<float> h_input2 = {1.5f, 2.5f}; // d^2 = (0.5)^2 + (0.5)^2 = 0.25 + 0.25 = 0.5
    std::vector<int> h_labels = {1}; // Similar

    float expected_loss = (1.0f-1.5f)*(1.0f-1.5f) + (2.0f-2.5f)*(2.0f-2.5f); // 0.25 + 0.25 = 0.5

    std::vector<float> h_loss = run_forward(h_input1, h_input2, h_labels, batch_size, feature_dim, margin);
    
    ASSERT_EQ(h_loss.size(), 1);
    EXPECT_NEAR(h_loss[0], expected_loss, 1e-5);
}

TEST_F(ContrastiveLossTest, ForwardPassDissimilarPairWithinMargin) {
    int batch_size = 1;
    int feature_dim = 2;
    float margin = 1.0f;
    // d = sqrt((0.1-0.3)^2 + (0.2-0.4)^2) = sqrt((-0.2)^2 + (-0.2)^2) = sqrt(0.04+0.04) = sqrt(0.08) approx 0.2828
    // d = 0.2828. margin = 1.0. margin - d = 1.0 - 0.2828 = 0.7172
    // loss = (margin - d)^2 = (0.7172)^2 approx 0.5143
    std::vector<float> h_input1 = {0.1f, 0.2f};
    std::vector<float> h_input2 = {0.3f, 0.4f};
    std::vector<int> h_labels = {0}; // Dissimilar

    float d_sq = (0.1f-0.3f)*(0.1f-0.3f) + (0.2f-0.4f)*(0.2f-0.4f); // 0.04 + 0.04 = 0.08
    float d = sqrtf(d_sq);
    float expected_loss = powf(fmaxf(0.0f, margin - d), 2.0f);

    std::vector<float> h_loss = run_forward(h_input1, h_input2, h_labels, batch_size, feature_dim, margin);

    ASSERT_EQ(h_loss.size(), 1);
    EXPECT_NEAR(h_loss[0], expected_loss, 1e-5);
}

TEST_F(ContrastiveLossTest, ForwardPassDissimilarPairOutsideMargin) {
    int batch_size = 1;
    int feature_dim = 2;
    float margin = 1.0f;
    // d = sqrt((1-3)^2 + (2-4)^2) = sqrt((-2)^2 + (-2)^2) = sqrt(4+4) = sqrt(8) approx 2.828
    // d = 2.828. margin = 1.0. margin - d = 1.0 - 2.828 = -1.828. max(0, -1.828) = 0.
    // loss = 0^2 = 0
    std::vector<float> h_input1 = {1.0f, 2.0f};
    std::vector<float> h_input2 = {3.0f, 4.0f};
    std::vector<int> h_labels = {0}; // Dissimilar

    float d_sq = (1.0f-3.0f)*(1.0f-3.0f) + (2.0f-4.0f)*(2.0f-4.0f); // 4 + 4 = 8
    float d = sqrtf(d_sq);
    float expected_loss = powf(fmaxf(0.0f, margin - d), 2.0f); // Should be 0

    std::vector<float> h_loss = run_forward(h_input1, h_input2, h_labels, batch_size, feature_dim, margin);
    
    ASSERT_EQ(h_loss.size(), 1);
    EXPECT_NEAR(h_loss[0], expected_loss, 1e-5);
}


TEST_F(ContrastiveLossTest, BackwardPassSimilarPair) {
    int batch_size = 1;
    int feature_dim = 2;
    float margin = 1.0f;
    std::vector<float> h_input1 = {1.0f, 2.0f}; // x1
    std::vector<float> h_input2 = {1.5f, 2.5f}; // x2
    std::vector<int> h_labels = {1}; // Similar

    // Expected gradients:
    // dL/dx1_k = 2 * (x1_k - x2_k)
    // dL/dx2_k = -2 * (x1_k - x2_k)
    std::vector<float> expected_grad1 = { 2.0f * (1.0f - 1.5f), 2.0f * (2.0f - 2.5f) }; // {-1.0, -1.0}
    std::vector<float> expected_grad2 = { -2.0f * (1.0f - 1.5f), -2.0f * (2.0f - 2.5f) };// {1.0, 1.0}

    auto grads = run_backward(h_input1, h_input2, h_labels, batch_size, feature_dim, margin);
    
    ASSERT_EQ(grads.first.size(), feature_dim);
    ASSERT_EQ(grads.second.size(), feature_dim);
    for(int i=0; i<feature_dim; ++i) {
        EXPECT_NEAR(grads.first[i], expected_grad1[i], 1e-5);
        EXPECT_NEAR(grads.second[i], expected_grad2[i], 1e-5);
    }
}

TEST_F(ContrastiveLossTest, BackwardPassDissimilarPairWithinMargin) {
    int batch_size = 1;
    int feature_dim = 2;
    float margin = 1.0f;
    std::vector<float> h_input1 = {0.1f, 0.2f}; // x1
    std::vector<float> h_input2 = {0.3f, 0.4f}; // x2
    std::vector<int> h_labels = {0}; // Dissimilar

    float d_sq = (0.1f-0.3f)*(0.1f-0.3f) + (0.2f-0.4f)*(0.2f-0.4f); // 0.04 + 0.04 = 0.08
    float d = sqrtf(d_sq); // approx 0.2828427
    float epsilon = 1e-8f;

    // Expected gradients:
    // dL/dx1_k = -2 * (margin - d) * (x1_k - x2_k) / (d + epsilon)
    // dL/dx2_k =  2 * (margin - d) * (x1_k - x2_k) / (d + epsilon)
    std::vector<float> expected_grad1(feature_dim);
    std::vector<float> expected_grad2(feature_dim);

    if (d < margin) {
        float common_factor = -2.0f * (margin - d) / (d + epsilon);
        expected_grad1[0] = common_factor * (h_input1[0] - h_input2[0]);
        expected_grad1[1] = common_factor * (h_input1[1] - h_input2[1]);
        expected_grad2[0] = -expected_grad1[0];
        expected_grad2[1] = -expected_grad1[1];
    } else {
        std::fill(expected_grad1.begin(), expected_grad1.end(), 0.0f);
        std::fill(expected_grad2.begin(), expected_grad2.end(), 0.0f);
    }
    
    auto grads = run_backward(h_input1, h_input2, h_labels, batch_size, feature_dim, margin);
    
    ASSERT_EQ(grads.first.size(), feature_dim);
    ASSERT_EQ(grads.second.size(), feature_dim);
    for(int i=0; i<feature_dim; ++i) {
        EXPECT_NEAR(grads.first[i], expected_grad1[i], 1e-4); // Relaxed tolerance due to division
        EXPECT_NEAR(grads.second[i], expected_grad2[i], 1e-4);
    }
}

TEST_F(ContrastiveLossTest, BackwardPassDissimilarPairOutsideMargin) {
    int batch_size = 1;
    int feature_dim = 2;
    float margin = 1.0f;
    std::vector<float> h_input1 = {1.0f, 2.0f};
    std::vector<float> h_input2 = {3.0f, 4.0f}; // d = sqrt(8) > margin
    std::vector<int> h_labels = {0}; // Dissimilar

    // Expected gradients: 0 for both
    std::vector<float> expected_grad1 = {0.0f, 0.0f};
    std::vector<float> expected_grad2 = {0.0f, 0.0f};

    auto grads = run_backward(h_input1, h_input2, h_labels, batch_size, feature_dim, margin);
    
    ASSERT_EQ(grads.first.size(), feature_dim);
    ASSERT_EQ(grads.second.size(), feature_dim);
    for(int i=0; i<feature_dim; ++i) {
        EXPECT_NEAR(grads.first[i], expected_grad1[i], 1e-5);
        EXPECT_NEAR(grads.second[i], expected_grad2[i], 1e-5);
    }
}

// It might be good to add a test with batch_size > 1 to ensure indexing is correct.
TEST_F(ContrastiveLossTest, ForwardPassBatch) {
    int batch_size = 2;
    int feature_dim = 1;
    float margin = 1.0f;

    // Pair 0: Similar
    // x1_0 = {0.5}, x2_0 = {0.7}, label_0 = 1
    // d_0_sq = (0.5-0.7)^2 = (-0.2)^2 = 0.04. Loss_0 = 0.04
    // Pair 1: Dissimilar, d < margin
    // x1_1 = {0.1}, x2_1 = {0.2}, label_1 = 0
    // d_1_sq = (0.1-0.2)^2 = (-0.1)^2 = 0.01. d_1 = 0.1
    // Loss_1 = max(0, 1.0 - 0.1)^2 = (0.9)^2 = 0.81
    std::vector<float> h_input1 = {0.5f, 0.1f}; 
    std::vector<float> h_input2 = {0.7f, 0.2f}; 
    std::vector<int> h_labels = {1, 0}; 

    std::vector<float> expected_losses = {0.04f, 0.81f};

    std::vector<float> h_loss = run_forward(h_input1, h_input2, h_labels, batch_size, feature_dim, margin);
    
    ASSERT_EQ(h_loss.size(), batch_size);
    for(int i=0; i<batch_size; ++i) {
        EXPECT_NEAR(h_loss[i], expected_losses[i], 1e-5);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
