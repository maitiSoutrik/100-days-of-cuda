#include "gtest/gtest.h"
#include "jsd_loss.cuh"
#include <vector>
#include <cmath> // For fabsf
#include <iomanip> // For std::setprecision

// Helper to compare floats with a tolerance
::testing::AssertionResult AssertFloatsNear(float val1, float val2, float tolerance = 1e-4f) {
    if (fabsf(val1 - val2) < tolerance) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << val1 << " and " << val2 
                                         << " are not within tolerance " << tolerance
                                         << " (diff: " << fabsf(val1 - val2) << ")";
}

// Test fixture for JSD Loss tests
class JSDLossTest : public ::testing::Test {
protected:
    int num_distributions_small = 2;
    int num_elements_small = 4;
    float epsilon = 1e-7f; // Use a slightly larger epsilon for tests to avoid very small number issues

    std::vector<float> h_P_small, h_Q_small;
    float *d_P_small, *d_Q_small, *d_loss_gpu_small, *d_grad_P_small, *d_grad_Q_small;

    void SetUp() override {
        h_P_small = {0.1f, 0.2f, 0.3f, 0.4f,  // Dist 1
                     0.5f, 0.1f, 0.1f, 0.3f}; // Dist 2
        h_Q_small = {0.4f, 0.3f, 0.2f, 0.1f,  // Dist 1
                     0.2f, 0.2f, 0.3f, 0.3f}; // Dist 2
        // Ensure rows sum to 1 (already do in this example)

        size_t matrix_size_bytes = num_distributions_small * num_elements_small * sizeof(float);
        CHECK_CUDA_ERROR(cudaMalloc(&d_P_small, matrix_size_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_Q_small, matrix_size_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_loss_gpu_small, sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_grad_P_small, matrix_size_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_grad_Q_small, matrix_size_bytes));

        CHECK_CUDA_ERROR(cudaMemcpy(d_P_small, h_P_small.data(), matrix_size_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_Q_small, h_Q_small.data(), matrix_size_bytes, cudaMemcpyHostToDevice));
    }

    void TearDown() override {
        cudaFree(d_P_small);
        cudaFree(d_Q_small);
        cudaFree(d_loss_gpu_small);
        cudaFree(d_grad_P_small);
        cudaFree(d_grad_Q_small);
    }
};

TEST_F(JSDLossTest, CPU_ForwardPass_Beta0_5_Symmetric) {
    float beta = 0.5f;
    float cpu_loss = jsd_loss_forward_cpu(h_P_small, h_Q_small, num_distributions_small, num_elements_small, beta, epsilon);
    
    // Manual calculation for one element of first distribution for sanity check:
    // P1 = [0.1, 0.2, 0.3, 0.4], Q1 = [0.4, 0.3, 0.2, 0.1]
    // For p1=0.1, q1=0.4: m1 = 0.5*(0.1+0.4) = 0.25
    // kl_p_m = 0.1 * log(0.1/0.25) = 0.1 * log(0.4) = 0.1 * -0.91629 = -0.091629
    // kl_q_m = 0.4 * log(0.4/0.25) = 0.4 * log(1.6) = 0.4 * 0.47000 = 0.18800
    // jsd_contrib = 0.5 * (-0.091629) + 0.5 * (0.18800) = 0.5 * 0.096371 = 0.0481855
    // This is just one element. The actual CPU loss is a sum over all.
    // Expected value based on a separate calculation/known good run would be better.
    // For now, just check it's non-negative.
    ASSERT_GE(cpu_loss, 0.0f);
    // A more concrete expected value: (Using an external calculator for this specific input)
    // For P1=[0.1,0.2,0.3,0.4], Q1=[0.4,0.3,0.2,0.1], M1=[0.25,0.25,0.25,0.25]
    // D(P1||M1) = 0.1ln(0.1/0.25)+0.2ln(0.2/0.25)+0.3ln(0.3/0.25)+0.4ln(0.4/0.25)
    //           = 0.1(-0.9163)+0.2(-0.2231)+0.3(0.1823)+0.4(0.4700)
    //           = -0.09163 -0.04462 +0.05469 +0.18800 = 0.10644
    // D(Q1||M1) = 0.4ln(0.4/0.25)+0.3ln(0.3/0.25)+0.2ln(0.2/0.25)+0.1ln(0.1/0.25)
    //           = 0.4(0.4700)+0.3(0.1823)+0.2(-0.2231)+0.1(-0.9163)
    //           = 0.18800 +0.05469 -0.04462 -0.09163 = 0.10644
    // JSD1 = 0.5 * 0.10644 + 0.5 * 0.10644 = 0.10644
    // For P2=[0.5,0.1,0.1,0.3], Q2=[0.2,0.2,0.3,0.3], M2=[0.35,0.15,0.20,0.30]
    // D(P2||M2) = 0.5ln(0.5/0.35)+0.1ln(0.1/0.15)+0.1ln(0.1/0.20)+0.3ln(0.3/0.30)
    //           = 0.5(0.3567)+0.1(-0.4055)+0.1(-0.6931)+0.3(0)
    //           = 0.17835 -0.04055 -0.06931 +0 = 0.06849
    // D(Q2||M2) = 0.2ln(0.2/0.35)+0.2ln(0.2/0.15)+0.3ln(0.3/0.20)+0.3ln(0.3/0.30)
    //           = 0.2(-0.5596)+0.2(0.2877)+0.3(0.4055)+0.3(0)
    //           = -0.11192 +0.05754 +0.12165 +0 = 0.06727
    // JSD2 = 0.5 * 0.06849 + 0.5 * 0.06727 = 0.5 * 0.13576 = 0.06788
    // Total JSD = 0.10644 + 0.06788 = 0.17432
    EXPECT_TRUE(AssertFloatsNear(cpu_loss, 0.17432f, 1e-5f));
}

TEST_F(JSDLossTest, GPU_vs_CPU_ForwardPass_Beta0_5) {
    float beta = 0.5f;
    float cpu_loss = jsd_loss_forward_cpu(h_P_small, h_Q_small, num_distributions_small, num_elements_small, beta, epsilon);

    jsd_loss_gpu(d_P_small, d_Q_small, d_loss_gpu_small, d_grad_P_small, d_grad_Q_small, 
                 num_distributions_small, num_elements_small, beta, epsilon);
    float h_gpu_loss;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_gpu_loss, d_loss_gpu_small, sizeof(float), cudaMemcpyDeviceToHost));
    
    EXPECT_TRUE(AssertFloatsNear(h_gpu_loss, cpu_loss, 1e-4f)) 
        << "GPU loss: " << h_gpu_loss << ", CPU loss: " << cpu_loss;
}

TEST_F(JSDLossTest, GPU_vs_CPU_ForwardPass_Beta1_0) { // P || M
    float beta = 1.0f;
    float cpu_loss = jsd_loss_forward_cpu(h_P_small, h_Q_small, num_distributions_small, num_elements_small, beta, epsilon);
    // Expected for beta = 1.0 (sum of D(P||M)): 0.10644 (for row1) + 0.06849 (for row2) = 0.17493
    EXPECT_TRUE(AssertFloatsNear(cpu_loss, 0.17493f, 1e-5f));


    jsd_loss_gpu(d_P_small, d_Q_small, d_loss_gpu_small, d_grad_P_small, d_grad_Q_small, 
                 num_distributions_small, num_elements_small, beta, epsilon);
    float h_gpu_loss;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_gpu_loss, d_loss_gpu_small, sizeof(float), cudaMemcpyDeviceToHost));
    
    EXPECT_TRUE(AssertFloatsNear(h_gpu_loss, cpu_loss, 1e-4f))
        << "GPU loss: " << h_gpu_loss << ", CPU loss: " << cpu_loss;
}

TEST_F(JSDLossTest, GPU_vs_CPU_ForwardPass_Beta0_0) { // Q || M
    float beta = 0.0f;
    float cpu_loss = jsd_loss_forward_cpu(h_P_small, h_Q_small, num_distributions_small, num_elements_small, beta, epsilon);
    // Expected for beta = 0.0 (sum of D(Q||M)): 0.10644 (for row1) + 0.06727 (for row2) = 0.17371
    EXPECT_TRUE(AssertFloatsNear(cpu_loss, 0.17371f, 1e-5f));

    jsd_loss_gpu(d_P_small, d_Q_small, d_loss_gpu_small, d_grad_P_small, d_grad_Q_small, 
                 num_distributions_small, num_elements_small, beta, epsilon);
    float h_gpu_loss;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_gpu_loss, d_loss_gpu_small, sizeof(float), cudaMemcpyDeviceToHost));
    
    EXPECT_TRUE(AssertFloatsNear(h_gpu_loss, cpu_loss, 1e-4f))
        << "GPU loss: " << h_gpu_loss << ", CPU loss: " << cpu_loss;
}

TEST_F(JSDLossTest, GradientsNonZero_Beta0_5) {
    float beta = 0.5f;
    jsd_loss_gpu(d_P_small, d_Q_small, d_loss_gpu_small, d_grad_P_small, d_grad_Q_small, 
                 num_distributions_small, num_elements_small, beta, epsilon);
    
    std::vector<float> h_grad_P_gpu(num_distributions_small * num_elements_small);
    std::vector<float> h_grad_Q_gpu(num_distributions_small * num_elements_small);
    CHECK_CUDA_ERROR(cudaMemcpy(h_grad_P_gpu.data(), d_grad_P_small, h_grad_P_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_grad_Q_gpu.data(), d_grad_Q_small, h_grad_Q_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool p_grad_non_zero = false;
    bool q_grad_non_zero = false;
    for(size_t i = 0; i < h_grad_P_gpu.size(); ++i) {
        if (fabsf(h_grad_P_gpu[i]) > 1e-7) p_grad_non_zero = true;
        if (fabsf(h_grad_Q_gpu[i]) > 1e-7) q_grad_non_zero = true;
    }
    EXPECT_TRUE(p_grad_non_zero) << "Gradients w.r.t P are all zero.";
    EXPECT_TRUE(q_grad_non_zero) << "Gradients w.r.t Q are all zero.";

    // Example gradient check for one element (P[0][0] = 0.1, Q[0][0] = 0.4, M[0][0] = 0.25)
    // For beta = 0.5:
    // log_p_m = log(0.1/0.25) = -0.91629
    // log_q_m = log(0.4/0.25) = 0.47000
    // grad_kl_p_m_dp = -0.91629 + 1 - 0.5 * (0.1/0.25) = -0.91629 + 1 - 0.2 = -0.11629
    // grad_kl_q_m_dp = -0.5 * (0.4/0.25) = -0.5 * 1.6 = -0.8
    // grad_p = 0.5 * (-0.11629) + 0.5 * (-0.8) = 0.5 * (-0.91629) = -0.458145
    // This is a rough check, actual values can be complex.
    // std::cout << "h_grad_P_gpu[0] = " << h_grad_P_gpu[0] << std::endl;
    EXPECT_NE(h_grad_P_gpu[0], 0.0f);
}


// Test with identical distributions P = Q
TEST_F(JSDLossTest, IdenticalDistributions_LossIsZero) {
    std::vector<float> h_P_identical = {0.25f, 0.25f, 0.25f, 0.25f, 0.5f, 0.5f, 0.0f, 0.0f};
    // For the second dist, make Q2 same as P2 to test 0 loss for that row
    // h_Q_small was {0.4f, 0.3f, 0.2f, 0.1f,  0.2f, 0.2f, 0.3f, 0.3f};
    // Let's make h_Q_identical same as h_P_identical
    std::vector<float> h_Q_identical = h_P_identical;


    CHECK_CUDA_ERROR(cudaMemcpy(d_P_small, h_P_identical.data(), h_P_identical.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_Q_small, h_Q_identical.data(), h_Q_identical.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    float beta = 0.5f; // Symmetric JSD
    float cpu_loss = jsd_loss_forward_cpu(h_P_identical, h_Q_identical, num_distributions_small, num_elements_small, beta, epsilon);
    EXPECT_TRUE(AssertFloatsNear(cpu_loss, 0.0f, 1e-6f)) << "CPU loss for identical P,Q should be 0, but was " << cpu_loss;

    jsd_loss_gpu(d_P_small, d_Q_small, d_loss_gpu_small, d_grad_P_small, d_grad_Q_small, 
                 num_distributions_small, num_elements_small, beta, epsilon);
    float h_gpu_loss;
    CHECK_CUDA_ERROR(cudaMemcpy(&h_gpu_loss, d_loss_gpu_small, sizeof(float), cudaMemcpyDeviceToHost));
    EXPECT_TRUE(AssertFloatsNear(h_gpu_loss, 0.0f, 1e-6f)) << "GPU loss for identical P,Q should be 0, but was " << h_gpu_loss;
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
