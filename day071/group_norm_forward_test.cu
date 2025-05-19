#include "group_norm_forward.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath> // For fabsf
#include <algorithm> // For std::generate, std::fill

// Helper to initialize data for tests
void initializeTestData(float* data, int size, float val_start = 0.1f, float val_step = 0.1f) {
    for (int i = 0; i < size; ++i) {
        data[i] = val_start + i * val_step;
    }
}

void initializeConstantData(float* data, int size, float val) {
    std::fill(data, data + size, val);
}


// Test fixture for GroupNorm tests
class GroupNormTest : public ::testing::Test {
protected:
    int N, C, H, W, G;
    float epsilon;
    std::vector<float> h_input, h_gamma, h_beta;
    std::vector<float> h_output_gpu, h_output_cpu;

    float *d_input, *d_gamma, *d_beta, *d_output;

    void SetUp(int n, int c, int h, int w, int g, float eps = 1e-5f) {
        N = n; C = c; H = h; W = w; G = g; epsilon = eps;

        if (C % G != 0) {
            FAIL() << "Number of channels C (" << C << ") must be divisible by G (" << G << ")";
        }

        int input_size = N * C * H * W;
        int params_size = C;

        h_input.resize(input_size);
        h_gamma.resize(params_size);
        h_beta.resize(params_size);
        h_output_gpu.resize(input_size);
        h_output_cpu.resize(input_size);

        initializeTestData(h_input.data(), input_size, 0.1f, 0.05f);
        initializeConstantData(h_gamma.data(), params_size, 1.0f); // Gamma = 1
        initializeConstantData(h_beta.data(), params_size, 0.0f);  // Beta = 0

        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, input_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gamma, params_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_beta, params_size * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, input_size * sizeof(float)));

        CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_gamma, h_gamma.data(), params_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_beta, h_beta.data(), params_size * sizeof(float), cudaMemcpyHostToDevice));
    }

    void TearDown() override {
        CHECK_CUDA_ERROR(cudaFree(d_input));
        CHECK_CUDA_ERROR(cudaFree(d_gamma));
        CHECK_CUDA_ERROR(cudaFree(d_beta));
        CHECK_CUDA_ERROR(cudaFree(d_output));
    }

    void RunAndCompare(float tolerance = 1e-4f) {
        // Run GPU version
        groupNormForward(d_output, d_input, N, C, H, W, G, d_gamma, d_beta, epsilon);
        CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, N * C * H * W * sizeof(float), cudaMemcpyDeviceToHost));

        // Run CPU version
        groupNormForwardCPU(h_output_cpu.data(), h_input.data(), N, C, H, W, G, h_gamma.data(), h_beta.data(), epsilon);

        // Compare
        for (int i = 0; i < N * C * H * W; ++i) {
            ASSERT_NEAR(h_output_cpu[i], h_output_gpu[i], tolerance)
                << "Mismatch at index " << i;
        }
    }
};

TEST_F(GroupNormTest, BasicTest) {
    SetUp(2, 4, 2, 2, 2); // N=2, C=4, H=2, W=2, G=2
    RunAndCompare();
}

TEST_F(GroupNormTest, SingleGroup) { // Equivalent to Layer Normalization (almost)
    SetUp(1, 8, 4, 4, 1); // N=1, C=8, H=4, W=4, G=1
    RunAndCompare();
}

TEST_F(GroupNormTest, GroupsEqualToChannels) { // Equivalent to Instance Normalization
    SetUp(1, 4, 3, 3, 4); // N=1, C=4, H=3, W=3, G=4
    RunAndCompare();
}

TEST_F(GroupNormTest, LargerDimensions) {
    SetUp(4, 16, 8, 8, 4); // N=4, C=16, H=8, W=8, G=4
    RunAndCompare();
}

TEST_F(GroupNormTest, NonUnitGammaBeta) {
    SetUp(2, 6, 2, 2, 3); // N=2, C=6, H=2, W=2, G=3
    initializeTestData(h_gamma.data(), C, 0.5f, 0.1f);
    initializeTestData(h_beta.data(), C, -0.2f, 0.05f);
    CHECK_CUDA_ERROR(cudaMemcpy(d_gamma, h_gamma.data(), C * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_beta, h_beta.data(), C * sizeof(float), cudaMemcpyHostToDevice));
    RunAndCompare();
}

TEST_F(GroupNormTest, MinimalDimensions) {
    SetUp(1, 2, 1, 1, 1); // N=1, C=2, H=1, W=1, G=1
    RunAndCompare();
}

TEST_F(GroupNormTest, MinimalDimensionsInstanceNormLike) {
    SetUp(1, 2, 1, 1, 2); // N=1, C=2, H=1, W=1, G=2 (each channel is a group)
    RunAndCompare();
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
