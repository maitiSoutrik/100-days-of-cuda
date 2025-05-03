#include "adahessian.h"
#include <gtest/gtest.h> // Include Google Test header
#include <vector>
#include <cmath> // For fabsf

// CPU version for verification (single element) - Keep this helper
void adaHessianUpdateCPU(
    float* theta,
    const float grad,
    const float gradPerturbed,
    float* m,
    float* v,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float delta
) {
    float h_diag = (delta != 0.0f) ? (gradPerturbed - grad) / delta : 0.0f;
    *m = beta1 * (*m) + (1.0f - beta1) * grad;
    *v = beta2 * (*v) + (1.0f - beta2) * (h_diag * h_diag);
    float denom = sqrtf(*v) + epsilon;
    if (denom != 0.0f) {
        *theta -= lr * (*m) / denom;
    }
}

// Define a test fixture class if we need setup/teardown (optional for this simple case)
class AdaHessianTest : public ::testing::Test {
protected:
    // Test parameters
    const int N = 10;
    const float lr = 0.01f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-7f;
    const float delta = 1e-4f;
    const float tolerance = 1e-6f;
    size_t bytes;

    // Host arrays
    std::vector<float> h_theta, h_grad, h_gradPerturbed, h_m, h_v;
    std::vector<float> h_theta_initial, h_m_initial, h_v_initial;
    std::vector<float> h_theta_gpu, h_m_gpu, h_v_gpu; // To store GPU results

    // Device pointers
    float *d_theta = nullptr, *d_grad = nullptr, *d_gradPerturbed = nullptr, *d_m = nullptr, *d_v = nullptr;

    // Setup runs before each test in the fixture
    void SetUp() override {
        bytes = N * sizeof(float);

        // Resize vectors
        h_theta.resize(N); h_grad.resize(N); h_gradPerturbed.resize(N);
        h_m.resize(N); h_v.resize(N);
        h_theta_initial.resize(N); h_m_initial.resize(N); h_v_initial.resize(N);
        h_theta_gpu.resize(N); h_m_gpu.resize(N); h_v_gpu.resize(N);

        // Initialize arrays with simple, predictable data
        for (int i = 0; i < N; i++) {
            h_theta_initial[i] = h_theta[i] = 1.0f;
            h_grad[i] = 0.1f * (i + 1); // Simple gradient
            h_gradPerturbed[i] = h_grad[i] + (0.05f * (i+1)) * delta; // Simple perturbation
            h_m_initial[i] = h_m[i] = 0.0f;
            h_v_initial[i] = h_v[i] = 0.0f;
        }

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_theta, bytes));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_grad, bytes));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gradPerturbed, bytes));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_m, bytes));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_v, bytes));

        // Copy initial data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_theta, h_theta.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_grad, h_grad.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_gradPerturbed, h_gradPerturbed.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_m, h_m.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_v, h_v.data(), bytes, cudaMemcpyHostToDevice));
    }

    // TearDown runs after each test in the fixture
    void TearDown() override {
        // Free GPU memory
        if (d_theta) CHECK_CUDA_ERROR(cudaFree(d_theta));
        if (d_grad) CHECK_CUDA_ERROR(cudaFree(d_grad));
        if (d_gradPerturbed) CHECK_CUDA_ERROR(cudaFree(d_gradPerturbed));
        if (d_m) CHECK_CUDA_ERROR(cudaFree(d_m));
        if (d_v) CHECK_CUDA_ERROR(cudaFree(d_v));
    }
};

// Define the test case using the fixture
TEST_F(AdaHessianTest, BasicUpdateVerification) {
    // Launch the kernel
    int blockSize = 32; // Smaller block size for small N
    int gridSize = (N + blockSize - 1) / blockSize;
    adaHessianUpdateKernel<<<gridSize, blockSize>>>(
        d_theta, d_grad, d_gradPerturbed, d_m, d_v,
        lr, beta1, beta2, epsilon, delta, N
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel completion

    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_theta_gpu.data(), d_theta, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_m_gpu.data(), d_m, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_v_gpu.data(), d_v, bytes, cudaMemcpyDeviceToHost));

    // --- CPU Verification & Comparison (element by element) ---
    for(int i = 0; i < N; ++i) {
        float theta_cpu = h_theta_initial[i];
        float m_cpu = h_m_initial[i];
        float v_cpu = h_v_initial[i];

        adaHessianUpdateCPU(&theta_cpu, h_grad[i], h_gradPerturbed[i], &m_cpu, &v_cpu,
                            lr, beta1, beta2, epsilon, delta);

        // Use Google Test assertions
        ASSERT_NEAR(h_theta_gpu[i], theta_cpu, tolerance) << "Mismatch in theta at index " << i;
        ASSERT_NEAR(h_m_gpu[i], m_cpu, tolerance) << "Mismatch in m at index " << i;
        ASSERT_NEAR(h_v_gpu[i], v_cpu, tolerance) << "Mismatch in v at index " << i;
    }
}

// Main function to run the tests (provided by gtest_main)
// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
// We link against gtest_main, so we don't need the main function here.
