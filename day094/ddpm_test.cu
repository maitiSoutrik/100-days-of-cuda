#include "ddpm_kernels.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm> // For std::all_of

// Helper to compare two float vectors with a tolerance
::testing::AssertionResult AreVectorsNear(const std::vector<float>& vec1, const std::vector<float>& vec2, float tolerance) {
    if (vec1.size() != vec2.size()) {
        return ::testing::AssertionFailure() << "Vector sizes differ: " << vec1.size() << " vs " << vec2.size();
    }
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::abs(vec1[i] - vec2[i]) > tolerance) {
            return ::testing::AssertionFailure() << "Mismatch at index " << i << ": " << vec1[i] << " vs " << vec2[i]
                                                 << ", diff " << std::abs(vec1[i] - vec2[i]);
        }
    }
    return ::testing::AssertionSuccess();
}

class DDPMKernelsTest : public ::testing::Test {
protected:
    const int N = 256 * 256; // Test with a smaller N for speed
    const int threads_per_block_val = 256;
    dim3 threads_per_block;
    dim3 num_blocks;

    float *d_x_in, *d_x_out, *d_epsilon_pred;
    curandState *d_states;
    std::vector<float> h_x_in, h_x_out_gpu;

    DDPMKernelsTest() : threads_per_block(threads_per_block_val), 
                        num_blocks((N + threads_per_block_val - 1) / threads_per_block_val) {}

    void SetUp() override {
        CHECK_CUDA_ERROR(cudaMalloc(&d_x_in, N * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_x_out, N * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_epsilon_pred, N * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_states, N * sizeof(curandState)));

        h_x_in.resize(N);
        h_x_out_gpu.resize(N);

        // Initialize input data (e.g., all ones)
        std::fill(h_x_in.begin(), h_x_in.end(), 1.0f);
        CHECK_CUDA_ERROR(cudaMemcpy(d_x_in, h_x_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        // Initialize cuRAND states
        unsigned long seed = 4321UL;
        launch_setup_curand_states(d_states, seed, N, threads_per_block, num_blocks);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    void TearDown() override {
        CHECK_CUDA_ERROR(cudaFree(d_x_in));
        CHECK_CUDA_ERROR(cudaFree(d_x_out));
        CHECK_CUDA_ERROR(cudaFree(d_epsilon_pred));
        CHECK_CUDA_ERROR(cudaFree(d_states));
    }
};

TEST_F(DDPMKernelsTest, ForwardDiffusionStep) {
    float beta_t = 0.01f; // Example beta

    launch_forward_diffusion_step(d_x_in, d_x_out, beta_t, d_states, N, threads_per_block, num_blocks);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(h_x_out_gpu.data(), d_x_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Basic checks:
    // 1. Output should be different from input if beta_t > 0
    // 2. Values should be somewhat reasonable (not NaN or Inf)
    // More rigorous tests would involve checking statistical properties or comparing with a CPU implementation.

    bool changed = false;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(h_x_out_gpu[i] - h_x_in[i]) > 1e-6) {
            changed = true;
            break;
        }
    }
    if (beta_t > 0) {
      ASSERT_TRUE(changed) << "Output data did not change after forward diffusion with beta_t > 0.";
    }


    ASSERT_TRUE(std::all_of(h_x_out_gpu.begin(), h_x_out_gpu.end(), [](float val){
        return std::isfinite(val);
    })) << "Output contains non-finite values.";

    // Check mean and stddev (approximate)
    // Expected mean for x_t = sqrt(1-beta_t) * mean(x_{t-1})
    // Expected variance for x_t = (1-beta_t) * var(x_{t-1}) + beta_t (if var(x_{t-1}) is small and mean is 0)
    // For x_in all 1.0f, mean(x_in) = 1.0, var(x_in) = 0.0
    // Expected mean(x_out) approx sqrt(1-beta_t) * 1.0
    // Expected stddev(x_out) approx sqrt(beta_t)
    
    double sum = 0.0;
    for(float val : h_x_out_gpu) sum += val;
    double mean_gpu = sum / N;

    double sq_sum_diff = 0.0;
    for(float val : h_x_out_gpu) sq_sum_diff += (val - mean_gpu) * (val - mean_gpu);
    double stddev_gpu = std::sqrt(sq_sum_diff / N);

    float expected_mean = sqrtf(1.0f - beta_t) * 1.0f;
    float expected_stddev = sqrtf(beta_t);

    // These are statistical properties, so allow for some deviation
    EXPECT_NEAR(mean_gpu, expected_mean, 0.1f) << "Mean of output is too far from expected.";
    EXPECT_NEAR(stddev_gpu, expected_stddev, 0.1f) << "Stddev of output is too far from expected.";
}

TEST_F(DDPMKernelsTest, SimplifiedReverseDiffusionStep) {
    // First, create some noisy data x_t using the forward step
    float beta_t_forward = 0.01f;
    launch_forward_diffusion_step(d_x_in, d_x_out, beta_t_forward, d_states, N, threads_per_block, num_blocks);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // d_x_out now contains x_t (noisy data)

    // Parameters for reverse step
    float alpha_t = 1.0f - beta_t_forward; // Corresponds to the beta_t used for forward
    // For one_minus_alpha_bar_t_sqrt, let's assume this is the first step, so alpha_bar_t = alpha_t
    float alpha_bar_t = alpha_t; 
    float one_minus_alpha_bar_t_sqrt = sqrtf(1.0f - alpha_bar_t);
    if (one_minus_alpha_bar_t_sqrt < 1e-6f) one_minus_alpha_bar_t_sqrt = 1e-6f;

    float sigma_t = sqrtf(beta_t_forward); // Stochastic reverse step
    // float sigma_t = 0.0f; // Deterministic reverse step

    // For epsilon_predicted, use the kernel's internal placeholder by passing nullptr to d_epsilon_pred
    // The kernel will use x_t[idx] * 0.1f as a placeholder.
    // d_x_in will store x_{t-1} (denoised data)
    launch_simplified_reverse_diffusion_step(
        d_x_out, d_x_in, 
        nullptr, // d_epsilon_predicted
        alpha_t, one_minus_alpha_bar_t_sqrt, sigma_t,
        d_states, N, threads_per_block, num_blocks
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(h_x_out_gpu.data(), d_x_in, N * sizeof(float), cudaMemcpyDeviceToHost)); // Store result in h_x_out_gpu

    // Basic checks:
    // 1. Output should be somewhat different from the noisy input.
    // 2. Values should be finite.
    // A more rigorous test would involve providing a known epsilon_predicted and checking if the math holds.
    // With the current placeholder epsilon, we expect the output to be "less noisy" than d_x_out,
    // meaning its variance might be smaller or mean closer to original if placeholder is somewhat helpful.

    std::vector<float> h_x_t_noisy(N);
    CHECK_CUDA_ERROR(cudaMemcpy(h_x_t_noisy.data(), d_x_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    bool changed = false;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(h_x_out_gpu[i] - h_x_t_noisy[i]) > 1e-6) {
            changed = true;
            break;
        }
    }
    ASSERT_TRUE(changed) << "Output data did not change after reverse diffusion.";

    ASSERT_TRUE(std::all_of(h_x_out_gpu.begin(), h_x_out_gpu.end(), [](float val){
        return std::isfinite(val);
    })) << "Reversed output contains non-finite values.";

    // If the simplified reverse step is somewhat effective, the mean should move back towards original mean (1.0)
    // and stddev might decrease compared to h_x_t_noisy.
    double sum_noisy = 0.0; for(float val : h_x_t_noisy) sum_noisy += val;
    double mean_noisy = sum_noisy / N;
    double sq_sum_diff_noisy = 0.0; for(float val : h_x_t_noisy) sq_sum_diff_noisy += (val - mean_noisy) * (val - mean_noisy);
    double stddev_noisy = std::sqrt(sq_sum_diff_noisy / N);

    double sum_reversed = 0.0; for(float val : h_x_out_gpu) sum_reversed += val;
    double mean_reversed = sum_reversed / N;
    double sq_sum_diff_reversed = 0.0; for(float val : h_x_out_gpu) sq_sum_diff_reversed += (val - mean_reversed) * (val - mean_reversed);
    double stddev_reversed = std::sqrt(sq_sum_diff_reversed / N);
    
    // This is a very weak test due to the placeholder epsilon.
    // We expect the mean to be closer to 1.0 than mean_noisy was.
    EXPECT_LT(std::abs(mean_reversed - 1.0f), std::abs(mean_noisy - 1.0f) + 0.1f) << "Mean of reversed data didn't get closer to original mean.";
    // We expect stddev_reversed to be less than or similar to stddev_noisy.
    EXPECT_LT(stddev_reversed, stddev_noisy + 0.1f) << "Stddev of reversed data didn't decrease or stay similar.";

}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
