#include "ddpm_kernels.cuh"
#include <iostream>
#include <vector>
#include <numeric> // For iota
#include <algorithm> // For generate, transform
#include <cmath> // For M_PI, cos, sin
#include <iomanip> // For std::fixed, std::setprecision

// Helper function to print a vector (or a portion of it)
void print_vector(const std::string& name, const std::vector<float>& vec, int count = 10) {
    std::cout << name << ": [";
    for (int i = 0; i < std::min((int)vec.size(), count); ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i] << (i == std::min((int)vec.size(), count) - 1 ? "" : ", ");
    }
    if (vec.size() > count) {
        std::cout << "...";
    }
    std::cout << "]" << std::endl;
}

// Helper function to calculate mean and stddev of a vector
void print_stats(const std::string& name, const std::vector<float>& vec) {
    if (vec.empty()) {
        std::cout << name << ": (empty)" << std::endl;
        return;
    }
    double sum = 0.0;
    for (float val : vec) {
        sum += val;
    }
    double mean = sum / vec.size();

    double sq_sum_diff = 0.0;
    for (float val : vec) {
        sq_sum_diff += (val - mean) * (val - mean);
    }
    double stddev = std::sqrt(sq_sum_diff / vec.size());

    std::cout << name << " Stats: Mean = " << std::fixed << std::setprecision(4) << mean
              << ", StdDev = " << std::fixed << std::setprecision(4) << stddev << std::endl;
}


int main() {
    // --- Parameters ---
    const int N = 1024 * 1024; // Number of data points (e.g., pixels in a large image)
    const int T_total_steps = 1000; // Total diffusion steps in a typical DDPM
    const float beta_min = 0.0001f;
    const float beta_max = 0.02f;

    // --- Host Data Initialization ---
    std::vector<float> h_x0(N);
    // Simple initial data: a sine wave for example
    for (int i = 0; i < N; ++i) {
        h_x0[i] = sinf(2.0f * M_PI * (float)i / (N / 16.0f)); // 16 cycles
    }
    std::vector<float> h_x_current = h_x0; // This will be x_t at various t
    std::vector<float> h_x_t_output(N);
    std::vector<float> h_epsilon_predicted(N); // For reverse step

    print_vector("Initial Data (h_x0)", h_x0);
    print_stats("Initial Data (h_x0)", h_x0);

    // --- Device Data Allocation ---
    float *d_x_current, *d_x_t_output, *d_epsilon_predicted;
    curandState *d_states;

    CHECK_CUDA_ERROR(cudaMalloc(&d_x_current, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x_t_output, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_epsilon_predicted, N * sizeof(float))); // For reverse step
    CHECK_CUDA_ERROR(cudaMalloc(&d_states, N * sizeof(curandState)));

    // --- CUDA Kernel Launch Configuration ---
    const int threads_per_block_val = 256;
    dim3 threads_per_block(threads_per_block_val);
    dim3 num_blocks((N + threads_per_block_val - 1) / threads_per_block_val);

    // Initialize cuRAND states
    unsigned long seed = 1234UL;
    launch_setup_curand_states(d_states, seed, N, threads_per_block, num_blocks);
    std::cout << "\n--- cuRAND States Initialized ---\n" << std::endl;

    // --- Simulate Forward Diffusion for a few steps ---
    std::cout << "--- Forward Diffusion Simulation ---" << std::endl;
    int num_forward_steps_to_simulate = 5;
    float current_beta;

    // Copy initial data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_x_current, h_x_current.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    for (int t = 1; t <= num_forward_steps_to_simulate; ++t) {
        // Linear beta schedule: beta_t = beta_min + (beta_max - beta_min) * t / T_total_steps
        current_beta = beta_min + (beta_max - beta_min) * (float)t / T_total_steps;
        
        launch_forward_diffusion_step(d_x_current, d_x_t_output, current_beta, d_states, N, threads_per_block, num_blocks);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Sync after each step for accurate timing/stats if needed

        // Copy output back to d_x_current for next iteration
        CHECK_CUDA_ERROR(cudaMemcpy(d_x_current, d_x_t_output, N * sizeof(float), cudaMemcpyDeviceToDevice));
        
        // Copy to host for printing stats
        CHECK_CUDA_ERROR(cudaMemcpy(h_x_t_output.data(), d_x_t_output, N * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "Forward Step t=" << t << " (beta_t=" << std::fixed << std::setprecision(5) << current_beta << "):" << std::endl;
        print_vector("  Data (h_x_t)", h_x_t_output);
        print_stats("  Data (h_x_t)", h_x_t_output);
    }
    
    // At this point, d_x_current (and h_x_t_output) holds x_t after num_forward_steps_to_simulate

    // --- Simulate Simplified Reverse Diffusion for a few steps ---
    std::cout << "\n--- Simplified Reverse Diffusion Simulation ---" << std::endl;
    // Let's assume d_x_current is our x_t from the forward pass. We want to get x_{t-1}, x_{t-2}, ...
    // We need alpha_t and alpha_bar_t values.
    // For simplicity, we'll use the parameters from the last forward step.
    // A real DDPM would iterate T times from pure noise. Here we just reverse a few steps.

    // For the reverse step, we need a "predicted noise".
    // In a real DDPM, a neural net predicts this. Here, we'll use a placeholder.
    // Option 1: Epsilon_predicted is zero (model perfectly denoises what it can, ignores stochastic part)
    // Option 2: Epsilon_predicted is some fraction of x_t (as in the kernel's nullptr fallback)
    // Option 3: Epsilon_predicted is the actual noise added in the last step (oracle, for testing)
    
    // For this demo, let's use the kernel's internal placeholder for epsilon_predicted by passing nullptr.
    // (The kernel uses x_t[idx] * 0.1f as a placeholder for epsilon_predicted[idx])

    float alpha_t, one_minus_alpha_bar_t_sqrt, sigma_t;
    std::vector<float> h_x_reverse_output(N);

    // Reverse the steps we just did
    for (int t = num_forward_steps_to_simulate; t >= 1; --t) {
        current_beta = beta_min + (beta_max - beta_min) * (float)t / T_total_steps;
        alpha_t = 1.0f - current_beta;
        
        // Calculate alpha_bar_t. For this simple demo, we'll approximate or use a fixed value.
        // A proper implementation would precompute all alpha_bar_t values.
        // Let's assume alpha_bar_t for step t.
        float alpha_bar_t_approx = 1.0f; // Placeholder, this needs to be calculated based on product of alphas
        for(int s=1; s<=t; ++s) {
            float beta_s = beta_min + (beta_max - beta_min) * (float)s / T_total_steps;
            alpha_bar_t_approx *= (1.0f - beta_s);
        }
        one_minus_alpha_bar_t_sqrt = sqrtf(1.0f - alpha_bar_t_approx);
        if (one_minus_alpha_bar_t_sqrt < 1e-6f) one_minus_alpha_bar_t_sqrt = 1e-6f; // Avoid division by zero

        // Sigma_t can be sqrt(beta_t) or a more complex term. Let's use sqrt(beta_t) for stochasticity.
        sigma_t = sqrtf(current_beta); 
        // sigma_t = 0.0f; // For deterministic reverse step based on epsilon_predicted only

        // d_x_current holds x_t, we want to compute x_{t-1} into d_x_t_output
        launch_simplified_reverse_diffusion_step(
            d_x_current, d_x_t_output, 
            nullptr, // d_epsilon_predicted (using kernel's internal placeholder)
            alpha_t, one_minus_alpha_bar_t_sqrt, sigma_t,
            d_states, // Pass cuRAND states for stochasticity if sigma_t > 0
            N, threads_per_block, num_blocks
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy output back to d_x_current for next iteration
        CHECK_CUDA_ERROR(cudaMemcpy(d_x_current, d_x_t_output, N * sizeof(float), cudaMemcpyDeviceToDevice));

        CHECK_CUDA_ERROR(cudaMemcpy(h_x_reverse_output.data(), d_x_t_output, N * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "Reverse Step, target t=" << t-1 << " (from t=" << t << "):" << std::endl;
        print_vector("  Data (h_x_reverse)", h_x_reverse_output);
        print_stats("  Data (h_x_reverse)", h_x_reverse_output);
    }


    // --- Cleanup ---
    CHECK_CUDA_ERROR(cudaFree(d_x_current));
    CHECK_CUDA_ERROR(cudaFree(d_x_t_output));
    CHECK_CUDA_ERROR(cudaFree(d_epsilon_predicted));
    CHECK_CUDA_ERROR(cudaFree(d_states));

    std::cout << "\n--- DDPM Demo Finished ---" << std::endl;
    return 0;
}
