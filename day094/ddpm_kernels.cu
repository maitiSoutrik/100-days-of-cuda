#include "ddpm_kernels.cuh"
#include <cmath> // For sqrtf, etc.

// --- cuRAND State Initialization ---
__global__ void setup_curand_states_kernel(curandState *states, unsigned long seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Initialize each state with a unique sequence and offset
        curand_init(seed, idx, 0, &states[idx]);
    }
}

void launch_setup_curand_states(curandState *d_states, unsigned long seed, int N,
                                dim3 threads_per_block, dim3 num_blocks) {
    setup_curand_states_kernel<<<num_blocks, threads_per_block>>>(d_states, seed, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


// --- Forward Diffusion Step ---
__global__ void forward_diffusion_step_kernel(
    float* x_t_minus_1,
    float* x_t,
    float beta_t,
    curandState* states,
    int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Generate Gaussian noise epsilon ~ N(0, 1)
        float epsilon = curand_normal(&states[idx]);

        float sqrt_one_minus_beta_t = sqrtf(1.0f - beta_t);
        float sqrt_beta_t = sqrtf(beta_t);

        x_t[idx] = sqrt_one_minus_beta_t * x_t_minus_1[idx] + sqrt_beta_t * epsilon;
    }
}

void launch_forward_diffusion_step(
    float* d_x_t_minus_1,
    float* d_x_t,
    float beta_t,
    curandState* d_states,
    int N,
    dim3 threads_per_block,
    dim3 num_blocks) {

    forward_diffusion_step_kernel<<<num_blocks, threads_per_block>>>(
        d_x_t_minus_1, d_x_t, beta_t, d_states, N
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    // No cudaDeviceSynchronize here, allow async execution if caller desires
}

// --- Simplified Reverse Diffusion Step ---
__global__ void simplified_reverse_diffusion_step_kernel(
    float* x_t,
    float* x_t_minus_1,
    float* epsilon_predicted, // This could be d_x_t itself if we assume epsilon_predicted = x_t * some_factor
    float alpha_t,
    float one_minus_alpha_bar_t_sqrt, // sqrt(1 - alpha_bar_t)
    float sigma_t,
    curandState* states,
    int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float pred_noise_term;
        if (epsilon_predicted != nullptr) {
             pred_noise_term = ((1.0f - alpha_t) / one_minus_alpha_bar_t_sqrt) * epsilon_predicted[idx];
        } else {
            // Extremely simplified placeholder if no epsilon_predicted is provided:
            // Assume epsilon_predicted is proportional to x_t. This is NOT how DDPMs work,
            // but serves as a structural placeholder.
            // A real DDPM would have a neural net predict epsilon.
            // For this example, let's assume epsilon_predicted is just a fraction of x_t
            // This is a gross simplification for kernel structure demonstration.
            pred_noise_term = ((1.0f - alpha_t) / one_minus_alpha_bar_t_sqrt) * (x_t[idx] * 0.1f); // Example: 10% of x_t
        }


        float term1 = (1.0f / sqrtf(alpha_t)) * (x_t[idx] - pred_noise_term);
        
        float z = 0.0f;
        if (sigma_t > 0.0f && states != nullptr) {
            z = curand_normal(&states[idx]);
        }
        
        x_t_minus_1[idx] = term1 + sigma_t * z;
    }
}

void launch_simplified_reverse_diffusion_step(
    float* d_x_t,
    float* d_x_t_minus_1,
    float* d_epsilon_predicted, // Can be nullptr for very basic test
    float alpha_t,
    float one_minus_alpha_bar_t_sqrt,
    float sigma_t,
    curandState* d_states, // Can be nullptr if sigma_t is 0
    int N,
    dim3 threads_per_block,
    dim3 num_blocks) {

    simplified_reverse_diffusion_step_kernel<<<num_blocks, threads_per_block>>>(
        d_x_t, d_x_t_minus_1, d_epsilon_predicted,
        alpha_t, one_minus_alpha_bar_t_sqrt, sigma_t,
        d_states, N
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    // No cudaDeviceSynchronize here
}
